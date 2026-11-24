#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include "../../include/strata_glm53.h"
#include "../../include/strata_embedded_shaders.h"
#import "../../include/strata_metal_cache.h"

#include <fcntl.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <mach/mach_time.h>
#include <os/lock.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/* Unique Objective-C class name so linking alongside `strata_ane_probe.o` has zero symbol collisions */
@interface StrataGlm53AneFeatureProvider : NSObject <MLFeatureProvider>
@property (nonatomic, strong, readonly) MLFeatureValue* xValue;
@property (nonatomic, strong, readonly) NSSet<NSString*>* names;
- (instancetype)initWithMultiArray:(MLMultiArray*)multiArray;
@end

@implementation StrataGlm53AneFeatureProvider
- (instancetype)initWithMultiArray:(MLMultiArray*)multiArray {
    if ((self = [super init])) {
        _xValue = [MLFeatureValue featureValueWithMultiArray:multiArray];
        _names  = [NSSet setWithObject:@"x"];
    }
    return self;
}
- (NSSet<NSString*>*)featureNames {
    return _names;
}
- (nullable MLFeatureValue*)featureValueForName:(NSString*)featureName {
    if ([featureName isEqualToString:@"x"]) {
        return _xValue;
    }
    return nil;
}
@end

namespace {

constexpr uint32_t kMaxPrefillChunk    = 512U; // 8x M_TILE=64 for Metal 4.1 NAX + bounded workspace!
constexpr uint32_t kMaxSpecBatch       = 8U;   // Supports M = 1 .. 5 (`draft_k = 0 .. 4`)
constexpr uint32_t kMaxDecodeSplits    = 16U;
constexpr uint32_t kMaxMoeBinTiles     = 1024U;
constexpr uint32_t kNgramHashSize      = 65536U;
constexpr uint32_t kAneSpatialAlign    = 16U;  // Physical 16-core ANE spatial width S=16
constexpr uint64_t kCoreMLSpecFP16     = 7ULL;
constexpr uint64_t kCoreMLArrayFloat16 = 65552ULL;
constexpr uint64_t kNvmeStripeChunk    = 1572864ULL; // 1.50 MiB = 6 * 256 KiB (`14,155,776 / 9`, exact!)

constexpr uint32_t kOutF16    = 0U;
constexpr uint32_t kOutF32    = 1U;

/* ---- Host <-> Metal 4 Parameter Struct Mirrors ---- */

struct Sg53EmbedParamsHost {
    uint32_t M;
    uint32_t cols;
    uint32_t vocab_size;
    uint32_t hc_tile_count;
    uint32_t pos_start;
    uint32_t zero_at_pos0;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53GemvBatchParamsHost {
    uint32_t M;
    uint32_t rows;
    uint32_t cols;
    uint32_t out_mode;
};

struct Sg53NaxGemmParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t N_logical;
    uint32_t N_padded;
    uint32_t K;
    uint32_t out_mode;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53QuantI8ParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t K;
    uint32_t pad0;
};

struct Sg53RmsNormParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t dim;
    uint32_t src_stride;
    uint32_t src_col_off;
    uint32_t dst_f16_stride;
    uint32_t dst_f16_col_off;
    float    eps;
    uint32_t write_f16;
    uint32_t write_f32;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53MhcNormParamsHost {
    uint32_t M_valid;
    uint32_t hc_dim;
    float    eps;
    uint32_t pad0;
};

struct Sg53MhcSinkhornParamsHost {
    uint32_t M_valid;
    uint32_t hc_mult;
    uint32_t sinkhorn_iters;
    float    hc_eps;
};

struct Sg53MhcCollapseParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t hc_mult;
    float    eps;
    uint32_t write_f32;
    uint32_t write_unit_f16;
    uint32_t pad1;
};

struct Sg53MhcFusedCollapseParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    uint32_t hc_mult;
    uint32_t sinkhorn_iters;
    float    hc_eps;
    float    norm_eps;
    uint32_t write_f32;
    uint32_t write_unit_f16;
    uint32_t split_k;
    uint32_t apply_hc_inv_rms;
    uint32_t write_x_rot_f16;
};

struct Sg53FrspecPatchParamsHost {
    uint32_t count;
    uint32_t cols;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53MhcExpandParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t hc_mult;
    uint32_t pad0;
};

struct Sg53ResAddParamsHost {
    uint32_t M_valid;
    uint32_t D;
};

struct Sg53HyperHeadParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    float    eps;
    uint32_t write_f16;
    uint32_t write_f32;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53KdaFbgSplitParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t nh;
    uint32_t hd;
};

struct Sg53KdaConvParamsHost {
    uint32_t M;
    uint32_t channels;
    uint32_t save_snap_mask;
    uint32_t snap_layer_idx;
    uint32_t num_kda_layers;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct Sg53KdaParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t nh;
    uint32_t hd;
    float    l2_eps;
    float    norm_eps;
    float    gate_lower_bound;
    uint32_t save_snap_mask;
    uint32_t snap_layer_idx;
    uint32_t num_kda_layers;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53KdaWyParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t num_chunks;
    uint32_t nh;
    float    l2_eps;
    float    gate_lower_bound;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53IndexerParamsHost {
    uint32_t M_valid;
    uint32_t idx_n_heads;
    uint32_t idx_head_dim;
    uint32_t index_kpool;
    uint32_t select_k_pools;
    uint32_t seq_pos_start;
    uint32_t first_pool;
    uint32_t num_new_pools;
    float    layernorm_eps;
    uint32_t max_pools_stride;
    uint32_t max_sel_words;
    uint32_t pad0;
};

struct Sg53MlaBatchedGemmParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t nh;
    uint32_t N;
    uint32_t K;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct Sg53MlaKvWriteParamsHost {
    uint32_t M_valid;
    uint32_t kvr;
    uint32_t src_stride;
    uint32_t src_col_off;
    uint32_t seq_pos_start;
    uint32_t paged;
    float    eps;
    uint32_t pad0;
};

struct Sg53MlaDecodeParamsHost {
    uint32_t M_valid;
    uint32_t nh;
    uint32_t kv_lora_rank;
    uint32_t q_per_tg;
    uint32_t seq_pos_start;
    uint32_t max_seq_alloc;
    uint32_t max_splits;
    uint32_t index_kpool;
    uint32_t select_k_pools;
    float    attn_scale;
    uint32_t m_base;
    uint32_t paged;
    uint32_t m_end;
    uint32_t M_padded;
    uint32_t total_kv;
    uint32_t pad0;
};

struct Sg53SwigluParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t d_ff;
    float    limit;
};

struct Sg53MoeRouterParamsHost {
    uint32_t M_valid;
    uint32_t D;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t norm_topk_prob;
    float    routed_scaling_factor;
    uint32_t n_group;
    uint32_t topk_group;
};

struct Sg53MoeIndirectParamsHost {
    uint64_t expert_record_bytes;
    uint64_t expert_proj_bytes;
    uint64_t expert_weight_bytes;
    uint64_t expert_scales_bytes;
    uint32_t M_valid;
    uint32_t top_k;
    uint32_t D;
    uint32_t d_moe;
    float    swiglu_limit;
    uint32_t num_tiles;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53MoeWorkTileHost {
    uint32_t source_buf;
    uint32_t slot_idx;
    uint32_t pair_start;
    uint32_t pair_count;
};

struct Sg53MoeCombineParamsHost {
    uint32_t M_valid;
    uint32_t top_k;
    uint32_t D;
    uint32_t pad0;
};

struct Sg53SampleParamsHost {
    uint32_t M;
    uint32_t vocab_size;
    uint32_t greedy;
    float    inv_temp;
    uint32_t top_k;
    float    top_p;
    uint32_t rng_lo;
    uint32_t rng_hi;
};

struct Sg53DraftSampleParamsHost {
    uint32_t M;
    uint32_t vocab_size;
    uint32_t greedy;
    float    inv_temp;
    uint32_t top_k;
    float    top_p;
    uint32_t rng_lo;
    uint32_t rng_hi;
    uint32_t use_token_map;
    uint32_t want_pmax;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53Rht64ParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t K;
    uint32_t pad0;
};

struct Sg53MoeV2WorkTileHost {
    uint32_t expert_id;
    uint32_t pair_start;
    uint32_t pair_count;
    uint32_t pad0;
};

struct Sg53MoeBinParamsHost {
    uint32_t M_valid;
    uint32_t top_k;
    uint32_t num_experts;
    uint32_t d_moe;
    uint32_t D;
    uint32_t max_tiles_per_bin;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53MoeE8ParamsHost {
    uint64_t layer_byte_off;
    uint64_t e8_record_bytes;
    uint64_t e8_proj_bytes;
    uint64_t e8_lo16_bytes;
    uint64_t e8_hi2_bytes;
    uint32_t M_valid;
    uint32_t top_k;
    uint32_t D;
    uint32_t d_moe;
    float    swiglu_limit;
    uint32_t total_pairs_pad;
    uint32_t tile_base;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sg53LogitPenaltyParamsHost {
    uint32_t M;
    uint32_t vocab_size;
    uint32_t num_hist_ids;
    uint32_t extra_count;
    uint32_t extra_mode;
    float    presence_penalty;
    float    frequency_penalty;
    float    repetition_penalty;
    uint32_t drop_toks[8];
};

/* ---- Timing, FP16, File & Lock Helpers ---- */

inline double sg53_ticks_to_sec(uint64_t ticks) {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return static_cast<double>(ticks) * static_cast<double>(tb.numer) / static_cast<double>(tb.denom) * 1e-9;
}

inline uint16_t sg53_fast_f32_to_fp16(float f) {
    __fp16 h = static_cast<__fp16>(f);
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

inline float sg53_fast_fp16_to_f32(uint16_t b) {
    __fp16 h;
    std::memcpy(&h, &b, sizeof(h));
    return static_cast<float>(h);
}

struct Sg53ScopedLock {
    os_unfair_lock* lk;
    explicit Sg53ScopedLock(os_unfair_lock* l) : lk(l) { os_unfair_lock_lock(lk); }
    ~Sg53ScopedLock() { os_unfair_lock_unlock(lk); }
};

struct Sg53ScopedUrlCleanup {
    NSURL* url = nil;
    ~Sg53ScopedUrlCleanup() { reset(); }
    void reset() {
        if (url != nil) {
            [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
            url = nil;
        }
    }
};

std::string sg53_read_text_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

std::string sg53_resolve_sibling_path(const char* rel_from_src_gpu) {
    std::string self_file = __FILE__;
    size_t slash = self_file.find_last_of('/');
    if (slash != std::string::npos) {
        std::string candidate = self_file.substr(0, slash + 1) + rel_from_src_gpu;
        std::ifstream test(candidate);
        if (test.good()) return candidate;
    }
    return std::string("src/gpu/") + rel_from_src_gpu;
}

void sg53_prefault_pages_rw(void* base, size_t bytes) {
    if (!base || bytes == 0U) return;
    constexpr size_t kPage = STRATA_GLM53_PAGE_BYTES;
    const size_t num_pages = (bytes + kPage - 1U) / kPage;
    auto* p = static_cast<volatile uint8_t*>(base);
    if (num_pages < 4096U) {
        for (size_t i = 0U; i < num_pages; ++i) {
            p[i * kPage] = 0U;
        }
        return;
    }
    const size_t workers = 8U;
    dispatch_apply(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
        const size_t beg = (num_pages * w) / workers;
        const size_t end = (num_pages * (w + 1U)) / workers;
        for (size_t i = beg; i < end; ++i) {
            p[i * kPage] = 0U;
        }
    });
}

void sg53_prefault_pages_ro(const void* base, size_t bytes) {
    if (!base || bytes == 0U) return;
    constexpr size_t kPage = STRATA_GLM53_PAGE_BYTES;
    const size_t num_pages = (bytes + kPage - 1U) / kPage;
    const auto* p = static_cast<const volatile uint8_t*>(base);
    if (num_pages < 4096U) {
        uint8_t sink = 0U;
        for (size_t i = 0U; i < num_pages; ++i) sink ^= p[i * kPage];
        (void)sink;
        return;
    }
    const size_t workers = 8U;
    dispatch_apply(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
        uint8_t sink = 0U;
        const size_t beg = (num_pages * w) / workers;
        const size_t end = (num_pages * (w + 1U)) / workers;
        for (size_t i = beg; i < end; ++i) {
            sink ^= p[i * kPage];
        }
        (void)sink;
    });
}

id<MTLBuffer> sg53_alloc_shared_prefaulted(id<MTLDevice> dev, size_t bytes) {
    const size_t aligned = std::max<size_t>(STRATA_GLM53_PAGE_BYTES,
                                            (bytes + STRATA_GLM53_PAGE_BYTES - 1U) & ~(STRATA_GLM53_PAGE_BYTES - 1U));
    id<MTLBuffer> buf = [dev newBufferWithLength:aligned options:MTLResourceStorageModeShared];
    if (buf) {
        if (aligned <= (64ULL << 20)) {
            std::memset([buf contents], 0, aligned);
        } else {
            sg53_prefault_pages_rw([buf contents], aligned);
        }
    }
    return buf;
}

id<MTLBuffer> sg53_alloc_private_aligned(id<MTLDevice> dev, size_t bytes) {
    const size_t aligned = std::max<size_t>(4096U, (bytes + 4095U) & ~static_cast<size_t>(4095U));
    return [dev newBufferWithLength:aligned options:MTLResourceStorageModePrivate];
}

void sg53_encode_zero_buffer(
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
    const uint32_t tg = 256U;
    [enc dispatchThreadgroups:MTLSizeMake((count_u4 + tg - 1U) / tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void sg53_encode_copy_buffer(
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
    const uint32_t tg = 256U;
    [enc dispatchThreadgroups:MTLSizeMake((count_u4 + tg - 1U) / tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

id<MTLResidencySet> sg53_create_residency_set(
    id<MTLDevice>           dev,
    id<MTLCommandQueue>     queue,
    NSArray<id<MTLBuffer>>* buffers
) {
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor* desc = [[MTLResidencySetDescriptor alloc] init];
        desc.initialCapacity = std::max<NSUInteger>(buffers.count + 8U, 1U);
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

void sg53_detach_residency_set(id<MTLCommandQueue> queue, id<MTLResidencySet> rset) {
    if (!rset) return;
    if (@available(macOS 15.0, *)) {
        if (queue) [queue removeResidencySet:rset];
        [rset endResidency];
    }
}

uint64_t sg53_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30ULL)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27ULL)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31ULL);
}

float sg53_host_rand_u01(uint64_t seed, uint64_t step) {
    uint64_t z   = sg53_splitmix64(seed + step * 0x9E3779B97F4A7C15ULL);
    uint32_t u24 = static_cast<uint32_t>(z >> 40ULL) & 0xFFFFFFU;
    return (static_cast<float>(u24) + 0.5f) * (1.0f / 16777216.0f);
}

float sg53_lookup_sparse_prob(
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

/* ============================================================================
 * Pure C++20 CoreML Spec-v7 Protobuf Builder for 16-Core `ANE` (`MLComputePlan`)
 * Multi-Horizon Router Lookahead (`W_ane[sub, e, c] = W_gate[e, c] * post_norm[c]`)
 * ============================================================================ */

struct Sg53ProtobufWriter {
    std::vector<uint8_t> buf;

    void varint(uint64_t v) {
        while (v >= 0x80ULL) {
            buf.push_back(static_cast<uint8_t>((v & 0x7FULL) | 0x80ULL));
            v >>= 7U;
        }
        buf.push_back(static_cast<uint8_t>(v));
    }
    void tag(uint32_t field_num, uint32_t wire_type) {
        varint((static_cast<uint64_t>(field_num) << 3U) | static_cast<uint64_t>(wire_type));
    }
    void u64_field(uint32_t field_num, uint64_t val) {
        tag(field_num, 0U);
        varint(val);
    }
    void bytes_field(uint32_t field_num, const void* data, size_t len) {
        tag(field_num, 2U);
        varint(static_cast<uint64_t>(len));
        const auto* p = static_cast<const uint8_t*>(data);
        buf.insert(buf.end(), p, p + len);
    }
    void str_field(uint32_t field_num, const char* s) {
        bytes_field(field_num, s, std::strlen(s));
    }
    void msg_field(uint32_t field_num, const Sg53ProtobufWriter& sub) {
        bytes_field(field_num, sub.buf.data(), sub.buf.size());
    }
};

Sg53ProtobufWriter sg53_pb_shape_4d(uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    Sg53ProtobufWriter t;
    t.u64_field(1U, 4ULL);
    t.u64_field(2U, n);
    t.u64_field(2U, c);
    t.u64_field(2U, h);
    t.u64_field(2U, w);
    return t;
}

Sg53ProtobufWriter sg53_pb_feat_fp16_4d(const char* name, uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    Sg53ProtobufWriter aft;
    aft.u64_field(1U, n);
    aft.u64_field(1U, c);
    aft.u64_field(1U, h);
    aft.u64_field(1U, w);
    aft.u64_field(2U, kCoreMLArrayFloat16);

    Sg53ProtobufWriter ft;
    ft.msg_field(5U, aft);

    Sg53ProtobufWriter fd;
    fd.str_field(1U, name);
    fd.msg_field(3U, ft);
    return fd;
}

std::vector<uint8_t> sg53_serialize_ane_router_mlmodel(
    uint32_t        d_pad,
    uint32_t        out_pad,
    uint32_t        spatial_s,
    const uint16_t* w_fp16_rowmajor
) {
    Sg53ProtobufWriter wp;
    wp.bytes_field(2U, w_fp16_rowmajor, static_cast<size_t>(out_pad) * d_pad * sizeof(uint16_t));

    Sg53ProtobufWriter border;
    Sg53ProtobufWriter valid_pad;
    valid_pad.msg_field(1U, border);

    Sg53ProtobufWriter conv;
    conv.u64_field(1U, out_pad);
    conv.u64_field(2U, d_pad);
    conv.u64_field(10U, 1ULL);
    conv.u64_field(20U, 1ULL);
    conv.u64_field(20U, 1ULL);
    conv.u64_field(30U, 1ULL);
    conv.u64_field(30U, 1ULL);
    conv.u64_field(40U, 1ULL);
    conv.u64_field(40U, 1ULL);
    conv.msg_field(50U, valid_pad);
    conv.u64_field(60U, 0ULL);
    conv.u64_field(70U, 0ULL);
    conv.msg_field(90U, wp);

    Sg53ProtobufWriter layer;
    layer.str_field(1U, "router_conv");
    layer.str_field(2U, "x");
    layer.str_field(3U, "y");
    layer.msg_field(4U, sg53_pb_shape_4d(1U, d_pad, 1U, spatial_s));
    layer.msg_field(5U, sg53_pb_shape_4d(1U, out_pad, 1U, spatial_s));
    layer.msg_field(100U, conv);

    Sg53ProtobufWriter desc;
    desc.msg_field(1U, sg53_pb_feat_fp16_4d("x", 1U, d_pad, 1U, spatial_s));
    desc.msg_field(10U, sg53_pb_feat_fp16_4d("y", 1U, out_pad, 1U, spatial_s));

    Sg53ProtobufWriter nn;
    nn.msg_field(1U, layer);
    nn.u64_field(5U, 1ULL); // EXACT_ARRAY_MAPPING

    Sg53ProtobufWriter model;
    model.u64_field(1U, kCoreMLSpecFP16);
    model.msg_field(2U, desc);
    model.msg_field(500U, nn);
    return model.buf;
}

struct Sg53AneWindow {
    os_unfair_lock                 lock = OS_UNFAIR_LOCK_INIT;
    uint32_t                       moe_layer_beg = 0U;
    uint32_t                       num_layers    = 0U; // 1..4
    uint32_t                       d_model       = 0U;
    uint32_t                       d_pad         = 0U;
    uint32_t                       num_experts   = 0U;
    uint32_t                       out_channels  = 0U;
    uint32_t                       out_pad       = 0U;

    MLModel*                       coreml_model  = nil;
    MLMultiArray*                  in_array      = nil;
    MLMultiArray*                  out_array     = nil;
    StrataGlm53AneFeatureProvider* feat_provider = nil;
    MLPredictionOptions*           pred_options  = nil;

    uint16_t*                      in_fp16_buf   = nullptr;
    uint16_t*                      out_fp16_buf  = nullptr;
    bool                           verified_ane  = false;

    std::vector<uint16_t>          cpu_w_fp16;   // Fallback if sandbox blocks IOKit
    std::vector<float>             e_biases;     // [num_layers * num_experts]

    Sg53AneWindow() = default;
    Sg53AneWindow(const Sg53AneWindow&) = delete;
    Sg53AneWindow& operator=(const Sg53AneWindow&) = delete;
    Sg53AneWindow(Sg53AneWindow&& o) noexcept
        : moe_layer_beg(o.moe_layer_beg),
          num_layers(o.num_layers),
          d_model(o.d_model),
          d_pad(o.d_pad),
          num_experts(o.num_experts),
          out_channels(o.out_channels),
          out_pad(o.out_pad),
          coreml_model(o.coreml_model),
          in_array(o.in_array),
          out_array(o.out_array),
          feat_provider(o.feat_provider),
          pred_options(o.pred_options),
          in_fp16_buf(o.in_fp16_buf),
          out_fp16_buf(o.out_fp16_buf),
          verified_ane(o.verified_ane),
          cpu_w_fp16(std::move(o.cpu_w_fp16)),
          e_biases(std::move(o.e_biases)) {
        o.coreml_model  = nil;
        o.in_array      = nil;
        o.out_array     = nil;
        o.feat_provider = nil;
        o.pred_options  = nil;
        o.in_fp16_buf   = nullptr;
        o.out_fp16_buf  = nullptr;
    }
    ~Sg53AneWindow() {
        pred_options  = nil;
        feat_provider = nil;
        in_array      = nil;
        out_array     = nil;
        coreml_model  = nil;
        if (in_fp16_buf)  { std::free(in_fp16_buf);  in_fp16_buf  = nullptr; }
        if (out_fp16_buf) { std::free(out_fp16_buf); out_fp16_buf = nullptr; }
    }
};

struct Sg53LayerHotState {
    uint32_t              base_slot  = 0U;
    uint32_t              cap_slots  = 0U;
    uint32_t              used_slots = 0U;
    uint32_t              clock_hand = 0U;
    std::vector<int32_t>  slot_of_e;  // [E] -> -1 or local slot `0 .. cap_slots - 1`
    std::vector<int32_t>  e_of_slot;  // [cap_slots] -> -1 or `0 .. E - 1`
    std::vector<uint32_t> freq_of_e;  // [E] -> LFU touch count
};

struct Sg53SpecLayerSnap {
    uint32_t          moe_l = 0U;
    Sg53LayerHotState snap[kMaxSpecBatch];
    uint64_t          promotions_cum[kMaxSpecBatch] = {};
};

} // namespace

struct StrataGlm53ContextSlot;
struct Sg53AneKvProbe;

struct StrataGlm53GpuEngine {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLResidencySet>         residency_set;
    bool                        has_nax;

    // Metal 4.1 Compute Pipeline States (v1 Regime II + v2 RAM-Resident E8S-18-V4 + NAX Prefill)
    id<MTLComputePipelineState> pso_zero_u32;
    id<MTLComputePipelineState> pso_copy_u32;
    id<MTLComputePipelineState> pso_embed_q4;
    id<MTLComputePipelineState> pso_q4_gemv_batch;
    id<MTLComputePipelineState> pso_f16_gemv_batch;
    id<MTLComputePipelineState> pso_f32_gemv_batch;
    id<MTLComputePipelineState> pso_q4_gemv_spec[5];
    id<MTLComputePipelineState> pso_q4_gemv_k128_spec[5];
    id<MTLComputePipelineState> pso_f16_gemv_spec[5];
    id<MTLComputePipelineState> pso_f32_gemv_splitk4_spec[5];
    id<MTLComputePipelineState> pso_mla_batched_gemv_spec[5];
    id<MTLComputePipelineState> pso_q4_gemm_nax_m16;
    id<MTLComputePipelineState> pso_q4_gemm_nax_m64;
    id<MTLComputePipelineState> pso_q4_gemm_nax_bk64_m16;
    id<MTLComputePipelineState> pso_f16_gemm_nax_m16;
    id<MTLComputePipelineState> pso_f16_gemm_nax_m64;
    id<MTLComputePipelineState> pso_quant_f16_to_i8_row;
    id<MTLComputePipelineState> pso_q4_gemm_nax_i8_m64;
    id<MTLComputePipelineState> pso_q4_gemm_nax_i8_m128;
    id<MTLComputePipelineState> pso_rmsnorm_strided;
    id<MTLComputePipelineState> pso_mhc_norm;
    id<MTLComputePipelineState> pso_mhc_sinkhorn;
    id<MTLComputePipelineState> pso_mhc_collapse_norm;
    id<MTLComputePipelineState> pso_mhc_sinkhorn_collapse;
    id<MTLComputePipelineState> pso_mhc_expand;
    id<MTLComputePipelineState> pso_single_res_add;
    id<MTLComputePipelineState> pso_hyper_head_norm;
    id<MTLComputePipelineState> pso_kda_fbg_split;
    id<MTLComputePipelineState> pso_kda_conv1d;
    id<MTLComputePipelineState> pso_kda_rec;
    id<MTLComputePipelineState> pso_kda_prep_batch;
    id<MTLComputePipelineState> pso_kda_rec_batch;
    id<MTLComputePipelineState> pso_kda_post_batch;
    id<MTLComputePipelineState> pso_kda_wy_prep_chunk32;
    id<MTLComputePipelineState> pso_kda_wy_recur_chunk32;
    id<MTLComputePipelineState> pso_idx_k_layernorm;
    id<MTLComputePipelineState> pso_idx_update_pools;
    id<MTLComputePipelineState> pso_idx_commit_tail;
    id<MTLComputePipelineState> pso_idx_scale_weights;
    id<MTLComputePipelineState> pso_idx_score_pools;
    id<MTLComputePipelineState> pso_idx_score_tiled;
    id<MTLComputePipelineState> pso_idx_q_to_hm;
    id<MTLComputePipelineState> pso_idx_score_nax;
    id<MTLComputePipelineState> pso_idx_score_decode_nax;
    id<MTLComputePipelineState> pso_idx_topk_pools;
    id<MTLComputePipelineState> pso_mla_batched_gemv;
    id<MTLComputePipelineState> pso_mla_batched_nax_m16;
    id<MTLComputePipelineState> pso_mla_batched_nax_m64;
    id<MTLComputePipelineState> pso_mla_q_to_hm;
    id<MTLComputePipelineState> pso_mla_batched_nax_hm_m16;
    id<MTLComputePipelineState> pso_mla_batched_nax_hm_m64;
    id<MTLComputePipelineState> pso_mla_kv_write;
    id<MTLComputePipelineState> pso_mla_splitkv;
    id<MTLComputePipelineState> pso_mla_decode_splitkv_nax;
    id<MTLComputePipelineState> pso_mla_reduce;
    id<MTLComputePipelineState> pso_mla_prefill_dense_nax;
    id<MTLComputePipelineState> pso_mla_prefill_sparse_nax;
    id<MTLComputePipelineState> pso_limited_swiglu;
    id<MTLComputePipelineState> pso_rht64_f16;
    id<MTLComputePipelineState> pso_limited_swiglu_rht64;
    id<MTLComputePipelineState> pso_moe_router_topk;
    id<MTLComputePipelineState> pso_moe_router_logits;
    id<MTLComputePipelineState> pso_moe_router_select;
    id<MTLComputePipelineState> pso_moe_gather_gu;
    id<MTLComputePipelineState> pso_moe_gather_dn;
    id<MTLComputePipelineState> pso_moe_grouped_gu;
    id<MTLComputePipelineState> pso_moe_grouped_dn;
    id<MTLComputePipelineState> pso_moe_combine;
    id<MTLComputePipelineState> pso_moe_bin_v2;
    id<MTLComputePipelineState> pso_moe_bin_nax;
    id<MTLComputePipelineState> pso_moe_gather_x_f16;
    id<MTLComputePipelineState> pso_moe_combine_v2;
    id<MTLComputePipelineState> pso_moe_combine_mhc_expand;
    id<MTLComputePipelineState> pso_moe_e8s18_gather_gu;
    id<MTLComputePipelineState> pso_moe_e8s18_gather_gu_256;
    id<MTLComputePipelineState> pso_moe_e8s18_gather_dn;
    id<MTLComputePipelineState> pso_moe_e8s18_gather_dn_256;
    id<MTLComputePipelineState> pso_moe_e8s18_grouped_gu;
    id<MTLComputePipelineState> pso_moe_e8s18_grouped_dn;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_gu_m64;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_gu_m32;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_gu_m16;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_dn_m64;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_dn_m32;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_dn_m16;
    id<MTLComputePipelineState> pso_moe_gather_x_i8;
    id<MTLComputePipelineState> pso_moe_quant_mid_i8;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_i8_gu_m64;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_i8_gu_m32;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_i8_gu_m16;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_i8_dn_m64;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_i8_dn_m32;
    id<MTLComputePipelineState> pso_moe_e8s18_nax_i8_dn_m16;
    id<MTLComputePipelineState> pso_moe_q4_uma_gather_gu;
    id<MTLComputePipelineState> pso_moe_q4_uma_gather_dn;
    id<MTLComputePipelineState> pso_moe_q4_uma_grouped_gu;
    id<MTLComputePipelineState> pso_moe_q4_uma_grouped_dn;
    id<MTLComputePipelineState> pso_sample_vocab;
    id<MTLComputePipelineState> pso_sample_stage1;
    id<MTLComputePipelineState> pso_sample_stage2;
    id<MTLComputePipelineState> pso_apply_logit_penalties;
    id<MTLComputePipelineState> pso_sample_draft_stage1;
    id<MTLComputePipelineState> pso_sample_draft_stage2;
    id<MTLComputePipelineState> pso_apply_logit_penalties_shortlist;
    id<MTLComputePipelineState> pso_frspec_patch_rows;

    // Bundle & Header
    const StrataGlm53Bundle*    bundle;
    StrataGlm53SpineHeader      hdr;
    StrataGlm53Config           cfg;
    bool                        has_mtp;
    bool                        is_ram_resident;
    bool                        force_serial_kda;
    bool                        ngram_draft_enabled;
    bool                        int8_prefill_enabled = false;
    bool                        int8_moe_enabled     = true;
    bool                        adaptive_k_enabled   = true;
    bool                        frspec_enabled       = true;
    bool                        frspec_widened       = false;
    uint32_t                    frspec_shortlist_size = 0U;
    uint32_t                    frspec_dyn_base       = 0U;
    uint32_t                    frspec_dyn_cap        = 0U;
    uint32_t                    frspec_dyn_cursor     = 0U;
    uint32_t                    shortlist_miss_mask  = 0U;
    uint32_t                    metal_cache_hit      = 0U;
    double                      pso_load_ms          = 0.0;

    // 3-State SpecCtl Adaptive-K (`ActiveProbe` / `HighConfFused` / `CooldownK0`) + Async `cb_fix`
    enum SpecState {
        kSpecActiveProbe   = 0,
        kSpecHighConfFused = 1,
        kSpecCooldownK0    = 2
    };
    SpecState                   spec_state        = kSpecActiveProbe;
    uint32_t                    spec_low_streak   = 0U;
    uint32_t                    spec_high_streak  = 0U;
    uint32_t                    spec_cooldown_rem = 0U;
    uint32_t                    spec_locked_k     = 0U;
    double                      ema_acc_cal       = 0.90;
    double                      ema_cal_num       = 1.8;
    double                      ema_cal_den       = 2.0;
    double                      ema_geom_p        = 0.55;
    double                      ema_geom_num      = 1.1;
    double                      ema_geom_den      = 2.0;
    double                      ema_draft_ms      = 2.2;
    double                      ema_ver_ms[kMaxSpecBatch] = {20.5, 27.5, 33.5, 39.0, 44.0, 49.0, 54.0, 59.0};
    id<MTLCommandBuffer>        pending_fix_cb    = nil;

    // Runtime Mode & Adaptive Gating
    StrataGlm53ExecMode         exec_mode;
    float                       adaptive_gate_tau;
    uint32_t                    adaptive_cold_cap;
    bool                        adaptive_renorm;

    // Sequence & MTP State Tracking
    uint32_t                    max_seq_len;
    uint32_t                    max_seq_alloc;
    uint32_t                    max_pools_alloc;
    uint32_t                    max_sel_words;
    uint32_t                    seq_pos;
    uint32_t                    mtp_seq_pos;
    uint64_t                    sample_step;
    StrataGlm53PrefillProgressFn progress_cb;
    void*                       progress_user_ctx;

    // Disaggregated UMA Hot-Expert Cache (v1) & Zero-Copy RAM-Resident Dual-Buffer UMA (v2)
    id<MTLBuffer>               b_spine;
    id<MTLBuffer>               b_experts_part0;
    id<MTLBuffer>               b_experts_part1;
    id<MTLBuffer>               b_mtp_experts;
    id<MTLBuffer>               b_e8s18_lut;
    uint64_t                    uma_part0_mapped_bytes;
    uint64_t                    uma_part1_mapped_bytes;
    uint64_t                    mtp_experts_mapped_bytes;
    id<MTLBuffer>               b_hot_experts;
    id<MTLBuffer>               b_ring_dma;
    uint8_t*                    hot_experts_ptr;
    uint8_t*                    ring_dma_ptr;

    uint32_t                        hot_capacity_experts;
    uint64_t                        hot_cache_bytes;
    uint32_t                        ring_dma_slots;
    uint32_t                        fg_ring_slots;
    uint32_t                        bg_ring_slots;
    uint64_t                        ring_dma_bytes;
    uint32_t                        fg_ring_cursor;
    uint32_t                        bg_ring_cursor;
    uint64_t                        cb_epoch;

    std::vector<Sg53LayerHotState>  layer_hot;         // [num_total_moe_layers] per-layer UMA LFU partitions
    std::vector<int32_t>            phys_hot_gid;      // [hot_capacity_experts] actual GID physically in b_hot_experts[s]
    std::vector<int32_t>            fg_slot_of_gid;    // [total_routed_experts] -> -1 or 0..fg_ring_slots-1
    std::vector<int32_t>            gid_of_fg_slot;    // [fg_ring_slots]

    os_unfair_lock              bg_ring_lock = OS_UNFAIR_LOCK_INIT;
    std::vector<int32_t>        gid_of_bg_slot;    // [bg_ring_slots]
    std::vector<uint8_t>        bg_slot_ready;     // [bg_ring_slots]
    std::vector<uint64_t>       bg_slot_pin_epoch; // [bg_ring_slots]

    // Sanctioned 16-Core `ANE` (`MLComputePlan`) Multi-Horizon Router Lookahead
    bool                        ane_verified_physical;
    uint32_t                    ane_lookahead_horizon;
    uint32_t                    ane_prefetch_budget;
    std::atomic<uint32_t>       fg_nvme_active{0U};
    std::atomic<uint32_t>       cur_moe_layer{0U};
    std::atomic<uint64_t>       ane_step_epoch{1ULL};
    bool                        profile_nvme = false;
    std::atomic<uint64_t>       prof_fg_reads{0ULL};
    std::atomic<uint64_t>       prof_bg_reads{0ULL};
    std::atomic<uint64_t>       prof_fixup_reads{0ULL};
    std::atomic<uint64_t>       prof_mtp_reads{0ULL};
    std::vector<int32_t>        moe_to_global_layer; // [num_main_moe_layers]
    std::vector<Sg53AneWindow>  ane_windows;
    dispatch_queue_t            ane_bg_queue;

    // Telemetry & Byte Accounting
    char                        device_name[128];
    bool                        zero_copy_spine;
    double                      load_time_ms;
    uint64_t                    spine_mapped_bytes;
    uint64_t                    conv_state_bytes;
    uint64_t                    rec_state_bytes;
    uint64_t                    idx_tail_bytes;
    uint64_t                    kv_cache_bytes_allocated;
    uint64_t                    kv_cache_bytes_per_token;
    uint64_t                    workspace_bytes;

    os_unfair_lock              telemetry_lock = OS_UNFAIR_LOCK_INIT;
    StrataGlm53PerfCounters     counters;

    // Per-Layer State Strides
    uint64_t                    conv_layer_stride;
    uint64_t                    rec_layer_stride;
    uint64_t                    kv_layer_stride;
    uint64_t                    idx_pool_layer_stride;
    uint64_t                    idx_tail_layer_stride;

    // Persistent GPU State Buffers (`34` KDA + `12` DSA MLA + Native MTP)
    id<MTLBuffer>               b_conv_state;
    id<MTLBuffer>               b_rec_state;
    id<MTLBuffer>               b_mla_kv_cache;
    id<MTLBuffer>               b_idx_pool_k;
    id<MTLBuffer>               b_idx_tail_ring;
    id<MTLBuffer>               b_mtp_h_prev;      // [D] FP32 (`H_main[seq_pos - 1]` post-`final_norm`)

    // Speculative Rollback Snapshot Arenas (`4` boundary slots)
    id<MTLBuffer>               b_rec_snaps;
    id<MTLBuffer>               b_conv_snaps;

    // Pre-Allocated Private GPU Scratch Activations (`0` runtime allocations!)
    id<MTLBuffer>               b_hc;              // [kMaxPrefillChunk, 4 * D] FP32
    id<MTLBuffer>               b_mtp_x_f32;       // [kMaxPrefillChunk, D] FP32 (single-stream MTP)
    id<MTLBuffer>               b_mtp_e_f32;       // [kMaxPrefillChunk, D] FP32
    id<MTLBuffer>               b_mtp_eh_cat_f16;  // [kMaxPrefillChunk, 2 * D] FP16
    id<MTLBuffer>               b_mtp_h_out_f32;   // [kMaxPrefillChunk, D] FP32 (`h_mtp_out`)
    id<MTLBuffer>               b_h_main_f32;      // [kMaxPrefillChunk, D] FP32 (`H_main` post-`final_norm`)
    id<MTLBuffer>               b_mhc_z_f32;       // [kMaxPrefillChunk, 4 * D] FP32
    id<MTLBuffer>               b_mhc_mix_f32;     // [kMaxPrefillChunk, 24] FP32
    id<MTLBuffer>               b_mhc_pre_f32;     // [kMaxPrefillChunk, 4] FP32
    id<MTLBuffer>               b_mhc_post_f32;    // [kMaxPrefillChunk, 4] FP32
    id<MTLBuffer>               b_mhc_comb_f32;    // [kMaxPrefillChunk, 16] FP32
    id<MTLBuffer>               b_x_in_f16;        // [kMaxPrefillChunk, D] FP16
    id<MTLBuffer>               b_x_rot_f16;       // [kMaxPrefillChunk, D] FP16 (`RHT-64` rotated x for E8S-18-V4!)
    id<MTLBuffer>               b_x_in_f32;        // [kMaxPrefillChunk, D] FP32
    id<MTLBuffer>               b_block_y_f32;     // [kMaxPrefillChunk, D] FP32

    // KDA Scratch
    id<MTLBuffer>               b_kda_qkv_f16;
    id<MTLBuffer>               b_kda_qkv_conv_f16;
    id<MTLBuffer>               b_kda_fbg_f32;
    id<MTLBuffer>               b_kda_fa_f16;
    id<MTLBuffer>               b_kda_b_f32;
    id<MTLBuffer>               b_kda_ga_f16;
    id<MTLBuffer>               b_kda_a_f32;
    id<MTLBuffer>               b_kda_gate_f16;
    id<MTLBuffer>               b_kda_qk_hat_f32;
    id<MTLBuffer>               b_kda_decay_f32;
    id<MTLBuffer>               b_kda_beta_f32;
    id<MTLBuffer>               b_kda_raw_out_f32;
    id<MTLBuffer>               b_kda_y_f16;

    // DSA MLA + Glm5NextIndexer Scratch
    id<MTLBuffer>               b_mla_qkv_a_f32;
    id<MTLBuffer>               b_mla_q_resid_f16;
    id<MTLBuffer>               b_mla_q_f16;
    id<MTLBuffer>               b_mla_q_latent_f16;
    id<MTLBuffer>               b_mla_c_out_f16;
    id<MTLBuffer>               b_mla_v_out_f16;
    id<MTLBuffer>               b_idx_wk_f32;
    id<MTLBuffer>               b_idx_kgate_f32;
    id<MTLBuffer>               b_idx_cur_kg_f32;
    id<MTLBuffer>               b_idx_q_f32;
    id<MTLBuffer>               b_idx_whead_f32;
    id<MTLBuffer>               b_idx_scores_f32;
    id<MTLBuffer>               b_idx_sel_bits;
    id<MTLBuffer>               b_idx_sel_pools;
    id<MTLBuffer>               b_part_m;
    id<MTLBuffer>               b_part_l;
    id<MTLBuffer>               b_part_o;

    // Dense & MoE Scratch
    id<MTLBuffer>               b_ffn_gate_up_f32;
    id<MTLBuffer>               b_ffn_mid_f16;
    id<MTLBuffer>               b_shared_out_f32;
    id<MTLBuffer>               b_moe_mid_f16;
    id<MTLBuffer>               b_moe_down_part;
    id<MTLBuffer>               b_prefill_x_i8;    // [kMaxPrefillChunk, 16384] INT8 (opt-in --int8-prefill)
    id<MTLBuffer>               b_prefill_x_rsc;   // [kMaxPrefillChunk] FP16 row scale
    id<MTLBuffer>               b_prefill_x_sums;  // [kMaxPrefillChunk, 256] FP16 per-group-64 sums
    id<MTLBuffer>               b_moe_x_rsc;       // [kMaxPrefillChunk * top_k + 64] FP16
    id<MTLBuffer>               b_moe_mid_rsc;     // [kMaxPrefillChunk * top_k + 64] FP16
    id<MTLBuffer>               b_moe_mid_i8;      // [kMaxPrefillChunk * top_k + 64, d_moe] INT8
    id<MTLBuffer>               b_sample_stage_v;
    id<MTLBuffer>               b_sample_stage_i;
    id<MTLBuffer>               b_sample_stage_ms; // [kMaxSpecBatch, 64] float2 (block_max, block_sumexp)

    // 20K SLC-Pinned FR-Spec Draft-Head Shortlist (`45.00 MiB < 48 MiB` M5 Max SLC!)
    id<MTLBuffer>               b_frspec_lm_head;       // Private compacted Q4_A [V_short, D]
    id<MTLBuffer>               b_frspec_short_to_full; // Shared uint32 [V_short]
    id<MTLBuffer>               b_frspec_full_to_short; // Shared uint32 [V] (0xFFFFFFFF if absent)
    id<MTLBuffer>               b_frspec_patch_pairs;   // Shared uint32 [2 * 1024] (slot, full_row)
    StrataGlm53QuantMatrix      frspec_lm_head_mat{};

    // Shared Host <-> GPU I/O & Indirect MoE Dispatch Buffers
    id<MTLBuffer>               b_unit_x_f16;      // [16, D] FP16 for 16-Core ANE Lookahead!
    id<MTLBuffer>               b_topk_idx;        // [kMaxPrefillChunk, top_k]
    id<MTLBuffer>               b_topk_w;          // [kMaxPrefillChunk, top_k]
    id<MTLBuffer>               b_expert_dispatch; // [kMaxPrefillChunk * top_k] StrataGlm53ExpertDispatch
    id<MTLBuffer>               b_moe_sorted_pairs;// [kMaxPrefillChunk * top_k]
    id<MTLBuffer>               b_moe_work_tiles;  // [max(kMaxPrefillChunk * top_k + E, 3 * kMaxMoeBinTiles)]
    id<MTLBuffer>               b_moe_indirect_args;// [4096 B] GPU-populated MTLDispatchThreadgroupsIndirectArguments
    id<MTLBuffer>               b_pen_ids;         // [vocab_size] uint32 unique active penalty token IDs
    id<MTLBuffer>               b_pen_counts;      // [vocab_size] uint16 occurrence counts in active window
    std::vector<uint32_t>       pen_hist;
    std::vector<uint32_t>       pen_pos;
    uint32_t                    pen_num_ids;
    uint32_t                    pen_window_start;
    uint32_t                    pen_window_end;
    id<MTLBuffer>               b_token_ids;
    id<MTLBuffer>               b_mtp_token_ids;
    id<MTLBuffer>               b_mtp_replay_toks; // Dedicated async `cb_fix` replay tokens (zero CPU-GPU race!)
    id<MTLBuffer>               b_logits;
    id<MTLBuffer>               b_next_tokens;
    id<MTLBuffer>               b_draft_pmax;      // Shared float [STRATA_GLM53_MAX_SPEC_DRAFT_K]
    id<MTLBuffer>               b_sample_top_ids;
    id<MTLBuffer>               b_sample_top_probs;
    id<MTLBuffer>               b_sample_keep_k;
    id<MTLBuffer>               b_draft_top_ids;
    id<MTLBuffer>               b_draft_top_probs;
    id<MTLBuffer>               b_draft_keep_k;

    // Step 4 Compaction Context Slots + Paged DSA MLA KV + 16-Core ANE `indexer_wq_b` Lookahead
    StrataGlm53ContextSlot*     default_slot        = nullptr;
    StrataGlm53ContextSlot*     bound_slot          = nullptr;
    id<MTLBuffer>               b_block_loc         = nil;
    id<MTLBuffer>               b_block_loc_dummy   = nil;
    id<MTLBuffer>               b_kv_probe_qresid   = nil; // [11, qr] FP16
    id<MTLBuffer>               b_kv_probe_whead    = nil; // [11, idx_nh] FP32
    uint32_t                    kv_stride_tokens    = 0U;
    bool                        kv_paged            = false;
    bool                        paged_error         = false;
    Sg53AneKvProbe*             ane_kv              = nullptr;
    bool                        ane_kv_init_tried   = false;
    const void*                 mlocked_text_ptr    = nullptr;
    size_t                      mlocked_text_len    = 0U;
    const void*                 mlocked_rodata_ptr  = nullptr;
    size_t                      mlocked_rodata_len  = 0U;
};

namespace {

/* ============================================================================
 * 8-FD Parallel `F_NOCACHE` NVMe Expert Streamer (`13.50 MiB = 9 * 1.50 MiB`)
 * ============================================================================ */

struct Sg53NvmeExpertCopyTask {
    uint32_t gid;
    uint8_t* dst_ptr;
};

int sg53_stream_experts_nvme_batch(
    StrataGlm53GpuEngine*                      eng,
    const std::vector<Sg53NvmeExpertCopyTask>& tasks,
    bool                                       count_telemetry,
    bool                                       is_foreground = true
) {
    if (tasks.empty()) return 0;
    const StrataGlm53Bundle* bundle = eng->bundle;
    const uint64_t rec_bytes = eng->hdr.expert_record_bytes;
    if (rec_bytes == 0ULL) return -1;

    struct FgGuard {
        std::atomic<uint32_t>* flag;
        ~FgGuard() { if (flag) flag->fetch_sub(1U, std::memory_order_release); }
    };
    FgGuard fg_guard{is_foreground ? &eng->fg_nvme_active : nullptr};
    if (is_foreground) eng->fg_nvme_active.fetch_add(1U, std::memory_order_acquire);

    // Sort multi-expert batches by ascending GID (ascending file offset) for sequential ANS4 NVMe access!
    std::vector<Sg53NvmeExpertCopyTask> sorted_tasks;
    const Sg53NvmeExpertCopyTask* tasks_ptr = tasks.data();
    if (tasks.size() > 1U) {
        sorted_tasks = tasks;
        std::sort(sorted_tasks.begin(), sorted_tasks.end(),
                  [](const Sg53NvmeExpertCopyTask& a, const Sg53NvmeExpertCopyTask& b) {
                      return a.gid < b.gid;
                  });
        tasks_ptr = sorted_tasks.data();
    }

    const uint64_t t0 = mach_absolute_time();

    if (bundle->num_experts_fds == 0U) {
        if (!bundle->synthetic_experts_base) return -1;
        for (size_t i = 0U; i < tasks.size(); ++i) {
            const uint64_t file_off = static_cast<uint64_t>(tasks_ptr[i].gid) * rec_bytes;
            std::memcpy(tasks_ptr[i].dst_ptr, bundle->synthetic_experts_base + file_off, static_cast<size_t>(rec_bytes));
        }
    } else {
        const uint32_t num_fds       = std::min<uint32_t>(bundle->num_experts_fds, STRATA_GLM53_NUM_NVME_FDS);
        const uint64_t stripe_sz     = (rec_bytes >= kNvmeStripeChunk && (rec_bytes % kNvmeStripeChunk) == 0ULL)
                                           ? kNvmeStripeChunk
                                           : rec_bytes;
        const uint32_t stripes_per_e = static_cast<uint32_t>(rec_bytes / stripe_sz);
        const size_t   total_jobs    = tasks.size() * static_cast<size_t>(stripes_per_e);

        std::atomic<int> io_err{0};
        const int*  fds_ptr   = bundle->experts_fds;

        if (total_jobs == 1U) {
            const uint64_t off = static_cast<uint64_t>(tasks_ptr[0].gid) * rec_bytes;
            size_t done = 0U;
            while (done < rec_bytes) {
                ssize_t r = ::pread(fds_ptr[0], tasks_ptr[0].dst_ptr + done,
                                    static_cast<size_t>(rec_bytes - done), static_cast<off_t>(off + done));
                if (r <= 0) return -1;
                done += static_cast<size_t>(r);
            }
        } else {
            dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
            std::atomic<int>* err_p = &io_err;
            dispatch_apply(total_jobs, q, ^(size_t job_idx) {
                if (err_p->load(std::memory_order_relaxed) != 0) return;
                const size_t   e_idx  = job_idx / stripes_per_e;
                const size_t   s_idx  = job_idx % stripes_per_e;
                const int      fd     = fds_ptr[job_idx % num_fds];
                const uint64_t f_off  = static_cast<uint64_t>(tasks_ptr[e_idx].gid) * rec_bytes +
                                        static_cast<uint64_t>(s_idx) * stripe_sz;
                uint8_t*       dst    = tasks_ptr[e_idx].dst_ptr + static_cast<size_t>(s_idx) * stripe_sz;
                size_t         rem    = static_cast<size_t>(stripe_sz);
                size_t         got    = 0U;
                while (got < rem) {
                    ssize_t r = ::pread(fd, dst + got, rem - got, static_cast<off_t>(f_off + got));
                    if (r <= 0) {
                        err_p->store(-1, std::memory_order_relaxed);
                        return;
                    }
                    got += static_cast<size_t>(r);
                }
            });
            if (io_err.load(std::memory_order_relaxed) != 0) return -1;
        }
    }

    const uint64_t t1     = mach_absolute_time();
    const double   dt_sec = sg53_ticks_to_sec(t1 - t0);

    if (count_telemetry) {
        Sg53ScopedLock lk(&eng->telemetry_lock);
        const uint64_t cnt   = static_cast<uint64_t>(tasks.size());
        const uint64_t bytes = cnt * rec_bytes;
        eng->counters.nvme_cold_reads += cnt;
        eng->counters.nvme_bytes_read += bytes;
        eng->counters.nvme_io_time_sec += dt_sec;
        if (eng->counters.nvme_io_time_sec > 1e-9) {
            eng->counters.nvme_bandwidth_gbps =
                (static_cast<double>(eng->counters.nvme_bytes_read) / 1e9) / eng->counters.nvme_io_time_sec;
        }
    }
    return 0;
}

/* ============================================================================
 * 16-Core `ANE` (`MLComputePlan`) Multi-Horizon Router Lookahead Init & Exec
 * ============================================================================ */

bool sg53_init_ane_lookahead(StrataGlm53GpuEngine* eng) {
    eng->ane_verified_physical = false;
    eng->ane_lookahead_horizon = STRATA_GLM53_DEFAULT_LOOKAHEAD_HORIZON;
    eng->moe_to_global_layer.clear();

    for (uint32_t l = 0U; l < eng->cfg.num_hidden_layers; ++l) {
        if (eng->hdr.layers[l].mlp_type == STRATA_GLM53_MLP_SPARSE_MOE &&
            eng->hdr.layers[l].moe_layer_idx >= 0) {
            eng->moe_to_global_layer.push_back(static_cast<int32_t>(l));
        }
    }

    const uint32_t num_moe = static_cast<uint32_t>(eng->moe_to_global_layer.size());
    if (num_moe == 0U) return false;

    const uint32_t H       = eng->ane_lookahead_horizon;
    const uint32_t num_win = (num_moe + H - 1U) / H;
    const uint32_t D       = eng->cfg.hidden_size;
    const uint32_t E       = eng->cfg.n_routed_experts;
    const uint32_t d_pad   = std::max<uint32_t>(64U, (D + 15U) & ~15U);

    eng->ane_windows.resize(num_win);

    // Check physical ANE presence & IOKit IOSurface sandbox accessibility
    NSInteger ane_cores = 0;
    for (id<MLComputeDeviceProtocol> dev in MLAllComputeDevices()) {
        if ([dev isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
            ane_cores = static_cast<MLNeuralEngineComputeDevice*>(dev).totalCoreCount;
            break;
        }
    }

    IOSurfaceRef probe_surf = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(kAneSpatialAlign),
        (id)kIOSurfaceHeight: @(16),
        (id)kIOSurfaceBytesPerElement: @(2),
        (id)kIOSurfaceAllocSize: @(STRATA_GLM53_PAGE_BYTES)
    });
    const bool iokit_ok = (probe_surf != nullptr) && (ane_cores > 0);
    if (probe_surf) CFRelease(probe_surf);

    static std::atomic<uint64_t> s_ane_nonce{1ULL};
    std::atomic<bool> all_ane_ok{iokit_ok};

    Sg53AneWindow* win_ptr = eng->ane_windows.data();
    const int32_t* map_ptr = eng->moe_to_global_layer.data();
    const uint8_t* sp_base = eng->bundle->spine_base;
    const StrataGlm53LayerDesc* layers_ptr = eng->hdr.layers;
    std::atomic<bool>* ok_ptr = &all_ane_ok;
    std::atomic<uint64_t>* nonce_ptr = &s_ane_nonce;

    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(num_win, q, ^(size_t w_sz) {
        @autoreleasepool {
            const uint32_t w       = static_cast<uint32_t>(w_sz);
            const uint32_t moe_beg = w * H;
            const uint32_t moe_end = std::min<uint32_t>(num_moe, moe_beg + H);
            const uint32_t win_cnt = moe_end - moe_beg;
            const uint32_t out_ch  = win_cnt * E;
            const uint32_t out_pad = std::max<uint32_t>(64U, (out_ch + 15U) & ~15U);

            Sg53AneWindow& wb = win_ptr[w];
            wb.moe_layer_beg = moe_beg;
            wb.num_layers    = win_cnt;
            wb.d_model       = D;
            wb.d_pad         = d_pad;
            wb.num_experts   = E;
            wb.out_channels  = out_ch;
            wb.out_pad       = out_pad;
            wb.cpu_w_fp16.assign(static_cast<size_t>(out_pad) * d_pad, 0U);
            wb.e_biases.assign(static_cast<size_t>(out_ch), 0.0f);

            // Fold `post_attention_layernorm` into `moe_router_gate` -> exact `W_ane[sub, e, c]`!
            for (uint32_t sub = 0U; sub < win_cnt; ++sub) {
                const uint32_t g_l = static_cast<uint32_t>(map_ptr[moe_beg + sub]);
                const StrataGlm53LayerDesc& ld = layers_ptr[g_l];
                const auto* norm_w = reinterpret_cast<const float*>(sp_base + ld.post_attention_layernorm.offset);
                const auto* gate_w = reinterpret_cast<const uint16_t*>(sp_base + ld.moe_router_gate.offset);
                const auto* bias_w = reinterpret_cast<const float*>(sp_base + ld.moe_e_score_bias.offset);

                std::memcpy(&wb.e_biases[static_cast<size_t>(sub) * E], bias_w, static_cast<size_t>(E) * sizeof(float));
                for (uint32_t e = 0U; e < E; ++e) {
                    uint16_t* dst_row = &wb.cpu_w_fp16[static_cast<size_t>(sub * E + e) * d_pad];
                    const uint16_t* src_row = gate_w + static_cast<size_t>(e) * D;
                    for (uint32_t c = 0U; c < D; ++c) {
                        float folded = sg53_fast_fp16_to_f32(src_row[c]) * norm_w[c];
                        dst_row[c]   = sg53_fast_f32_to_fp16(folded);
                    }
                }
            }

            if (!ok_ptr->load(std::memory_order_relaxed)) return;

            std::vector<uint8_t> pb = sg53_serialize_ane_router_mlmodel(
                d_pad, out_pad, kAneSpatialAlign, wb.cpu_w_fp16.data());

            const uint64_t nonce = nonce_ptr->fetch_add(1ULL, std::memory_order_relaxed);
            std::string tmp_root([NSTemporaryDirectory() UTF8String]);
            if (!tmp_root.empty() && tmp_root.back() != '/') tmp_root.push_back('/');
            std::string dir_tmpl = tmp_root + "strata_glm53_ane_XXXXXX";
            std::vector<char> dir_buf(dir_tmpl.begin(), dir_tmpl.end());
            dir_buf.push_back('\0');
            char* staged_dir = ::mkdtemp(dir_buf.data());
            if (!staged_dir) {
                ok_ptr->store(false, std::memory_order_relaxed);
                return;
            }
            NSURL* dirURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:staged_dir]];
            Sg53ScopedUrlCleanup srcGuard{dirURL};
            NSString* fname = [NSString stringWithFormat:@"w%u_%llu.mlmodel",
                               w, static_cast<unsigned long long>(nonce)];
            NSURL* srcURL = [dirURL URLByAppendingPathComponent:fname];

            NSData* pbData = [NSData dataWithBytesNoCopy:pb.data() length:pb.size() freeWhenDone:NO];
            NSError* wErr = nil;
            if (![pbData writeToURL:srcURL options:NSDataWritingAtomic error:&wErr]) {
                ok_ptr->store(false, std::memory_order_relaxed);
                return;
            }

            __block NSURL* compiledURL = nil;
            dispatch_semaphore_t cSem = dispatch_semaphore_create(0);
            [MLModel compileModelAtURL:srcURL completionHandler:^(NSURL* _Nullable url, NSError* _Nullable) {
                compiledURL = url;
                dispatch_semaphore_signal(cSem);
            }];
            dispatch_semaphore_wait(cSem, DISPATCH_TIME_FOREVER);
            srcGuard.reset(); // Unlink `.mlmodel` immediately!

            if (!compiledURL) {
                ok_ptr->store(false, std::memory_order_relaxed);
                return;
            }
            Sg53ScopedUrlCleanup compGuard{compiledURL};

            MLModelConfiguration* mlCfg = [[MLModelConfiguration alloc] init];
            mlCfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

            __block MLComputePlan* plan = nil;
            dispatch_semaphore_t pSem = dispatch_semaphore_create(0);
            [MLComputePlan loadContentsOfURL:compiledURL
                               configuration:mlCfg
                           completionHandler:^(MLComputePlan* _Nullable cp, NSError* _Nullable) {
                plan = cp;
                dispatch_semaphore_signal(pSem);
            }];
            dispatch_semaphore_wait(pSem, DISPATCH_TIME_FOREVER);

            bool win_on_ane = false;
            if (plan && plan.modelStructure.neuralNetwork) {
                NSArray<MLModelStructureNeuralNetworkLayer*>* ops = plan.modelStructure.neuralNetwork.layers;
                uint32_t ane_cnt = 0U;
                for (MLModelStructureNeuralNetworkLayer* op in ops) {
                    MLComputePlanDeviceUsage* usage = [plan computeDeviceUsageForNeuralNetworkLayer:op];
                    if (usage && [usage.preferredComputeDevice isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
                        ++ane_cnt;
                    }
                }
                win_on_ane = (ops.count >= 1U) && (ane_cnt == static_cast<uint32_t>(ops.count));
            }
            if (!win_on_ane) {
                ok_ptr->store(false, std::memory_order_relaxed);
                return;
            }

            NSError* lErr = nil;
            MLModel* mdl = [MLModel modelWithContentsOfURL:compiledURL configuration:mlCfg error:&lErr];
            compGuard.reset(); // Unlink `.mlmodelc` immediately (0 bytes left on disk)!
            if (!mdl || lErr) {
                ok_ptr->store(false, std::memory_order_relaxed);
                return;
            }

            const size_t in_sz  = sg53_align_up_u64(static_cast<uint64_t>(d_pad) * kAneSpatialAlign * sizeof(uint16_t),
                                                    STRATA_GLM53_PAGE_BYTES);
            const size_t out_sz = sg53_align_up_u64(static_cast<uint64_t>(out_pad) * kAneSpatialAlign * sizeof(uint16_t),
                                                    STRATA_GLM53_PAGE_BYTES);
            void* raw_in  = nullptr;
            void* raw_out = nullptr;
            if (::posix_memalign(&raw_in, STRATA_GLM53_PAGE_BYTES, in_sz) != 0 || !raw_in ||
                ::posix_memalign(&raw_out, STRATA_GLM53_PAGE_BYTES, out_sz) != 0 || !raw_out) {
                if (raw_in)  std::free(raw_in);
                if (raw_out) std::free(raw_out);
                ok_ptr->store(false, std::memory_order_relaxed);
                return;
            }
            std::memset(raw_in, 0, in_sz);
            std::memset(raw_out, 0, out_sz);
            wb.in_fp16_buf  = static_cast<uint16_t*>(raw_in);
            wb.out_fp16_buf = static_cast<uint16_t*>(raw_out);

            NSError* aErr = nil;
            wb.in_array = [[MLMultiArray alloc] initWithDataPointer:wb.in_fp16_buf
                                                              shape:@[@1, @(d_pad), @1, @(kAneSpatialAlign)]
                                                           dataType:MLMultiArrayDataTypeFloat16
                                                            strides:@[@(d_pad * kAneSpatialAlign), @(kAneSpatialAlign), @(kAneSpatialAlign), @1]
                                                        deallocator:nil
                                                              error:&aErr];
            wb.out_array = [[MLMultiArray alloc] initWithDataPointer:wb.out_fp16_buf
                                                               shape:@[@1, @(out_pad), @1, @(kAneSpatialAlign)]
                                                            dataType:MLMultiArrayDataTypeFloat16
                                                             strides:@[@(out_pad * kAneSpatialAlign), @(kAneSpatialAlign), @(kAneSpatialAlign), @1]
                                                         deallocator:nil
                                                               error:&aErr];
            if (!wb.in_array || !wb.out_array) {
                ok_ptr->store(false, std::memory_order_relaxed);
                return;
            }

            wb.feat_provider = [[StrataGlm53AneFeatureProvider alloc] initWithMultiArray:wb.in_array];
            wb.pred_options  = [[MLPredictionOptions alloc] init];
            wb.pred_options.outputBackings = @{@"y": wb.out_array};
            wb.coreml_model  = mdl;

            NSError* pErr = nil;
            id<MLFeatureProvider> warm = [wb.coreml_model predictionFromFeatures:wb.feat_provider
                                                                         options:wb.pred_options
                                                                           error:&pErr];
            if (!warm || pErr) {
                wb.coreml_model = nil;
                ok_ptr->store(false, std::memory_order_relaxed);
                return;
            }
            wb.verified_ane = true;
            // Free CPU duplicate weight vector once CoreML ANE `.hwx` is loaded
            std::vector<uint16_t>().swap(wb.cpu_w_fp16);
        }
    });

    eng->ane_verified_physical = all_ane_ok.load(std::memory_order_relaxed);
    return eng->ane_verified_physical;
}

void sg53_schedule_ane_lookahead_async(
    StrataGlm53GpuEngine* eng,
    uint32_t              win_idx,
    uint32_t              batch_B,
    bool                  do_bg_nvme_prefetch
) {
    if (!eng || win_idx >= eng->ane_windows.size() || !eng->ane_bg_queue || batch_B == 0U) return;

    const uint32_t B = std::min<uint32_t>(batch_B, kAneSpatialAlign);
    const uint32_t D = eng->cfg.hidden_size;

    // Copy snapshot of `[B, D]` unit-RMS FP16 vector synchronously (`<= 128 KiB`, `< 1 us`) before GPU overwrites `b_unit_x_f16`
    std::vector<uint16_t> x_snap(static_cast<size_t>(B) * D);
    std::memcpy(x_snap.data(), [eng->b_unit_x_f16 contents], x_snap.size() * sizeof(uint16_t));

    // Snapshot immutable `layer_hot` + `fg_ring` bits for upcoming next layer (`sub = 1`) in `win_idx` (read-only copy)
    std::vector<uint8_t> hot_bitmap;
    const uint32_t pref_budget = eng->ane_prefetch_budget;
    if (do_bg_nvme_prefetch && pref_budget > 0U && eng->bg_ring_slots > 0U && eng->hdr.total_routed_experts > 0U) {
        hot_bitmap.resize(eng->hdr.total_routed_experts, 0U);
        const uint32_t E_loc = eng->cfg.n_routed_experts;
        const Sg53AneWindow& wb_ref = eng->ane_windows[win_idx];
        const uint32_t sub_lim = std::min<uint32_t>(2U, wb_ref.num_layers); // Only immediate next layer (`sub == 1`)!
        for (uint32_t sub = 1U; sub < sub_lim; ++sub) {
            const uint32_t ml = wb_ref.moe_layer_beg + sub;
            if (ml < eng->layer_hot.size()) {
                const auto& lst = eng->layer_hot[ml];
                for (uint32_t e = 0U; e < E_loc && e < lst.slot_of_e.size(); ++e) {
                    const uint32_t gid = ml * E_loc + e;
                    if (lst.slot_of_e[e] >= 0 ||
                        (gid < eng->fg_slot_of_gid.size() && eng->fg_slot_of_gid[gid] >= 0)) {
                        hot_bitmap[gid] = 1U;
                    }
                }
            }
        }
    }

    const StrataGlm53ExecMode mode_snap = eng->exec_mode;
    const float               tau_snap  = eng->adaptive_gate_tau;
    const uint32_t            cap_snap  = eng->adaptive_cold_cap;
    const uint64_t            cur_epoch = eng->cb_epoch;
    const uint64_t            my_step   = eng->ane_step_epoch.load(std::memory_order_acquire);

    dispatch_async(eng->ane_bg_queue, ^{
        @autoreleasepool {
            Sg53AneWindow& wb = eng->ane_windows[win_idx];
            Sg53ScopedLock wlk(&wb.lock);

            const uint64_t t0 = mach_absolute_time();
            bool used_ane     = false;
            std::vector<float> logits(wb.out_channels, 0.0f);

            if (wb.verified_ane && wb.coreml_model != nil && wb.in_fp16_buf && wb.out_fp16_buf) {
                std::memset(wb.in_fp16_buf, 0, static_cast<size_t>(wb.d_pad) * kAneSpatialAlign * sizeof(uint16_t));
                for (uint32_t t = 0U; t < B; ++t) {
                    const uint16_t* row = x_snap.data() + static_cast<size_t>(t) * D;
                    for (uint32_t c = 0U; c < D; ++c) {
                        wb.in_fp16_buf[static_cast<size_t>(c) * kAneSpatialAlign + t] = row[c];
                    }
                }

                NSError* err = nil;
                id<MLFeatureProvider> outFeat = [wb.coreml_model predictionFromFeatures:wb.feat_provider
                                                                                options:wb.pred_options
                                                                                  error:&err];
                if (outFeat != nil && err == nil) {
                    MLMultiArray* outArr = [outFeat featureValueForName:@"y"].multiArrayValue;
                    if (outArr == wb.out_array) {
                        used_ane = true;
                        for (uint32_t oc = 0U; oc < wb.out_channels; ++oc) {
                            const uint16_t* ch = wb.out_fp16_buf + static_cast<size_t>(oc) * kAneSpatialAlign;
                            float best_v = sg53_fast_fp16_to_f32(ch[0]);
                            for (uint32_t t = 1U; t < B; ++t) {
                                const float v = sg53_fast_fp16_to_f32(ch[t]);
                                if (v > best_v) best_v = v;
                            }
                            logits[oc] = best_v;
                        }
                    } else if (outArr != nil) {
                        used_ane = true;
                        const uint32_t out_ch = wb.out_channels;
                        float* dst_log = logits.data();
                        [outArr getBytesWithHandler:^(const void* bytes, NSInteger size) {
                            if (size >= static_cast<NSInteger>(static_cast<size_t>(out_ch) * kAneSpatialAlign * sizeof(uint16_t))) {
                                const auto* ptr = static_cast<const uint16_t*>(bytes);
                                for (uint32_t oc = 0U; oc < out_ch; ++oc) {
                                    const uint16_t* ch = ptr + static_cast<size_t>(oc) * kAneSpatialAlign;
                                    float best_v = sg53_fast_fp16_to_f32(ch[0]);
                                    for (uint32_t t = 1U; t < B; ++t) {
                                        const float v = sg53_fast_fp16_to_f32(ch[t]);
                                        if (v > best_v) best_v = v;
                                    }
                                    dst_log[oc] = best_v;
                                }
                            }
                        }];
                    }
                }
            }

            if (!used_ane && !wb.cpu_w_fp16.empty()) {
                for (uint32_t oc = 0U; oc < wb.out_channels; ++oc) {
                    const uint16_t* wrow = &wb.cpu_w_fp16[static_cast<size_t>(oc) * wb.d_pad];
                    float acc = 0.0f;
                    for (uint32_t c = 0U; c < D; ++c) {
                        acc += sg53_fast_fp16_to_f32(x_snap[c]) * sg53_fast_fp16_to_f32(wrow[c]);
                    }
                    logits[oc] = acc;
                }
            }

            const uint64_t t1    = mach_absolute_time();
            const double   dt_ms = sg53_ticks_to_sec(t1 - t0) * 1000.0;

            {
                Sg53ScopedLock tlk(&eng->telemetry_lock);
                eng->counters.ane_lookahead_calls += 1ULL;
                if (used_ane) eng->counters.ran_on_ane += 1ULL;
                eng->counters.ane_total_time_ms += dt_ms;
            }

            // Contention-Free Background `F_NOCACHE` NVMe Prefetch for Immediate Next Layer (`sub == 1`):
            // Skip immediately if step ended, GPU already reached `moe_layer_beg + 1`, or foreground NVMe is active!
            if (!do_bg_nvme_prefetch || pref_budget == 0U || wb.num_layers <= 1U ||
                eng->bg_ring_slots == 0U || hot_bitmap.empty()) {
                return;
            }
            if (eng->ane_step_epoch.load(std::memory_order_acquire) != my_step ||
                eng->cur_moe_layer.load(std::memory_order_acquire) > wb.moe_layer_beg ||
                eng->fg_nvme_active.load(std::memory_order_acquire) > 0U) {
                return;
            }

            const uint32_t E          = wb.num_experts;
            const uint32_t top_k      = eng->cfg.num_experts_per_tok;
            const uint32_t pre_budget = (mode_snap == STRATA_GLM53_MODE_ADAPTIVE)
                                            ? std::min<uint32_t>(pref_budget, cap_snap)
                                            : std::min<uint32_t>(pref_budget, top_k);
            if (pre_budget == 0U) return;

            std::vector<Sg53NvmeExpertCopyTask> bg_tasks;
            std::vector<uint32_t>               bg_slots_claimed;

            const uint32_t sub_end = std::min<uint32_t>(2U, wb.num_layers);
            for (uint32_t sub = 1U; sub < sub_end; ++sub) {
                const uint32_t moe_l = wb.moe_layer_beg + sub;
                const float*   l_log = &logits[static_cast<size_t>(sub) * E];
                const float*   l_bi  = &wb.e_biases[static_cast<size_t>(sub) * E];

                struct Cand { uint32_t e; float choice; float raw_sc; };
                std::vector<Cand> cands(E);
                for (uint32_t e = 0U; e < E; ++e) {
                    float sc  = 1.0f / (1.0f + std::exp(-l_log[e]));
                    cands[e]  = Cand{e, sc + l_bi[e], sc};
                }
                const uint32_t k_sel = std::min<uint32_t>(top_k, E);
                std::partial_sort(cands.begin(), cands.begin() + k_sel, cands.end(),
                                  [](const Cand& a, const Cand& b) {
                                      return (a.choice > b.choice) || (a.choice == b.choice && a.e < b.e);
                                  });

                float sum_raw = 1e-20f;
                for (uint32_t k = 0U; k < k_sel; ++k) sum_raw += cands[k].raw_sc;
                const float w_scale = eng->cfg.routed_scaling_factor / sum_raw;

                uint32_t sub_queued = 0U;
                for (uint32_t k = 0U; k < k_sel && sub_queued < pre_budget; ++k) {
                    const float w_est = cands[k].raw_sc * w_scale;
                    if (mode_snap == STRATA_GLM53_MODE_ADAPTIVE && w_est < tau_snap) continue;

                    const uint32_t gid = moe_l * E + cands[k].e;
                    if (gid >= hot_bitmap.size() || hot_bitmap[gid] != 0U) continue;

                    int32_t chosen_bg = -1;
                    {
                        Sg53ScopedLock blk(&eng->bg_ring_lock);
                        bool already = false;
                        for (uint32_t s = 0U; s < eng->bg_ring_slots; ++s) {
                            if (eng->gid_of_bg_slot[s] == static_cast<int32_t>(gid)) {
                                already = true;
                                break;
                            }
                        }
                        if (!already) {
                            for (uint32_t tries = 0U; tries < eng->bg_ring_slots; ++tries) {
                                uint32_t s = (eng->bg_ring_cursor++) % eng->bg_ring_slots;
                                if (eng->bg_slot_pin_epoch[s] <= cur_epoch || eng->gid_of_bg_slot[s] < 0) {
                                    eng->gid_of_bg_slot[s]    = static_cast<int32_t>(gid);
                                    eng->bg_slot_ready[s]     = 0U;
                                    eng->bg_slot_pin_epoch[s] = cur_epoch + 8ULL;
                                    chosen_bg                 = static_cast<int32_t>(s);
                                    break;
                                }
                            }
                        }
                    }

                    if (chosen_bg >= 0) {
                        const uint32_t phys_slot = eng->fg_ring_slots + static_cast<uint32_t>(chosen_bg);
                        uint8_t* dst = eng->ring_dma_ptr + static_cast<uint64_t>(phys_slot) * eng->hdr.expert_record_bytes;
                        bg_tasks.push_back(Sg53NvmeExpertCopyTask{gid, dst});
                        bg_slots_claimed.push_back(static_cast<uint32_t>(chosen_bg));
                        ++sub_queued;
                    }
                }
            }

            if (!bg_tasks.empty()) {
                if (eng->ane_step_epoch.load(std::memory_order_acquire) != my_step ||
                    eng->cur_moe_layer.load(std::memory_order_acquire) > wb.moe_layer_beg ||
                    eng->fg_nvme_active.load(std::memory_order_acquire) > 0U) {
                    Sg53ScopedLock blk(&eng->bg_ring_lock);
                    for (uint32_t s : bg_slots_claimed) {
                        eng->gid_of_bg_slot[s] = -1;
                        eng->bg_slot_ready[s]  = 0U;
                    }
                    return;
                }
                if (eng->profile_nvme) {
                    eng->prof_bg_reads.fetch_add(bg_tasks.size(), std::memory_order_relaxed);
                }
                if (sg53_stream_experts_nvme_batch(eng, bg_tasks, /*count_telemetry=*/true, /*is_foreground=*/false) == 0) {
                    Sg53ScopedLock blk(&eng->bg_ring_lock);
                    for (uint32_t s : bg_slots_claimed) {
                        eng->bg_slot_ready[s] = 1U;
                    }
                } else {
                    Sg53ScopedLock blk(&eng->bg_ring_lock);
                    for (uint32_t s : bg_slots_claimed) {
                        eng->gid_of_bg_slot[s] = -1;
                        eng->bg_slot_ready[s]  = 0U;
                    }
                }
            }
        }
    });
}

/* ============================================================================
 * Linear & Layer Encoding Primitives
 * ============================================================================ */

void sg53_encode_q4_linear(
    id<MTLComputeCommandEncoder>   enc,
    StrataGlm53GpuEngine*          eng,
    const StrataGlm53QuantMatrix&  mat,
    id<MTLBuffer>                  x_in_f16,
    NSUInteger                     x_off,
    id<MTLBuffer>                  y_f16,
    NSUInteger                     y_f16_off,
    id<MTLBuffer>                  y_f32,
    NSUInteger                     y_f32_off,
    uint32_t                       M_valid,
    uint32_t                       M_padded,
    uint32_t                       out_mode,
    bool                           force_gemv,
    id<MTLBuffer>                  w_buf_override = nil
) {
    id<MTLBuffer> w_buf = w_buf_override ? w_buf_override : eng->b_spine;
    const bool use_nax = !force_gemv && eng->has_nax && ((mat.cols % 64U) == 0U);
    if (use_nax) {
        // Opt-In `--int8-prefill` (`M >= 32` ONLY — decode `M <= 5` NEVER touches INT8!)
        const bool can_i8 =
            eng->int8_prefill_enabled &&
            (M_valid >= 32U) &&
            (M_padded >= 64U) && ((M_padded % 64U) == 0U) &&
            (mat.cols >= 512U) && (mat.cols <= 16384U) && ((mat.cols % 128U) == 0U) &&
            ((mat.rows_padded % 64U) == 0U) &&
            (eng->b_prefill_x_i8 != nil) && (eng->b_prefill_x_rsc != nil) && (eng->b_prefill_x_sums != nil) &&
            (static_cast<size_t>(M_padded) * mat.cols <= [eng->b_prefill_x_i8 length]);
        if (can_i8) {
            Sg53QuantI8ParamsHost qp{M_valid, M_padded, mat.cols, 0U};
            [enc setComputePipelineState:eng->pso_quant_f16_to_i8_row];
            [enc setBuffer:x_in_f16 offset:x_off atIndex:0];
            [enc setBuffer:eng->b_prefill_x_i8 offset:0 atIndex:1];
            [enc setBuffer:eng->b_prefill_x_rsc offset:0 atIndex:2];
            [enc setBuffer:eng->b_prefill_x_sums offset:0 atIndex:3];
            [enc setBytes:&qp length:sizeof(qp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(M_padded, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            const bool     use_m128 = (M_padded >= 128U) && ((M_padded % 128U) == 0U);
            const uint32_t m_tile   = use_m128 ? 128U : 64U;
            Sg53NaxGemmParamsHost p{
                M_valid, M_padded, mat.rows_logical, mat.rows_padded, mat.cols, out_mode, 0U, 0U
            };
            [enc setComputePipelineState:(use_m128 ? eng->pso_q4_gemm_nax_i8_m128 : eng->pso_q4_gemm_nax_i8_m64)];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.weight_off) atIndex:0];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.scales_off) atIndex:1];
            [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.biases_off) atIndex:2];
            [enc setBuffer:eng->b_prefill_x_i8 offset:0 atIndex:3];
            [enc setBuffer:eng->b_prefill_x_rsc offset:0 atIndex:4];
            [enc setBuffer:eng->b_prefill_x_sums offset:0 atIndex:5];
            [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:(y_f16 ? y_f16_off : 0U) atIndex:6];
            [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:(y_f32 ? y_f32_off : 0U) atIndex:7];
            [enc setBytes:&p length:sizeof(p) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(mat.rows_padded / 64U, M_padded / m_tile, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }

        const bool     bk128   = ((mat.cols % 128U) == 0U);
        const bool     use_m16 = !bk128 || (M_padded < 64U);
        const uint32_t m_tile  = use_m16 ? 16U : 64U;
        const uint32_t th_tg   = (bk128 && use_m16) ? 64U : 128U; // m16 BK=128 uses SG=2 (64 th/TG)!
        Sg53NaxGemmParamsHost p{
            M_valid, M_padded, mat.rows_logical, mat.rows_padded, mat.cols, out_mode, 0U, 0U
        };
        id<MTLComputePipelineState> pso = !bk128
            ? eng->pso_q4_gemm_nax_bk64_m16
            : (use_m16 ? eng->pso_q4_gemm_nax_m16 : eng->pso_q4_gemm_nax_m64);
        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.biases_off) atIndex:2];
        [enc setBuffer:x_in_f16 offset:x_off atIndex:3];
        [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:(y_f16 ? y_f16_off : 0U) atIndex:4];
        [enc setBuffer:(y_f32 ? y_f32 : x_in_f16) offset:(y_f32 ? y_f32_off : 0U) atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(mat.rows_padded / 64U, (M_padded + m_tile - 1U) / m_tile, 1)
            threadsPerThreadgroup:MTLSizeMake(th_tg, 1, 1)];
        return;
    }

    for (uint32_t m0 = 0U; m0 < M_valid; m0 += kMaxSpecBatch) {
        const uint32_t mb = std::min<uint32_t>(kMaxSpecBatch, M_valid - m0);
        Sg53GemvBatchParamsHost p{mb, mat.rows_logical, mat.cols, out_mode};
        const bool use_k128 = (mat.cols == 128U) && (mb >= 1U && mb <= 5U) &&
                              (eng->pso_q4_gemv_k128_spec[mb - 1U] != nil);
        id<MTLComputePipelineState> pso = use_k128
            ? eng->pso_q4_gemv_k128_spec[mb - 1U]
            : ((mb >= 1U && mb <= 5U && eng->pso_q4_gemv_spec[mb - 1U])
                   ? eng->pso_q4_gemv_spec[mb - 1U]
                   : eng->pso_q4_gemv_batch);
        [enc setComputePipelineState:pso];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.weight_off) atIndex:0];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.scales_off) atIndex:1];
        [enc setBuffer:w_buf offset:static_cast<NSUInteger>(mat.biases_off) atIndex:2];
        [enc setBuffer:x_in_f16 offset:(x_off + static_cast<NSUInteger>(m0) * mat.cols * sizeof(uint16_t)) atIndex:3];
        [enc setBuffer:(y_f16 ? y_f16 : x_in_f16)
                offset:(y_f16 ? (y_f16_off + static_cast<NSUInteger>(m0) * mat.rows_logical * sizeof(uint16_t)) : 0U)
               atIndex:4];
        [enc setBuffer:(y_f32 ? y_f32 : x_in_f16)
                offset:(y_f32 ? (y_f32_off + static_cast<NSUInteger>(m0) * mat.rows_logical * sizeof(float)) : 0U)
               atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(use_k128 ? ((mat.rows_logical + 31U) / 32U) : ((mat.rows_logical + 7U) / 8U), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    // Zero-pad trailing rows `[M_valid .. M_padded - 1]` if `out_mode == kOutF16 && M_padded > M_valid`
    if (out_mode == kOutF16 && y_f16 && M_padded > M_valid) {
        const size_t pad_bytes = static_cast<size_t>(M_padded - M_valid) * mat.rows_logical * sizeof(uint16_t);
        const NSUInteger pad_off = y_f16_off + static_cast<NSUInteger>(M_valid) * mat.rows_logical * sizeof(uint16_t);
        const uint32_t count_u4 = static_cast<uint32_t>(pad_bytes / 16ULL);
        if (count_u4 > 0U && (pad_off % 16U) == 0U) {
            [enc setComputePipelineState:eng->pso_zero_u32];
            [enc setBuffer:y_f16 offset:pad_off atIndex:0];
            [enc setBytes:&count_u4 length:sizeof(uint32_t) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake((count_u4 + 255U) / 256U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }
}

void sg53_encode_f32_linear(
    id<MTLComputeCommandEncoder>   enc,
    StrataGlm53GpuEngine*          eng,
    const StrataGlm53TensorSlice&  sl,
    id<MTLBuffer>                  x_f32,
    NSUInteger                     x_off,
    id<MTLBuffer>                  y_f32,
    NSUInteger                     y_off,
    uint32_t                       M_valid,
    uint32_t                       out_mode
) {
    Sg53GemvBatchParamsHost p{M_valid, sl.shape0, sl.shape1, out_mode};
    [enc setComputePipelineState:eng->pso_f32_gemv_batch];
    [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(sl.offset) atIndex:0];
    [enc setBuffer:x_f32 offset:x_off atIndex:1];
    [enc setBuffer:y_f32 offset:y_off atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(sl.shape0, (M_valid + 7U) / 8U, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sg53_encode_mla_batched_f16(
    id<MTLComputeCommandEncoder>   enc,
    StrataGlm53GpuEngine*          eng,
    const StrataGlm53TensorSlice&  w_slice,
    id<MTLBuffer>                  x_in,
    id<MTLBuffer>                  y_out,
    uint32_t                       M_valid,
    uint32_t                       M_padded,
    uint32_t                       nh,
    uint32_t                       N,
    uint32_t                       K,
    bool                           force_gemv
) {
    const bool use_nax = !force_gemv && eng->has_nax && ((N % 64U) == 0U) && ((K % 64U) == 0U);
    Sg53MlaBatchedGemmParamsHost p{M_valid, M_padded, nh, N, K, 0U, 0U, 0U};

    if (use_nax) {
        const bool     use_m16 = (M_padded < 64U);
        const uint32_t m_tile  = use_m16 ? 16U : 64U;
        [enc setComputePipelineState:(use_m16 ? eng->pso_mla_batched_nax_m16 : eng->pso_mla_batched_nax_m64)];
        [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(w_slice.offset) atIndex:0];
        [enc setBuffer:x_in offset:0 atIndex:1];
        [enc setBuffer:y_out offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(N / 64U, (M_padded + m_tile - 1U) / m_tile, nh)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }

    for (uint32_t m0 = 0U; m0 < M_valid; m0 += kMaxSpecBatch) {
        const uint32_t mb  = std::min<uint32_t>(kMaxSpecBatch, M_valid - m0);
        const uint32_t mp  = (m0 + mb >= M_valid) ? (M_padded - m0) : mb;
        Sg53MlaBatchedGemmParamsHost bp{mb, mp, nh, N, K, 0U, 0U, 0U};
        id<MTLComputePipelineState> pso = (mb >= 1U && mb <= 5U && eng->pso_mla_batched_gemv_spec[mb - 1U])
            ? eng->pso_mla_batched_gemv_spec[mb - 1U]
            : eng->pso_mla_batched_gemv;
        [enc setComputePipelineState:pso];
        [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(w_slice.offset) atIndex:0];
        [enc setBuffer:x_in offset:(static_cast<NSUInteger>(m0) * nh * K * sizeof(uint16_t)) atIndex:1];
        [enc setBuffer:y_out offset:(static_cast<NSUInteger>(m0) * nh * N * sizeof(uint16_t)) atIndex:2];
        [enc setBytes:&bp length:sizeof(bp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + 7U) / 8U, nh, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
}

void sg53_encode_mla_batched_f16_hm(
    id<MTLComputeCommandEncoder>   enc,
    StrataGlm53GpuEngine*          eng,
    const StrataGlm53TensorSlice&  w_slice,
    id<MTLBuffer>                  x_hm,
    id<MTLBuffer>                  y_out,
    uint32_t                       M_valid,
    uint32_t                       M_padded,
    uint32_t                       nh,
    uint32_t                       N,
    uint32_t                       K,
    bool                           out_head_major
) {
    const bool     use_m16 = (M_padded < 64U);
    const uint32_t m_tile  = use_m16 ? 16U : 64U;
    const uint32_t th_tg   = use_m16 ? 64U : 128U;
    Sg53MlaBatchedGemmParamsHost p{M_valid, M_padded, nh, N, K, out_head_major ? 1U : 0U, 0U, 0U};
    [enc setComputePipelineState:(use_m16 ? eng->pso_mla_batched_nax_hm_m16 : eng->pso_mla_batched_nax_hm_m64)];
    [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(w_slice.offset) atIndex:0];
    [enc setBuffer:x_hm offset:0 atIndex:1];
    [enc setBuffer:y_out offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((N + 63U) / 64U, (M_padded + m_tile - 1U) / m_tile, nh)
        threadsPerThreadgroup:MTLSizeMake(th_tg, 1, 1)];
}

void sg53_fast_memcpy_expert(uint8_t* dst, const uint8_t* src, uint64_t bytes) {
    if (bytes < (1ULL << 20)) {
        std::memcpy(dst, src, static_cast<size_t>(bytes));
        return;
    }
    constexpr size_t kStripes = 6U;
    const size_t chunk = (static_cast<size_t>(bytes) + kStripes - 1U) / kStripes;
    dispatch_apply(kStripes, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t si) {
        const size_t off = si * chunk;
        if (off >= static_cast<size_t>(bytes)) return;
        const size_t n = std::min<size_t>(chunk, static_cast<size_t>(bytes) - off);
        std::memcpy(dst + off, src + off, n);
    });
}

bool sg53_encode_sublayer_entry(
    id<MTLComputeCommandEncoder>   enc,
    StrataGlm53GpuEngine*          eng,
    bool                           has_mhc,
    const StrataGlm53MhcDesc&      mhc,
    const StrataGlm53TensorSlice&  layernorm_w,
    id<MTLBuffer>                  stream_buf, // `b_hc` (`[M, 4*D]`) or `b_mtp_x_f32` (`[M, D]`)
    uint32_t                       M_valid,
    uint32_t                       M_padded,
    bool                           write_f32,
    bool                           write_unit_f16,
    bool                           write_x_rot_f16 = false
) {
    const uint32_t D      = eng->cfg.hidden_size;
    const uint32_t H      = eng->cfg.hc_mult;
    const uint32_t hc_dim = H * D;
    const float    eps    = eng->cfg.rms_norm_eps;

    if (has_mhc) {
        const bool use_fused_mhc =
            (M_valid >= 1U) && (M_valid <= 5U) && (H == 4U) &&
            (mhc.fn.shape0 == 24U) && ((mhc.fn.shape1 % 16U) == 0U) &&
            (eng->pso_f32_gemv_splitk4_spec[M_valid - 1U] != nil) &&
            (eng->pso_mhc_sinkhorn_collapse != nil);
        if (use_fused_mhc) {
            // Linearity: `hc_fn @ (stream * inv_rms) == (hc_fn @ stream) * inv_rms` -> skips `pso_mhc_norm` (-90 barriers/tok!)
            Sg53GemvBatchParamsHost gp{M_valid, mhc.fn.shape0, mhc.fn.shape1, kOutF32};
            [enc setComputePipelineState:eng->pso_f32_gemv_splitk4_spec[M_valid - 1U]];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(mhc.fn.offset) atIndex:0];
            [enc setBuffer:stream_buf offset:0 atIndex:1];
            [enc setBuffer:eng->b_mhc_mix_f32 offset:0 atIndex:2];
            [enc setBytes:&gp length:sizeof(gp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(mhc.fn.shape0, 4, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            const bool fuse_rot = write_x_rot_f16 && ((D & 63U) == 0U) && (eng->b_x_rot_f16 != nil);
            Sg53MhcFusedCollapseParamsHost fcp{
                M_valid, M_padded, D, H, eng->cfg.hc_sinkhorn_iters, eng->cfg.hc_eps, eps,
                write_f32 ? 1U : 0U,
                write_unit_f16 ? 1U : 0U,
                4U /* split_k */,
                1U /* apply_hc_inv_rms */,
                fuse_rot ? 1U : 0U
            };
            [enc setComputePipelineState:eng->pso_mhc_sinkhorn_collapse];
            [enc setBuffer:stream_buf offset:0 atIndex:0];
            [enc setBuffer:eng->b_mhc_mix_f32 offset:0 atIndex:1];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(mhc.base.offset) atIndex:2];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(mhc.scale.offset) atIndex:3];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(layernorm_w.offset) atIndex:4];
            [enc setBuffer:eng->b_mhc_post_f32 offset:0 atIndex:5];
            [enc setBuffer:eng->b_mhc_comb_f32 offset:0 atIndex:6];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:7];
            [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:8];
            [enc setBuffer:eng->b_unit_x_f16 offset:0 atIndex:9];
            [enc setBuffer:(eng->b_x_rot_f16 ? eng->b_x_rot_f16 : eng->b_x_in_f16) offset:0 atIndex:10];
            [enc setBytes:&fcp length:sizeof(fcp) atIndex:11];
            [enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            return fuse_rot;
        } else {
            {
                Sg53MhcNormParamsHost np{M_valid, hc_dim, eps, 0U};
                [enc setComputePipelineState:eng->pso_mhc_norm];
                [enc setBuffer:stream_buf offset:0 atIndex:0];
                [enc setBuffer:eng->b_mhc_z_f32 offset:0 atIndex:1];
                [enc setBytes:&np length:sizeof(np) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            sg53_encode_f32_linear(enc, eng, mhc.fn, eng->b_mhc_z_f32, 0, eng->b_mhc_mix_f32, 0, M_valid, kOutF32);
            {
                Sg53MhcSinkhornParamsHost sp{M_valid, H, eng->cfg.hc_sinkhorn_iters, eng->cfg.hc_eps};
                [enc setComputePipelineState:eng->pso_mhc_sinkhorn];
                [enc setBuffer:eng->b_mhc_mix_f32 offset:0 atIndex:0];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(mhc.base.offset) atIndex:1];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(mhc.scale.offset) atIndex:2];
                [enc setBuffer:eng->b_mhc_pre_f32 offset:0 atIndex:3];
                [enc setBuffer:eng->b_mhc_post_f32 offset:0 atIndex:4];
                [enc setBuffer:eng->b_mhc_comb_f32 offset:0 atIndex:5];
                [enc setBytes:&sp length:sizeof(sp) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((M_valid + 31U) / 32U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            {
                Sg53MhcCollapseParamsHost cp{
                    M_valid, M_padded, D, H, eps,
                    write_f32 ? 1U : 0U,
                    write_unit_f16 ? 1U : 0U,
                    0U
                };
                [enc setComputePipelineState:eng->pso_mhc_collapse_norm];
                [enc setBuffer:stream_buf offset:0 atIndex:0];
                [enc setBuffer:eng->b_mhc_pre_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(layernorm_w.offset) atIndex:2];
                [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:3];
                [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:4];
                [enc setBuffer:eng->b_unit_x_f16 offset:0 atIndex:5];
                [enc setBytes:&cp length:sizeof(cp) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }
    } else {
        Sg53RmsNormParamsHost rp{
            M_valid, M_padded, D, D, 0U, D, 0U, eps, 1U, write_f32 ? 1U : 0U, 0U, 0U
        };
        [enc setComputePipelineState:eng->pso_rmsnorm_strided];
        [enc setBuffer:stream_buf offset:0 atIndex:0];
        [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(layernorm_w.offset) atIndex:1];
        [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:3];
        [enc setBytes:&rp length:sizeof(rp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return false;
}

void sg53_encode_sublayer_exit(
    id<MTLComputeCommandEncoder> enc,
    StrataGlm53GpuEngine*        eng,
    bool                         has_mhc,
    id<MTLBuffer>                stream_buf,
    uint32_t                     M_valid
) {
    const uint32_t D  = eng->cfg.hidden_size;
    const uint32_t d4 = D / 4U;

    if (has_mhc) {
        Sg53MhcExpandParamsHost ep{M_valid, D, eng->cfg.hc_mult, 0U};
        [enc setComputePipelineState:eng->pso_mhc_expand];
        [enc setBuffer:stream_buf offset:0 atIndex:0];
        [enc setBuffer:eng->b_block_y_f32 offset:0 atIndex:1];
        [enc setBuffer:eng->b_mhc_post_f32 offset:0 atIndex:2];
        [enc setBuffer:eng->b_mhc_comb_f32 offset:0 atIndex:3];
        [enc setBytes:&ep length:sizeof(ep) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_valid, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        Sg53ResAddParamsHost rp{M_valid, D};
        [enc setComputePipelineState:eng->pso_single_res_add];
        [enc setBuffer:stream_buf offset:0 atIndex:0];
        [enc setBuffer:eng->b_block_y_f32 offset:0 atIndex:1];
        [enc setBytes:&rp length:sizeof(rp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_valid, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}

/* ============================================================================
 * Step 4 Two-Tier Compaction Slot, Paged DSA MLA KV & 16-Core ANE `indexer_wq_b` Probe
 * ============================================================================ */

constexpr uint8_t  kSg53BlkUnalloc         = 0U;
constexpr uint8_t  kSg53BlkResident        = 1U;
constexpr uint8_t  kSg53BlkPaged           = 2U;
constexpr uint32_t kSg53PhysFree           = 0xFFFFFFFFU;
constexpr uint32_t kSg53MaxIoBatchBlks     = 1024U;
constexpr uint32_t kSg53AneMaxPrefetchLyr  = 32U;
constexpr float    kSg53AnePrefetchMargin  = 0.90f;
constexpr uint32_t kSg53CompactOff         = 0U;
constexpr uint32_t kSg53CompactPaged       = 2U;

struct Sg53PagedLayerState {
    std::vector<uint8_t>  st;
    std::vector<uint8_t>  on_nvme;
    std::vector<uint32_t> phys_owner;
    std::vector<uint32_t> free_list;
    std::vector<float>    u;
    std::vector<uint32_t> pin_epoch;
    std::vector<uint32_t> pf_epoch;
    uint32_t              resident    = 0U;
    float                 last_thresh = 0.0f;
    bool                  has_thresh  = false;
};

struct Sg53MGroup {
    uint32_t              beg = 0U;
    uint32_t              end = 0U;
    std::vector<uint32_t> need;
};

struct Sg53PagedCounters {
    uint64_t paged_out = 0ULL;
    uint64_t recalled  = 0ULL;
    uint64_t wbytes    = 0ULL;
    uint64_t rbytes    = 0ULL;
    double   wms       = 0.0;
    double   rms       = 0.0;
    uint64_t misses    = 0ULL;
    uint64_t pf_issued = 0ULL;
    uint64_t pf_hits   = 0ULL;
    bool     err       = false;
};

struct Sg53AneKvModelBundle {
    uint32_t                       layer_idx = 0U;
    uint32_t                       in_ch     = 0U;
    uint32_t                       out_ch    = 0U;
    MLModel*                       model     = nil;
    MLMultiArray*                  in_arr    = nil;
    MLMultiArray*                  out_arr   = nil;
    StrataGlm53AneFeatureProvider* fp        = nil;
    MLPredictionOptions*           opts      = nil;
    uint16_t*                      in_buf    = nullptr;
    uint16_t*                      out_buf   = nullptr;
};

} // namespace

struct Sg53AneKvProbe {
    bool                              ready     = false;
    std::string                       note;
    std::vector<Sg53AneKvModelBundle> bundles;
    uint32_t                          n_layers  = 0U; // 11 main DSA layers
    uint32_t                          qr        = 0U; // 1536
    uint32_t                          idx_nh    = 0U; // 32
    uint32_t                          idx_hd    = 0U; // 128
    uint32_t                          ane_ops   = 0U;
    uint32_t                          total_ops = 0U;
    long                              ane_cores = 0;
    std::vector<uint16_t>             qresid_snap; // [n_layers * qr]
    std::vector<float>                whead_snap;  // [n_layers * idx_nh]
    std::vector<float>                q_hat;       // [n_layers * idx_nh * idx_hd]
    dispatch_queue_t                  queue     = nil;
    double                            init_ms   = 0.0;

    ~Sg53AneKvProbe() {
        for (auto& b : bundles) {
            b.opts = nil; b.fp = nil; b.in_arr = nil; b.out_arr = nil; b.model = nil;
            if (b.in_buf)  { std::free(b.in_buf);  b.in_buf  = nullptr; }
            if (b.out_buf) { std::free(b.out_buf); b.out_buf = nullptr; }
        }
    }
};

struct StrataGlm53ContextSlot {
    StrataGlm53GpuEngine*          eng              = nullptr;
    bool                           is_default       = false;
    bool                           paged            = false;
    uint32_t                       pool_tokens      = 0U;
    uint32_t                       pool_blocks      = 0U;
    uint32_t                       max_blocks       = 0U;
    uint32_t                       n_layers         = 0U;
    uint32_t                       kv_stride_tokens = 0U;
    uint64_t                       kv_layer_stride  = 0ULL;
    uint64_t                       bytes            = 0ULL;

    id<MTLBuffer>                  b_conv_state     = nil;
    id<MTLBuffer>                  b_rec_state      = nil;
    id<MTLBuffer>                  b_mla_kv_cache   = nil;
    id<MTLBuffer>                  b_idx_pool_k     = nil;
    id<MTLBuffer>                  b_idx_tail_ring  = nil;
    id<MTLBuffer>                  b_mtp_h_prev     = nil;
    id<MTLBuffer>                  b_block_loc      = nil;
    id<MTLResidencySet>            rset             = nil;

    uint32_t                       seq_pos          = 0U;
    uint32_t                       mtp_seq_pos      = 0U;
    uint64_t                       sample_step      = 0ULL;

    int                            swap_fd          = -1;
    char                           swap_path[256]   = {};
    uint8_t*                       io_stage         = nullptr;
    uint8_t*                       ane_stage        = nullptr;
    std::vector<Sg53PagedLayerState> layers;
    uint32_t                       pin_epoch_ctr    = 1U;
    uint32_t                       step_epoch       = 1U;
    bool                           q_valid          = false;
    bool                           probe_pending    = false;
    dispatch_group_t               probe_grp        = nil;
    double                         ane_us_total     = 0.0;
    StrataGlm53CompactionStats     stats            = {};
    std::vector<uint32_t>          ngram_hist;
    std::vector<uint32_t>          ngram5_ht;
    std::vector<uint32_t>          ngram4_ht;
    uint32_t                       ngram_cooldown   = 0U;
};

namespace {

inline size_t sg53_paged_block_bytes(const StrataGlm53ContextSlot* s) {
    return static_cast<size_t>(STRATA_GLM53_KV_BLOCK_TOKENS) * s->eng->cfg.kv_lora_rank * sizeof(uint16_t);
}

inline uint8_t* sg53_pool_blk_ptr(StrataGlm53ContextSlot* s, uint32_t ls, uint32_t phys) {
    const size_t blk_b = sg53_paged_block_bytes(s);
    return static_cast<uint8_t*>([s->b_mla_kv_cache contents]) +
           static_cast<size_t>(ls) * s->kv_layer_stride + static_cast<size_t>(phys) * blk_b;
}

inline const uint8_t* sg53_pool_blk_cptr(const StrataGlm53ContextSlot* s, uint32_t ls, uint32_t phys) {
    const size_t blk_b = sg53_paged_block_bytes(s);
    return static_cast<const uint8_t*>([s->b_mla_kv_cache contents]) +
           static_cast<size_t>(ls) * s->kv_layer_stride + static_cast<size_t>(phys) * blk_b;
}

inline uint32_t* sg53_loc_row(StrataGlm53ContextSlot* s, uint32_t ls) {
    return static_cast<uint32_t*>([s->b_block_loc contents]) + static_cast<size_t>(ls) * s->max_blocks;
}

bool sg53_paged_io_batch(StrataGlm53ContextSlot* s, uint32_t ls,
                         std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                         bool write, uint8_t* stage_buf, uint32_t stage_cap_blocks, Sg53PagedCounters& c) {
    if (pairs.empty()) return true;
    if (!stage_buf || stage_cap_blocks == 0U) { c.err = true; return false; }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    const size_t blk_bytes = sg53_paged_block_bytes(s);
    struct Run { uint32_t idx0; uint32_t len; off_t foff; };
    std::vector<Run> runs;

    for (size_t base = 0U; base < pairs.size(); base += stage_cap_blocks) {
        const uint32_t cnt = static_cast<uint32_t>(std::min<size_t>(stage_cap_blocks, pairs.size() - base));
        if (write) {
            for (uint32_t i = 0U; i < cnt; ++i) {
                std::memcpy(stage_buf + static_cast<size_t>(i) * blk_bytes,
                            sg53_pool_blk_cptr(s, ls, pairs[base + i].second), blk_bytes);
            }
        }
        runs.clear();
        uint32_t r0 = 0U;
        while (r0 < cnt) {
            uint32_t rlen = 1U;
            while (r0 + rlen < cnt && pairs[base + r0 + rlen].first == pairs[base + r0].first + rlen) ++rlen;
            const off_t foff = static_cast<off_t>(
                (static_cast<uint64_t>(ls) * s->max_blocks + pairs[base + r0].first) * blk_bytes);
            runs.push_back(Run{r0, rlen, foff});
            r0 += rlen;
        }
        std::atomic<bool> io_fail{false};
        std::atomic<bool>* fail_ptr = &io_fail;
        const Run* run_ptr = runs.data();
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
        const double ms = sg53_ticks_to_sec(mach_absolute_time() - t0) * 1000.0;
        if (io_fail.load(std::memory_order_relaxed)) { c.err = true; return false; }
        if (!write) {
            for (uint32_t i = 0U; i < cnt; ++i) {
                std::memcpy(sg53_pool_blk_ptr(s, ls, pairs[base + i].second),
                            stage_buf + static_cast<size_t>(i) * blk_bytes, blk_bytes);
            }
        }
        const uint64_t bytes_done = static_cast<uint64_t>(cnt) * blk_bytes;
        if (write) { c.paged_out += cnt; c.wbytes += bytes_done; c.wms += ms; }
        else       { c.recalled  += cnt; c.rbytes += bytes_done; c.rms += ms; }
    }
    return true;
}

uint32_t sg53_paged_take_phys(StrataGlm53ContextSlot* s, Sg53PagedLayerState& L, uint32_t b) {
    if (b < s->pool_blocks && L.phys_owner[b] == kSg53PhysFree) return b;
    while (!L.free_list.empty()) {
        const uint32_t p = L.free_list.back();
        L.free_list.pop_back();
        if (L.phys_owner[p] == kSg53PhysFree) return p;
    }
    return kSg53PhysFree;
}

void sg53_paged_release(StrataGlm53ContextSlot* s, Sg53PagedLayerState& L, uint32_t ls, uint32_t b, uint8_t new_st) {
    uint32_t* loc = sg53_loc_row(s, ls);
    if (L.st[b] == kSg53BlkResident) {
        const uint32_t phys = loc[b] & STRATA_GLM53_KV_LOC_IDX_MASK;
        if (phys < s->pool_blocks) {
            L.phys_owner[phys] = kSg53PhysFree;
            L.free_list.push_back(phys);
        }
        if (L.resident > 0U) --L.resident;
    }
    L.st[b] = new_st;
    loc[b]  = (new_st == kSg53BlkPaged) ? (b & STRATA_GLM53_KV_LOC_IDX_MASK) : 0U;
}

uint32_t sg53_paged_evict_staged(StrataGlm53ContextSlot* s, uint32_t ls, uint32_t count, uint32_t newest_blk,
                                 bool allow_protected, uint8_t* stage_buf, uint32_t stage_cap, Sg53PagedCounters& c) {
    if (count == 0U) return 0U;
    Sg53PagedLayerState& L = s->layers[ls];
    const uint32_t sink  = STRATA_GLM53_COMPACT_SINK_BLOCKS;
    const uint32_t local = STRATA_GLM53_COMPACT_LOCAL_BLOCKS;
    const uint32_t lim   = std::min<uint32_t>(s->max_blocks, newest_blk + 1U);
    std::vector<uint32_t> cand;
    cand.reserve(L.resident);
    for (uint32_t b = 0U; b < s->max_blocks; ++b) {
        if (L.st[b] != kSg53BlkResident || L.pin_epoch[b] == s->pin_epoch_ctr) continue;
        if (!allow_protected && (b < sink || (b < lim && b + local > newest_blk))) continue;
        cand.push_back(b);
    }
    const size_t k = std::min<size_t>(count, cand.size());
    if (k == 0U) return 0U;
    auto cmp = [&](uint32_t a, uint32_t b2) { return (L.u[a] < L.u[b2]) || (L.u[a] == L.u[b2] && a < b2); };
    if (k < cand.size()) std::nth_element(cand.begin(), cand.begin() + static_cast<ptrdiff_t>(k), cand.end(), cmp);

    uint32_t* loc = sg53_loc_row(s, ls);
    std::vector<std::pair<uint32_t, uint32_t>> dirty;
    dirty.reserve(k);
    for (size_t i = 0U; i < k; ++i) {
        const uint32_t b = cand[i];
        if (!L.on_nvme[b]) dirty.emplace_back(b, loc[b] & STRATA_GLM53_KV_LOC_IDX_MASK);
    }
    if (!dirty.empty()) {
        if (!sg53_paged_io_batch(s, ls, dirty, /*write=*/true, stage_buf, stage_cap, c)) return 0U;
        for (const auto& pr : dirty) L.on_nvme[pr.first] = 1U;
    }
    for (size_t i = 0U; i < k; ++i) sg53_paged_release(s, L, ls, cand[i], kSg53BlkPaged);
    return static_cast<uint32_t>(k);
}

bool sg53_paged_ensure_staged(StrataGlm53ContextSlot* s, uint32_t ls, const std::vector<uint32_t>& need,
                              uint32_t newest_blk, bool exact_check,
                              uint8_t* stage_buf, uint32_t stage_cap, Sg53PagedCounters& c) {
    Sg53PagedLayerState& L = s->layers[ls];
    uint32_t* loc = sg53_loc_row(s, ls);
    std::vector<uint32_t> missing;
    for (uint32_t b : need) {
        if (b >= s->max_blocks) continue;
        if (L.st[b] == kSg53BlkResident) {
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
        want -= sg53_paged_evict_staged(s, ls, want, newest_blk, false, stage_buf, stage_cap, c);
        if (want > 0U) want -= sg53_paged_evict_staged(s, ls, want, newest_blk, true, stage_buf, stage_cap, c);
        if (want > 0U || c.err) { c.err = true; return false; }
    }
    const size_t blk_bytes = sg53_paged_block_bytes(s);
    std::vector<std::pair<uint32_t, uint32_t>> to_read;
    to_read.reserve(missing.size());
    for (uint32_t b : missing) {
        const uint32_t phys = sg53_paged_take_phys(s, L, b);
        if (phys == kSg53PhysFree) { c.err = true; return false; }
        L.phys_owner[phys] = b;
        if (L.st[b] == kSg53BlkPaged && L.on_nvme[b]) {
            to_read.emplace_back(b, phys);
        } else {
            std::memset(sg53_pool_blk_ptr(s, ls, phys), 0, blk_bytes);
            L.on_nvme[b] = 0U;
        }
        L.st[b] = kSg53BlkResident;
        loc[b]  = STRATA_GLM53_KV_LOC_UMA_BIT | phys;
        ++L.resident;
    }
    if (!to_read.empty()) {
        if (!sg53_paged_io_batch(s, ls, to_read, /*write=*/false, stage_buf, stage_cap, c)) return false;
        if (exact_check) c.misses += to_read.size();
    }
    return true;
}

bool sg53_paged_ensure(StrataGlm53ContextSlot* s, uint32_t ls, const std::vector<uint32_t>& need,
                       uint32_t newest_blk, bool exact_check, Sg53PagedCounters& c) {
    return sg53_paged_ensure_staged(s, ls, need, newest_blk, exact_check, s->io_stage, kSg53MaxIoBatchBlks, c);
}

void sg53_paged_merge(StrataGlm53ContextSlot* s, const Sg53PagedCounters& c) {
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

void sg53_paged_join_probe(StrataGlm53ContextSlot* s) {
    if (s && s->probe_pending && s->probe_grp) {
        dispatch_group_wait(s->probe_grp, DISPATCH_TIME_FOREVER);
        s->probe_pending = false;
    }
}

void sg53_paged_reset_tables(StrataGlm53ContextSlot* s) {
    if (!s || !s->paged) return;
    sg53_paged_join_probe(s);
    for (uint32_t ls = 0U; ls < s->n_layers; ++ls) {
        Sg53PagedLayerState& L = s->layers[ls];
        std::fill(L.st.begin(), L.st.end(), kSg53BlkUnalloc);
        std::fill(L.on_nvme.begin(), L.on_nvme.end(), 0U);
        std::fill(L.phys_owner.begin(), L.phys_owner.end(), kSg53PhysFree);
        std::fill(L.u.begin(), L.u.end(), 0.0f);
        std::fill(L.pin_epoch.begin(), L.pin_epoch.end(), 0U);
        std::fill(L.pf_epoch.begin(), L.pf_epoch.end(), 0U);
        L.free_list.resize(s->pool_blocks);
        for (uint32_t i = 0U; i < s->pool_blocks; ++i) L.free_list[i] = s->pool_blocks - 1U - i;
        L.resident    = 0U;
        L.has_thresh  = false;
        L.last_thresh = 0.0f;
    }
    std::memset([s->b_block_loc contents], 0, static_cast<size_t>(s->n_layers) * s->max_blocks * sizeof(uint32_t));
    s->q_valid = false;
}

void sg53_paged_on_rewind(StrataGlm53ContextSlot* s, uint32_t seq_pos, uint32_t mtp_seq_pos) {
    if (!s || !s->paged) return;
    sg53_paged_join_probe(s);
    Sg53PagedCounters c;
    for (uint32_t ls = 0U; ls < s->n_layers; ++ls) {
        Sg53PagedLayerState& L = s->layers[ls];
        const uint32_t valid_pos  = (ls + 1U == s->n_layers && s->eng->has_mtp) ? mtp_seq_pos : seq_pos;
        const uint32_t first_dead = (valid_pos + STRATA_GLM53_KV_BLOCK_TOKENS - 1U) / STRATA_GLM53_KV_BLOCK_TOKENS;
        for (uint32_t b = first_dead; b < s->max_blocks; ++b) {
            if (L.st[b] != kSg53BlkUnalloc) sg53_paged_release(s, L, ls, b, kSg53BlkUnalloc);
            L.on_nvme[b] = 0U;
            L.u[b]       = 0.0f;
        }
        if ((valid_pos % STRATA_GLM53_KV_BLOCK_TOKENS) != 0U) {
            const uint32_t bb = valid_pos / STRATA_GLM53_KV_BLOCK_TOKENS;
            ++s->pin_epoch_ctr;
            L.pin_epoch[bb] = s->pin_epoch_ctr;
            std::vector<uint32_t> need{bb};
            (void)sg53_paged_ensure(s, ls, need, bb, false, c);
            L.on_nvme[bb] = 0U;
        }
        L.has_thresh = false;
    }
    s->q_valid = false;
    sg53_paged_merge(s, c);
}

void sg53_paged_commit_wait(StrataGlm53GpuEngine* eng,
                            id<MTLComputeCommandEncoder> __strong& enc,
                            id<MTLCommandBuffer> __strong& cb) {
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError) {
        eng->paged_error = true;
    }
    ++eng->cb_epoch;
    cb  = [eng->queue commandBuffer];
    enc = [cb computeCommandEncoder];
}

void sg53_paged_alloc_range(StrataGlm53GpuEngine* eng,
                            id<MTLComputeCommandEncoder> __strong& enc,
                            id<MTLCommandBuffer> __strong& cb,
                            uint32_t ls, uint32_t pos_start, uint32_t M) {
    StrataGlm53ContextSlot* s = eng->bound_slot;
    sg53_paged_join_probe(s);
    Sg53PagedLayerState& L = s->layers[ls];
    const uint32_t b0 = pos_start / STRATA_GLM53_KV_BLOCK_TOKENS;
    const uint32_t b1 = (pos_start + M - 1U) / STRATA_GLM53_KV_BLOCK_TOKENS;
    ++s->pin_epoch_ctr;
    std::vector<uint32_t> need;
    uint32_t missing_cnt = 0U;
    /* This path serves DENSE launches, which read every block `0 .. b1` through `block_loc` without a residency
     * sync. Going forward nothing below the write range can be paged out (the dense range is smaller than the
     * pool), but a checkpoint restore from the sparse phase rewinds into it with older blocks still on NVMe;
     * their table entry then holds the logical index and the kernels would read whatever lives at that physical
     * block. Recall them here (at most `select_k` blocks, which the pool always holds). */
    for (uint32_t b = 0U; b < b0 && b < s->max_blocks; ++b) {
        if (L.st[b] == kSg53BlkPaged) {
            L.pin_epoch[b] = s->pin_epoch_ctr;
            need.push_back(b);
            ++missing_cnt;
        }
    }
    const size_t n_recalled = need.size();
    for (uint32_t b = b0; b <= b1 && b < s->max_blocks; ++b) {
        L.pin_epoch[b] = s->pin_epoch_ctr;
        need.push_back(b);
        if (L.st[b] != kSg53BlkResident) ++missing_cnt;
    }
    if (L.resident + missing_cnt > s->pool_blocks) {
        /* Need to evict dirty UMA blocks to kv_swap: flush prior GPU writes first */
        sg53_paged_commit_wait(eng, enc, cb);
    }
    Sg53PagedCounters c;
    if (!sg53_paged_ensure(s, ls, need, b1, false, c)) {
        eng->paged_error = true;
        sg53_paged_merge(s, c);
        return;
    }
    /* Only the write range is about to change; recalled blocks keep their valid `kv_swap` copy. */
    for (size_t i = n_recalled; i < need.size(); ++i) L.on_nvme[need[i]] = 0U;
    sg53_paged_merge(s, c);
}

void sg53_paged_make_group_resident(StrataGlm53GpuEngine* eng, uint32_t ls, uint32_t newest_blk,
                                    const Sg53MGroup& g, const Sg53MGroup* next_g) {
    StrataGlm53ContextSlot* s = eng->bound_slot;
    Sg53PagedLayerState& L = s->layers[ls];
    ++s->pin_epoch_ctr;
    for (uint32_t b : g.need) L.pin_epoch[b] = s->pin_epoch_ctr;
    if (next_g) {
        for (uint32_t b : next_g->need) if (b < s->max_blocks) L.u[b] += 1000.0f;
    }
    Sg53PagedCounters c;
    const bool ok = sg53_paged_ensure(s, ls, g.need, newest_blk, /*exact_check=*/true, c);
    if (next_g) {
        for (uint32_t b : next_g->need) if (b < s->max_blocks) L.u[b] -= 1000.0f;
    }
    if (!ok) eng->paged_error = true;
    sg53_paged_merge(s, c);
}

void sg53_paged_sync_point(StrataGlm53GpuEngine* eng,
                           id<MTLComputeCommandEncoder> __strong& enc,
                           id<MTLCommandBuffer> __strong& cb,
                           uint32_t ls, uint32_t pos_start, uint32_t M,
                           uint32_t select_k, uint32_t kpool,
                           std::vector<Sg53MGroup>* groups_out) {
    StrataGlm53ContextSlot* s = eng->bound_slot;
    sg53_paged_join_probe(s);
    sg53_paged_commit_wait(eng, enc, cb);
    s->stats.paged_sync_points += 1ULL;
    if (ls < eng->cfg.num_dsa_layers) s->q_valid = true;

    Sg53PagedLayerState& L = s->layers[ls];
    const uint32_t newest_blk = (pos_start + M - 1U) / kpool;

    {
        const uint32_t q_last = pos_start + M - 1U;
        const uint32_t nc     = (q_last + 1U) / kpool;
        if (nc > select_k) {
            const float* sc = static_cast<const float*>([eng->b_idx_scores_f32 contents]) +
                              static_cast<size_t>(M - 1U) * eng->max_pools_alloc;
            std::vector<float> tmp(sc, sc + nc);
            for (uint32_t b = 0U; b < nc; ++b) L.u[b] = 0.98f * L.u[b] + sc[b];
            std::nth_element(tmp.begin(), tmp.begin() + static_cast<ptrdiff_t>(select_k - 1U), tmp.end(), std::greater<float>());
            L.last_thresh = tmp[select_k - 1U];
            L.has_thresh  = true;
        }
    }

    const uint32_t* sel = static_cast<const uint32_t*>([eng->b_idx_sel_pools contents]);
    const uint32_t b0   = pos_start / kpool;
    std::vector<Sg53MGroup>& groups = *groups_out;
    groups.clear();
    auto open_group = [&](uint32_t beg) {
        Sg53MGroup g;
        g.beg = beg;
        ++s->pin_epoch_ctr;
        for (uint32_t b = b0; b <= newest_blk && b < s->max_blocks; ++b) {
            L.pin_epoch[b] = s->pin_epoch_ctr;
            g.need.push_back(b);
        }
        groups.push_back(std::move(g));
    };
    open_group(0U);
    std::vector<uint32_t> mine;
    mine.reserve(select_k + 8U);
    for (uint32_t m = 0U; m < M; ++m) {
        const uint32_t q_pos = pos_start + m;
        const uint32_t nc    = (q_pos + 1U) / kpool;
        mine.clear();
        if (nc > select_k) {
            const uint32_t* row = sel + static_cast<size_t>(m) * select_k;
            for (uint32_t i = 0U; i < select_k; ++i) mine.push_back(row[i]);
        } else {
            for (uint32_t b = 0U; b < nc; ++b) mine.push_back(b);
        }
        uint32_t fresh = 0U;
        for (uint32_t b : mine) if (b < s->max_blocks && L.pin_epoch[b] != s->pin_epoch_ctr) ++fresh;
        if (groups.back().need.size() + fresh > s->pool_blocks && m > groups.back().beg) {
            groups.back().end = m;
            open_group(m);
        }
        for (uint32_t b : mine) {
            if (b < s->max_blocks && L.pin_epoch[b] != s->pin_epoch_ctr) {
                L.pin_epoch[b] = s->pin_epoch_ctr;
                groups.back().need.push_back(b);
            }
        }
    }
    groups.back().end = M;

    Sg53PagedCounters c;
    ++s->pin_epoch_ctr;
    for (uint32_t b : groups[0].need) L.pin_epoch[b] = s->pin_epoch_ctr;
    const Sg53MGroup* g1 = (groups.size() > 1U) ? &groups[1] : nullptr;
    if (g1) {
        for (uint32_t b : g1->need) if (b < s->max_blocks) L.u[b] += 1000.0f;
    }
    const bool ok = sg53_paged_ensure(s, ls, groups[0].need, newest_blk, /*exact_check=*/true, c);
    if (g1) {
        for (uint32_t b : g1->need) if (b < s->max_blocks) L.u[b] -= 1000.0f;
    }
    if (!ok) {
        eng->paged_error = true;
        sg53_paged_merge(s, c);
        return;
    }
    for (uint32_t b = b0; b <= newest_blk && b < s->max_blocks; ++b) L.on_nvme[b] = 0U;
    sg53_paged_merge(s, c);
}

/* Dequantizes `Q4 [rows, cols]` (`group_size=64`) into `FP16 [rows, cols]` for CoreML ANE compilation. */
void sg53_dequant_q4_to_fp16(const uint8_t* spine_base, const StrataGlm53QuantMatrix& qm,
                             uint32_t rows, uint32_t cols, uint16_t* dst_fp16) {
    const auto* w_u32  = reinterpret_cast<const uint32_t*>(spine_base + qm.weight_off);
    const auto* scales = reinterpret_cast<const uint16_t*>(spine_base + qm.scales_off);
    const auto* biases = reinterpret_cast<const uint16_t*>(spine_base + qm.biases_off);
    const uint32_t u32_per_row = cols / 8U;
    const uint32_t grp_per_row = cols / 64U;
    for (uint32_t r = 0U; r < rows; ++r) {
        const uint32_t* rw = w_u32 + static_cast<size_t>(r) * u32_per_row;
        const uint16_t* rs = scales + static_cast<size_t>(r) * grp_per_row;
        const uint16_t* rb = biases + static_cast<size_t>(r) * grp_per_row;
        uint16_t*       dr = dst_fp16 + static_cast<size_t>(r) * cols;
        for (uint32_t g = 0U; g < grp_per_row; ++g) {
            const float sc = sg53_fast_fp16_to_f32(rs[g]);
            const float bi = sg53_fast_fp16_to_f32(rb[g]);
            for (uint32_t u = 0U; u < 8U; ++u) {
                const uint32_t pack = rw[g * 8U + u];
                for (uint32_t nib = 0U; nib < 8U; ++nib) {
                    const float q = static_cast<float>((pack >> (4U * nib)) & 0xFU);
                    dr[g * 64U + u * 8U + nib] = sg53_fast_f32_to_fp16(q * sc + bi);
                }
            }
        }
    }
}

bool sg53_ane_kv_build_bundle(uint32_t layer_idx, uint32_t in_ch, uint32_t out_ch, const uint16_t* w_fp16,
                              Sg53AneKvModelBundle* out, uint32_t* ane_ops, uint32_t* total_ops, std::string* err) {
    @autoreleasepool {
        static std::atomic<uint64_t> s_nonce{1ULL};
        std::vector<uint8_t> pb = sg53_serialize_ane_router_mlmodel(in_ch, out_ch, kAneSpatialAlign, w_fp16);
        std::string tmp_root([NSTemporaryDirectory() UTF8String]);
        if (!tmp_root.empty() && tmp_root.back() != '/') tmp_root.push_back('/');
        std::string dir_tmpl = tmp_root + "strata_glm53_kv_ane_XXXXXX";
        std::vector<char> dir_buf(dir_tmpl.begin(), dir_tmpl.end());
        dir_buf.push_back('\0');
        char* staged_dir = ::mkdtemp(dir_buf.data());
        if (!staged_dir) { *err = "mkdtemp failed"; return false; }
        NSURL* dirURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:staged_dir]];
        Sg53ScopedUrlCleanup srcGuard{dirURL};
        NSString* fname = [NSString stringWithFormat:@"l%u_%llu.mlmodel",
                           layer_idx, static_cast<unsigned long long>(s_nonce.fetch_add(1ULL))];
        NSURL* src = [dirURL URLByAppendingPathComponent:fname];
        NSData* data = [NSData dataWithBytesNoCopy:pb.data() length:pb.size() freeWhenDone:NO];
        if (![data writeToURL:src options:NSDataWritingAtomic error:nil]) { *err = "stage .mlmodel failed"; return false; }

        __block NSURL* compiled = nil;
        dispatch_semaphore_t cSem = dispatch_semaphore_create(0);
        [MLModel compileModelAtURL:src completionHandler:^(NSURL* _Nullable u, NSError* _Nullable) {
            compiled = u;
            dispatch_semaphore_signal(cSem);
        }];
        dispatch_semaphore_wait(cSem, DISPATCH_TIME_FOREVER);
        srcGuard.reset();
        if (!compiled) { *err = "CoreML compile failed"; return false; }
        Sg53ScopedUrlCleanup compGuard{compiled};

        MLModelConfiguration* mlCfg = [[MLModelConfiguration alloc] init];
        mlCfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        __block MLComputePlan* plan = nil;
        dispatch_semaphore_t pSem = dispatch_semaphore_create(0);
        [MLComputePlan loadContentsOfURL:compiled configuration:mlCfg completionHandler:^(MLComputePlan* _Nullable cp, NSError* _Nullable) {
            plan = cp;
            dispatch_semaphore_signal(pSem);
        }];
        dispatch_semaphore_wait(pSem, DISPATCH_TIME_FOREVER);
        if (!plan || !plan.modelStructure.neuralNetwork) { *err = "MLComputePlan failed"; return false; }

        uint32_t ane = 0U, tot = 0U;
        for (MLModelStructureNeuralNetworkLayer* op in plan.modelStructure.neuralNetwork.layers) {
            ++tot;
            MLComputePlanDeviceUsage* usage = [plan computeDeviceUsageForNeuralNetworkLayer:op];
            if (usage && [usage.preferredComputeDevice isKindOfClass:[MLNeuralEngineComputeDevice class]]) ++ane;
        }
        *ane_ops   += ane;
        *total_ops += tot;
        if (tot == 0U || ane != tot) { *err = "MLComputePlan preferred non-ANE device"; return false; }

        NSError* lErr = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:compiled configuration:mlCfg error:&lErr];
        compGuard.reset();
        if (!model || lErr) { *err = "MLModel load failed"; return false; }

        const size_t in_b  = sg53_align_up_u64(static_cast<uint64_t>(in_ch) * kAneSpatialAlign * 2ULL, STRATA_GLM53_PAGE_BYTES);
        const size_t out_b = sg53_align_up_u64(static_cast<uint64_t>(out_ch) * kAneSpatialAlign * 2ULL, STRATA_GLM53_PAGE_BYTES);
        void* ib = nullptr;
        void* ob = nullptr;
        if (posix_memalign(&ib, STRATA_GLM53_PAGE_BYTES, in_b) != 0 ||
            posix_memalign(&ob, STRATA_GLM53_PAGE_BYTES, out_b) != 0) {
            if (ib) std::free(ib);
            if (ob) std::free(ob);
            *err = "posix_memalign failed";
            return false;
        }
        std::memset(ib, 0, in_b);
        std::memset(ob, 0, out_b);
        out->layer_idx = layer_idx;
        out->in_ch     = in_ch;
        out->out_ch    = out_ch;
        out->in_buf    = static_cast<uint16_t*>(ib);
        out->out_buf   = static_cast<uint16_t*>(ob);
        NSError* aErr  = nil;
        out->in_arr = [[MLMultiArray alloc] initWithDataPointer:ib
                                                          shape:@[@1, @(in_ch), @1, @(kAneSpatialAlign)]
                                                       dataType:MLMultiArrayDataTypeFloat16
                                                        strides:@[@(in_ch * kAneSpatialAlign), @(kAneSpatialAlign), @(kAneSpatialAlign), @1]
                                                    deallocator:nil
                                                          error:&aErr];
        out->out_arr = [[MLMultiArray alloc] initWithDataPointer:ob
                                                           shape:@[@1, @(out_ch), @1, @(kAneSpatialAlign)]
                                                        dataType:MLMultiArrayDataTypeFloat16
                                                         strides:@[@(out_ch * kAneSpatialAlign), @(kAneSpatialAlign), @(kAneSpatialAlign), @1]
                                                     deallocator:nil
                                                           error:&aErr];
        if (!out->in_arr || !out->out_arr) { *err = "MLMultiArray init failed"; return false; }
        out->fp   = [[StrataGlm53AneFeatureProvider alloc] initWithMultiArray:out->in_arr];
        out->opts = [[MLPredictionOptions alloc] init];
        out->opts.outputBackings = @{@"y": out->out_arr};
        out->model = model;

        NSError* wErr = nil;
        id<MLFeatureProvider> warm = [model predictionFromFeatures:out->fp options:out->opts error:&wErr];
        if (!warm || wErr) { out->model = nil; *err = "ANE warmup failed"; return false; }
        return true;
    }
}

void sg53_ane_kv_probe_init(StrataGlm53GpuEngine* eng) {
    if (eng->ane_kv_init_tried) return;
    eng->ane_kv_init_tried = true;
    const auto t0 = std::chrono::steady_clock::now();
    auto* pr = new Sg53AneKvProbe();
    pr->n_layers = eng->cfg.num_dsa_layers;
    pr->qr       = eng->cfg.q_lora_rank;
    pr->idx_nh   = eng->cfg.index_n_heads;
    pr->idx_hd   = eng->cfg.index_head_dim;
    pr->queue    = dispatch_queue_create("strata.glm53.kv_ane_lookahead", DISPATCH_QUEUE_SERIAL);
    const uint32_t q_rows = pr->idx_nh * pr->idx_hd;
    pr->qresid_snap.assign(static_cast<size_t>(pr->n_layers) * pr->qr, 0U);
    pr->whead_snap.assign(static_cast<size_t>(pr->n_layers) * pr->idx_nh, 0.0f);
    pr->q_hat.assign(static_cast<size_t>(pr->n_layers) * q_rows, 0.0f);

    @autoreleasepool {
        for (id<MLComputeDeviceProtocol> dev in MLAllComputeDevices()) {
            if ([dev isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
                pr->ane_cores = static_cast<MLNeuralEngineComputeDevice*>(dev).totalCoreCount;
            }
        }
    }

    std::string err;
    bool all_ok = (pr->ane_cores > 0) && (pr->n_layers > 0U) && (pr->qr >= 64U) && (q_rows >= 64U);
    std::vector<uint16_t> w_fp16(static_cast<size_t>(q_rows) * pr->qr, 0U);
    uint32_t dsa_i = 0U;
    for (uint32_t l = 0U; l < eng->cfg.num_hidden_layers && dsa_i < pr->n_layers && all_ok; ++l) {
        const StrataGlm53LayerDesc& ld = eng->hdr.layers[l];
        if (ld.attn_type != STRATA_GLM53_ATTN_DSA_MLA) continue;
        sg53_dequant_q4_to_fp16(eng->bundle->spine_base, ld.indexer_wq_b, q_rows, pr->qr, w_fp16.data());
        Sg53AneKvModelBundle bdl{};
        if (!sg53_ane_kv_build_bundle(dsa_i, pr->qr, q_rows, w_fp16.data(), &bdl, &pr->ane_ops, &pr->total_ops, &err)) {
            if (bdl.in_buf)  std::free(bdl.in_buf);
            if (bdl.out_buf) std::free(bdl.out_buf);
            all_ok = false;
            break;
        }
        pr->bundles.push_back(bdl);
        ++dsa_i;
    }
    pr->ready = all_ok && (pr->bundles.size() == pr->n_layers);
    if (pr->ready) {
        dispatch_sync(pr->queue, ^{
            @autoreleasepool {
                for (Sg53AneKvModelBundle& bdl : pr->bundles) {
                    NSError* wErr = nil;
                    (void)[bdl.model predictionFromFeatures:bdl.fp options:bdl.opts error:&wErr];
                }
            }
        });
        pr->init_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        char nb[384];
        std::snprintf(nb, sizeof(nb),
                      "CoreML spec-v7 MLComputePlan-verified %ld-core ANE (MLNeuralEngineComputeDevice %u/%u ops, "
                      "%zu DSA MLA layers x %u->%u FP16 indexer_wq_b, init %.0f ms)",
                      pr->ane_cores, pr->ane_ops, pr->total_ops, pr->bundles.size(), pr->qr, q_rows, pr->init_ms);
        pr->note = nb;
    } else {
        pr->note = "ANE KV probe unavailable: " + err;
    }
    eng->ane_kv = pr;
}

void sg53_ane_kv_probe_execute(StrataGlm53GpuEngine* eng, StrataGlm53ContextSlot* s, uint32_t q_pos) {
    Sg53AneKvProbe* pr = eng->ane_kv;
    const uint64_t t0 = mach_absolute_time();
    const uint32_t q_rows = pr->idx_nh * pr->idx_hd;
    bool on_ane = pr->ready;
    @autoreleasepool {
        for (Sg53AneKvModelBundle& bdl : pr->bundles) {
            const uint32_t S = kAneSpatialAlign;
            const uint16_t* qs = &pr->qresid_snap[static_cast<size_t>(bdl.layer_idx) * pr->qr];
            for (uint32_t c = 0U; c < pr->qr; ++c) bdl.in_buf[static_cast<size_t>(c) * S] = qs[c];
            NSError* err = nil;
            id<MLFeatureProvider> outF = [bdl.model predictionFromFeatures:bdl.fp options:bdl.opts error:&err];
            if (!outF || err) { on_ane = false; continue; }
            MLMultiArray* oa = [outF featureValueForName:@"y"].multiArrayValue;
            float* qh = &pr->q_hat[static_cast<size_t>(bdl.layer_idx) * q_rows];
            if (oa == bdl.out_arr || !oa) {
                const __fp16* ob = reinterpret_cast<const __fp16*>(bdl.out_buf);
                for (uint32_t oc = 0U; oc < q_rows; ++oc) qh[oc] = static_cast<float>(ob[static_cast<size_t>(oc) * S]);
            } else {
                [oa getBytesWithHandler:^(const void* bytes, NSInteger size) {
                    if (size >= static_cast<NSInteger>(static_cast<size_t>(q_rows) * S * 2U)) {
                        const __fp16* ob = static_cast<const __fp16*>(bytes);
                        for (uint32_t oc = 0U; oc < q_rows; ++oc) qh[oc] = static_cast<float>(ob[static_cast<size_t>(oc) * S]);
                    }
                }];
            }
        }
    }
    const uint64_t t_ane_done = mach_absolute_time();

    const uint32_t nc        = (q_pos + 1U) / STRATA_GLM53_KV_BLOCK_TOKENS;
    const uint32_t n_lyr     = std::min<uint32_t>(pr->n_layers, s->n_layers);
    const uint8_t* pk_base   = static_cast<const uint8_t*>([s->b_idx_pool_k contents]);
    const float    inv_sqrt  = 1.0f / std::sqrt(static_cast<float>(pr->idx_hd));
    const uint32_t pin_epoch = ++s->pin_epoch_ctr;
    const size_t   blk_bytes = sg53_paged_block_bytes(s);
    std::vector<Sg53PagedCounters> lc(n_lyr);
    Sg53PagedCounters* lcp = lc.data();

    dispatch_apply(n_lyr, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t li) {
        const uint32_t ls = static_cast<uint32_t>(li);
        Sg53PagedLayerState& L = s->layers[ls];
        if (!L.has_thresh || !s->ane_stage) return;
        uint8_t* l_stage   = s->ane_stage + li * static_cast<size_t>(kSg53AneMaxPrefetchLyr) * blk_bytes;
        const __fp16* pk   = reinterpret_cast<const __fp16*>(pk_base + static_cast<size_t>(ls) * eng->idx_pool_layer_stride);
        const float*  qh   = &pr->q_hat[static_cast<size_t>(ls) * q_rows];
        const float*  wh   = &pr->whead_snap[static_cast<size_t>(ls) * pr->idx_nh];
        const float   thr  = L.last_thresh * kSg53AnePrefetchMargin;
        std::vector<std::pair<float, uint32_t>> cand;
        for (uint32_t b = 0U; b < nc && b < s->max_blocks; ++b) {
            if (L.st[b] != kSg53BlkPaged) continue;
            const __fp16* row = pk + static_cast<size_t>(b) * pr->idx_hd;
            float tot = 0.0f;
            for (uint32_t h = 0U; h < pr->idx_nh; ++h) {
                const float* q = qh + static_cast<size_t>(h) * pr->idx_hd;
                float d = 0.0f;
                for (uint32_t k = 0U; k < pr->idx_hd; ++k) d += q[k] * static_cast<float>(row[k]);
                tot += wh[h] * std::max(0.0f, d * inv_sqrt);
            }
            if (tot >= thr) cand.emplace_back(tot, b);
        }
        if (cand.empty()) return;
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b2) { return a.first > b2.first; });
        if (cand.size() > kSg53AneMaxPrefetchLyr) cand.resize(kSg53AneMaxPrefetchLyr);
        std::vector<uint32_t> need;
        for (const auto& pc : cand) { L.pin_epoch[pc.second] = pin_epoch; need.push_back(pc.second); }
        const uint32_t newest   = (nc > 0U) ? nc - 1U : 0U;
        const uint32_t free_cnt = s->pool_blocks - L.resident;
        if (need.size() > free_cnt) {
            const uint32_t got = sg53_paged_evict_staged(
                s, ls, static_cast<uint32_t>(need.size()) - free_cnt, newest, false, l_stage, kSg53AneMaxPrefetchLyr, lcp[li]);
            const uint32_t cap = free_cnt + got;
            if (need.size() > cap) need.resize(cap);
        }
        if (need.empty()) return;
        (void)sg53_paged_ensure_staged(s, ls, need, newest, false, l_stage, kSg53AneMaxPrefetchLyr, lcp[li]);
        for (uint32_t b : need) {
            if (L.st[b] == kSg53BlkResident) { L.pf_epoch[b] = s->step_epoch; lcp[li].pf_issued += 1ULL; }
        }
    });

    Sg53PagedCounters tot;
    for (const auto& c : lc) {
        tot.paged_out += c.paged_out; tot.recalled += c.recalled; tot.wbytes += c.wbytes; tot.rbytes += c.rbytes;
        tot.wms += c.wms; tot.rms += c.rms; tot.pf_issued += c.pf_issued; tot.err = tot.err || c.err;
    }
    sg53_paged_merge(s, tot);
    s->stats.ane_probe_calls += 1ULL;
    if (on_ane) {
        s->stats.ane_probe_on_ane += 1ULL;
        s->stats.ran_on_ane = 1U;
    }
    s->ane_us_total += sg53_ticks_to_sec(t_ane_done - t0) * 1e6;
}

void sg53_paged_step_begin(StrataGlm53GpuEngine* eng, uint32_t q_pos_last) {
    StrataGlm53ContextSlot* s = eng->bound_slot;
    if (!s || !s->paged) return;
    sg53_paged_join_probe(s);
    s->step_epoch += 1U;
    const uint32_t kpool    = std::max<uint32_t>(1U, eng->cfg.index_kpool);
    const uint32_t select_k = std::max<uint32_t>(1U, eng->cfg.index_topk / kpool);
    const uint32_t nc       = (q_pos_last + 1U) / kpool;
    if (!eng->ane_kv || !eng->ane_kv->ready || !s->q_valid || nc <= select_k) return;
    Sg53AneKvProbe* pr = eng->ane_kv;
    std::memcpy(pr->qresid_snap.data(), [eng->b_kv_probe_qresid contents], pr->qresid_snap.size() * sizeof(uint16_t));
    std::memcpy(pr->whead_snap.data(), [eng->b_kv_probe_whead contents], pr->whead_snap.size() * sizeof(float));
    s->probe_pending = true;
    dispatch_group_async(s->probe_grp, pr->queue, ^{
        sg53_ane_kv_probe_execute(eng, s, q_pos_last);
    });
}

void sg53_encode_attention_sublayer(
    id<MTLCommandBuffer> __strong*         io_cb,
    id<MTLComputeCommandEncoder> __strong* io_enc,
    StrataGlm53GpuEngine*                  eng,
    const StrataGlm53LayerDesc&            ld,
    id<MTLBuffer>                          stream_buf,
    uint32_t                               kda_slot,
    uint32_t                               dsa_slot,
    uint32_t                               seq_pos_start,
    uint32_t                               M_valid,
    uint32_t                               M_padded,
    uint32_t                     save_snap_mask,
    bool                         force_gemv
) {
    /* `enc` ALIASES the caller's encoder (not a copy): the paged-KV helpers below end it, commit `*io_cb` and replace
     * both. Every return path, including the `paged_error` early returns, must leave the caller holding the live
     * `(command buffer, encoder)` pair; with a stale copy the caller would end a finished encoder and commit a
     * buffer whose encoder is still open, which Metal answers with an abort. */
    id<MTLComputeCommandEncoder> __strong& enc = *io_enc;
    const bool need_x_f32 = (ld.attn_type == STRATA_GLM53_ATTN_DSA_MLA); // `indexer_compress_gate` reads FP32 `x`!
    sg53_encode_sublayer_entry(enc, eng, ld.has_mhc != 0U, ld.hc_attn, ld.input_layernorm,
                               stream_buf, M_valid, M_padded, need_x_f32, /*write_unit_f16=*/false);

    if (ld.attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) {
        const uint32_t nh       = eng->cfg.linear_num_heads;
        const uint32_t hd       = eng->cfg.linear_head_dim;
        const uint32_t conv_dim = 3U * nh * hd;

        sg53_encode_q4_linear(enc, eng, ld.lin_qkv_proj,
                              eng->b_x_in_f16, 0, eng->b_kda_qkv_f16, 0, nil, 0,
                              M_valid, M_padded, kOutF16, force_gemv);
        sg53_encode_q4_linear(enc, eng, ld.lin_fbg_a_proj,
                              eng->b_x_in_f16, 0, nil, 0, eng->b_kda_fbg_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);

        {
            Sg53KdaFbgSplitParamsHost sp{M_valid, M_padded, nh, hd};
            [enc setComputePipelineState:eng->pso_kda_fbg_split];
            [enc setBuffer:eng->b_kda_fbg_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_kda_fa_f16 offset:0 atIndex:1];
            [enc setBuffer:eng->b_kda_b_f32 offset:0 atIndex:2];
            [enc setBuffer:eng->b_kda_ga_f16 offset:0 atIndex:3];
            [enc setBytes:&sp length:sizeof(sp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        sg53_encode_q4_linear(enc, eng, ld.lin_f_b_proj,
                              eng->b_kda_fa_f16, 0, nil, 0, eng->b_kda_a_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
        sg53_encode_q4_linear(enc, eng, ld.lin_g_b_proj,
                              eng->b_kda_ga_f16, 0, eng->b_kda_gate_f16, 0, nil, 0,
                              M_valid, M_padded, kOutF16, force_gemv);

        {
            Sg53KdaConvParamsHost cp{
                M_valid, conv_dim, save_snap_mask, kda_slot, eng->cfg.num_kda_layers, 0U, 0U, 0U
            };
            [enc setComputePipelineState:eng->pso_kda_conv1d];
            [enc setBuffer:eng->b_kda_qkv_f16 offset:0 atIndex:0];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.lin_qkv_conv1d.offset) atIndex:1];
            [enc setBuffer:eng->b_conv_state offset:static_cast<NSUInteger>(static_cast<uint64_t>(kda_slot) * eng->conv_layer_stride) atIndex:2];
            [enc setBuffer:eng->b_kda_qkv_conv_f16 offset:0 atIndex:3];
            [enc setBuffer:(eng->b_conv_snaps ? eng->b_conv_snaps : eng->b_conv_state) offset:0 atIndex:4];
            [enc setBytes:&cp length:sizeof(cp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((conv_dim + 255U) / 256U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        Sg53KdaParamsHost kp{
            M_valid, M_padded, nh, hd, 1e-6f, eng->cfg.rms_norm_eps, eng->cfg.linear_gate_lower_bound,
            save_snap_mask, kda_slot, eng->cfg.num_kda_layers, 0U, 0U
        };
        const NSUInteger rec_off = static_cast<NSUInteger>(static_cast<uint64_t>(kda_slot) * eng->rec_layer_stride);

        if (force_gemv || save_snap_mask != 0U || hd != 128U) {
            [enc setComputePipelineState:eng->pso_kda_rec];
            [enc setBuffer:eng->b_kda_qkv_conv_f16 offset:0 atIndex:0];
            [enc setBuffer:eng->b_kda_a_f32 offset:0 atIndex:1];
            [enc setBuffer:eng->b_kda_b_f32 offset:0 atIndex:2];
            [enc setBuffer:eng->b_kda_gate_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.lin_A_log.offset) atIndex:4];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.lin_dt_bias.offset) atIndex:5];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.lin_o_norm.offset) atIndex:6];
            [enc setBuffer:eng->b_rec_state offset:rec_off atIndex:7];
            [enc setBuffer:eng->b_kda_y_f16 offset:0 atIndex:8];
            [enc setBuffer:(eng->b_rec_snaps ? eng->b_rec_snaps : eng->b_rec_state) offset:0 atIndex:9];
            [enc setBytes:&kp length:sizeof(kp) atIndex:10];
            [enc dispatchThreadgroups:MTLSizeMake(nh, 1, 1)
                threadsPerThreadgroup:MTLSizeMake((hd == 128U) ? 512U : std::max<uint32_t>(32U, hd), 1, 1)];
        } else {
            // Chunk-32 WY Metal 4.1 NAX KDA (aliases dead `b_moe_down_part`, 0 B extra VRAM!)
            const uint32_t M_wy       = (M_valid + 31U) & ~31U;
            const uint32_t num_chunks = M_wy / 32U;
            const uint64_t qk_bytes   = sg53_align_up_u64(static_cast<uint64_t>(nh) * M_wy * 128ULL * sizeof(uint16_t), 256ULL);
            const uint64_t p_bytes    = sg53_align_up_u64(static_cast<uint64_t>(nh) * num_chunks * 1024ULL * sizeof(uint16_t), 256ULL);
            const uint64_t gl_bytes   = sg53_align_up_u64(static_cast<uint64_t>(nh) * num_chunks * 128ULL * sizeof(float), 256ULL);
            const uint64_t q_sc_off   = 0ULL;
            const uint64_t k_end_off  = q_sc_off  + qk_bytes;
            const uint64_t w_off      = k_end_off + qk_bytes;
            const uint64_t u_off      = w_off     + qk_bytes;
            const uint64_t p_off      = u_off     + qk_bytes;
            const uint64_t gl_off     = p_off     + p_bytes;
            const uint64_t wy_total   = gl_off    + gl_bytes;

            const bool use_wy = (!eng->force_serial_kda) && eng->has_nax && (M_valid > kMaxSpecBatch) &&
                                (M_wy <= M_padded) && ([eng->b_moe_down_part length] >= wy_total);
            if (use_wy) {
                Sg53KdaWyParamsHost wyp{
                    M_valid, M_wy, num_chunks, nh, 1e-6f, eng->cfg.linear_gate_lower_bound, 0U, 0U
                };
                [enc setComputePipelineState:eng->pso_kda_wy_prep_chunk32];
                [enc setBuffer:eng->b_kda_qkv_conv_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_kda_a_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_kda_b_f32 offset:0 atIndex:2];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.lin_A_log.offset) atIndex:3];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.lin_dt_bias.offset) atIndex:4];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(q_sc_off) atIndex:5];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(k_end_off) atIndex:6];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(w_off) atIndex:7];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(u_off) atIndex:8];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(p_off) atIndex:9];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(gl_off) atIndex:10];
                [enc setBytes:&wyp length:sizeof(wyp) atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake(num_chunks, nh, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_kda_wy_recur_chunk32];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(q_sc_off) atIndex:0];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(k_end_off) atIndex:1];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(w_off) atIndex:2];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(u_off) atIndex:3];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(p_off) atIndex:4];
                [enc setBuffer:eng->b_moe_down_part offset:static_cast<NSUInteger>(gl_off) atIndex:5];
                [enc setBuffer:eng->b_rec_state offset:rec_off atIndex:6];
                [enc setBuffer:eng->b_kda_raw_out_f32 offset:0 atIndex:7];
                [enc setBytes:&wyp length:sizeof(wyp) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(4, nh, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            } else {
                [enc setComputePipelineState:eng->pso_kda_prep_batch];
                [enc setBuffer:eng->b_kda_qkv_conv_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_kda_a_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_kda_b_f32 offset:0 atIndex:2];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.lin_A_log.offset) atIndex:3];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.lin_dt_bias.offset) atIndex:4];
                [enc setBuffer:eng->b_kda_qk_hat_f32 offset:0 atIndex:5];
                [enc setBuffer:eng->b_kda_decay_f32 offset:0 atIndex:6];
                [enc setBuffer:eng->b_kda_beta_f32 offset:0 atIndex:7];
                [enc setBytes:&kp length:sizeof(kp) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(3U * nh + 1U, M_valid, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_kda_rec_batch];
                [enc setBuffer:eng->b_kda_qkv_conv_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_kda_qk_hat_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_kda_decay_f32 offset:0 atIndex:2];
                [enc setBuffer:eng->b_kda_beta_f32 offset:0 atIndex:3];
                [enc setBuffer:eng->b_rec_state offset:rec_off atIndex:4];
                [enc setBuffer:eng->b_kda_raw_out_f32 offset:0 atIndex:5];
                [enc setBytes:&kp length:sizeof(kp) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(4, nh, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            [enc setComputePipelineState:eng->pso_kda_post_batch];
            [enc setBuffer:eng->b_kda_raw_out_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_kda_gate_f16 offset:0 atIndex:1];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.lin_o_norm.offset) atIndex:2];
            [enc setBuffer:eng->b_kda_y_f16 offset:0 atIndex:3];
            [enc setBytes:&kp length:sizeof(kp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(nh, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        sg53_encode_q4_linear(enc, eng, ld.lin_o_proj,
                              eng->b_kda_y_f16, 0, nil, 0, eng->b_block_y_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
    } else {
        // 12-Layer DSA NoPE MLA + Glm5NextIndexer
        const uint32_t nh        = eng->cfg.num_attention_heads;
        const uint32_t qr        = eng->cfg.q_lora_rank;
        const uint32_t kvr       = eng->cfg.kv_lora_rank;
        const uint32_t q_nope    = eng->cfg.qk_nope_head_dim;
        const uint32_t v_dim     = eng->cfg.v_head_dim;
        const uint32_t idx_nh    = eng->cfg.index_n_heads;
        const uint32_t idx_hd    = eng->cfg.index_head_dim;
        const uint32_t kpool     = std::max<uint32_t>(1U, eng->cfg.index_kpool);
        const uint32_t select_k  = std::max<uint32_t>(1U, eng->cfg.index_topk / kpool);
        const float    eps       = eng->cfg.rms_norm_eps;
        const uint32_t total_kv  = seq_pos_start + M_valid;

        const NSUInteger kv_layer_off   = static_cast<NSUInteger>(static_cast<uint64_t>(dsa_slot) * eng->kv_layer_stride);
        const NSUInteger pool_layer_off = static_cast<NSUInteger>(static_cast<uint64_t>(dsa_slot) * eng->idx_pool_layer_stride);
        const NSUInteger tail_layer_off = static_cast<NSUInteger>(static_cast<uint64_t>(dsa_slot) * eng->idx_tail_layer_stride);

        // 1. `qkv_a = mla_qkv_a_proj(x)` -> `q_resid` & compressed latent `c_kv [512]` written straight to KV cache!
        sg53_encode_q4_linear(enc, eng, ld.mla_qkv_a_proj,
                              eng->b_x_in_f16, 0, nil, 0, eng->b_mla_qkv_a_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);

        {
            Sg53RmsNormParamsHost qnp{
                M_valid, M_padded, qr, qr + kvr, 0U, qr, 0U, eps, 1U, 0U, 0U, 0U
            };
            [enc setComputePipelineState:eng->pso_rmsnorm_strided];
            [enc setBuffer:eng->b_mla_qkv_a_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.mla_q_a_layernorm.offset) atIndex:1];
            [enc setBuffer:eng->b_mla_q_resid_f16 offset:0 atIndex:2];
            [enc setBuffer:eng->b_mla_qkv_a_f32 offset:0 atIndex:3];
            [enc setBytes:&qnp length:sizeof(qnp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // 2. `q = mla_q_b_proj(q_resid)` -> `q_latent = mla_w_uk @ q`
        //    During Prefill (`M > 8`, `kvr == 512`, `kpool == 4`): use Head-Major Direct-Device `static_slice`
        //    `W_UK`/`W_UV` + Zero-`part_o` Metal 4.1 NAX Flash-MLA!
        const bool use_flash_mla =
            (!force_gemv) && eng->has_nax && (M_valid > kMaxSpecBatch) &&
            (kvr == 512U) && (kpool == 4U) && ((M_padded % 16U) == 0U) &&
            ((q_nope % 64U) == 0U) && ((v_dim % 64U) == 0U) &&
            (eng->kv_paged || ((eng->kv_stride_tokens % 32U) == 0U)) &&
            ([eng->b_mla_v_out_f16 length] >= static_cast<size_t>(nh) * M_padded * q_nope * sizeof(uint16_t));

        sg53_encode_q4_linear(enc, eng, ld.mla_q_b_proj,
                              eng->b_mla_q_resid_f16, 0, eng->b_mla_q_f16, 0, nil, 0,
                              M_valid, M_padded, kOutF16, force_gemv);
        if (use_flash_mla) {
            Sg53MlaBatchedGemmParamsHost tp{M_valid, M_padded, nh, 0U, q_nope, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_mla_q_to_hm];
            [enc setBuffer:eng->b_mla_q_f16 offset:0 atIndex:0];
            [enc setBuffer:eng->b_mla_v_out_f16 offset:0 atIndex:1]; // dead alias -> q_hm [nh, M_padded, q_nope]
            [enc setBytes:&tp length:sizeof(tp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(nh, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            sg53_encode_mla_batched_f16_hm(enc, eng, ld.mla_w_uk,
                                           eng->b_mla_v_out_f16, eng->b_mla_q_latent_f16,
                                           M_valid, M_padded, nh, kvr, q_nope, /*out_head_major=*/true);
        } else {
            sg53_encode_mla_batched_f16(enc, eng, ld.mla_w_uk,
                                        eng->b_mla_q_f16, eng->b_mla_q_latent_f16,
                                        M_valid, M_padded, nh, kvr, q_nope, force_gemv);
        }

        // 3. `Glm5NextIndexer` key/gate update + 4-way per-channel pool softmax compression
        sg53_encode_q4_linear(enc, eng, ld.indexer_wk,
                              eng->b_x_in_f16, 0, nil, 0, eng->b_idx_wk_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
        sg53_encode_f32_linear(enc, eng, ld.indexer_compress_gate,
                               eng->b_x_in_f32, 0, eng->b_idx_kgate_f32, 0,
                               M_valid, kOutF32);

        const uint32_t first_pool     = seq_pos_start / kpool;
        const uint32_t end_pool       = total_kv / kpool;
        const uint32_t num_new_pools  = (end_pool > first_pool) ? (end_pool - first_pool) : 0U;
        const uint32_t max_q_complete = total_kv / kpool;
        const bool     any_sparse     = (max_q_complete > select_k);

        Sg53IndexerParamsHost ip{
            M_valid, idx_nh, idx_hd, kpool, select_k,
            seq_pos_start, first_pool, num_new_pools,
            eng->cfg.indexer_layernorm_eps,
            eng->max_pools_alloc, eng->max_sel_words, 0U
        };

        {
            [enc setComputePipelineState:eng->pso_idx_k_layernorm];
            [enc setBuffer:eng->b_idx_wk_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_idx_kgate_f32 offset:0 atIndex:1];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.indexer_k_norm_weight.offset) atIndex:2];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.indexer_k_norm_bias.offset) atIndex:3];
            [enc setBuffer:eng->b_idx_cur_kg_f32 offset:0 atIndex:4];
            [enc setBytes:&ip length:sizeof(ip) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::max<uint32_t>(32U, idx_hd), 1, 1)];
        }

        if (num_new_pools > 0U) {
            [enc setComputePipelineState:eng->pso_idx_update_pools];
            [enc setBuffer:eng->b_idx_tail_ring offset:tail_layer_off atIndex:0];
            [enc setBuffer:eng->b_idx_cur_kg_f32 offset:0 atIndex:1];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.indexer_compress_ape.offset) atIndex:2];
            [enc setBuffer:eng->b_idx_pool_k offset:pool_layer_off atIndex:3];
            [enc setBytes:&ip length:sizeof(ip) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(num_new_pools, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::max<uint32_t>(32U, idx_hd), 1, 1)];
        }

        {
            const uint32_t keep_tail = std::min<uint32_t>(M_valid, 16U);
            [enc setComputePipelineState:eng->pso_idx_commit_tail];
            [enc setBuffer:eng->b_idx_cur_kg_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_idx_tail_ring offset:tail_layer_off atIndex:1];
            [enc setBytes:&ip length:sizeof(ip) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(keep_tail, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::max<uint32_t>(32U, idx_hd), 1, 1)];
        }

        // Sparse candidate selection (`q_pos >= 2051`; skipped completely on fast dense path `q_pos < 2051`!)
        if (any_sparse) {
            sg53_encode_q4_linear(enc, eng, ld.indexer_wq_b,
                                  eng->b_mla_q_resid_f16, 0, nil, 0, eng->b_idx_q_f32, 0,
                                  M_valid, M_padded, kOutF32, force_gemv);
            sg53_encode_q4_linear(enc, eng, ld.indexer_weights_proj,
                                  eng->b_x_in_f16, 0, nil, 0, eng->b_idx_whead_f32, 0,
                                  M_valid, M_padded, kOutF32, force_gemv);

            {
                const uint32_t tot_w = M_valid * idx_nh;
                [enc setComputePipelineState:eng->pso_idx_scale_weights];
                [enc setBuffer:eng->b_idx_whead_f32 offset:0 atIndex:0];
                [enc setBytes:&ip length:sizeof(ip) atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake((tot_w + 255U) / 256U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            const bool use_idx_nax =
                (!force_gemv) && eng->has_nax && (idx_nh == 32U) && (idx_hd == 128U) &&
                (M_valid > kMaxSpecBatch) && ((M_padded % 64U) == 0U) &&
                ((eng->max_pools_alloc % 64U) == 0U) &&
                ([eng->b_ffn_mid_f16 length] >= static_cast<size_t>(idx_nh) * M_padded * idx_hd * sizeof(uint16_t));
            const bool use_idx_decode_nax =
                eng->has_nax && (idx_nh == 32U) && (idx_hd == 128U) &&
                ((eng->max_pools_alloc % 64U) == 0U);
            const bool use_tiled = (!force_gemv) && (idx_nh == 32U) && (idx_hd == 128U);
            if (use_idx_nax) {
                Sg53IndexerParamsHost ip_nax = ip;
                ip_nax.pad0 = M_padded;
                [enc setComputePipelineState:eng->pso_idx_q_to_hm];
                [enc setBuffer:eng->b_idx_q_f32 offset:0 atIndex:0];
                [enc setBuffer:eng->b_ffn_mid_f16 offset:0 atIndex:1]; // dead alias -> q_idx_hm_f16 [32, M_padded, 128]
                [enc setBytes:&ip_nax length:sizeof(ip_nax) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(idx_nh, M_padded, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:eng->pso_idx_score_nax];
                [enc setBuffer:eng->b_ffn_mid_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_idx_whead_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_idx_pool_k offset:pool_layer_off atIndex:2];
                [enc setBuffer:eng->b_idx_scores_f32 offset:0 atIndex:3];
                [enc setBytes:&ip_nax length:sizeof(ip_nax) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((max_q_complete + 63U) / 64U, M_padded / 64U, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else if (use_idx_decode_nax) {
                [enc setComputePipelineState:eng->pso_idx_score_decode_nax];
                [enc setBuffer:eng->b_idx_q_f32 offset:0 atIndex:0];
                [enc setBuffer:eng->b_idx_whead_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_idx_pool_k offset:pool_layer_off atIndex:2];
                [enc setBuffer:eng->b_idx_scores_f32 offset:0 atIndex:3];
                [enc setBytes:&ip length:sizeof(ip) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((max_q_complete + 63U) / 64U, M_valid, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else if (use_tiled) {
                [enc setComputePipelineState:eng->pso_idx_score_tiled];
                [enc setBuffer:eng->b_idx_q_f32 offset:0 atIndex:0];
                [enc setBuffer:eng->b_idx_whead_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_idx_pool_k offset:pool_layer_off atIndex:2];
                [enc setBuffer:eng->b_idx_scores_f32 offset:0 atIndex:3];
                [enc setBytes:&ip length:sizeof(ip) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((max_q_complete + 63U) / 64U, (M_valid + 7U) / 8U, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [enc setComputePipelineState:eng->pso_idx_score_pools];
                [enc setBuffer:eng->b_idx_q_f32 offset:0 atIndex:0];
                [enc setBuffer:eng->b_idx_whead_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_idx_pool_k offset:pool_layer_off atIndex:2];
                [enc setBuffer:eng->b_idx_scores_f32 offset:0 atIndex:3];
                [enc setBytes:&ip length:sizeof(ip) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((max_q_complete + 7U) / 8U, M_valid, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            [enc setComputePipelineState:eng->pso_idx_topk_pools];
            [enc setBuffer:eng->b_idx_scores_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_idx_sel_bits offset:0 atIndex:1];
            [enc setBuffer:eng->b_idx_sel_pools offset:0 atIndex:2];
            [enc setBytes:&ip length:sizeof(ip) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }

        // Step 4 Tier 2: exact selected-pool residency check + physical block allocation
        const uint32_t   paged_u = eng->kv_paged ? 1U : 0U;
        id<MTLBuffer>    loc_buf = eng->b_block_loc;
        const NSUInteger loc_off = eng->kv_paged
            ? static_cast<NSUInteger>(static_cast<uint64_t>(dsa_slot) * eng->max_pools_alloc * sizeof(uint32_t))
            : 0U;
        std::vector<Sg53MGroup> mgroups;
        if (eng->kv_paged) {
            if (any_sparse) {
                if (dsa_slot < eng->cfg.num_dsa_layers) {
                    sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                            eng->b_mla_q_resid_f16, static_cast<NSUInteger>(M_valid - 1U) * qr * sizeof(uint16_t),
                                            eng->b_kv_probe_qresid, static_cast<NSUInteger>(dsa_slot) * qr * sizeof(uint16_t),
                                            static_cast<size_t>(qr) * sizeof(uint16_t));
                    sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                            eng->b_idx_whead_f32, static_cast<NSUInteger>(M_valid - 1U) * idx_nh * sizeof(float),
                                            eng->b_kv_probe_whead, static_cast<NSUInteger>(dsa_slot) * idx_nh * sizeof(float),
                                            static_cast<size_t>(idx_nh) * sizeof(float));
                }
                sg53_paged_sync_point(eng, enc, *io_cb, dsa_slot, seq_pos_start, M_valid, select_k, kpool, &mgroups);
            } else {
                sg53_paged_alloc_range(eng, enc, *io_cb, dsa_slot, seq_pos_start, M_valid);
            }
            if (eng->paged_error) return;
        }

        // Write normalized compressed latent `c_kv [M_valid, kvr]` into linear or paged `b_mla_kv_cache`!
        {
            Sg53MlaKvWriteParamsHost kvp{
                M_valid, kvr, qr + kvr, qr, seq_pos_start, paged_u, eps, 0U
            };
            [enc setComputePipelineState:eng->pso_mla_kv_write];
            [enc setBuffer:eng->b_mla_qkv_a_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.mla_kv_a_layernorm.offset) atIndex:1];
            [enc setBuffer:eng->b_mla_kv_cache offset:kv_layer_off atIndex:2];
            [enc setBuffer:loc_buf offset:loc_off atIndex:3];
            [enc setBytes:&kvp length:sizeof(kvp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, M_valid, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // 4. Absorbed Multi-Query (`64:1`) Latent Attention + `W_UV` Unembed + `mla_o_proj`
        const uint32_t q_per_tg      = std::min<uint32_t>(4U, nh);
        const uint32_t num_q_groups  = (nh + q_per_tg - 1U) / q_per_tg;
        const uint32_t max_eff_kv    = any_sparse ? (select_k * kpool + kpool) : total_kv;
        const uint32_t launch_splits = std::clamp<uint32_t>((max_eff_kv + 63U) / 64U, 1U, kMaxDecodeSplits);
        const float    attn_scale    = 1.0f / std::sqrt(static_cast<float>(q_nope));
        const uint32_t newest_blk    = (total_kv - 1U) / kpool;

        Sg53MlaDecodeParamsHost mp{
            M_valid, nh, kvr, q_per_tg, seq_pos_start, eng->kv_stride_tokens,
            kMaxDecodeSplits, kpool, select_k, attn_scale, 0U, paged_u,
            M_valid, M_padded, total_kv, 0U
        };

        if (use_flash_mla) {
            if (!any_sparse && !eng->kv_paged) {
                [enc setComputePipelineState:eng->pso_mla_prefill_dense_nax];
                [enc setBuffer:eng->b_mla_q_latent_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_mla_kv_cache offset:kv_layer_off atIndex:1];
                [enc setBuffer:eng->b_mla_c_out_f16 offset:0 atIndex:2];
                [enc setBytes:&mp length:sizeof(mp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(M_padded / 16U, nh, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                const size_t n_groups = mgroups.empty() ? 1U : mgroups.size();
                for (size_t g = 0U; g < n_groups; ++g) {
                    const uint32_t g_beg = mgroups.empty() ? 0U : mgroups[g].beg;
                    const uint32_t g_end = mgroups.empty() ? M_valid : mgroups[g].end;
                    if (g > 0U) {
                        sg53_paged_commit_wait(eng, enc, *io_cb);
                        const Sg53MGroup* ng = (g + 1U < n_groups) ? &mgroups[g + 1U] : nullptr;
                        sg53_paged_make_group_resident(eng, dsa_slot, newest_blk, mgroups[g], ng);
                        if (eng->paged_error) return;
                    }
                    mp.m_base = g_beg;
                    mp.m_end  = g_end;
                    const uint32_t g_launch_end = (g + 1U == n_groups) ? M_padded : g_end;
                    const uint32_t q_tiles      = (g_launch_end - g_beg + 15U) / 16U;
                    [enc setComputePipelineState:eng->pso_mla_prefill_sparse_nax];
                    [enc setBuffer:eng->b_mla_q_latent_f16 offset:0 atIndex:0];
                    [enc setBuffer:eng->b_mla_kv_cache offset:kv_layer_off atIndex:1];
                    [enc setBuffer:eng->b_idx_sel_pools offset:0 atIndex:2];
                    [enc setBuffer:eng->b_mla_c_out_f16 offset:0 atIndex:3];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                    [enc setBuffer:loc_buf offset:loc_off atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(q_tiles, nh, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            }

            sg53_encode_mla_batched_f16_hm(enc, eng, ld.mla_w_uv,
                                           eng->b_mla_c_out_f16, eng->b_mla_v_out_f16,
                                           M_valid, M_padded, nh, v_dim, kvr, /*out_head_major=*/false);
        } else {
            const size_t n_groups = mgroups.empty() ? 1U : mgroups.size();
            for (size_t g = 0U; g < n_groups; ++g) {
                const uint32_t g_beg = mgroups.empty() ? 0U : mgroups[g].beg;
                const uint32_t g_end = mgroups.empty() ? M_valid : mgroups[g].end;
                if (g > 0U) {
                    sg53_paged_commit_wait(eng, enc, *io_cb);
                    const Sg53MGroup* ng = (g + 1U < n_groups) ? &mgroups[g + 1U] : nullptr;
                    sg53_paged_make_group_resident(eng, dsa_slot, newest_blk, mgroups[g], ng);
                    if (eng->paged_error) return;
                }
                // Slice into batches of `<= kMaxSpecBatch` so `b_part_m/l/o` (16 MiB) NEVER overflows!
                const bool use_mla_decode_nax = eng->has_nax && (kvr == 512U) && ((nh % 16U) == 0U);
                for (uint32_t mb = g_beg; mb < g_end; mb += kMaxSpecBatch) {
                    const uint32_t mcnt = std::min<uint32_t>(kMaxSpecBatch, g_end - mb);
                    mp.m_base = mb;
                    mp.m_end  = mb + mcnt;
                    if (use_mla_decode_nax) {
                        [enc setComputePipelineState:eng->pso_mla_decode_splitkv_nax];
                        [enc setBuffer:eng->b_mla_q_latent_f16 offset:0 atIndex:0];
                        [enc setBuffer:eng->b_mla_kv_cache offset:kv_layer_off atIndex:1];
                        [enc setBuffer:eng->b_idx_sel_pools offset:0 atIndex:2];
                        [enc setBuffer:eng->b_part_m offset:0 atIndex:3];
                        [enc setBuffer:eng->b_part_l offset:0 atIndex:4];
                        [enc setBuffer:eng->b_part_o offset:0 atIndex:5];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                        [enc setBuffer:loc_buf offset:loc_off atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(launch_splits, nh / 16U, mcnt)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    } else {
                        [enc setComputePipelineState:eng->pso_mla_splitkv];
                        [enc setBuffer:eng->b_mla_q_latent_f16 offset:0 atIndex:0];
                        [enc setBuffer:eng->b_mla_kv_cache offset:kv_layer_off atIndex:1];
                        [enc setBuffer:eng->b_idx_sel_pools offset:0 atIndex:2];
                        [enc setBuffer:eng->b_part_m offset:0 atIndex:3];
                        [enc setBuffer:eng->b_part_l offset:0 atIndex:4];
                        [enc setBuffer:eng->b_part_o offset:0 atIndex:5];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                        [enc setBuffer:loc_buf offset:loc_off atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(launch_splits, num_q_groups, mcnt)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }

                    [enc setComputePipelineState:eng->pso_mla_reduce];
                    [enc setBuffer:eng->b_part_m offset:0 atIndex:0];
                    [enc setBuffer:eng->b_part_l offset:0 atIndex:1];
                    [enc setBuffer:eng->b_part_o offset:0 atIndex:2];
                    [enc setBuffer:eng->b_mla_c_out_f16 offset:0 atIndex:3];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(nh, mcnt, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }

            sg53_encode_mla_batched_f16(enc, eng, ld.mla_w_uv,
                                        eng->b_mla_c_out_f16, eng->b_mla_v_out_f16,
                                        M_valid, M_padded, nh, v_dim, kvr, force_gemv);
        }
        sg53_encode_q4_linear(enc, eng, ld.mla_o_proj,
                              eng->b_mla_v_out_f16, 0, nil, 0, eng->b_block_y_f32, 0,
                              M_valid, M_padded, kOutF32, force_gemv);
    }

    sg53_encode_sublayer_exit(enc, eng, ld.has_mhc != 0U, stream_buf, M_valid);
    /* `enc` aliases `*io_enc`: nothing to hand back. */
}

void sg53_encode_dense_ffn_sublayer(
    id<MTLComputeCommandEncoder> enc,
    StrataGlm53GpuEngine*        eng,
    const StrataGlm53LayerDesc&  ld,
    id<MTLBuffer>                stream_buf,
    uint32_t                     M_valid,
    uint32_t                     M_padded,
    bool                         force_gemv
) {
    const uint32_t d_ff = eng->cfg.intermediate_size;
    sg53_encode_sublayer_entry(enc, eng, ld.has_mhc != 0U, ld.hc_ffn, ld.post_attention_layernorm,
                               stream_buf, M_valid, M_padded, /*write_f32=*/false, /*write_unit_f16=*/false);

    sg53_encode_q4_linear(enc, eng, ld.dense_gate_up_proj,
                          eng->b_x_in_f16, 0, nil, 0, eng->b_ffn_gate_up_f32, 0,
                          M_valid, M_padded, kOutF32, force_gemv);
    {
        Sg53SwigluParamsHost sp{M_valid, M_padded, d_ff, eng->cfg.swiglu_limit};
        [enc setComputePipelineState:eng->pso_limited_swiglu];
        [enc setBuffer:eng->b_ffn_gate_up_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_ffn_mid_f16 offset:0 atIndex:1];
        [enc setBytes:&sp length:sizeof(sp) atIndex:2];
        const uint32_t d4 = d_ff / 4U;
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    sg53_encode_q4_linear(enc, eng, ld.dense_down_proj,
                          eng->b_ffn_mid_f16, 0, nil, 0, eng->b_block_y_f32, 0,
                          M_valid, M_padded, kOutF32, force_gemv);

    sg53_encode_sublayer_exit(enc, eng, ld.has_mhc != 0U, stream_buf, M_valid);
}

bool sg53_encode_moe_router_step_a(
    id<MTLComputeCommandEncoder> enc,
    StrataGlm53GpuEngine*        eng,
    const StrataGlm53LayerDesc&  ld,
    id<MTLBuffer>                stream_buf,
    uint32_t                     M_valid,
    uint32_t                     M_padded,
    bool                         write_ane_unit_x,
    bool                         write_x_rot_f16 = false
) {
    const bool fused_rot = sg53_encode_sublayer_entry(
        enc, eng, ld.has_mhc != 0U, ld.hc_ffn, ld.post_attention_layernorm,
        stream_buf, M_valid, M_padded, /*write_f32=*/true, write_ane_unit_x, write_x_rot_f16);

    Sg53MoeRouterParamsHost rp{
        M_valid,
        eng->cfg.hidden_size,
        eng->cfg.n_routed_experts,
        eng->cfg.num_experts_per_tok,
        eng->cfg.norm_topk_prob,
        eng->cfg.routed_scaling_factor,
        eng->cfg.n_group,
        eng->cfg.topk_group
    };
    const bool use_split_router =
        ([eng->b_ffn_gate_up_f32 length] >=
         static_cast<size_t>(M_valid) * eng->cfg.n_routed_experts * sizeof(float));
    if (use_split_router) {
        // 40-Core Parallel Router Logits (reads `router_gate` ONCE across M=1..8!) + `< 2 us` Top-K Select!
        // Uses dead `b_ffn_gate_up_f32` scratch (overwritten next by `shared_gate_up_proj`), 100.0000% bit-exact!
        [enc setComputePipelineState:eng->pso_moe_router_logits];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.moe_router_gate.offset) atIndex:1];
        [enc setBuffer:eng->b_ffn_gate_up_f32 offset:0 atIndex:2];
        [enc setBytes:&rp length:sizeof(rp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((eng->cfg.n_routed_experts + 3U) / 4U, (M_valid + 7U) / 8U, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:eng->pso_moe_router_select];
        [enc setBuffer:eng->b_ffn_gate_up_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.moe_e_score_bias.offset) atIndex:1];
        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:2];
        [enc setBuffer:eng->b_topk_w offset:0 atIndex:3];
        [enc setBytes:&rp length:sizeof(rp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    } else {
        [enc setComputePipelineState:eng->pso_moe_router_topk];
        [enc setBuffer:eng->b_x_in_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.moe_router_gate.offset) atIndex:1];
        [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(ld.moe_e_score_bias.offset) atIndex:2];
        [enc setBuffer:eng->b_topk_idx offset:0 atIndex:3];
        [enc setBuffer:eng->b_topk_w offset:0 atIndex:4];
        [enc setBytes:&rp length:sizeof(rp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    }
    return fused_rot;
}

void sg53_encode_shared_expert(
    id<MTLComputeCommandEncoder> enc,
    StrataGlm53GpuEngine*        eng,
    const StrataGlm53LayerDesc&  ld,
    uint32_t                     M_valid,
    uint32_t                     M_padded,
    bool                         force_gemv
) {
    const uint32_t d_sh = eng->cfg.moe_intermediate_size * std::max<uint32_t>(1U, eng->cfg.n_shared_experts);
    sg53_encode_q4_linear(enc, eng, ld.shared_gate_up_proj,
                          eng->b_x_in_f16, 0, nil, 0, eng->b_ffn_gate_up_f32, 0,
                          M_valid, M_padded, kOutF32, force_gemv);
    {
        Sg53SwigluParamsHost sp{M_valid, M_padded, d_sh, eng->cfg.swiglu_limit};
        [enc setComputePipelineState:eng->pso_limited_swiglu];
        [enc setBuffer:eng->b_ffn_gate_up_f32 offset:0 atIndex:0];
        [enc setBuffer:eng->b_ffn_mid_f16 offset:0 atIndex:1];
        [enc setBytes:&sp length:sizeof(sp) atIndex:2];
        const uint32_t d4 = d_sh / 4U;
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_padded, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    sg53_encode_q4_linear(enc, eng, ld.shared_down_proj,
                          eng->b_ffn_mid_f16, 0, nil, 0, eng->b_shared_out_f32, 0,
                          M_valid, M_padded, kOutF32, force_gemv);
}

/* ============================================================================
 * Disaggregated Regime II MoE Execution (56 GiB Hot LFU + 2 GiB Ring_DMA +
 * 16-Core ANE Lookahead + 8-FD Parallel F_NOCACHE NVMe Overlap)
 * ============================================================================ */

uint32_t sg53_claim_fg_ring_slot(
    StrataGlm53GpuEngine*       eng,
    uint32_t                    gid,
    const std::vector<uint8_t>& pinned_fg
) {
    for (uint32_t tries = 0U; tries < eng->fg_ring_slots; ++tries) {
        const uint32_t s = (eng->fg_ring_cursor++) % eng->fg_ring_slots;
        if (pinned_fg[s] == 0U) {
            const int32_t old_gid = eng->gid_of_fg_slot[s];
            if (old_gid >= 0) eng->fg_slot_of_gid[static_cast<size_t>(old_gid)] = -1;
            eng->gid_of_fg_slot[s] = static_cast<int32_t>(gid);
            eng->fg_slot_of_gid[gid] = static_cast<int32_t>(s);
            return s;
        }
    }
    const uint32_t s = (eng->fg_ring_cursor++) % eng->fg_ring_slots;
    const int32_t old_gid = eng->gid_of_fg_slot[s];
    if (old_gid >= 0) eng->fg_slot_of_gid[static_cast<size_t>(old_gid)] = -1;
    eng->gid_of_fg_slot[s] = static_cast<int32_t>(gid);
    eng->fg_slot_of_gid[gid] = static_cast<int32_t>(s);
    return s;
}

inline uint32_t sg53_total_hot_resident(const StrataGlm53GpuEngine* eng) {
    uint32_t sum = 0U;
    for (const auto& st : eng->layer_hot) sum += st.used_slots;
    return sum;
}

struct Sg53TokenRouteDecision {
    uint8_t  keep[16]     = {};
    float    eff_w[16]    = {};
    uint64_t skipped_cold = 0ULL;
    uint64_t promotions   = 0ULL;
};

Sg53TokenRouteDecision sg53_step_layer_token(
    Sg53LayerHotState&          st,
    const StrataGlm53GpuEngine* eng,
    const uint32_t*             m_idx,
    const float*                m_w,
    uint32_t                    top_k,
    bool                        mutate_state,
    std::vector<uint8_t>*       io_batch_pin
) {
    Sg53TokenRouteDecision dec{};
    if (top_k > 16U) top_k = 16U;
    const bool is_adaptive = (eng->exec_mode == STRATA_GLM53_MODE_ADAPTIVE);

    float    orig_sum = 0.0f;
    float    kept_sum = 0.0f;
    uint32_t kept_cnt = 0U;

    if (!is_adaptive) {
        for (uint32_t k = 0U; k < top_k; ++k) {
            dec.keep[k] = 1U;
            orig_sum   += m_w[k];
            kept_sum   += m_w[k];
            ++kept_cnt;
        }
    } else {
        struct ColdCand { uint32_t k; float w; };
        ColdCand colds[16];
        uint32_t n_colds = 0U;

        for (uint32_t k = 0U; k < top_k; ++k) {
            orig_sum += m_w[k];
            const uint32_t e = m_idx[k];
            if (e < st.slot_of_e.size() && st.slot_of_e[e] >= 0) {
                dec.keep[k] = 1U;
                kept_sum   += m_w[k];
                ++kept_cnt;
            } else {
                colds[n_colds++] = ColdCand{k, m_w[k]};
            }
        }

        std::sort(colds, colds + n_colds, [](const ColdCand& a, const ColdCand& b) {
            return (a.w > b.w) || (a.w == b.w && a.k < b.k);
        });

        uint32_t cold_taken = 0U;
        for (uint32_t ci = 0U; ci < n_colds; ++ci) {
            if (cold_taken < eng->adaptive_cold_cap && colds[ci].w >= eng->adaptive_gate_tau) {
                dec.keep[colds[ci].k] = 1U;
                kept_sum += colds[ci].w;
                ++kept_cnt;
                ++cold_taken;
            } else {
                dec.skipped_cold += 1ULL;
            }
        }

        if (kept_cnt == 0U && n_colds > 0U && eng->adaptive_cold_cap > 0U) {
            dec.keep[colds[0].k] = 1U;
            kept_sum += colds[0].w;
            ++kept_cnt;
            if (dec.skipped_cold > 0ULL) dec.skipped_cold -= 1ULL;
        }
    }

    const float renorm = (is_adaptive && eng->adaptive_renorm && kept_cnt > 0U && kept_cnt < top_k && kept_sum > 1e-20f)
                             ? (orig_sum / kept_sum)
                             : 1.0f;
    for (uint32_t k = 0U; k < top_k; ++k) {
        dec.eff_w[k] = dec.keep[k] ? (m_w[k] * renorm) : 0.0f;
    }

    if (!mutate_state || st.cap_slots == 0U) {
        if (mutate_state) {
            for (uint32_t k = 0U; k < top_k; ++k) {
                if (!dec.keep[k]) continue;
                const uint32_t e = m_idx[k];
                if (e < st.freq_of_e.size()) st.freq_of_e[e] += 1U;
            }
        }
        return dec;
    }

    std::vector<uint8_t> tok_pin(st.cap_slots, 0U);
    for (uint32_t k = 0U; k < top_k; ++k) {
        if (!dec.keep[k]) continue;
        const uint32_t e = m_idx[k];
        if (e < st.freq_of_e.size()) st.freq_of_e[e] += 1U;
        if (e < st.slot_of_e.size() && st.slot_of_e[e] >= 0) {
            const uint32_t ls = static_cast<uint32_t>(st.slot_of_e[e]);
            tok_pin[ls] = 1U;
            if (io_batch_pin) (*io_batch_pin)[ls] = 1U;
        }
    }

    for (uint32_t k = 0U; k < top_k; ++k) {
        if (!dec.keep[k]) continue;
        const uint32_t e = m_idx[k];
        if (e >= st.slot_of_e.size() || st.slot_of_e[e] >= 0) continue;

        if (st.used_slots < st.cap_slots) {
            const uint32_t ls = st.used_slots++;
            st.slot_of_e[e]  = static_cast<int32_t>(ls);
            st.e_of_slot[ls] = static_cast<int32_t>(e);
            tok_pin[ls]      = 1U;
            if (io_batch_pin) (*io_batch_pin)[ls] = 1U;
            dec.promotions  += 1ULL;
        } else {
            const uint32_t my_freq = st.freq_of_e[e];
            if (my_freq < 2U) continue;

            uint32_t best_ls   = UINT32_MAX;
            uint32_t best_freq = UINT32_MAX;
            const uint32_t probes = std::min<uint32_t>(32U, st.cap_slots);
            for (uint32_t i = 0U; i < probes; ++i) {
                const uint32_t ls = (st.clock_hand++) % st.cap_slots;
                if (tok_pin[ls] != 0U) continue;
                if (io_batch_pin && (*io_batch_pin)[ls] != 0U) continue;
                const int32_t owner  = st.e_of_slot[ls];
                const uint32_t ofreq = (owner >= 0) ? st.freq_of_e[static_cast<size_t>(owner)] : 0U;
                if (ofreq < best_freq) {
                    best_freq = ofreq;
                    best_ls   = ls;
                    if (ofreq <= 1U) break;
                }
            }

            if (best_ls != UINT32_MAX && my_freq >= best_freq + 2U) {
                const int32_t old_e = st.e_of_slot[best_ls];
                if (old_e >= 0) st.slot_of_e[static_cast<size_t>(old_e)] = -1;
                st.slot_of_e[e]       = static_cast<int32_t>(best_ls);
                st.e_of_slot[best_ls] = static_cast<int32_t>(e);
                tok_pin[best_ls]      = 1U;
                if (io_batch_pin) (*io_batch_pin)[best_ls] = 1U;
                dec.promotions       += 1ULL;
            }
        }
    }
    return dec;
}

int sg53_execute_moe_ffn_pipelined(
    StrataGlm53GpuEngine*               eng,
    const StrataGlm53LayerDesc&         ld,
    id<MTLBuffer>                       stream_buf,
    uint32_t                            M_valid,
    uint32_t                            M_padded,
    bool                                force_gemv,
    bool                                is_speculative,
    bool                                allow_ane_lookahead,
    id<MTLCommandBuffer> __strong*      io_cb,
    id<MTLComputeCommandEncoder> __strong* io_enc,
    std::vector<Sg53SpecLayerSnap>*     spec_snaps_out
) {
    const uint32_t moe_l = static_cast<uint32_t>(ld.moe_layer_idx);
    const uint32_t E     = eng->cfg.n_routed_experts;
    const uint32_t top_k = eng->cfg.num_experts_per_tok;
    const uint32_t D     = eng->cfg.hidden_size;
    const uint32_t d_moe = eng->cfg.moe_intermediate_size;
    const uint32_t H_ane = std::max<uint32_t>(1U, eng->ane_lookahead_horizon);
    const uint64_t rec_b = eng->hdr.expert_record_bytes;

    const bool is_main_moe  = (moe_l < eng->moe_to_global_layer.size());
    if (is_main_moe) {
        eng->cur_moe_layer.store(moe_l, std::memory_order_release);
    }
    const bool fire_ane_win = allow_ane_lookahead && is_main_moe &&
                              !eng->ane_windows.empty() && ((moe_l % H_ane) == 0U);

    // 1. Encode `hc_ffn` + `post_attention_layernorm` + `moe_router_topk` into active encoder and synchronize!
    sg53_encode_moe_router_step_a(*io_enc, eng, ld, stream_buf, M_valid, M_padded, fire_ane_win);
    [*io_enc endEncoding];
    [*io_cb commit];
    [*io_cb waitUntilCompleted];
    /* From here until a fresh pair is opened further down, `*io_cb` / `*io_enc` are the FINISHED router pass. Every
     * caller answers a non-zero return with `[enc endEncoding]; [cb commit];`, and Metal aborts on a second commit,
     * so each error return must first hand back a live `(command buffer, encoder)` pair. */
    auto fail_with = [&](int rc) -> int {
        *io_cb  = [eng->queue commandBuffer];
        *io_enc = [*io_cb computeCommandEncoder];
        return rc;
    };
    if ([*io_cb status] == MTLCommandBufferStatusError) {
        std::fprintf(stderr, "[strata_glm53_gpu] GPU router CB error at layer=%u\n", ld.global_layer_idx);
        return fail_with(-4);
    }
    ++eng->cb_epoch;

    // 2. Kick off 16-Core ANE Multi-Horizon Router Lookahead asynchronously on `ane_bg_queue`!
    if (fire_ane_win) {
        const uint32_t win_idx = moe_l / H_ane;
        bool win_has_cold = false;
        const Sg53AneWindow& wb_ref = eng->ane_windows[win_idx];
        for (uint32_t sub = 1U; sub < wb_ref.num_layers; ++sub) {
            const uint32_t ml = wb_ref.moe_layer_beg + sub;
            if (ml < eng->layer_hot.size() && eng->layer_hot[ml].used_slots < E) {
                win_has_cold = true;
                break;
            }
        }
        const bool do_bg_io = (M_valid <= 16U) && win_has_cold;
        sg53_schedule_ane_lookahead_async(eng, win_idx, M_valid, do_bg_io);
    }

    // 3. Deterministic Per-Layer `LOSSLESS` / `ADAPTIVE` Gating & Per-Token Cache Transition!
    const auto* topk_idx = reinterpret_cast<const uint32_t*>([eng->b_topk_idx contents]);
    const auto* topk_w   = reinterpret_cast<const float*>([eng->b_topk_w contents]);
    auto*       disp_out = reinterpret_cast<StrataGlm53ExpertDispatch*>([eng->b_expert_dispatch contents]);

    Sg53LayerHotState dummy_st{};
    dummy_st.slot_of_e.assign(E, -1);
    dummy_st.freq_of_e.assign(E, 0U);
    Sg53LayerHotState& committed_st = (moe_l < eng->layer_hot.size()) ? eng->layer_hot[moe_l] : dummy_st;
    const uint32_t     pre_used     = committed_st.used_slots;

    Sg53LayerHotState sim_st;
    Sg53SpecLayerSnap spec_rec;
    std::vector<int32_t> first_unused_ls_for_e;
    if (is_speculative) {
        sim_st         = committed_st;
        spec_rec.moe_l = moe_l;
        if (spec_snaps_out) {
            first_unused_ls_for_e.assign(E, -1);
        }
    }

    std::vector<uint8_t> batch_pin;
    std::vector<uint8_t>* batch_pin_ptr = nullptr;
    if (!is_speculative && M_valid > 1U && committed_st.cap_slots > 0U) {
        batch_pin.assign(committed_st.cap_slots, 0U);
        batch_pin_ptr = &batch_pin;
    }

    struct PairNeed {
        uint32_t m;
        uint32_t k;
        uint32_t e;
        uint32_t gid;
        float    w;
    };
    std::vector<PairNeed> active_pairs;
    active_pairs.reserve(static_cast<size_t>(M_valid) * top_k);

    uint64_t skipped_cold = 0ULL;
    uint64_t promos_delta = 0ULL;
    std::vector<uint8_t> unused_ls_claimed(std::max<size_t>(1U, committed_st.cap_slots), 0U);

    for (uint32_t m = 0U; m < M_valid; ++m) {
        const uint32_t* m_idx = topk_idx + static_cast<size_t>(m) * top_k;
        const float*    m_w   = topk_w   + static_cast<size_t>(m) * top_k;

        Sg53LayerHotState& target_st = is_speculative ? sim_st : committed_st;
        const bool         do_mut    = (!is_speculative) || (spec_snaps_out != nullptr);
        const Sg53TokenRouteDecision dec = sg53_step_layer_token(
            target_st, eng, m_idx, m_w, top_k, do_mut, batch_pin_ptr);

        skipped_cold += dec.skipped_cold;
        promos_delta += dec.promotions;

        if (is_speculative && spec_snaps_out && m < kMaxSpecBatch) {
            spec_rec.snap[m]           = sim_st;
            spec_rec.promotions_cum[m] = promos_delta;
            for (uint32_t ls = pre_used; ls < sim_st.used_slots; ++ls) {
                if (unused_ls_claimed[ls] == 0U && sim_st.e_of_slot[ls] >= 0) {
                    const uint32_t owner_e = static_cast<uint32_t>(sim_st.e_of_slot[ls]);
                    if (owner_e < E && first_unused_ls_for_e[owner_e] < 0) {
                        first_unused_ls_for_e[owner_e] = static_cast<int32_t>(ls);
                        unused_ls_claimed[ls]          = 1U;
                    }
                }
            }
        }

        for (uint32_t k = 0U; k < top_k; ++k) {
            const size_t pid = static_cast<size_t>(m) * top_k + k;
            if (!dec.keep[k]) {
                disp_out[pid] = StrataGlm53ExpertDispatch{m_idx[k], 0U, 0U, 0.0f};
            } else {
                const uint32_t e   = m_idx[k];
                const uint32_t gid = moe_l * E + e;
                disp_out[pid]      = StrataGlm53ExpertDispatch{e, 0U, 0U, dec.eff_w[k]};
                active_pairs.push_back(PairNeed{m, k, e, gid, dec.eff_w[k]});
            }
        }
    }

    if (is_speculative && spec_snaps_out) {
        spec_snaps_out->push_back(std::move(spec_rec));
    } else if (!is_speculative && promos_delta > 0ULL) {
        Sg53ScopedLock tlk(&eng->telemetry_lock);
        eng->counters.lfu_promotions += promos_delta;
    }

    // 4. Deduplicate unique experts needed in this layer (`<= 288`) & materialize UMA Hot / Ring_DMA residency!
    struct UniqueExp {
        uint32_t              e;
        uint32_t              gid;
        int32_t               src_buf;          // -1 if cold NVMe miss, else 0 (HOT_UMA) or 1 (RING_DMA)
        uint32_t              slot_idx;
        int32_t               target_phys_hot;  // >= 0 if cold NVMe read should land directly in `b_hot_experts`!
        std::vector<uint32_t> pair_ids;         // `m * top_k + k`
    };
    int32_t e_to_uniq[512];
    for (uint32_t i = 0U; i < E && i < 512U; ++i) e_to_uniq[i] = -1;

    std::vector<UniqueExp> uniq_exps;
    uniq_exps.reserve(std::min<size_t>(active_pairs.size(), E));

    uint64_t hot_hits_delta = 0ULL;
    uint64_t ane_hits_delta = 0ULL;
    std::vector<uint8_t> pinned_fg(std::max<size_t>(1U, eng->fg_ring_slots), 0U);
    std::vector<uint32_t> cold_uniq_indices;

    auto try_find_ready_bg_slot = [&](uint32_t gid, uint64_t pin_horizon) -> int32_t {
        Sg53ScopedLock blk(&eng->bg_ring_lock);
        for (uint32_t s = 0U; s < eng->bg_ring_slots; ++s) {
            if (eng->gid_of_bg_slot[s] == static_cast<int32_t>(gid) && eng->bg_slot_ready[s] != 0U) {
                eng->bg_slot_pin_epoch[s] = eng->cb_epoch + pin_horizon;
                return static_cast<int32_t>(s);
            }
        }
        return -1;
    };

    for (const auto& pr : active_pairs) {
        if (pr.e >= 512U) continue;
        int32_t uidx = e_to_uniq[pr.e];
        if (uidx < 0) {
            uidx = static_cast<int32_t>(uniq_exps.size());
            e_to_uniq[pr.e] = uidx;
            UniqueExp ue{};
            ue.e               = pr.e;
            ue.gid             = pr.gid;
            ue.src_buf         = -1;
            ue.slot_idx        = 0U;
            ue.target_phys_hot = -1;

            int32_t assigned_phys_hot = -1;
            if (!is_speculative) {
                if (pr.e < committed_st.slot_of_e.size() && committed_st.slot_of_e[pr.e] >= 0) {
                    assigned_phys_hot = static_cast<int32_t>(
                        committed_st.base_slot + static_cast<uint32_t>(committed_st.slot_of_e[pr.e]));
                }
            } else {
                if (pr.e < committed_st.slot_of_e.size() && committed_st.slot_of_e[pr.e] >= 0) {
                    assigned_phys_hot = static_cast<int32_t>(
                        committed_st.base_slot + static_cast<uint32_t>(committed_st.slot_of_e[pr.e]));
                } else if (spec_snaps_out && pr.e < first_unused_ls_for_e.size() && first_unused_ls_for_e[pr.e] >= 0) {
                    assigned_phys_hot = static_cast<int32_t>(
                        committed_st.base_slot + static_cast<uint32_t>(first_unused_ls_for_e[pr.e]));
                }
            }

            if (assigned_phys_hot >= 0) {
                const uint32_t phys_hs = static_cast<uint32_t>(assigned_phys_hot);
                if (eng->phys_hot_gid[phys_hs] == static_cast<int32_t>(pr.gid)) {
                    ue.src_buf  = static_cast<int32_t>(STRATA_GLM53_EXPERT_SRC_HOT_UMA);
                    ue.slot_idx = phys_hs;
                } else if (eng->fg_slot_of_gid[pr.gid] >= 0) {
                    const uint32_t fs = static_cast<uint32_t>(eng->fg_slot_of_gid[pr.gid]);
                    sg53_fast_memcpy_expert(eng->hot_experts_ptr + static_cast<uint64_t>(phys_hs) * rec_b,
                                            eng->ring_dma_ptr + static_cast<uint64_t>(fs) * rec_b,
                                            rec_b);
                    eng->phys_hot_gid[phys_hs] = static_cast<int32_t>(pr.gid);
                    ue.src_buf  = static_cast<int32_t>(STRATA_GLM53_EXPERT_SRC_HOT_UMA);
                    ue.slot_idx = phys_hs;
                } else {
                    const int32_t bg_hit = try_find_ready_bg_slot(pr.gid, 8ULL);
                    if (bg_hit >= 0) {
                        const uint32_t phys_rs = eng->fg_ring_slots + static_cast<uint32_t>(bg_hit);
                        sg53_fast_memcpy_expert(eng->hot_experts_ptr + static_cast<uint64_t>(phys_hs) * rec_b,
                                                eng->ring_dma_ptr + static_cast<uint64_t>(phys_rs) * rec_b,
                                                rec_b);
                        eng->phys_hot_gid[phys_hs] = static_cast<int32_t>(pr.gid);
                        ue.src_buf  = static_cast<int32_t>(STRATA_GLM53_EXPERT_SRC_HOT_UMA);
                        ue.slot_idx = phys_hs;
                        ane_hits_delta += 1ULL;
                    } else {
                        ue.target_phys_hot = assigned_phys_hot;
                        cold_uniq_indices.push_back(static_cast<uint32_t>(uidx));
                    }
                }
            } else {
                if (eng->fg_slot_of_gid[pr.gid] >= 0) {
                    const uint32_t fs = static_cast<uint32_t>(eng->fg_slot_of_gid[pr.gid]);
                    ue.src_buf  = static_cast<int32_t>(STRATA_GLM53_EXPERT_SRC_RING_DMA);
                    ue.slot_idx = fs;
                    pinned_fg[fs] = 1U;
                } else {
                    const int32_t bg_hit = try_find_ready_bg_slot(pr.gid, 16ULL);
                    if (bg_hit >= 0) {
                        const uint32_t phys_rs = eng->fg_ring_slots + static_cast<uint32_t>(bg_hit);
                        ue.src_buf  = static_cast<int32_t>(STRATA_GLM53_EXPERT_SRC_RING_DMA);
                        ue.slot_idx = phys_rs;
                        ane_hits_delta += 1ULL;
                    } else {
                        cold_uniq_indices.push_back(static_cast<uint32_t>(uidx));
                    }
                }
            }
            uniq_exps.push_back(std::move(ue));
        }

        const UniqueExp& cur_ue = uniq_exps[static_cast<size_t>(uidx)];
        if (cur_ue.src_buf >= 0) {
            hot_hits_delta += 1ULL;
        }
        uniq_exps[static_cast<size_t>(uidx)].pair_ids.push_back(pr.m * top_k + pr.k);
    }

    {
        Sg53ScopedLock tlk(&eng->telemetry_lock);
        eng->counters.active_experts_evaluated += static_cast<uint64_t>(active_pairs.size());
        eng->counters.adaptive_cold_skipped    += skipped_cold;
        eng->counters.uma_hot_hits             += hot_hits_delta;
        eng->counters.ane_lookahead_hits       += ane_hits_delta;
    }

    // Helper counting how many cold misses actually require `fg_ring` slots (`target_phys_hot < 0`)
    uint32_t cold_fg_needed = 0U;
    for (uint32_t ui : cold_uniq_indices) {
        if (uniq_exps[ui].target_phys_hot < 0) ++cold_fg_needed;
    }

    // 5. Overlap `shared_experts` + ALL ALREADY-RESIDENT HOT EXPERTS on GPU with 8-FD `F_NOCACHE` NVMe Cold Streaming!
    const bool use_multi_wave = (cold_fg_needed > eng->fg_ring_slots);
    bool experts_already_encoded = false;

    if (cold_uniq_indices.empty()) {
        // 100% UMA / ANE Prefetch Hit -> Zero NVMe Stall! Open next fused CB immediately!
        for (const auto& ue : uniq_exps) {
            for (uint32_t pid : ue.pair_ids) {
                disp_out[pid].source_buf = static_cast<uint32_t>(ue.src_buf);
                disp_out[pid].slot_idx   = ue.slot_idx;
            }
        }

        *io_cb  = [eng->queue commandBuffer];
        *io_enc = [*io_cb computeCommandEncoder];
        sg53_encode_shared_expert(*io_enc, eng, ld, M_valid, M_padded, force_gemv);
    } else if (!use_multi_wave) {
        // Single-wave Cold Fetch: Execute `shared_experts` + ALL already-resident UMA Hot/Ring experts in `cb_sh`
        // concurrently on all 40 M5 Max GPU cores while 8-FD NVMe streams ONLY the cold miss experts!
        auto* sorted_pairs = reinterpret_cast<uint32_t*>([eng->b_moe_sorted_pairs contents]);
        auto* work_tiles   = reinterpret_cast<Sg53MoeWorkTileHost*>([eng->b_moe_work_tiles contents]);

        uint32_t hot_p = 0U, hot_t = 0U;
        for (const auto& ue : uniq_exps) {
            if (ue.src_buf < 0) continue;
            const uint32_t cnt   = static_cast<uint32_t>(ue.pair_ids.size());
            const uint32_t p_beg = hot_p;
            for (uint32_t pid : ue.pair_ids) {
                disp_out[pid].source_buf = static_cast<uint32_t>(ue.src_buf);
                disp_out[pid].slot_idx   = ue.slot_idx;
                sorted_pairs[hot_p++]    = pid;
            }
            for (uint32_t st = 0U; st < cnt; st += 8U) {
                work_tiles[hot_t++] = Sg53MoeWorkTileHost{
                    static_cast<uint32_t>(ue.src_buf), ue.slot_idx, p_beg + st, std::min<uint32_t>(8U, cnt - st)
                };
            }
        }

        id<MTLCommandBuffer>         cb_sh  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc_sh = [cb_sh computeCommandEncoder];
        sg53_encode_shared_expert(enc_sh, eng, ld, M_valid, M_padded, force_gemv);

        if (hot_t > 0U) {
            Sg53MoeIndirectParamsHost mp_hot{
                eng->hdr.expert_record_bytes, eng->hdr.expert_proj_bytes,
                eng->hdr.expert_weight_bytes, eng->hdr.expert_scales_bytes,
                M_valid, top_k, D, d_moe, eng->cfg.swiglu_limit, hot_t, 0U, 0U
            };
            [enc_sh setComputePipelineState:eng->pso_moe_grouped_gu];
            [enc_sh setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [enc_sh setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [enc_sh setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
            [enc_sh setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [enc_sh setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
            [enc_sh setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
            [enc_sh setBytes:&mp_hot length:sizeof(mp_hot) atIndex:6];
            [enc_sh dispatchThreadgroups:MTLSizeMake((d_moe + 3U) / 4U, hot_t, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc_sh setComputePipelineState:eng->pso_moe_grouped_dn];
            [enc_sh setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [enc_sh setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [enc_sh setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
            [enc_sh setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [enc_sh setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
            [enc_sh setBuffer:eng->b_moe_down_part offset:0 atIndex:5];
            [enc_sh setBytes:&mp_hot length:sizeof(mp_hot) atIndex:6];
            [enc_sh dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, hot_t, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [enc_sh endEncoding];
        [cb_sh commit];

        std::vector<Sg53NvmeExpertCopyTask> io_tasks;
        io_tasks.reserve(cold_uniq_indices.size());

        // Write cold tiles at cache-line-disjoint offsets so `cb_sh` can read `[0..hot_t-1]` race-free!
        uint32_t cold_p = (hot_p + 15U) & ~15U;
        const uint32_t cold_t_base = (hot_t + 3U) & ~3U;
        uint32_t cold_t = 0U;

        for (uint32_t ui : cold_uniq_indices) {
            UniqueExp& ue = uniq_exps[ui];
            if (ue.target_phys_hot >= 0) {
                const uint32_t hs = static_cast<uint32_t>(ue.target_phys_hot);
                ue.src_buf  = static_cast<int32_t>(STRATA_GLM53_EXPERT_SRC_HOT_UMA);
                ue.slot_idx = hs;
                eng->phys_hot_gid[hs] = static_cast<int32_t>(ue.gid);
                uint8_t* dst = eng->hot_experts_ptr + static_cast<uint64_t>(hs) * rec_b;
                io_tasks.push_back(Sg53NvmeExpertCopyTask{ue.gid, dst});
            } else {
                const uint32_t fs = sg53_claim_fg_ring_slot(eng, ue.gid, pinned_fg);
                pinned_fg[fs] = 1U;
                ue.src_buf  = static_cast<int32_t>(STRATA_GLM53_EXPERT_SRC_RING_DMA);
                ue.slot_idx = fs;
                uint8_t* dst = eng->ring_dma_ptr + static_cast<uint64_t>(fs) * rec_b;
                io_tasks.push_back(Sg53NvmeExpertCopyTask{ue.gid, dst});
            }
            const uint32_t cnt   = static_cast<uint32_t>(ue.pair_ids.size());
            const uint32_t p_beg = cold_p;
            for (uint32_t pid : ue.pair_ids) {
                disp_out[pid].source_buf = static_cast<uint32_t>(ue.src_buf);
                disp_out[pid].slot_idx   = ue.slot_idx;
                sorted_pairs[cold_p++]   = pid;
            }
            for (uint32_t st = 0U; st < cnt; st += 8U) {
                work_tiles[cold_t_base + cold_t++] = Sg53MoeWorkTileHost{
                    static_cast<uint32_t>(ue.src_buf), ue.slot_idx, p_beg + st, std::min<uint32_t>(8U, cnt - st)
                };
            }
        }

        if (eng->profile_nvme) {
            if (is_main_moe) eng->prof_fg_reads.fetch_add(io_tasks.size(), std::memory_order_relaxed);
            else             eng->prof_mtp_reads.fetch_add(io_tasks.size(), std::memory_order_relaxed);
        }
        if (sg53_stream_experts_nvme_batch(eng, io_tasks, /*count_telemetry=*/true, /*is_foreground=*/true) != 0) {
            [cb_sh waitUntilCompleted];
            return fail_with(-3);
        }

        *io_cb  = [eng->queue commandBuffer];
        *io_enc = [*io_cb computeCommandEncoder];

        if (cold_t > 0U) {
            const NSUInteger cold_tile_off = static_cast<NSUInteger>(cold_t_base) * sizeof(Sg53MoeWorkTileHost);
            Sg53MoeIndirectParamsHost mp_cold{
                eng->hdr.expert_record_bytes, eng->hdr.expert_proj_bytes,
                eng->hdr.expert_weight_bytes, eng->hdr.expert_scales_bytes,
                M_valid, top_k, D, d_moe, eng->cfg.swiglu_limit, cold_t, 0U, 0U
            };
            [*io_enc setComputePipelineState:eng->pso_moe_grouped_gu];
            [*io_enc setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [*io_enc setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [*io_enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
            [*io_enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [*io_enc setBuffer:eng->b_moe_work_tiles offset:cold_tile_off atIndex:4];
            [*io_enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
            [*io_enc setBytes:&mp_cold length:sizeof(mp_cold) atIndex:6];
            [*io_enc dispatchThreadgroups:MTLSizeMake((d_moe + 3U) / 4U, cold_t, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [*io_enc setComputePipelineState:eng->pso_moe_grouped_dn];
            [*io_enc setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [*io_enc setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [*io_enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
            [*io_enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [*io_enc setBuffer:eng->b_moe_work_tiles offset:cold_tile_off atIndex:4];
            [*io_enc setBuffer:eng->b_moe_down_part offset:0 atIndex:5];
            [*io_enc setBytes:&mp_cold length:sizeof(mp_cold) atIndex:6];
            [*io_enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, cold_t, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        experts_already_encoded = true;
    } else {
        // Multi-wave Cold Fetch (`cold_fg_needed > fg_ring_slots` on cold large-batch prefill):
        // Wave 0 runs `shared_experts` + all UMA hot experts; subsequent waves stream `fg_ring_slots` cold experts each!
        id<MTLCommandBuffer>         cb_w0  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc_w0 = [cb_w0 computeCommandEncoder];
        sg53_encode_shared_expert(enc_w0, eng, ld, M_valid, M_padded, force_gemv);

        auto* sorted_pairs = reinterpret_cast<uint32_t*>([eng->b_moe_sorted_pairs contents]);
        auto* work_tiles   = reinterpret_cast<Sg53MoeWorkTileHost*>([eng->b_moe_work_tiles contents]);

        uint32_t p_cur = 0U, t_cur = 0U;
        for (const auto& ue : uniq_exps) {
            if (ue.src_buf < 0) continue;
            const uint32_t cnt   = static_cast<uint32_t>(ue.pair_ids.size());
            const uint32_t p_beg = p_cur;
            for (uint32_t pid : ue.pair_ids) {
                disp_out[pid].source_buf = static_cast<uint32_t>(ue.src_buf);
                disp_out[pid].slot_idx   = ue.slot_idx;
                sorted_pairs[p_cur++]    = pid;
            }
            for (uint32_t st = 0U; st < cnt; st += 8U) {
                work_tiles[t_cur++] = Sg53MoeWorkTileHost{
                    static_cast<uint32_t>(ue.src_buf), ue.slot_idx, p_beg + st, std::min<uint32_t>(8U, cnt - st)
                };
            }
        }

        if (t_cur > 0U) {
            Sg53MoeIndirectParamsHost mp{
                eng->hdr.expert_record_bytes, eng->hdr.expert_proj_bytes,
                eng->hdr.expert_weight_bytes, eng->hdr.expert_scales_bytes,
                M_valid, top_k, D, d_moe, eng->cfg.swiglu_limit, t_cur, 0U, 0U
            };
            [enc_w0 setComputePipelineState:eng->pso_moe_grouped_gu];
            [enc_w0 setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [enc_w0 setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [enc_w0 setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
            [enc_w0 setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [enc_w0 setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
            [enc_w0 setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
            [enc_w0 setBytes:&mp length:sizeof(mp) atIndex:6];
            [enc_w0 dispatchThreadgroups:MTLSizeMake((d_moe + 3U) / 4U, t_cur, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc_w0 setComputePipelineState:eng->pso_moe_grouped_dn];
            [enc_w0 setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [enc_w0 setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [enc_w0 setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
            [enc_w0 setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [enc_w0 setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
            [enc_w0 setBuffer:eng->b_moe_down_part offset:0 atIndex:5];
            [enc_w0 setBytes:&mp length:sizeof(mp) atIndex:6];
            [enc_w0 dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, t_cur, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [enc_w0 endEncoding];
        [cb_w0 commit];
        [cb_w0 waitUntilCompleted];
        ++eng->cb_epoch;

        const uint32_t wave_cap = std::max<uint32_t>(1U, eng->fg_ring_slots);
        for (size_t c_beg = 0U; c_beg < cold_uniq_indices.size(); c_beg += wave_cap) {
            const size_t c_end = std::min<size_t>(cold_uniq_indices.size(), c_beg + wave_cap);
            std::fill(pinned_fg.begin(), pinned_fg.end(), 0U);

            std::vector<Sg53NvmeExpertCopyTask> io_tasks;
            io_tasks.reserve(c_end - c_beg);
            p_cur = 0U;
            t_cur = 0U;

            for (size_t ci = c_beg; ci < c_end; ++ci) {
                UniqueExp& ue = uniq_exps[cold_uniq_indices[ci]];
                if (ue.target_phys_hot >= 0) {
                    const uint32_t hs = static_cast<uint32_t>(ue.target_phys_hot);
                    ue.src_buf  = static_cast<int32_t>(STRATA_GLM53_EXPERT_SRC_HOT_UMA);
                    ue.slot_idx = hs;
                    eng->phys_hot_gid[hs] = static_cast<int32_t>(ue.gid);
                    io_tasks.push_back(Sg53NvmeExpertCopyTask{
                        ue.gid, eng->hot_experts_ptr + static_cast<uint64_t>(hs) * rec_b
                    });
                } else {
                    const uint32_t fs = sg53_claim_fg_ring_slot(eng, ue.gid, pinned_fg);
                    pinned_fg[fs] = 1U;
                    ue.src_buf  = static_cast<int32_t>(STRATA_GLM53_EXPERT_SRC_RING_DMA);
                    ue.slot_idx = fs;
                    io_tasks.push_back(Sg53NvmeExpertCopyTask{
                        ue.gid, eng->ring_dma_ptr + static_cast<uint64_t>(fs) * rec_b
                    });
                }

                const uint32_t cnt   = static_cast<uint32_t>(ue.pair_ids.size());
                const uint32_t p_beg = p_cur;
                for (uint32_t pid : ue.pair_ids) {
                    disp_out[pid].source_buf = static_cast<uint32_t>(ue.src_buf);
                    disp_out[pid].slot_idx   = ue.slot_idx;
                    sorted_pairs[p_cur++]    = pid;
                }
                for (uint32_t st = 0U; st < cnt; st += 8U) {
                    work_tiles[t_cur++] = Sg53MoeWorkTileHost{
                        static_cast<uint32_t>(ue.src_buf), ue.slot_idx, p_beg + st, std::min<uint32_t>(8U, cnt - st)
                    };
                }
            }

            if (sg53_stream_experts_nvme_batch(eng, io_tasks, /*count_telemetry=*/true) != 0) return fail_with(-3);

            id<MTLCommandBuffer>         cb_cw  = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_cw = [cb_cw computeCommandEncoder];
            Sg53MoeIndirectParamsHost mp{
                eng->hdr.expert_record_bytes, eng->hdr.expert_proj_bytes,
                eng->hdr.expert_weight_bytes, eng->hdr.expert_scales_bytes,
                M_valid, top_k, D, d_moe, eng->cfg.swiglu_limit, t_cur, 0U, 0U
            };
            [enc_cw setComputePipelineState:eng->pso_moe_grouped_gu];
            [enc_cw setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [enc_cw setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [enc_cw setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
            [enc_cw setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [enc_cw setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
            [enc_cw setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
            [enc_cw setBytes:&mp length:sizeof(mp) atIndex:6];
            [enc_cw dispatchThreadgroups:MTLSizeMake((d_moe + 3U) / 4U, t_cur, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc_cw setComputePipelineState:eng->pso_moe_grouped_dn];
            [enc_cw setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [enc_cw setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [enc_cw setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
            [enc_cw setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [enc_cw setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
            [enc_cw setBuffer:eng->b_moe_down_part offset:0 atIndex:5];
            [enc_cw setBytes:&mp length:sizeof(mp) atIndex:6];
            [enc_cw dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, t_cur, 1)
                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc_cw endEncoding];
            [cb_cw commit];
            [cb_cw waitUntilCompleted];
            ++eng->cb_epoch;
        }

        // Open fresh CB for `moe_combine` + `sublayer_exit` + next layer!
        *io_cb  = [eng->queue commandBuffer];
        *io_enc = [*io_cb computeCommandEncoder];

        Sg53MoeCombineParamsHost cp{M_valid, top_k, D, 0U};
        [*io_enc setComputePipelineState:eng->pso_moe_combine];
        [*io_enc setBuffer:eng->b_moe_down_part offset:0 atIndex:0];
        [*io_enc setBuffer:eng->b_expert_dispatch offset:0 atIndex:1];
        [*io_enc setBuffer:eng->b_shared_out_f32 offset:0 atIndex:2];
        [*io_enc setBuffer:eng->b_block_y_f32 offset:0 atIndex:3];
        [*io_enc setBytes:&cp length:sizeof(cp) atIndex:4];
        const uint32_t d4 = D / 4U;
        [*io_enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_valid, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        sg53_encode_sublayer_exit(*io_enc, eng, ld.has_mhc != 0U, stream_buf, M_valid);
        return 0;
    }

    // 6. Encode Single-Pass Indirect Multi-Buffer (`b_hot_experts` + `b_ring_dma`) MoE Kernels + Combine!
    //    For `M_valid > 1`, use zero-spill `B`-unrolled grouped tiles so duplicate 13.5 MiB experts are read ONCE!
    if (!experts_already_encoded) {
        if (force_gemv && M_valid == 1U) {
            Sg53MoeIndirectParamsHost mp{
                eng->hdr.expert_record_bytes, eng->hdr.expert_proj_bytes,
                eng->hdr.expert_weight_bytes, eng->hdr.expert_scales_bytes,
                M_valid, top_k, D, d_moe, eng->cfg.swiglu_limit, 0U, 0U, 0U
            };
            [*io_enc setComputePipelineState:eng->pso_moe_gather_gu];
            [*io_enc setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [*io_enc setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [*io_enc setBuffer:eng->b_expert_dispatch offset:0 atIndex:2];
            [*io_enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:3];
            [*io_enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:4];
            [*io_enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [*io_enc dispatchThreadgroups:MTLSizeMake((d_moe + 3U) / 4U, top_k, M_valid)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [*io_enc setComputePipelineState:eng->pso_moe_gather_dn];
            [*io_enc setBuffer:eng->b_hot_experts offset:0 atIndex:0];
            [*io_enc setBuffer:eng->b_ring_dma offset:0 atIndex:1];
            [*io_enc setBuffer:eng->b_expert_dispatch offset:0 atIndex:2];
            [*io_enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
            [*io_enc setBuffer:eng->b_moe_down_part offset:0 atIndex:4];
            [*io_enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [*io_enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, top_k, M_valid)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            auto* sorted_pairs = reinterpret_cast<uint32_t*>([eng->b_moe_sorted_pairs contents]);
            auto* work_tiles   = reinterpret_cast<Sg53MoeWorkTileHost*>([eng->b_moe_work_tiles contents]);
            uint32_t p_cur = 0U, t_cur = 0U;
            for (const auto& ue : uniq_exps) {
                const uint32_t cnt   = static_cast<uint32_t>(ue.pair_ids.size());
                const uint32_t p_beg = p_cur;
                for (uint32_t pid : ue.pair_ids) {
                    sorted_pairs[p_cur++] = pid;
                }
                for (uint32_t st = 0U; st < cnt; st += 8U) {
                    work_tiles[t_cur++] = Sg53MoeWorkTileHost{
                        static_cast<uint32_t>(ue.src_buf), ue.slot_idx, p_beg + st, std::min<uint32_t>(8U, cnt - st)
                    };
                }
            }
            if (t_cur > 0U) {
                Sg53MoeIndirectParamsHost mp{
                    eng->hdr.expert_record_bytes, eng->hdr.expert_proj_bytes,
                    eng->hdr.expert_weight_bytes, eng->hdr.expert_scales_bytes,
                    M_valid, top_k, D, d_moe, eng->cfg.swiglu_limit, t_cur, 0U, 0U
                };
                [*io_enc setComputePipelineState:eng->pso_moe_grouped_gu];
                [*io_enc setBuffer:eng->b_hot_experts offset:0 atIndex:0];
                [*io_enc setBuffer:eng->b_ring_dma offset:0 atIndex:1];
                [*io_enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
                [*io_enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                [*io_enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                [*io_enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
                [*io_enc setBytes:&mp length:sizeof(mp) atIndex:6];
                [*io_enc dispatchThreadgroups:MTLSizeMake((d_moe + 3U) / 4U, t_cur, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [*io_enc setComputePipelineState:eng->pso_moe_grouped_dn];
                [*io_enc setBuffer:eng->b_hot_experts offset:0 atIndex:0];
                [*io_enc setBuffer:eng->b_ring_dma offset:0 atIndex:1];
                [*io_enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
                [*io_enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                [*io_enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                [*io_enc setBuffer:eng->b_moe_down_part offset:0 atIndex:5];
                [*io_enc setBytes:&mp length:sizeof(mp) atIndex:6];
                [*io_enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, t_cur, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        }
    }

    {
        Sg53MoeCombineParamsHost cp{M_valid, top_k, D, 0U};
        [*io_enc setComputePipelineState:eng->pso_moe_combine];
        [*io_enc setBuffer:eng->b_moe_down_part offset:0 atIndex:0];
        [*io_enc setBuffer:eng->b_expert_dispatch offset:0 atIndex:1];
        [*io_enc setBuffer:eng->b_shared_out_f32 offset:0 atIndex:2];
        [*io_enc setBuffer:eng->b_block_y_f32 offset:0 atIndex:3];
        [*io_enc setBytes:&cp length:sizeof(cp) atIndex:4];
        const uint32_t d4 = D / 4U;
        [*io_enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_valid, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    sg53_encode_sublayer_exit(*io_enc, eng, ld.has_mhc != 0U, stream_buf, M_valid);
    return 0;
}

/* ============================================================================
 * v2 100% RAM-Resident Single-CB GPU-Binned `RHT-64 + E8S-18-V4` & UMA MTP MoE
 * (ZERO Mid-Step CPU Syncs — Entire 45-Layer Pass Encoded in 1 Command Buffer!)
 * ============================================================================ */

int sg53_encode_v2_moe_ffn(
    StrataGlm53GpuEngine*                  eng,
    const StrataGlm53LayerDesc&            ld,
    id<MTLBuffer>                          stream_buf,
    uint32_t                               M_valid,
    uint32_t                               M_padded,
    bool                                   force_gemv,
    id<MTLCommandBuffer> __strong*         io_cb,
    id<MTLComputeCommandEncoder> __strong* io_enc
) {
    const uint32_t moe_l = static_cast<uint32_t>(ld.moe_layer_idx);
    const uint32_t E     = eng->cfg.n_routed_experts;
    const uint32_t top_k = eng->cfg.num_experts_per_tok;
    const uint32_t D     = eng->cfg.hidden_size;
    const uint32_t d_moe = eng->cfg.moe_intermediate_size;
    id<MTLComputeCommandEncoder> enc = *io_enc;

    // 1. Determine Main `E8S-18-V4` (dual UMA part0/part1) vs Layer-45 MTP `Q4_A` (`b_mtp_experts`)
    const bool is_main_e8 =
        (moe_l < eng->hdr.num_main_moe_layers) &&
        (moe_l < 48U) &&
        (eng->hdr.moe_layer_quant[moe_l] == STRATA_GLM53_EXPERT_DTYPE_E8S18_V4);

    // 2. Sublayer entry (+ fused RHT-64 on `b_x_rot_f16` during decode -> -42 barriers/tok!) + 288E sigmoid router
    const bool fused_rot_done = sg53_encode_moe_router_step_a(
        enc, eng, ld, stream_buf, M_valid, M_padded, /*write_ane_unit_x=*/false,
        /*write_x_rot_f16=*/is_main_e8);

    // 3. Un-gated `shared_experts` (reads un-rotated `b_x_in_f16` -> writes `b_shared_out_f32`)
    sg53_encode_shared_expert(enc, eng, ld, M_valid, M_padded, force_gemv);

    if (is_main_e8) {
        const bool     in_part0       = (moe_l < eng->hdr.part0_moe_layers) || (!eng->b_experts_part1);
        id<MTLBuffer>  part_buf       = in_part0 ? eng->b_experts_part0 : eng->b_experts_part1;
        const uint32_t local_l        = in_part0 ? moe_l : (moe_l - eng->hdr.part0_moe_layers);
        const uint64_t layer_byte_off = static_cast<uint64_t>(local_l) * static_cast<uint64_t>(E) * eng->hdr.e8_record_bytes;

        // Warp-shuffle `RHT-64` on `b_x_in_f16 [M_valid, D]` -> `b_x_rot_f16 [M_valid, D]` (skipped if fused above!)
        if (!fused_rot_done) {
            Sg53Rht64ParamsHost rp{M_valid, M_valid, D, 0U};
            [enc setComputePipelineState:eng->pso_rht64_f16];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:0];
            [enc setBuffer:eng->b_x_rot_f16 offset:0 atIndex:1];
            [enc setBytes:&rp length:sizeof(rp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(((D / 64U) + 3U) / 4U, M_valid, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        const uint32_t total_pairs = M_valid * top_k;
        Sg53MoeE8ParamsHost ep{
            layer_byte_off,
            eng->hdr.e8_record_bytes,
            eng->hdr.e8_proj_bytes,
            eng->hdr.e8_lo16_bytes,
            eng->hdr.e8_hi2_bytes,
            M_valid, top_k, D, d_moe,
            eng->cfg.swiglu_limit,
            total_pairs + 64U,
            0U, 0U, 0U
        };

        if (M_valid == 1U) {
            // Fast `K = 0` Decode (`256`-th/TG, `constant` 64 KiB LUT + 2x LUT prefetch + dual-rate `half4` inner FMA!)
            const bool use_256 = (eng->pso_moe_e8s18_gather_gu_256 != nil) && (eng->pso_moe_e8s18_gather_dn_256 != nil);
            [enc setComputePipelineState:(use_256 ? eng->pso_moe_e8s18_gather_gu_256 : eng->pso_moe_e8s18_gather_gu)];
            [enc setBuffer:part_buf offset:0 atIndex:0];
            [enc setBuffer:eng->b_e8s18_lut offset:0 atIndex:1];
            [enc setBuffer:eng->b_topk_idx offset:0 atIndex:2];
            [enc setBuffer:eng->b_x_rot_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:4];
            [enc setBytes:&ep length:sizeof(ep) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(use_256 ? ((d_moe + 15U) / 16U) : ((d_moe + 7U) / 8U), top_k, 1)
                threadsPerThreadgroup:MTLSizeMake(use_256 ? 256 : 128, 1, 1)];

            // In-place `RHT-64` on `b_moe_mid_f16 [top_k, d_moe]` before `down_proj`
            {
                Sg53Rht64ParamsHost mrp{top_k, top_k, d_moe, 0U};
                [enc setComputePipelineState:eng->pso_rht64_f16];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:1];
                [enc setBytes:&mrp length:sizeof(mrp) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(((d_moe / 64U) + 3U) / 4U, top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            [enc setComputePipelineState:(use_256 ? eng->pso_moe_e8s18_gather_dn_256 : eng->pso_moe_e8s18_gather_dn)];
            [enc setBuffer:part_buf offset:0 atIndex:0];
            [enc setBuffer:eng->b_e8s18_lut offset:0 atIndex:1];
            [enc setBuffer:eng->b_topk_idx offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
            [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:4];
            [enc setBytes:&ep length:sizeof(ep) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((D + 15U) / 16U, top_k, 1)
                threadsPerThreadgroup:MTLSizeMake(use_256 ? 256 : 128, 1, 1)];
        } else if (force_gemv || M_valid <= kMaxSpecBatch || !eng->has_nax ||
                   ((D % 128U) != 0U) || ((d_moe % 128U) != 0U)) {
            // Spec Verify `M = 2..8` (GPU-binned indirect GEMV, 100.0000% bit-exact to `M = 1`!)
            Sg53MoeBinParamsHost bp{M_valid, top_k, E, d_moe, D, kMaxMoeBinTiles, 0U, 0U};
            [enc setComputePipelineState:eng->pso_moe_bin_v2];
            [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
            [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
            [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:3];
            [enc setBytes:&bp length:sizeof(bp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            [enc setComputePipelineState:eng->pso_moe_e8s18_grouped_gu];
            [enc setBuffer:part_buf offset:0 atIndex:0];
            [enc setBuffer:eng->b_e8s18_lut offset:0 atIndex:1];
            [enc setBuffer:eng->b_x_rot_f16 offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
            [enc setBytes:&ep length:sizeof(ep) atIndex:6];
            [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                   indirectBufferOffset:32
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            // In-place `RHT-64` on `b_moe_mid_f16 [M_valid * top_k, d_moe]`
            {
                Sg53Rht64ParamsHost mrp{total_pairs, total_pairs, d_moe, 0U};
                [enc setComputePipelineState:eng->pso_rht64_f16];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:1];
                [enc setBytes:&mrp length:sizeof(mrp) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(((d_moe / 64U) + 3U) / 4U, total_pairs, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            [enc setComputePipelineState:eng->pso_moe_e8s18_grouped_dn];
            [enc setBuffer:part_buf offset:0 atIndex:0];
            [enc setBuffer:eng->b_e8s18_lut offset:0 atIndex:1];
            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
            [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
            [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:5];
            [enc setBytes:&ep length:sizeof(ep) atIndex:6];
            [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                   indirectBufferOffset:44
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            // Prefill `M > 8`: Tri-Bin (`BM = 64 / 32 / 16`, `BK = 128`) Metal 4.1 Concurrent NAX MoE (`>= 42 TF`!)
            // Fused `LimitedSwiGLU + RHT-64` epilogue inside `gu` -> ZERO extra TG mem, ZERO extra RHT kernel!
            Sg53MoeBinParamsHost bp{M_valid, top_k, E, d_moe, D, kMaxMoeBinTiles, 0U, 0U};
            [enc setComputePipelineState:eng->pso_moe_bin_nax];
            [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
            [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
            [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:3];
            [enc setBytes:&bp length:sizeof(bp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            const bool use_e8_i8 =
                eng->int8_prefill_enabled && eng->int8_moe_enabled &&
                (M_valid >= 32U) &&
                (eng->b_moe_x_rsc != nil) && (eng->b_moe_mid_rsc != nil) && (eng->b_moe_mid_i8 != nil) &&
                (eng->pso_moe_gather_x_i8 != nil) && (eng->pso_moe_quant_mid_i8 != nil);

            Sg53MoeE8ParamsHost ep64 = ep; ep64.tile_base = 0U;
            Sg53MoeE8ParamsHost ep32 = ep; ep32.tile_base = kMaxMoeBinTiles;
            Sg53MoeE8ParamsHost ep16 = ep; ep16.tile_base = 2U * kMaxMoeBinTiles;

            if (use_e8_i8) {
                // Opt-In `--int8-prefill` E8S-18-V4 INT8×INT8->INT32 NAX (`cos = 0.99993`, `1.24x` vs FP16 NAX!)
                [enc setComputePipelineState:eng->pso_moe_gather_x_i8];
                [enc setBuffer:eng->b_x_rot_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
                [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:2]; // x_sorted_i8 alias
                [enc setBuffer:eng->b_moe_x_rsc offset:0 atIndex:3];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:4];
                [enc setBytes:&bp length:sizeof(bp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(total_pairs + 64U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];

                {
                    id<MTLComputeCommandEncoder> cenc_gu = [*io_cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    [cenc_gu setComputePipelineState:eng->pso_moe_e8s18_nax_i8_gu_m64];
                    [cenc_gu setBuffer:part_buf offset:0 atIndex:0];
                    [cenc_gu setBuffer:eng->b_e8s18_lut offset:0 atIndex:1];
                    [cenc_gu setBuffer:eng->b_moe_down_part offset:0 atIndex:2];
                    [cenc_gu setBuffer:eng->b_moe_x_rsc offset:0 atIndex:3];
                    [cenc_gu setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                    [cenc_gu setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:5];
                    [cenc_gu setBytes:&ep64 length:sizeof(ep64) atIndex:6];
                    [cenc_gu dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:0
                                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_gu setComputePipelineState:eng->pso_moe_e8s18_nax_i8_gu_m32];
                    [cenc_gu setBytes:&ep32 length:sizeof(ep32) atIndex:6];
                    [cenc_gu dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:12
                                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_gu setComputePipelineState:eng->pso_moe_e8s18_nax_i8_gu_m16];
                    [cenc_gu setBytes:&ep16 length:sizeof(ep16) atIndex:6];
                    [cenc_gu dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:24
                                              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [cenc_gu endEncoding];
                }

                {
                    id<MTLComputeCommandEncoder> qenc = [*io_cb computeCommandEncoder];
                    [qenc setComputePipelineState:eng->pso_moe_quant_mid_i8];
                    [qenc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:0];
                    [qenc setBuffer:eng->b_moe_mid_i8 offset:0 atIndex:1];
                    [qenc setBuffer:eng->b_moe_mid_rsc offset:0 atIndex:2];
                    [qenc setBytes:&bp length:sizeof(bp) atIndex:3];
                    [qenc dispatchThreadgroups:MTLSizeMake(total_pairs + 64U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [qenc endEncoding];
                }

                {
                    id<MTLComputeCommandEncoder> cenc_dn = [*io_cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    [cenc_dn setComputePipelineState:eng->pso_moe_e8s18_nax_i8_dn_m64];
                    [cenc_dn setBuffer:part_buf offset:0 atIndex:0];
                    [cenc_dn setBuffer:eng->b_e8s18_lut offset:0 atIndex:1];
                    [cenc_dn setBuffer:eng->b_moe_mid_i8 offset:0 atIndex:2];
                    [cenc_dn setBuffer:eng->b_moe_mid_rsc offset:0 atIndex:3];
                    [cenc_dn setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:4];
                    [cenc_dn setBuffer:eng->b_moe_work_tiles offset:0 atIndex:5];
                    [cenc_dn setBuffer:eng->b_moe_down_part offset:0 atIndex:6];
                    [cenc_dn setBytes:&ep64 length:sizeof(ep64) atIndex:7];
                    [cenc_dn dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:36
                                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_dn setComputePipelineState:eng->pso_moe_e8s18_nax_i8_dn_m32];
                    [cenc_dn setBytes:&ep32 length:sizeof(ep32) atIndex:7];
                    [cenc_dn dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:48
                                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_dn setComputePipelineState:eng->pso_moe_e8s18_nax_i8_dn_m16];
                    [cenc_dn setBytes:&ep16 length:sizeof(ep16) atIndex:7];
                    [cenc_dn dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:60
                                              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [cenc_dn endEncoding];
                }
            } else {
                // Gather `x_rot` into dead `b_moe_down_part` alias (`x_sorted`) + zero `+64` tails of `x_sorted` & `mid_sorted`!
                [enc setComputePipelineState:eng->pso_moe_gather_x_f16];
                [enc setBuffer:eng->b_x_rot_f16 offset:0 atIndex:0];
                [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
                [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:2];
                [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
                [enc setBytes:&bp length:sizeof(bp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(total_pairs + 64U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc endEncoding];

                {
                    id<MTLComputeCommandEncoder> cenc_gu = [*io_cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    [cenc_gu setComputePipelineState:eng->pso_moe_e8s18_nax_gu_m64];
                    [cenc_gu setBuffer:part_buf offset:0 atIndex:0];
                    [cenc_gu setBuffer:eng->b_e8s18_lut offset:0 atIndex:1];
                    [cenc_gu setBuffer:eng->b_moe_down_part offset:0 atIndex:2]; // x_sorted alias
                    [cenc_gu setBuffer:eng->b_moe_work_tiles offset:0 atIndex:3];
                    [cenc_gu setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:4];   // mid_sorted (RHT-64 rotated!)
                    [cenc_gu setBytes:&ep64 length:sizeof(ep64) atIndex:5];
                    [cenc_gu dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:0
                                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_gu setComputePipelineState:eng->pso_moe_e8s18_nax_gu_m32];
                    [cenc_gu setBytes:&ep32 length:sizeof(ep32) atIndex:5];
                    [cenc_gu dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:12
                                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_gu setComputePipelineState:eng->pso_moe_e8s18_nax_gu_m16];
                    [cenc_gu setBytes:&ep16 length:sizeof(ep16) atIndex:5];
                    [cenc_gu dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:24
                                              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [cenc_gu endEncoding];
                }

                {
                    id<MTLComputeCommandEncoder> cenc_dn = [*io_cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    [cenc_dn setComputePipelineState:eng->pso_moe_e8s18_nax_dn_m64];
                    [cenc_dn setBuffer:part_buf offset:0 atIndex:0];
                    [cenc_dn setBuffer:eng->b_e8s18_lut offset:0 atIndex:1];
                    [cenc_dn setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
                    [cenc_dn setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:3];
                    [cenc_dn setBuffer:eng->b_moe_work_tiles offset:0 atIndex:4];
                    [cenc_dn setBuffer:eng->b_moe_down_part offset:0 atIndex:5]; // x_sorted dead -> write down_part!
                    [cenc_dn setBytes:&ep64 length:sizeof(ep64) atIndex:6];
                    [cenc_dn dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:36
                                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_dn setComputePipelineState:eng->pso_moe_e8s18_nax_dn_m32];
                    [cenc_dn setBytes:&ep32 length:sizeof(ep32) atIndex:6];
                    [cenc_dn dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:48
                                              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_dn setComputePipelineState:eng->pso_moe_e8s18_nax_dn_m16];
                    [cenc_dn setBytes:&ep16 length:sizeof(ep16) atIndex:6];
                    [cenc_dn dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                               indirectBufferOffset:60
                                              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [cenc_dn endEncoding];
                }
            }

            enc     = [*io_cb computeCommandEncoder];
            *io_enc = enc;
        }
    } else if (eng->b_mtp_experts) {
        // Layer-45 MTP UMA `Q4_A` (`3.80 GiB`, ZERO CPU sync!)
        Sg53MoeIndirectParamsHost mp{
            eng->hdr.expert_record_bytes, eng->hdr.expert_proj_bytes,
            eng->hdr.expert_weight_bytes, eng->hdr.expert_scales_bytes,
            M_valid, top_k, D, d_moe, eng->cfg.swiglu_limit, 0U, 0U, 0U
        };
        if (M_valid == 1U) {
            [enc setComputePipelineState:eng->pso_moe_q4_uma_gather_gu];
            [enc setBuffer:eng->b_mtp_experts offset:0 atIndex:0];
            [enc setBuffer:eng->b_topk_idx offset:0 atIndex:1];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:3];
            [enc setBytes:&mp length:sizeof(mp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((d_moe + 3U) / 4U, top_k, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:eng->pso_moe_q4_uma_gather_dn];
            [enc setBuffer:eng->b_mtp_experts offset:0 atIndex:0];
            [enc setBuffer:eng->b_topk_idx offset:0 atIndex:1];
            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:3];
            [enc setBytes:&mp length:sizeof(mp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, top_k, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            Sg53MoeBinParamsHost bp{M_valid, top_k, E, d_moe, D, kMaxMoeBinTiles, 0U, 0U};
            [enc setComputePipelineState:eng->pso_moe_bin_v2];
            [enc setBuffer:eng->b_topk_idx offset:0 atIndex:0];
            [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:1];
            [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_indirect_args offset:0 atIndex:3];
            [enc setBytes:&bp length:sizeof(bp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            [enc setComputePipelineState:eng->pso_moe_q4_uma_grouped_gu];
            [enc setBuffer:eng->b_mtp_experts offset:0 atIndex:0];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:1];
            [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:3];
            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:4];
            [enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                   indirectBufferOffset:0
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:eng->pso_moe_q4_uma_grouped_dn];
            [enc setBuffer:eng->b_mtp_experts offset:0 atIndex:0];
            [enc setBuffer:eng->b_moe_mid_f16 offset:0 atIndex:1];
            [enc setBuffer:eng->b_moe_sorted_pairs offset:0 atIndex:2];
            [enc setBuffer:eng->b_moe_work_tiles offset:0 atIndex:3];
            [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:4];
            [enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [enc dispatchThreadgroupsWithIndirectBuffer:eng->b_moe_indirect_args
                                   indirectBufferOffset:12
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }

    // 4. Direct `b_topk_w` combine + sublayer exit (fused into ONE kernel when `has_mhc` -> -42 barriers/tok!)
    if (ld.has_mhc != 0U && eng->cfg.hc_mult == 4U && eng->pso_moe_combine_mhc_expand != nil) {
        Sg53MoeCombineParamsHost cp{M_valid, top_k, D, 0U};
        [enc setComputePipelineState:eng->pso_moe_combine_mhc_expand];
        [enc setBuffer:stream_buf offset:0 atIndex:0];
        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:1];
        [enc setBuffer:eng->b_topk_w offset:0 atIndex:2];
        [enc setBuffer:eng->b_shared_out_f32 offset:0 atIndex:3];
        [enc setBuffer:eng->b_mhc_post_f32 offset:0 atIndex:4];
        [enc setBuffer:eng->b_mhc_comb_f32 offset:0 atIndex:5];
        [enc setBytes:&cp length:sizeof(cp) atIndex:6];
        const uint32_t d4 = D / 4U;
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_valid, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        Sg53MoeCombineParamsHost cp{M_valid, top_k, D, 0U};
        [enc setComputePipelineState:eng->pso_moe_combine_v2];
        [enc setBuffer:eng->b_moe_down_part offset:0 atIndex:0];
        [enc setBuffer:eng->b_topk_w offset:0 atIndex:1];
        [enc setBuffer:eng->b_shared_out_f32 offset:0 atIndex:2];
        [enc setBuffer:eng->b_block_y_f32 offset:0 atIndex:3];
        [enc setBytes:&cp length:sizeof(cp) atIndex:4];
        const uint32_t d4 = D / 4U;
        [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_valid, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        sg53_encode_sublayer_exit(enc, eng, ld.has_mhc != 0U, stream_buf, M_valid);
    }

    {
        const uint64_t eval_cnt = static_cast<uint64_t>(M_valid) * top_k;
        Sg53ScopedLock tlk(&eng->telemetry_lock);
        eng->counters.active_experts_evaluated += eval_cnt;
        eng->counters.uma_hot_hits             += eval_cnt;
    }
    return 0;
}

int sg53_execute_decoder_layer(
    StrataGlm53GpuEngine*                  eng,
    const StrataGlm53LayerDesc&            ld,
    id<MTLBuffer>                          stream_buf,
    uint32_t                               kda_slot,
    uint32_t                               dsa_slot,
    uint32_t                               seq_pos_start,
    uint32_t                               M_valid,
    uint32_t                               M_padded,
    uint32_t                               save_snap_mask,
    bool                                   force_gemv,
    bool                                   is_speculative,
    bool                                   allow_ane_lookahead,
    id<MTLCommandBuffer> __strong*         io_cb,
    id<MTLComputeCommandEncoder> __strong* io_enc,
    std::vector<Sg53SpecLayerSnap>*        spec_snaps_out
) {
    sg53_encode_attention_sublayer(io_cb, io_enc, eng, ld, stream_buf, kda_slot, dsa_slot,
                                   seq_pos_start, M_valid, M_padded, save_snap_mask, force_gemv);
    if (eng->paged_error) return -4;
    if (ld.mlp_type == STRATA_GLM53_MLP_DENSE) {
        sg53_encode_dense_ffn_sublayer(*io_enc, eng, ld, stream_buf, M_valid, M_padded, force_gemv);
        return 0;
    }
    if (eng->is_ram_resident) {
        return sg53_encode_v2_moe_ffn(eng, ld, stream_buf, M_valid, M_padded, force_gemv, io_cb, io_enc);
    }
    return sg53_execute_moe_ffn_pipelined(
        eng, ld, stream_buf, M_valid, M_padded, force_gemv,
        is_speculative, allow_ane_lookahead, io_cb, io_enc, spec_snaps_out
    );
}

// Executes Native Layer-45 `MTP` (`layers.45`, Single-Stream `DSA MLA` + `288E` MoE):
//   - `e_tok = (mtp_pos == 0) ? zeros(D) : embed_tokens(shifted_tok)`
//   - `x_mtp = mtp.eh_proj(concat(enorm(e_tok), hnorm(h_in)))`
//   - `mtp.layer` (`dsa_slot = cfg.num_dsa_layers`, `moe_layer_idx = 42`, `has_mhc == 0`)
//   - If `run_head`: `h_mtp_out = rms_norm(x_mtp, mtp.shared_head_norm)` -> `b_mtp_h_out_f32` & `b_x_in_f16` -> `lm_head` -> `b_logits`!
int sg53_execute_mtp_pass(
    StrataGlm53GpuEngine*                  eng,
    id<MTLBuffer>                          h_in_f32_buf,
    NSUInteger                             h_in_byte_off,
    id<MTLBuffer>                          tok_ids_buf,
    NSUInteger                             tok_byte_off,
    uint32_t                               mtp_pos_start,
    uint32_t                               M_valid,
    bool                                   run_head,
    bool                                   force_gemv,
    bool                                   is_speculative,
    id<MTLCommandBuffer> __strong*         io_cb,
    id<MTLComputeCommandEncoder> __strong* io_enc,
    bool                                   use_frspec = false
) {
    if (!eng->has_mtp || M_valid == 0U) return 0;

    const StrataGlm53MtpDesc& mtp = eng->hdr.mtp;
    const uint32_t D        = eng->cfg.hidden_size;
    const uint32_t V        = eng->cfg.vocab_size;
    const float    eps      = eng->cfg.rms_norm_eps;
    const uint32_t M_padded = (!force_gemv && eng->has_nax) ? ((M_valid + 63U) & ~63U) : M_valid;

    // 1. `e_tok` (`zeros(D)` when `mtp_pos_start + m == 0`!) -> `b_mtp_e_f32 [M_valid, D]`
    {
        Sg53EmbedParamsHost ep{M_valid, D, V, 1U, mtp_pos_start, 1U /* zero_at_pos0! */, 0U, 0U};
        [*io_enc setComputePipelineState:eng->pso_embed_q4];
        [*io_enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.weight_off) atIndex:0];
        [*io_enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.scales_off) atIndex:1];
        [*io_enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.biases_off) atIndex:2];
        [*io_enc setBuffer:tok_ids_buf offset:tok_byte_off atIndex:3];
        [*io_enc setBuffer:eng->b_mtp_e_f32 offset:0 atIndex:4];
        [*io_enc setBytes:&ep length:sizeof(ep) atIndex:5];
        [*io_enc dispatchThreadgroups:MTLSizeMake(M_valid, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
    }

    // 2. `concat_8192(rms_norm(e_tok, mtp.enorm, 1e-5f), rms_norm(h_in, mtp.hnorm, 1e-5f))` -> `b_mtp_eh_cat_f16 [M_padded, 2*D]`
    {
        Sg53RmsNormParamsHost enp{M_valid, M_padded, D, D, 0U, 2U * D, 0U, eps, 1U, 0U, 0U, 0U};
        [*io_enc setComputePipelineState:eng->pso_rmsnorm_strided];
        [*io_enc setBuffer:eng->b_mtp_e_f32 offset:0 atIndex:0];
        [*io_enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(mtp.enorm.offset) atIndex:1];
        [*io_enc setBuffer:eng->b_mtp_eh_cat_f16 offset:0 atIndex:2];
        [*io_enc setBuffer:eng->b_mtp_e_f32 offset:0 atIndex:3];
        [*io_enc setBytes:&enp length:sizeof(enp) atIndex:4];
        [*io_enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        Sg53RmsNormParamsHost hnp{M_valid, M_padded, D, D, 0U, 2U * D, D, eps, 1U, 0U, 0U, 0U};
        [*io_enc setComputePipelineState:eng->pso_rmsnorm_strided];
        [*io_enc setBuffer:h_in_f32_buf offset:h_in_byte_off atIndex:0];
        [*io_enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(mtp.hnorm.offset) atIndex:1];
        [*io_enc setBuffer:eng->b_mtp_eh_cat_f16 offset:0 atIndex:2];
        [*io_enc setBuffer:eng->b_mtp_e_f32 offset:0 atIndex:3];
        [*io_enc setBytes:&hnp length:sizeof(hnp) atIndex:4];
        [*io_enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    // 3. `x_mtp = mtp.eh_proj(b_mtp_eh_cat_f16)` -> `b_mtp_x_f32 [M_valid, D]`
    sg53_encode_q4_linear(*io_enc, eng, mtp.eh_proj,
                          eng->b_mtp_eh_cat_f16, 0, nil, 0, eng->b_mtp_x_f32, 0,
                          M_valid, M_padded, kOutF32, force_gemv);

    // 4. Single-stream Layer-45 `DSA MLA` + `288E` MoE (`dsa_slot = cfg.num_dsa_layers`)
    const int rc = sg53_execute_decoder_layer(
        eng, mtp.layer, eng->b_mtp_x_f32,
        0U, eng->cfg.num_dsa_layers,
        mtp_pos_start, M_valid, M_padded, 0U,
        force_gemv, is_speculative, /*allow_ane_lookahead=*/false,
        io_cb, io_enc, nullptr
    );
    if (rc != 0) return rc;

    // 5. Optional `shared_head.norm` + `lm_head` (writes `b_mtp_h_out_f32` for chained `k >= 1` draft steps!)
    if (run_head) {
        Sg53RmsNormParamsHost snp{M_valid, M_padded, D, D, 0U, D, 0U, eps, 1U, 1U, 0U, 0U};
        [*io_enc setComputePipelineState:eng->pso_rmsnorm_strided];
        [*io_enc setBuffer:eng->b_mtp_x_f32 offset:0 atIndex:0];
        [*io_enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(mtp.shared_head_norm.offset) atIndex:1];
        [*io_enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
        [*io_enc setBuffer:eng->b_mtp_h_out_f32 offset:0 atIndex:3];
        [*io_enc setBytes:&snp length:sizeof(snp) atIndex:4];
        [*io_enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        if (use_frspec && eng->b_frspec_lm_head != nil && eng->frspec_shortlist_size > 0U) {
            sg53_encode_q4_linear(*io_enc, eng, eng->frspec_lm_head_mat,
                                  eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                                  M_valid, M_padded, kOutF32, true, eng->b_frspec_lm_head);
        } else {
            sg53_encode_q4_linear(*io_enc, eng, eng->hdr.lm_head,
                                  eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                                  M_valid, M_padded, kOutF32, true);
        }
    }
    return 0;
}

/* ============================================================================
 * Sliding-Window OpenAI/HF Logit Penalties & Longest-Match-First 5/4-Gram Drafter
 * ============================================================================ */

inline bool sg53_penalties_active(const StrataGlm53SamplingParams* s) {
    if (!s) return false;
    return (s->presence_penalty != 0.0f) ||
           (s->frequency_penalty != 0.0f) ||
           (s->repetition_penalty > 0.0f && s->repetition_penalty != 1.0f);
}

inline void sg53_penalties_inc(
    StrataGlm53GpuEngine* eng,
    uint32_t*             ids,
    uint16_t*             cnt,
    uint32_t              t
) {
    const uint32_t V = eng->cfg.vocab_size;
    if (t >= V) return;
    if (cnt[t] == 0U && eng->pen_num_ids < V) {
        const uint32_t idx = eng->pen_num_ids++;
        ids[idx]        = t;
        eng->pen_pos[t] = idx;
    }
    if (cnt[t] < 65535U) ++cnt[t];
}

inline void sg53_penalties_dec(
    StrataGlm53GpuEngine* eng,
    uint32_t*             ids,
    uint16_t*             cnt,
    uint32_t              t
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

inline void sg53_penalties_clear_active(StrataGlm53GpuEngine* eng) {
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

void sg53_penalties_sync_window(StrataGlm53GpuEngine* eng, uint32_t eff_window) {
    if (!eng || !eng->b_pen_ids || !eng->b_pen_counts) return;
    const uint32_t L = static_cast<uint32_t>(eng->pen_hist.size());
    const uint32_t target_start = (eff_window > 0U && L > eff_window) ? (L - eff_window) : 0U;
    if (eng->pen_window_start == target_start && eng->pen_window_end == L) return;

    uint32_t* ids = reinterpret_cast<uint32_t*>([eng->b_pen_ids contents]);
    uint16_t* cnt = reinterpret_cast<uint16_t*>([eng->b_pen_counts contents]);

    if (target_start >= eng->pen_window_end || eng->pen_window_end > L) {
        sg53_penalties_clear_active(eng);
        eng->pen_window_start = target_start;
        eng->pen_window_end   = target_start;
    }
    if (target_start < eng->pen_window_start) {
        for (uint32_t i = target_start; i < eng->pen_window_start; ++i) {
            sg53_penalties_inc(eng, ids, cnt, eng->pen_hist[i]);
        }
        eng->pen_window_start = target_start;
    }
    if (eng->pen_window_end < L) {
        for (uint32_t i = eng->pen_window_end; i < L; ++i) {
            sg53_penalties_inc(eng, ids, cnt, eng->pen_hist[i]);
        }
        eng->pen_window_end = L;
    }
    if (eng->pen_window_start < target_start) {
        for (uint32_t i = eng->pen_window_start; i < target_start; ++i) {
            sg53_penalties_dec(eng, ids, cnt, eng->pen_hist[i]);
        }
        eng->pen_window_start = target_start;
    }
}

void sg53_encode_logit_penalties(
    id<MTLComputeCommandEncoder>     enc,
    StrataGlm53GpuEngine*            eng,
    const StrataGlm53SamplingParams* sampling,
    uint32_t                         M,
    id<MTLBuffer>                    extra_toks_buf,
    uint32_t                         extra_mode,
    uint32_t                         extra_count
) {
    if (!sg53_penalties_active(sampling) || !eng->pso_apply_logit_penalties ||
        !eng->b_pen_ids || !eng->b_pen_counts) {
        return;
    }
    const uint32_t eff_window = (sampling->penalty_window == 0U)
        ? 0U
        : std::max<uint32_t>(sampling->penalty_window, 8U);
    sg53_penalties_sync_window(eng, eff_window);

    const uint32_t total_items = eng->pen_num_ids + extra_count;
    if (total_items == 0U) return;

    Sg53LogitPenaltyParamsHost pp{};
    pp.M                  = M;
    pp.vocab_size         = eng->cfg.vocab_size;
    pp.num_hist_ids       = eng->pen_num_ids;
    pp.extra_count        = extra_count;
    pp.extra_mode         = extra_mode;
    pp.presence_penalty   = sampling->presence_penalty;
    pp.frequency_penalty  = sampling->frequency_penalty;
    pp.repetition_penalty = sampling->repetition_penalty;
    for (uint32_t j = 0U; j < 8U; ++j) pp.drop_toks[j] = 0xFFFFFFFFU;
    if (eff_window > 0U && extra_count > 0U) {
        const uint32_t L = static_cast<uint32_t>(eng->pen_hist.size());
        const uint32_t max_depth = std::min<uint32_t>(extra_count, 8U);
        for (uint32_t j = 0U; j < max_depth; ++j) {
            if (L + j >= eff_window) {
                const uint32_t evict_idx = L + j - eff_window;
                if (evict_idx < L) pp.drop_toks[j] = eng->pen_hist[evict_idx];
            }
        }
    }

    [enc setComputePipelineState:eng->pso_apply_logit_penalties];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_pen_ids offset:0 atIndex:1];
    [enc setBuffer:eng->b_pen_counts offset:0 atIndex:2];
    [enc setBuffer:(extra_toks_buf ? extra_toks_buf : eng->b_token_ids) offset:0 atIndex:3];
    [enc setBytes:&pp length:sizeof(pp) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((total_items + 255U) / 256U, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sg53_encode_logit_penalties_shortlist(
    id<MTLComputeCommandEncoder>     enc,
    StrataGlm53GpuEngine*            eng,
    const StrataGlm53SamplingParams* sampling,
    id<MTLBuffer>                    extra_toks_buf,
    uint32_t                         extra_count
) {
    if (!sg53_penalties_active(sampling) || !eng->pso_apply_logit_penalties_shortlist ||
        !eng->b_pen_ids || !eng->b_pen_counts || !eng->b_frspec_full_to_short ||
        eng->frspec_shortlist_size == 0U) {
        return;
    }
    const uint32_t eff_window = (sampling->penalty_window == 0U)
        ? 0U
        : std::max<uint32_t>(sampling->penalty_window, 8U);
    sg53_penalties_sync_window(eng, eff_window);

    const uint32_t total_items = eng->pen_num_ids + extra_count;
    if (total_items == 0U) return;

    Sg53LogitPenaltyParamsHost pp{};
    pp.M                  = 1U;
    pp.vocab_size         = eng->frspec_shortlist_size;
    pp.num_hist_ids       = eng->pen_num_ids;
    pp.extra_count        = extra_count;
    pp.extra_mode         = 1U;
    pp.presence_penalty   = sampling->presence_penalty;
    pp.frequency_penalty  = sampling->frequency_penalty;
    pp.repetition_penalty = sampling->repetition_penalty;
    for (uint32_t j = 0U; j < 8U; ++j) pp.drop_toks[j] = 0xFFFFFFFFU;
    if (eff_window > 0U && extra_count > 0U) {
        const uint32_t L = static_cast<uint32_t>(eng->pen_hist.size());
        const uint32_t max_depth = std::min<uint32_t>(extra_count, 8U);
        for (uint32_t j = 0U; j < max_depth; ++j) {
            if (L + j >= eff_window) {
                const uint32_t evict_idx = L + j - eff_window;
                if (evict_idx < L) pp.drop_toks[j] = eng->pen_hist[evict_idx];
            }
        }
    }

    [enc setComputePipelineState:eng->pso_apply_logit_penalties_shortlist];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_pen_ids offset:0 atIndex:1];
    [enc setBuffer:eng->b_pen_counts offset:0 atIndex:2];
    [enc setBuffer:(extra_toks_buf ? extra_toks_buf : eng->b_mtp_token_ids) offset:0 atIndex:3];
    [enc setBuffer:eng->b_frspec_full_to_short offset:0 atIndex:4];
    [enc setBytes:&pp length:sizeof(pp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((total_items + 255U) / 256U, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

inline uint32_t sg53_ngram5_hash(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
    uint64_t h = (static_cast<uint64_t>(a) * 0x9E3779B97F4A7C15ULL) ^
                 (static_cast<uint64_t>(b) * 0xBF58476D1CE4E5B9ULL) ^
                 (static_cast<uint64_t>(c) * 0x94D049BB133111EBULL) ^
                 (static_cast<uint64_t>(d) * 0xD6E8FEB86659FD93ULL) ^
                 (static_cast<uint64_t>(e));
    h ^= (h >> 33ULL);
    h *= 0xFF51AFD7ED558CCDULL;
    return static_cast<uint32_t>((h >> 32ULL) & (kNgramHashSize - 1U));
}

inline uint32_t sg53_ngram4_hash(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint64_t h = (static_cast<uint64_t>(a) * 0xBF58476D1CE4E5B9ULL) ^
                 (static_cast<uint64_t>(b) * 0x94D049BB133111EBULL) ^
                 (static_cast<uint64_t>(c) * 0xD6E8FEB86659FD93ULL) ^
                 (static_cast<uint64_t>(d));
    h ^= (h >> 33ULL);
    h *= 0xC4CEB9FE1A85EC53ULL;
    return static_cast<uint32_t>((h >> 32ULL) & (kNgramHashSize - 1U));
}

inline void sg53_ngram_reset(StrataGlm53ContextSlot* s) {
    if (!s) return;
    s->ngram_hist.clear();
    if (!s->ngram5_ht.empty()) std::fill(s->ngram5_ht.begin(), s->ngram5_ht.end(), 0xFFFFFFFFU);
    if (!s->ngram4_ht.empty()) std::fill(s->ngram4_ht.begin(), s->ngram4_ht.end(), 0xFFFFFFFFU);
    s->ngram_cooldown = 0U;
}

inline void sg53_ngram_append(StrataGlm53ContextSlot* s, const uint32_t* toks, uint32_t n) {
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
        s->ngram4_ht[sg53_ngram4_hash(h[i - 3U], h[i - 2U], h[i - 1U], h[i])] = static_cast<uint32_t>(i);
        if (i >= 4U) {
            s->ngram5_ht[sg53_ngram5_hash(h[i - 4U], h[i - 3U], h[i - 2U], h[i - 1U], h[i])] = static_cast<uint32_t>(i);
        }
    }
}

inline void sg53_ngram_rewind(StrataGlm53ContextSlot* s, uint32_t new_len) {
    if (!s) return;
    if (new_len < s->ngram_hist.size()) s->ngram_hist.resize(new_len);
    if (!s->ngram5_ht.empty()) {
        std::fill(s->ngram5_ht.begin(), s->ngram5_ht.end(), 0xFFFFFFFFU);
        std::fill(s->ngram4_ht.begin(), s->ngram4_ht.end(), 0xFFFFFFFFU);
        const uint32_t* h   = s->ngram_hist.data();
        const size_t    end = s->ngram_hist.size();
        for (size_t i = 3U; i < end; ++i) {
            s->ngram4_ht[sg53_ngram4_hash(h[i - 3U], h[i - 2U], h[i - 1U], h[i])] = static_cast<uint32_t>(i);
            if (i >= 4U) {
                s->ngram5_ht[sg53_ngram5_hash(h[i - 4U], h[i - 3U], h[i - 2U], h[i - 1U], h[i])] = static_cast<uint32_t>(i);
            }
        }
    }
    s->ngram_cooldown = 0U;
}

inline uint32_t sg53_ngram_propose(
    const StrataGlm53ContextSlot* s,
    uint32_t                      pending_tok,
    uint32_t                      max_k,
    uint32_t*                     out_drafts
) {
    if (!s || !out_drafts || max_k == 0U || s->ngram_cooldown > 0U || s->ngram5_ht.empty()) return 0U;
    const size_t sz = s->ngram_hist.size();
    if (sz < 4U) return 0U;
    const uint32_t* h = s->ngram_hist.data();

    // 1. Longest-match-first: 5-gram hit (up to `max_k`)
    if (sz >= 5U) {
        const uint32_t a = h[sz - 4U], b = h[sz - 3U], c = h[sz - 2U], d = h[sz - 1U], e = pending_tok;
        const uint32_t pos = s->ngram5_ht[sg53_ngram5_hash(a, b, c, d, e)];
        if (pos != 0xFFFFFFFFU && pos >= 4U && static_cast<size_t>(pos) + 1U < sz &&
            h[pos - 4U] == a && h[pos - 3U] == b && h[pos - 2U] == c && h[pos - 1U] == d && h[pos] == e) {
            const uint32_t avail = static_cast<uint32_t>(sz - 1U - static_cast<size_t>(pos));
            const uint32_t min_k = std::min<uint32_t>(max_k, 3U);
            if (avail >= min_k) {
                const uint32_t eff_k = std::min<uint32_t>(max_k, avail);
                for (uint32_t k = 0U; k < eff_k; ++k) out_drafts[k] = h[static_cast<size_t>(pos) + 1U + k];
                return eff_k;
            }
        }
    }

    // 2. Fallback 4-gram hit (up to `max_k`)
    {
        const uint32_t b = h[sz - 3U], c = h[sz - 2U], d = h[sz - 1U], e = pending_tok;
        const uint32_t pos = s->ngram4_ht[sg53_ngram4_hash(b, c, d, e)];
        if (pos != 0xFFFFFFFFU && pos >= 3U && static_cast<size_t>(pos) + 1U < sz &&
            h[pos - 3U] == b && h[pos - 2U] == c && h[pos - 1U] == d && h[pos] == e) {
            const uint32_t avail = static_cast<uint32_t>(sz - 1U - static_cast<size_t>(pos));
            const uint32_t min_k = std::min<uint32_t>(max_k, 2U);
            if (avail >= min_k) {
                const uint32_t eff_k = std::min<uint32_t>(max_k, avail);
                for (uint32_t k = 0U; k < eff_k; ++k) out_drafts[k] = h[static_cast<size_t>(pos) + 1U + k];
                return eff_k;
            }
        }
    }
    return 0U;
}

void sg53_encode_sample_batch(
    id<MTLComputeCommandEncoder>     enc,
    StrataGlm53GpuEngine*            eng,
    const StrataGlm53SamplingParams* sampling,
    uint32_t                         M,
    id<MTLBuffer>                    out_tok_buf,
    NSUInteger                       out_tok_off,
    id<MTLBuffer>                    top_ids_buf,
    NSUInteger                       top_ids_off,
    id<MTLBuffer>                    top_probs_buf,
    NSUInteger                       top_probs_off,
    id<MTLBuffer>                    keep_k_buf,
    NSUInteger                       keep_k_off
) {
    Sg53SampleParamsHost sp{};
    sp.M          = M;
    sp.vocab_size = eng->cfg.vocab_size;

    const bool is_greedy = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
    if (is_greedy) {
        sp.greedy   = 1U;
        sp.inv_temp = 1.0f;
        sp.top_k    = 1U;
        sp.top_p    = 1.0f;
        sp.rng_lo   = 0U;
        sp.rng_hi   = 0U;
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

    [enc setComputePipelineState:eng->pso_sample_stage1];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:1];
    [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:2];
    [enc setBytes:&sp length:sizeof(sp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(64, M, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:eng->pso_sample_stage2];
    [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
    [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
    [enc setBuffer:out_tok_buf offset:out_tok_off atIndex:2];
    [enc setBuffer:(top_ids_buf ? top_ids_buf : out_tok_buf) offset:top_ids_off atIndex:3];
    [enc setBuffer:(top_probs_buf ? top_probs_buf : eng->b_logits) offset:top_probs_off atIndex:4];
    [enc setBuffer:(keep_k_buf ? keep_k_buf : out_tok_buf) offset:keep_k_off atIndex:5];
    [enc setBytes:&sp length:sizeof(sp) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sg53_encode_draft_sample_step(
    id<MTLComputeCommandEncoder>     enc,
    StrataGlm53GpuEngine*            eng,
    const StrataGlm53SamplingParams* sampling,
    bool                             use_frspec,
    bool                             want_pmax,
    id<MTLBuffer>                    out_tok_buf,
    NSUInteger                       out_tok_off,
    id<MTLBuffer>                    out_pmax_buf,
    NSUInteger                       out_pmax_off,
    id<MTLBuffer>                    top_ids_buf,
    NSUInteger                       top_ids_off,
    id<MTLBuffer>                    top_probs_buf,
    NSUInteger                       top_probs_off,
    id<MTLBuffer>                    keep_k_buf,
    NSUInteger                       keep_k_off
) {
    const bool use_map = use_frspec && (eng->b_frspec_short_to_full != nil) && (eng->frspec_shortlist_size > 0U);
    Sg53DraftSampleParamsHost sp{};
    sp.M             = 1U;
    sp.vocab_size    = use_map ? eng->frspec_shortlist_size : eng->cfg.vocab_size;
    sp.use_token_map = use_map ? 1U : 0U;
    sp.want_pmax     = want_pmax ? 1U : 0U;

    const bool is_greedy = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
    if (is_greedy) {
        sp.greedy   = 1U;
        sp.inv_temp = 1.0f;
        sp.top_k    = 1U;
        sp.top_p    = 1.0f;
        sp.rng_lo   = 0U;
        sp.rng_hi   = 0U;
    } else {
        sp.greedy   = 0U;
        sp.inv_temp = 1.0f / std::max<float>(sampling->temperature, 1e-5f);
        sp.top_k    = (sampling->top_k > 0U && sampling->top_k <= 64U) ? sampling->top_k : 20U;
        sp.top_p    = sampling->top_p;
        const uint64_t seed = sampling->rng_seed ^ (eng->sample_step * 0xD1342543DE82EF95ULL);
        sp.rng_lo = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
        sp.rng_hi = static_cast<uint32_t>(seed >> 32ULL);
        eng->sample_step += 1U;
    }

    [enc setComputePipelineState:eng->pso_sample_draft_stage1];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:1];
    [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:2];
    [enc setBuffer:eng->b_sample_stage_ms offset:0 atIndex:3];
    [enc setBytes:&sp length:sizeof(sp) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(64, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc setComputePipelineState:eng->pso_sample_draft_stage2];
    [enc setBuffer:eng->b_sample_stage_v offset:0 atIndex:0];
    [enc setBuffer:eng->b_sample_stage_i offset:0 atIndex:1];
    [enc setBuffer:eng->b_sample_stage_ms offset:0 atIndex:2];
    [enc setBuffer:(use_map ? eng->b_frspec_short_to_full : out_tok_buf) offset:0 atIndex:3];
    [enc setBuffer:out_tok_buf offset:out_tok_off atIndex:4];
    [enc setBuffer:(out_pmax_buf ? out_pmax_buf : eng->b_draft_pmax) offset:(out_pmax_buf ? out_pmax_off : 0U) atIndex:5];
    [enc setBuffer:(top_ids_buf ? top_ids_buf : out_tok_buf) offset:top_ids_off atIndex:6];
    [enc setBuffer:(top_probs_buf ? top_probs_buf : eng->b_logits) offset:top_probs_off atIndex:7];
    [enc setBuffer:(keep_k_buf ? keep_k_buf : out_tok_buf) offset:keep_k_off atIndex:8];
    [enc setBytes:&sp length:sizeof(sp) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void sg53_reset_spec_ctl(StrataGlm53GpuEngine* eng) {
    if (!eng) return;
    eng->spec_state          = StrataGlm53GpuEngine::kSpecActiveProbe;
    eng->spec_low_streak     = 0U;
    eng->spec_high_streak    = 0U;
    eng->spec_cooldown_rem   = 0U;
    eng->spec_locked_k       = 0U;
    eng->ema_acc_cal         = 0.90;
    eng->ema_cal_num         = 1.8;
    eng->ema_cal_den         = 2.0;
    eng->ema_geom_p          = 0.55;
    eng->ema_geom_num        = 1.1;
    eng->ema_geom_den        = 2.0;
    eng->ema_draft_ms        = 2.2;
    const double init_ver[kMaxSpecBatch] = {20.5, 27.5, 33.5, 39.0, 44.0, 49.0, 54.0, 59.0};
    for (uint32_t j = 0U; j < kMaxSpecBatch; ++j) eng->ema_ver_ms[j] = init_ver[j];
    eng->frspec_widened      = false;
    eng->shortlist_miss_mask = 0U;
}

bool sg53_drain_pending_fix(StrataGlm53GpuEngine* eng) {
    if (!eng || !eng->pending_fix_cb) return true;
    id<MTLCommandBuffer> cb = eng->pending_fix_cb;
    eng->pending_fix_cb = nil;
    [cb waitUntilCompleted];
    return [cb status] != MTLCommandBufferStatusError;
}

bool sg53_build_frspec_shortlist(StrataGlm53GpuEngine* eng) {
    eng->frspec_shortlist_size = 0U;
    eng->b_frspec_lm_head      = nil;
    eng->b_frspec_short_to_full= nil;
    eng->b_frspec_full_to_short= nil;
    std::memset(&eng->frspec_lm_head_mat, 0, sizeof(eng->frspec_lm_head_mat));

    if (!eng || !eng->has_mtp || !eng->bundle || !eng->bundle->spine_base) return true;
    const StrataGlm53QuantMatrix& full_lm = eng->hdr.lm_head;
    const uint32_t V = full_lm.rows_logical;
    const uint32_t D = full_lm.cols;
    if (V < 64U || D == 0U || (D % 64U) != 0U) return true;

    // 20,480 rows * (D/2 + D/32 + D/32) = 45.00 MiB < 48 MiB M5 Max SLC!
    // For tiny synthetic unit-test models (V <= 20480), pick half rounded down to 64 so FR-Spec + auto-widen is exercised.
    uint32_t V_short = 0U;
    if (V > 20480U) {
        V_short = 20480U;
    } else if (V >= 128U) {
        V_short = (V / 2U) & ~63U;
    } else {
        V_short = V;
    }
    if (V_short == 0U || V_short > V) return true;

    std::vector<uint8_t> selected(V, 0U);
    std::vector<uint32_t> short_ids;
    short_ids.reserve(V_short);

    // 1. Pin tail special/added tokens (`[154820..154879]` on GLM-5.3, or last `min(64, V_short/8)` tokens)
    const uint32_t tail_pin = std::min<uint32_t>(64U, std::max<uint32_t>(1U, V_short / 8U));
    for (uint32_t r = (V > tail_pin ? V - tail_pin : 0U); r < V && short_ids.size() < V_short; ++r) {
        if (!selected[r]) {
            selected[r] = 1U;
            short_ids.push_back(r);
        }
    }

    // 2. Pin lowest-ID common ASCII/BPE frequency-ordered tokens (`[0..17919]` for 20K; leaves `1024` dynamic ring slots!)
    const uint32_t low_pin = (V_short == 20480U) ? 17920U : ((V_short * 3U) / 4U);
    for (uint32_t r = 0U; r < low_pin && r < V && short_ids.size() < V_short; ++r) {
        if (!selected[r]) {
            selected[r] = 1U;
            short_ids.push_back(r);
        }
    }

    // 3. Fill remaining slots with highest `lm_head` scale-L2 norm rows (`< 2 ms` parallel scan!)
    const uint32_t rem_need = V_short - static_cast<uint32_t>(short_ids.size());
    if (rem_need > 0U) {
        const uint32_t grp_per_row = D / 64U;
        const auto* scales_base = reinterpret_cast<const uint16_t*>(eng->bundle->spine_base + full_lm.scales_off);
        std::vector<std::pair<float, uint32_t>> cands;
        cands.reserve(V);
        for (uint32_t r = 0U; r < V; ++r) {
            if (!selected[r]) cands.emplace_back(0.0f, r);
        }
        auto* cand_ptr = cands.data();
        const size_t n_cand = cands.size();
        const size_t workers = (n_cand >= 4096U) ? 8U : 1U;
        if (workers > 1U) {
            dispatch_apply(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
                const size_t beg = (n_cand * w) / workers;
                const size_t end = (n_cand * (w + 1U)) / workers;
                for (size_t i = beg; i < end; ++i) {
                    const uint32_t r = cand_ptr[i].second;
                    const uint16_t* rs = scales_base + static_cast<size_t>(r) * grp_per_row;
                    float l2 = 0.0f;
                    for (uint32_t g = 0U; g < grp_per_row; ++g) {
                        const float s = sg53_fast_fp16_to_f32(rs[g]);
                        l2 += s * s;
                    }
                    cand_ptr[i].first = l2;
                }
            });
        } else {
            for (size_t i = 0U; i < n_cand; ++i) {
                const uint32_t r = cand_ptr[i].second;
                const uint16_t* rs = scales_base + static_cast<size_t>(r) * grp_per_row;
                float l2 = 0.0f;
                for (uint32_t g = 0U; g < grp_per_row; ++g) {
                    const float s = sg53_fast_fp16_to_f32(rs[g]);
                    l2 += s * s;
                }
                cand_ptr[i].first = l2;
            }
        }
        const size_t take = std::min<size_t>(rem_need, n_cand);
        if (take < n_cand) {
            std::nth_element(cands.begin(), cands.begin() + static_cast<ptrdiff_t>(take), cands.end(),
                             [](const auto& a, const auto& b) {
                                 return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                             });
        }
        for (size_t i = 0U; i < take; ++i) {
            selected[cands[i].second] = 1U;
            short_ids.push_back(cands[i].second);
        }
    }

    V_short = static_cast<uint32_t>(short_ids.size());
    const uint32_t dyn_cap  = (V > 20480U && V_short == 20480U) ? 1024U : 0U;
    const uint32_t dyn_base = V_short - dyn_cap;
    // Sort ONLY the static zone `[0 .. dyn_base)` so special tail tokens `[V-64..V-1]` stay permanently pinned!
    std::sort(short_ids.begin(), short_ids.begin() + static_cast<ptrdiff_t>(dyn_base));

    const uint32_t rows_pad    = (V_short + 63U) & ~63U;
    const size_t   w_row_bytes = static_cast<size_t>(D / 2U);
    const size_t   s_row_bytes = static_cast<size_t>(D / 64U) * sizeof(uint16_t);
    const uint64_t w_off       = 0ULL;
    const uint64_t s_off       = sg53_align_up_u64(static_cast<uint64_t>(rows_pad) * w_row_bytes, 256ULL);
    const uint64_t b_off       = sg53_align_up_u64(s_off + static_cast<uint64_t>(rows_pad) * s_row_bytes, 256ULL);
    const uint64_t tot_bytes   = sg53_align_up_u64(b_off + static_cast<uint64_t>(rows_pad) * s_row_bytes, 4096ULL);

    eng->b_frspec_lm_head       = sg53_alloc_private_aligned(eng->device, static_cast<size_t>(tot_bytes));
    eng->b_frspec_short_to_full = sg53_alloc_shared_prefaulted(eng->device, static_cast<size_t>(V_short) * sizeof(uint32_t));
    eng->b_frspec_full_to_short = sg53_alloc_shared_prefaulted(eng->device, static_cast<size_t>(V) * sizeof(uint32_t));
    eng->b_frspec_patch_pairs   = (dyn_cap > 0U)
        ? sg53_alloc_shared_prefaulted(eng->device, static_cast<size_t>(dyn_cap) * 2U * sizeof(uint32_t))
        : nil;
    if (!eng->b_frspec_lm_head || !eng->b_frspec_short_to_full || !eng->b_frspec_full_to_short ||
        (dyn_cap > 0U && !eng->b_frspec_patch_pairs)) {
        eng->b_frspec_lm_head       = nil;
        eng->b_frspec_short_to_full = nil;
        eng->b_frspec_full_to_short = nil;
        eng->b_frspec_patch_pairs   = nil;
        return false;
    }

    auto* s2f = reinterpret_cast<uint32_t*>([eng->b_frspec_short_to_full contents]);
    auto* f2s = reinterpret_cast<uint32_t*>([eng->b_frspec_full_to_short contents]);
    std::memset(f2s, 0xFF, static_cast<size_t>(V) * sizeof(uint32_t));
    for (uint32_t s = 0U; s < V_short; ++s) {
        s2f[s]            = short_ids[s];
        f2s[short_ids[s]] = s;
    }

    bool upload_ok = false;
    @autoreleasepool {
        id<MTLBuffer> stage = [eng->device newBufferWithLength:static_cast<NSUInteger>(tot_bytes)
                                                       options:MTLResourceStorageModeShared];
        if (stage) {
            uint8_t* dst = static_cast<uint8_t*>([stage contents]);
            if (rows_pad > V_short) {
                std::memset(dst + static_cast<size_t>(V_short) * w_row_bytes, 0,
                            static_cast<size_t>(rows_pad - V_short) * w_row_bytes);
                std::memset(dst + s_off + static_cast<size_t>(V_short) * s_row_bytes, 0,
                            static_cast<size_t>(rows_pad - V_short) * s_row_bytes);
                std::memset(dst + b_off + static_cast<size_t>(V_short) * s_row_bytes, 0,
                            static_cast<size_t>(rows_pad - V_short) * s_row_bytes);
            }
            const uint8_t* src_w = eng->bundle->spine_base + full_lm.weight_off;
            const uint8_t* src_s = eng->bundle->spine_base + full_lm.scales_off;
            const uint8_t* src_b = eng->bundle->spine_base + full_lm.biases_off;
            const uint32_t* ids_ptr = short_ids.data();
            const size_t copy_workers = (V_short >= 2048U) ? 8U : 1U;
            if (copy_workers > 1U) {
                dispatch_apply(copy_workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
                    const uint32_t beg = static_cast<uint32_t>((static_cast<size_t>(V_short) * w) / copy_workers);
                    const uint32_t end = static_cast<uint32_t>((static_cast<size_t>(V_short) * (w + 1U)) / copy_workers);
                    for (uint32_t s = beg; s < end; ++s) {
                        const uint32_t r = ids_ptr[s];
                        std::memcpy(dst + w_off + static_cast<size_t>(s) * w_row_bytes,
                                    src_w + static_cast<size_t>(r) * w_row_bytes, w_row_bytes);
                        std::memcpy(dst + s_off + static_cast<size_t>(s) * s_row_bytes,
                                    src_s + static_cast<size_t>(r) * s_row_bytes, s_row_bytes);
                        std::memcpy(dst + b_off + static_cast<size_t>(s) * s_row_bytes,
                                    src_b + static_cast<size_t>(r) * s_row_bytes, s_row_bytes);
                    }
                });
            } else {
                for (uint32_t s = 0U; s < V_short; ++s) {
                    const uint32_t r = ids_ptr[s];
                    std::memcpy(dst + w_off + static_cast<size_t>(s) * w_row_bytes,
                                src_w + static_cast<size_t>(r) * w_row_bytes, w_row_bytes);
                    std::memcpy(dst + s_off + static_cast<size_t>(s) * s_row_bytes,
                                src_s + static_cast<size_t>(r) * s_row_bytes, s_row_bytes);
                    std::memcpy(dst + b_off + static_cast<size_t>(s) * s_row_bytes,
                                src_b + static_cast<size_t>(r) * s_row_bytes, s_row_bytes);
                }
            }
            id<MTLCommandBuffer>      cb_bl = [eng->queue commandBuffer];
            id<MTLBlitCommandEncoder> bl    = [cb_bl blitCommandEncoder];
            [bl copyFromBuffer:stage sourceOffset:0 toBuffer:eng->b_frspec_lm_head destinationOffset:0 size:static_cast<NSUInteger>(tot_bytes)];
            [bl endEncoding];
            [cb_bl commit];
            [cb_bl waitUntilCompleted];
            upload_ok = ([cb_bl status] != MTLCommandBufferStatusError);
        }
    }

    if (!upload_ok) {
        eng->b_frspec_lm_head       = nil;
        eng->b_frspec_short_to_full = nil;
        eng->b_frspec_full_to_short = nil;
        eng->b_frspec_patch_pairs   = nil;
        return false;
    }

    eng->frspec_lm_head_mat.rows_logical = V_short;
    eng->frspec_lm_head_mat.rows_padded  = rows_pad;
    eng->frspec_lm_head_mat.cols         = D;
    eng->frspec_lm_head_mat.group_size   = full_lm.group_size;
    eng->frspec_lm_head_mat.weight_off   = w_off;
    eng->frspec_lm_head_mat.scales_off   = s_off;
    eng->frspec_lm_head_mat.biases_off   = b_off;
    eng->frspec_shortlist_size           = V_short;
    eng->frspec_dyn_base                 = dyn_base;
    eng->frspec_dyn_cap                  = dyn_cap;
    eng->frspec_dyn_cursor               = 0U;
    return true;
}

// Self-Healing Dynamic GPU Patcher for 20K SLC-Pinned FR-Spec (`< 1 us/row`, 0 extra CBs!):
// Assigns missing tokens from `toks[0..n_toks-1]` into ring `[frspec_dyn_base .. V_short - 1]`, updates
// `short_to_full`/`full_to_short` atomically, and encodes `sg53_frspec_patch_rows` directly into `enc`.
void sg53_frspec_encode_patch_tokens(
    id<MTLComputeCommandEncoder> enc,
    StrataGlm53GpuEngine*        eng,
    const uint32_t*              toks,
    uint32_t                     n_toks
) {
    if (!enc || !eng || !toks || n_toks == 0U ||
        !eng->b_frspec_lm_head || !eng->b_frspec_short_to_full ||
        !eng->b_frspec_full_to_short || !eng->b_frspec_patch_pairs ||
        !eng->pso_frspec_patch_rows || eng->frspec_dyn_cap == 0U) {
        return;
    }
    const uint32_t V   = eng->hdr.lm_head.rows_logical;
    auto* s2f          = reinterpret_cast<uint32_t*>([eng->b_frspec_short_to_full contents]);
    auto* f2s          = reinterpret_cast<uint32_t*>([eng->b_frspec_full_to_short contents]);
    auto* pairs        = reinterpret_cast<uint32_t*>([eng->b_frspec_patch_pairs contents]);
    uint32_t n_patch   = 0U;
    for (uint32_t i = 0U; i < n_toks && n_patch < eng->frspec_dyn_cap; ++i) {
        const uint32_t tok = toks[i];
        if (tok >= V || f2s[tok] != 0xFFFFFFFFU) continue;
        const uint32_t slot = eng->frspec_dyn_base + (eng->frspec_dyn_cursor % eng->frspec_dyn_cap);
        eng->frspec_dyn_cursor += 1U;
        const uint32_t old_tok = s2f[slot];
        if (old_tok < V) f2s[old_tok] = 0xFFFFFFFFU;
        s2f[slot] = tok;
        f2s[tok]  = slot;
        pairs[2U * n_patch + 0U] = slot;
        pairs[2U * n_patch + 1U] = tok;
        ++n_patch;
    }
    if (n_patch == 0U) return;

    const StrataGlm53QuantMatrix& src_lm = eng->hdr.lm_head;
    const StrataGlm53QuantMatrix& dst_lm = eng->frspec_lm_head_mat;
    Sg53FrspecPatchParamsHost pp{n_patch, src_lm.cols, 0U, 0U};
    [enc setComputePipelineState:eng->pso_frspec_patch_rows];
    [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(src_lm.weight_off) atIndex:0];
    [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(src_lm.scales_off) atIndex:1];
    [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(src_lm.biases_off) atIndex:2];
    [enc setBuffer:eng->b_frspec_lm_head offset:static_cast<NSUInteger>(dst_lm.weight_off) atIndex:3];
    [enc setBuffer:eng->b_frspec_lm_head offset:static_cast<NSUInteger>(dst_lm.scales_off) atIndex:4];
    [enc setBuffer:eng->b_frspec_lm_head offset:static_cast<NSUInteger>(dst_lm.biases_off) atIndex:5];
    [enc setBuffer:eng->b_frspec_patch_pairs offset:0 atIndex:6];
    [enc setBytes:&pp length:sizeof(pp) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(n_patch, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

int sg53_commit_spec_snaps_post_verify(
    StrataGlm53GpuEngine*                 eng,
    const std::vector<Sg53SpecLayerSnap>& snaps,
    uint32_t                              accepted
) {
    const uint64_t rec_b = eng->hdr.expert_record_bytes;
    const uint32_t E     = eng->cfg.n_routed_experts;
    uint64_t tot_promos  = 0ULL;

    struct MemCopyTask { uint8_t* dst; const uint8_t* src; };
    std::vector<MemCopyTask>            mem_copies;
    std::vector<Sg53NvmeExpertCopyTask> fixup_io;

    for (const auto& sp : snaps) {
        const uint32_t moe_l = sp.moe_l;
        if (moe_l >= eng->layer_hot.size()) continue;

        const Sg53LayerHotState& target = sp.snap[accepted];
        tot_promos += sp.promotions_cum[accepted];

        for (uint32_t ls = 0U; ls < target.cap_slots; ++ls) {
            const uint32_t phys_hs = target.base_slot + ls;
            if (ls >= target.used_slots || target.e_of_slot[ls] < 0) {
                eng->phys_hot_gid[phys_hs] = -1;
                continue;
            }
            const uint32_t want_e   = static_cast<uint32_t>(target.e_of_slot[ls]);
            const uint32_t want_gid = moe_l * E + want_e;
            if (eng->phys_hot_gid[phys_hs] == static_cast<int32_t>(want_gid)) {
                continue;
            }

            uint8_t* dst = eng->hot_experts_ptr + static_cast<uint64_t>(phys_hs) * rec_b;
            bool copied  = false;
            if (want_gid < eng->fg_slot_of_gid.size() && eng->fg_slot_of_gid[want_gid] >= 0) {
                const uint32_t fs = static_cast<uint32_t>(eng->fg_slot_of_gid[want_gid]);
                mem_copies.push_back(MemCopyTask{dst, eng->ring_dma_ptr + static_cast<uint64_t>(fs) * rec_b});
                copied = true;
            } else {
                int32_t bg_s = -1;
                {
                    Sg53ScopedLock blk(&eng->bg_ring_lock);
                    for (uint32_t s = 0U; s < eng->bg_ring_slots; ++s) {
                        if (eng->gid_of_bg_slot[s] == static_cast<int32_t>(want_gid) && eng->bg_slot_ready[s] != 0U) {
                            eng->bg_slot_pin_epoch[s] = eng->cb_epoch + 8ULL;
                            bg_s = static_cast<int32_t>(s);
                            break;
                        }
                    }
                }
                if (bg_s >= 0) {
                    const uint32_t phys_rs = eng->fg_ring_slots + static_cast<uint32_t>(bg_s);
                    mem_copies.push_back(MemCopyTask{dst, eng->ring_dma_ptr + static_cast<uint64_t>(phys_rs) * rec_b});
                    copied = true;
                }
            }

            if (!copied) {
                fixup_io.push_back(Sg53NvmeExpertCopyTask{want_gid, dst});
            }
            eng->phys_hot_gid[phys_hs] = static_cast<int32_t>(want_gid);
        }

        eng->layer_hot[moe_l] = target;
    }

    if (mem_copies.size() == 1U) {
        sg53_fast_memcpy_expert(mem_copies[0].dst, mem_copies[0].src, rec_b);
    } else if (mem_copies.size() > 1U) {
        const MemCopyTask* tasks_ptr = mem_copies.data();
        dispatch_apply(mem_copies.size(), dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
            std::memcpy(tasks_ptr[i].dst, tasks_ptr[i].src, static_cast<size_t>(rec_b));
        });
    }

    if (!fixup_io.empty()) {
        if (eng->profile_nvme) {
            eng->prof_fixup_reads.fetch_add(fixup_io.size(), std::memory_order_relaxed);
        }
        if (sg53_stream_experts_nvme_batch(eng, fixup_io, /*count_telemetry=*/true, /*is_foreground=*/true) != 0) {
            return -3;
        }
    }
    if (tot_promos > 0ULL) {
        Sg53ScopedLock tlk(&eng->telemetry_lock);
        eng->counters.lfu_promotions += tot_promos;
    }
    return 0;
}

void sg53_drain_ane_queue(const StrataGlm53GpuEngine* eng) {
    if (eng && eng->ane_bg_queue) {
        dispatch_sync(eng->ane_bg_queue, ^{});
    }
}

} // namespace

extern "C" {

StrataGlm53GpuEngine* strata_glm53_gpu_create(
    const StrataGlm53Bundle* bundle,
    uint32_t                 max_seq_len,
    float                    hot_cache_gib,
    StrataGlm53ExecMode      exec_mode
) {
    if (!bundle || !bundle->spine_base || !bundle->header || bundle->spine_bytes == 0U) {
        return nullptr;
    }

    /* Engine geometry validation (Round 6 rules G2–G9, Round 7 rule G10) before shader compile / buffer allocation.
     *   G10: the paged-KV probe snapshots copy one `[index_n_heads]` FP32 row and one `[q_lora_rank]` FP16 row per DSA
     *        layer with `sg53_encode_copy_buffer`, which moves whole 16-byte units; rows that are not a multiple of
     *        16 bytes would be over-copied by 4..12 bytes past the row (past the buffer for the last DSA slot). */
    {
        const StrataGlm53Config& gc = bundle->cfg;
        const uint64_t max_ffn = std::max<uint64_t>(gc.moe_intermediate_size, gc.intermediate_size);
        const uint64_t sh_ffn  = static_cast<uint64_t>(gc.n_shared_experts) * gc.moe_intermediate_size;
        if (gc.hc_mult != 4U ||
            (gc.index_n_heads % 4U) != 0U || (gc.q_lora_rank % 8U) != 0U ||
            (gc.n_shared_experts > 0U && sh_ffn > max_ffn) ||
            gc.kv_lora_rank == 0U || gc.kv_lora_rank > 512U || (gc.kv_lora_rank % 32U) != 0U ||
            gc.linear_head_dim == 0U || gc.linear_head_dim > 128U ||
            gc.index_head_dim == 0U || gc.index_head_dim > 128U ||
            gc.linear_conv_kernel_dim != 4U ||
            gc.index_kpool != STRATA_GLM53_KV_BLOCK_TOKENS ||
            gc.index_topk < STRATA_GLM53_KV_BLOCK_TOKENS ||
            (gc.index_topk % STRATA_GLM53_KV_BLOCK_TOKENS) != 0U ||
            gc.index_topk > 2048U) {
            std::fprintf(stderr, "[strata_glm53_gpu] Unsupported GLM-5.3 geometry in bundle header\n");
            return nullptr;
        }
    }

    @autoreleasepool {
        const auto t_start = std::chrono::steady_clock::now();

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            std::fprintf(stderr, "[strata_glm53_gpu] No Metal GPU device available\n");
            return nullptr;
        }

        std::string hdr_src = strata_load_shader_asset("strata_glm53.h", "");
        if (hdr_src.empty()) hdr_src = sg53_read_text_file(sg53_resolve_sibling_path("../../include/strata_glm53.h"));
        std::string msl_src = strata_load_shader_asset("strata_glm53_kernels.metal", "");
        if (msl_src.empty()) msl_src = sg53_read_text_file(sg53_resolve_sibling_path("strata_glm53_kernels.metal"));
        if (hdr_src.empty() || msl_src.empty()) {
            std::fprintf(stderr, "[strata_glm53_gpu] Failed to read strata_glm53.h or strata_glm53_kernels.metal\n");
            return nullptr;
        }

        const std::string marker = "// @@STRATA_GLM53_HEADER@@";
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

        auto* eng = new StrataGlm53GpuEngine();
        eng->device              = dev;
        eng->queue               = [dev newCommandQueue];
        eng->has_nax             = compiled_v4;
        eng->bundle              = bundle;
        eng->hdr                 = *bundle->header;
        eng->cfg                 = bundle->cfg;
        eng->has_mtp             = (eng->cfg.num_nextn_predict_layers > 0U) && (eng->hdr.mtp.enabled != 0U);
        eng->is_ram_resident     = (eng->hdr.version == STRATA_GLM53_FORMAT_VERSION_V2) &&
                                   (eng->hdr.is_ram_resident != 0U) &&
                                   (bundle->is_ram_resident != 0U);
        eng->force_serial_kda    = (std::getenv("STRATA_SG53_SERIAL_KDA") != nullptr &&
                                    std::atoi(std::getenv("STRATA_SG53_SERIAL_KDA")) != 0);
        eng->ngram_draft_enabled = !(std::getenv("STRATA_GLM53_DISABLE_NGRAM") != nullptr &&
                                     std::atoi(std::getenv("STRATA_GLM53_DISABLE_NGRAM")) != 0);
        eng->int8_moe_enabled    = !(std::getenv("STRATA_SG53_INT8_MOE") != nullptr &&
                                     std::atoi(std::getenv("STRATA_SG53_INT8_MOE")) == 0);
        sg53_reset_spec_ctl(eng);
        eng->exec_mode           = exec_mode;
        eng->adaptive_gate_tau   = STRATA_GLM53_DEFAULT_ADAPTIVE_TAU;
        eng->adaptive_cold_cap   = STRATA_GLM53_DEFAULT_ADAPTIVE_COLD_CAP;
        eng->adaptive_renorm     = true;
        eng->uma_part0_mapped_bytes   = 0ULL;
        eng->uma_part1_mapped_bytes   = 0ULL;
        eng->mtp_experts_mapped_bytes = 0ULL;
        eng->pen_num_ids         = 0U;
        eng->pen_window_start    = 0U;
        eng->pen_window_end      = 0U;

        const uint32_t req_seq = (max_seq_len > 0U) ? max_seq_len : STRATA_GLM53_DEFAULT_MAX_SEQ_LEN;
        if (req_seq > STRATA_GLM53_MAX_SEQ_LEN_CAP) {
            std::fprintf(stderr, "[strata_glm53_gpu] max_seq_len=%u exceeds cap=%u\n",
                         req_seq, STRATA_GLM53_MAX_SEQ_LEN_CAP);
            delete eng;
            return nullptr;
        }

        const uint32_t kpool   = std::max<uint32_t>(1U, eng->cfg.index_kpool);
        eng->max_seq_len       = req_seq;
        eng->max_seq_alloc     = (std::max<uint32_t>(req_seq, 64U) + 63U) & ~63U;
        eng->max_pools_alloc   = std::max<uint32_t>(64U, ((eng->max_seq_alloc / kpool) + 63U) & ~63U);
        eng->max_sel_words     = std::max<uint32_t>(32U, (eng->max_pools_alloc + 31U) / 32U);
        eng->seq_pos           = 0U;
        eng->mtp_seq_pos       = 0U;
        eng->sample_step       = 0ULL;
        eng->progress_cb       = nullptr;
        eng->progress_user_ctx = nullptr;
        eng->cb_epoch          = 1ULL;
        eng->fg_ring_cursor    = 0U;
        eng->bg_ring_cursor    = 0U;
        eng->ane_prefetch_budget = 1U;
        if (const char* env_ap = std::getenv("STRATA_GLM53_ANE_PREFETCH")) {
            const long v = std::strtol(env_ap, nullptr, 10);
            if (v >= 0L && v <= 8L) eng->ane_prefetch_budget = static_cast<uint32_t>(v);
        }
        eng->profile_nvme = (std::getenv("STRATA_GLM53_PROFILE") != nullptr);
        std::memset(&eng->counters, 0, sizeof(eng->counters));
        std::snprintf(eng->device_name, sizeof(eng->device_name), "%s", [[dev name] UTF8String]);

        dispatch_queue_attr_t q_attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
        eng->ane_bg_queue = dispatch_queue_create("com.strata.glm53.ane_lookahead", q_attr);

        // Overlap Metal 4 library + 43-PSO compilation AND 16-Core ANE CoreML compilation concurrently with UMA wiring!
        __block bool shader_ok = true;
        dispatch_group_t init_grp = dispatch_group_create();
        dispatch_queue_t bg_q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        dispatch_group_async(init_grp, bg_q, ^{
            @autoreleasepool {
                NSError* err = nil;
                id<MTLLibrary> lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
                if (!lib && compiled_v4) {
                    if (@available(macOS 26.0, *)) {
                        std::fprintf(stderr, "[strata_glm53_gpu] Metal 4.1 compile failed (%s), falling back to Metal 4.0...\n",
                                     err ? [[err localizedDescription] UTF8String] : "unknown");
                        err = nil;
                        opts.languageVersion = MTLLanguageVersion4_0;
                        lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
                    }
                }
                if (!lib) {
                    std::fprintf(stderr, "[strata_glm53_gpu] Metal 4 compile error: %s\n",
                                 err ? [[err localizedDescription] UTF8String] : "unknown");
                    shader_ok = false;
                    return;
                }
                if (err) {
                    std::fprintf(stderr, "[strata_glm53_gpu] Metal 4 compiler warning:\n%s\n",
                                 [[err localizedDescription] UTF8String]);
                }
                eng->library = lib;

                StrataMetalPsoCache* pso_cache = strata_metal_cache_open_with_opts(
                    dev, lib, "glm53", msl_src.data(), msl_src.size(), opts);
                if (!pso_cache) {
                    std::fprintf(stderr, "[strata_glm53_gpu] Failed to open StrataMetalPsoCache\n");
                    shader_ok = false;
                    return;
                }

                struct PsoSlot {
                    const char*                           fn_name;
                    id<MTLComputePipelineState> __strong* dst;
                };
                PsoSlot pso_table[] = {
                    {"sg53_zero_u32",                      &eng->pso_zero_u32},
                    {"sg53_copy_u32",                      &eng->pso_copy_u32},
                    {"sg53_embed_lookup_q4",               &eng->pso_embed_q4},
                    {"sg53_q4_gemv_batch",                 &eng->pso_q4_gemv_batch},
                    {"sg53_f16_gemv_batch",                &eng->pso_f16_gemv_batch},
                    {"sg53_f32_gemv_batch",                &eng->pso_f32_gemv_batch},
                    {"sg53_q4_gemm_nax_m16",               &eng->pso_q4_gemm_nax_m16},
                    {"sg53_q4_gemm_nax_m64",               &eng->pso_q4_gemm_nax_m64},
                    {"sg53_q4_gemm_nax_bk64_m16",          &eng->pso_q4_gemm_nax_bk64_m16},
                    {"sg53_f16_gemm_nax_m16",              &eng->pso_f16_gemm_nax_m16},
                    {"sg53_f16_gemm_nax_m64",              &eng->pso_f16_gemm_nax_m64},
                    {"sg53_quant_f16_to_i8_row",           &eng->pso_quant_f16_to_i8_row},
                    {"sg53_q4_gemm_nax_i8_m64",            &eng->pso_q4_gemm_nax_i8_m64},
                    {"sg53_q4_gemm_nax_i8_m128",           &eng->pso_q4_gemm_nax_i8_m128},
                    {"sg53_rmsnorm_strided",               &eng->pso_rmsnorm_strided},
                    {"sg53_mhc_unweighted_norm",           &eng->pso_mhc_norm},
                    {"sg53_mhc_sinkhorn",                  &eng->pso_mhc_sinkhorn},
                    {"sg53_mhc_collapse_rmsnorm",          &eng->pso_mhc_collapse_norm},
                    {"sg53_mhc_sinkhorn_collapse_rmsnorm", &eng->pso_mhc_sinkhorn_collapse},
                    {"sg53_mhc_expand",                    &eng->pso_mhc_expand},
                    {"sg53_single_residual_add",           &eng->pso_single_res_add},
                    {"sg53_hyper_head_final_norm",         &eng->pso_hyper_head_norm},
                    {"sg53_kda_fbg_split",                 &eng->pso_kda_fbg_split},
                    {"sg53_kda_conv1d_silu",               &eng->pso_kda_conv1d},
                    {"sg53_kda_recurrent_gated_norm",      &eng->pso_kda_rec},
                    {"sg53_kda_prep_batch",                &eng->pso_kda_prep_batch},
                    {"sg53_kda_rec_batch",                 &eng->pso_kda_rec_batch},
                    {"sg53_kda_post_batch",                &eng->pso_kda_post_batch},
                    {"sg53_kda_wy_prep_chunk32",           &eng->pso_kda_wy_prep_chunk32},
                    {"sg53_kda_wy_recur_chunk32",          &eng->pso_kda_wy_recur_chunk32},
                    {"sg53_indexer_k_layernorm",           &eng->pso_idx_k_layernorm},
                    {"sg53_indexer_update_pools",          &eng->pso_idx_update_pools},
                    {"sg53_indexer_commit_tail",           &eng->pso_idx_commit_tail},
                    {"sg53_indexer_scale_weights",         &eng->pso_idx_scale_weights},
                    {"sg53_indexer_score_pools",           &eng->pso_idx_score_pools},
                    {"sg53_indexer_score_pools_tiled",     &eng->pso_idx_score_tiled},
                    {"sg53_indexer_q_to_hm_f16",           &eng->pso_idx_q_to_hm},
                    {"sg53_indexer_score_pools_nax",       &eng->pso_idx_score_nax},
                    {"sg53_indexer_score_pools_decode_nax",&eng->pso_idx_score_decode_nax},
                    {"sg53_indexer_topk_pools",            &eng->pso_idx_topk_pools},
                    {"sg53_mla_batched_f16_gemv",          &eng->pso_mla_batched_gemv},
                    {"sg53_mla_batched_f16_gemm_nax_m16",  &eng->pso_mla_batched_nax_m16},
                    {"sg53_mla_batched_f16_gemm_nax_m64",  &eng->pso_mla_batched_nax_m64},
                    {"sg53_mla_q_to_hm_f16",               &eng->pso_mla_q_to_hm},
                    {"sg53_mla_batched_f16_nax_hm_m16",    &eng->pso_mla_batched_nax_hm_m16},
                    {"sg53_mla_batched_f16_nax_hm_m64",    &eng->pso_mla_batched_nax_hm_m64},
                    {"sg53_mla_kv_rmsnorm_write",          &eng->pso_mla_kv_write},
                    {"sg53_mla_latent_splitkv",            &eng->pso_mla_splitkv},
                    {"sg53_mla_decode_splitkv_nax",        &eng->pso_mla_decode_splitkv_nax},
                    {"sg53_mla_latent_reduce",             &eng->pso_mla_reduce},
                    {"sg53_mla_prefill_dense_nax",         &eng->pso_mla_prefill_dense_nax},
                    {"sg53_mla_prefill_sparse_nax",        &eng->pso_mla_prefill_sparse_nax},
                    {"sg53_limited_swiglu_split",          &eng->pso_limited_swiglu},
                    {"sg53_rht64_f16",                     &eng->pso_rht64_f16},
                    {"sg53_limited_swiglu_rht64",          &eng->pso_limited_swiglu_rht64},
                    {"sg53_moe_router_topk",               &eng->pso_moe_router_topk},
                    {"sg53_moe_router_logits",             &eng->pso_moe_router_logits},
                    {"sg53_moe_router_select",             &eng->pso_moe_router_select},
                    {"sg53_moe_indirect_gather_gu",        &eng->pso_moe_gather_gu},
                    {"sg53_moe_indirect_gather_dn",        &eng->pso_moe_gather_dn},
                    {"sg53_moe_indirect_grouped_gu",       &eng->pso_moe_grouped_gu},
                    {"sg53_moe_indirect_grouped_dn",       &eng->pso_moe_grouped_dn},
                    {"sg53_moe_combine",                   &eng->pso_moe_combine},
                    {"sg53_moe_bin_tokens_v2",             &eng->pso_moe_bin_v2},
                    {"sg53_moe_bin_tokens_nax",            &eng->pso_moe_bin_nax},
                    {"sg53_moe_gather_x_f16",              &eng->pso_moe_gather_x_f16},
                    {"sg53_moe_combine_v2",                &eng->pso_moe_combine_v2},
                    {"sg53_moe_combine_mhc_expand",        &eng->pso_moe_combine_mhc_expand},
                    {"sg53_moe_e8s18_gather_gu",           &eng->pso_moe_e8s18_gather_gu},
                    {"sg53_moe_e8s18_gather_gu_256",       &eng->pso_moe_e8s18_gather_gu_256},
                    {"sg53_moe_e8s18_gather_dn",           &eng->pso_moe_e8s18_gather_dn},
                    {"sg53_moe_e8s18_gather_dn_256",       &eng->pso_moe_e8s18_gather_dn_256},
                    {"sg53_moe_e8s18_grouped_gu",          &eng->pso_moe_e8s18_grouped_gu},
                    {"sg53_moe_e8s18_grouped_dn",          &eng->pso_moe_e8s18_grouped_dn},
                    {"sg53_moe_e8s18_nax_gu_m64",          &eng->pso_moe_e8s18_nax_gu_m64},
                    {"sg53_moe_e8s18_nax_gu_m32",          &eng->pso_moe_e8s18_nax_gu_m32},
                    {"sg53_moe_e8s18_nax_gu_m16",          &eng->pso_moe_e8s18_nax_gu_m16},
                    {"sg53_moe_e8s18_nax_dn_m64",          &eng->pso_moe_e8s18_nax_dn_m64},
                    {"sg53_moe_e8s18_nax_dn_m32",          &eng->pso_moe_e8s18_nax_dn_m32},
                    {"sg53_moe_e8s18_nax_dn_m16",          &eng->pso_moe_e8s18_nax_dn_m16},
                    {"sg53_moe_gather_x_i8",               &eng->pso_moe_gather_x_i8},
                    {"sg53_moe_quant_mid_i8",              &eng->pso_moe_quant_mid_i8},
                    {"sg53_moe_e8s18_nax_i8_gu_m64",       &eng->pso_moe_e8s18_nax_i8_gu_m64},
                    {"sg53_moe_e8s18_nax_i8_gu_m32",       &eng->pso_moe_e8s18_nax_i8_gu_m32},
                    {"sg53_moe_e8s18_nax_i8_gu_m16",       &eng->pso_moe_e8s18_nax_i8_gu_m16},
                    {"sg53_moe_e8s18_nax_i8_dn_m64",       &eng->pso_moe_e8s18_nax_i8_dn_m64},
                    {"sg53_moe_e8s18_nax_i8_dn_m32",       &eng->pso_moe_e8s18_nax_i8_dn_m32},
                    {"sg53_moe_e8s18_nax_i8_dn_m16",       &eng->pso_moe_e8s18_nax_i8_dn_m16},
                    {"sg53_moe_q4_uma_gather_gu",          &eng->pso_moe_q4_uma_gather_gu},
                    {"sg53_moe_q4_uma_gather_dn",          &eng->pso_moe_q4_uma_gather_dn},
                    {"sg53_moe_q4_uma_grouped_gu",         &eng->pso_moe_q4_uma_grouped_gu},
                    {"sg53_moe_q4_uma_grouped_dn",         &eng->pso_moe_q4_uma_grouped_dn},
                    {"sg53_sample_vocab_batch",            &eng->pso_sample_vocab},
                    {"sg53_sample_vocab_stage1",           &eng->pso_sample_stage1},
                    {"sg53_sample_vocab_stage2",           &eng->pso_sample_stage2},
                    {"sg53_apply_logit_penalties",         &eng->pso_apply_logit_penalties},
                    {"sg53_sample_draft_stage1",           &eng->pso_sample_draft_stage1},
                    {"sg53_sample_draft_stage2",           &eng->pso_sample_draft_stage2},
                    {"sg53_apply_logit_penalties_shortlist", &eng->pso_apply_logit_penalties_shortlist},
                    {"sg53_frspec_patch_rows",             &eng->pso_frspec_patch_rows},
                };
                struct SpecPsoSlot {
                    const char*                           fn_name;
                    uint32_t                              const_val;
                    id<MTLComputePipelineState> __strong* dst;
                };
                SpecPsoSlot spec_table[] = {
                    {"sg53_q4_gemv_spec",              1U, &eng->pso_q4_gemv_spec[0]},
                    {"sg53_q4_gemv_spec",              2U, &eng->pso_q4_gemv_spec[1]},
                    {"sg53_q4_gemv_spec",              3U, &eng->pso_q4_gemv_spec[2]},
                    {"sg53_q4_gemv_spec",              4U, &eng->pso_q4_gemv_spec[3]},
                    {"sg53_q4_gemv_spec",              5U, &eng->pso_q4_gemv_spec[4]},
                    {"sg53_q4_gemv_k128_spec",         1U, &eng->pso_q4_gemv_k128_spec[0]},
                    {"sg53_q4_gemv_k128_spec",         2U, &eng->pso_q4_gemv_k128_spec[1]},
                    {"sg53_q4_gemv_k128_spec",         3U, &eng->pso_q4_gemv_k128_spec[2]},
                    {"sg53_q4_gemv_k128_spec",         4U, &eng->pso_q4_gemv_k128_spec[3]},
                    {"sg53_q4_gemv_k128_spec",         5U, &eng->pso_q4_gemv_k128_spec[4]},
                    {"sg53_f16_gemv_spec",             1U, &eng->pso_f16_gemv_spec[0]},
                    {"sg53_f16_gemv_spec",             2U, &eng->pso_f16_gemv_spec[1]},
                    {"sg53_f16_gemv_spec",             3U, &eng->pso_f16_gemv_spec[2]},
                    {"sg53_f16_gemv_spec",             4U, &eng->pso_f16_gemv_spec[3]},
                    {"sg53_f16_gemv_spec",             5U, &eng->pso_f16_gemv_spec[4]},
                    {"sg53_f32_gemv_splitk4_spec",     1U, &eng->pso_f32_gemv_splitk4_spec[0]},
                    {"sg53_f32_gemv_splitk4_spec",     2U, &eng->pso_f32_gemv_splitk4_spec[1]},
                    {"sg53_f32_gemv_splitk4_spec",     3U, &eng->pso_f32_gemv_splitk4_spec[2]},
                    {"sg53_f32_gemv_splitk4_spec",     4U, &eng->pso_f32_gemv_splitk4_spec[3]},
                    {"sg53_f32_gemv_splitk4_spec",     5U, &eng->pso_f32_gemv_splitk4_spec[4]},
                    {"sg53_mla_batched_f16_gemv_spec", 1U, &eng->pso_mla_batched_gemv_spec[0]},
                    {"sg53_mla_batched_f16_gemv_spec", 2U, &eng->pso_mla_batched_gemv_spec[1]},
                    {"sg53_mla_batched_f16_gemv_spec", 3U, &eng->pso_mla_batched_gemv_spec[2]},
                    {"sg53_mla_batched_f16_gemv_spec", 4U, &eng->pso_mla_batched_gemv_spec[3]},
                    {"sg53_mla_batched_f16_gemv_spec", 5U, &eng->pso_mla_batched_gemv_spec[4]},
                };
                constexpr size_t kNumPsos     = sizeof(pso_table) / sizeof(pso_table[0]);
                constexpr size_t kNumSpecPsos = sizeof(spec_table) / sizeof(spec_table[0]);
                PsoSlot*     pso_ptr  = pso_table;
                SpecPsoSlot* spec_ptr = spec_table;

                dispatch_apply(kNumPsos + kNumSpecPsos, bg_q, ^(size_t idx) {
                    @autoreleasepool {
                        NSError* p_err = nil;
                        if (idx < kNumPsos) {
                            id<MTLComputePipelineState> pso = strata_metal_cache_make_pso(
                                pso_cache, pso_ptr[idx].fn_name, nil, &p_err);
                            if (!pso) {
                                std::fprintf(stderr, "[strata_glm53_gpu] PSO compile error (%s): %s\n",
                                             pso_ptr[idx].fn_name,
                                             p_err ? [[p_err localizedDescription] UTF8String] : "unknown");
                            }
                            *pso_ptr[idx].dst = pso;
                        } else {
                            const size_t si = idx - kNumPsos;
                            id<MTLComputePipelineState> pso = strata_metal_cache_make_pso_const_u32(
                                pso_cache, spec_ptr[si].fn_name, 0U, spec_ptr[si].const_val, &p_err);
                            if (!pso) {
                                std::fprintf(stderr, "[strata_glm53_gpu] Spec PSO compile error (%s[M=%u]): %s\n",
                                             spec_ptr[si].fn_name, spec_ptr[si].const_val,
                                             p_err ? [[p_err localizedDescription] UTF8String] : "unknown");
                            }
                            *spec_ptr[si].dst = pso;
                        }
                    }
                });

                for (size_t i = 0U; i < kNumPsos; ++i) {
                    if (!(*pso_table[i].dst)) {
                        shader_ok = false;
                        break;
                    }
                }
                for (size_t i = 0U; i < kNumSpecPsos && shader_ok; ++i) {
                    if (!(*spec_table[i].dst)) {
                        shader_ok = false;
                        break;
                    }
                }
                strata_metal_cache_finish(pso_cache);
                eng->metal_cache_hit = pso_cache->cache_hit ? 1U : 0U;
                eng->pso_load_ms     = pso_cache->elapsed_ms;
                strata_metal_cache_destroy(pso_cache);
            }
        });

        // v1 Regime II only: compile 16-Core ANE router lookahead probes (v2 RAM-resident needs ZERO ANE router probes!)
        if (!eng->is_ram_resident) {
            dispatch_group_async(init_grp, bg_q, ^{
                @autoreleasepool {
                    (void)sg53_init_ane_lookahead(eng);
                }
            });
        } else {
            eng->ane_verified_physical = false;
            eng->ane_lookahead_horizon = 0U;
        }

        // 1. Zero-Copy `spine.strata` (`~5.26 GiB`) UMA Wiring:
        //    In v2 RAM-resident (`93.22 GiB` total), DO NOT CPU-prefault before `MTLResidencySet` (causes double-faulting
        //    and clean-page eviction -> `Steady MajFlt > 0`!). Match `strata_qwen_flash_gpu.mm`: `POSIX_MADV_WILLNEED` +
        //    `newBufferWithBytesNoCopy:` + `MTLResidencySet` wires all pages directly via Metal/AGX!
        (void)posix_madvise(const_cast<uint8_t*>(bundle->spine_base), bundle->spine_bytes, POSIX_MADV_WILLNEED);
        if (!eng->is_ram_resident) {
            sg53_prefault_pages_ro(bundle->spine_base, bundle->spine_bytes);
        }

        bool zc_spine = false;
        if ((reinterpret_cast<uintptr_t>(bundle->spine_base) % STRATA_GLM53_PAGE_BYTES) == 0U &&
            (bundle->spine_bytes % STRATA_GLM53_PAGE_BYTES) == 0U) {
            eng->b_spine = [dev newBufferWithBytesNoCopy:const_cast<uint8_t*>(bundle->spine_base)
                                                  length:bundle->spine_bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
            if (eng->b_spine) zc_spine = true;
        }
        if (!eng->b_spine) {
            eng->b_spine = [dev newBufferWithBytes:bundle->spine_base
                                            length:bundle->spine_bytes
                                           options:MTLResourceStorageModeShared];
        }
        if (!eng->b_spine) {
            std::fprintf(stderr, "[strata_glm53_gpu] Failed to bind spine.strata Metal buffer\n");
            dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
            strata_glm53_gpu_destroy(eng);
            return nullptr;
        }
        eng->zero_copy_spine    = zc_spine;
        eng->spine_mapped_bytes = static_cast<uint64_t>(bundle->spine_bytes);

        // 2. Expert Residency:
        //    - v2 RAM-Resident (`version == 2 && is_ram_resident == 1`): Zero-Copy `part0` (`42.08 GiB`),
        //      `part1` (`42.08 GiB`), `mtp_experts` (`3.80 GiB`), `64 KiB` `b_e8s18_lut`, ZERO `b_hot_experts` / `b_ring_dma`!
        //    - v1 Regime II (`version == 1`): Configurable UMA LFU Hot-Expert Cache + `Ring_DMA`!
        auto bind_zero_copy_ro = [&](const uint8_t* base, size_t bytes, const char* label) -> id<MTLBuffer> {
            if (!base || bytes == 0U) return nil;
            (void)posix_madvise(const_cast<uint8_t*>(base), bytes, POSIX_MADV_WILLNEED);
            id<MTLBuffer> buf = nil;
            if ((reinterpret_cast<uintptr_t>(base) % STRATA_GLM53_PAGE_BYTES) == 0U &&
                (bytes % STRATA_GLM53_PAGE_BYTES) == 0U) {
                buf = [dev newBufferWithBytesNoCopy:const_cast<uint8_t*>(base)
                                             length:bytes
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
            }
            if (!buf) {
                buf = [dev newBufferWithBytes:base length:bytes options:MTLResourceStorageModeShared];
            }
            if (!buf) {
                std::fprintf(stderr, "[strata_glm53_gpu] Failed to bind %s (%zu bytes)\n", label, bytes);
            }
            return buf;
        };

        if (eng->is_ram_resident) {
            eng->hot_capacity_experts = 0U;
            eng->hot_cache_bytes      = 0ULL;
            eng->ring_dma_slots       = 0U;
            eng->fg_ring_slots        = 0U;
            eng->bg_ring_slots        = 0U;
            eng->ring_dma_bytes       = 0ULL;
            eng->b_hot_experts        = nil;
            eng->b_ring_dma           = nil;
            eng->hot_experts_ptr      = nullptr;
            eng->ring_dma_ptr         = nullptr;

            // Zero-copy wire `part0`, `part1`, and `mtp_experts` (`<= 80.64 GiB` per buffer, `0 B` duplication!)
            if (bundle->part0_base && bundle->part0_bytes > 0U) {
                eng->b_experts_part0 = bind_zero_copy_ro(bundle->part0_base, bundle->part0_bytes, "experts_uma_part0.strata");
                if (!eng->b_experts_part0) {
                    dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
                    strata_glm53_gpu_destroy(eng);
                    return nullptr;
                }
                eng->uma_part0_mapped_bytes = static_cast<uint64_t>(bundle->part0_bytes);
            }
            if (bundle->part1_base && bundle->part1_bytes > 0U) {
                eng->b_experts_part1 = bind_zero_copy_ro(bundle->part1_base, bundle->part1_bytes, "experts_uma_part1.strata");
                if (!eng->b_experts_part1) {
                    dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
                    strata_glm53_gpu_destroy(eng);
                    return nullptr;
                }
                eng->uma_part1_mapped_bytes = static_cast<uint64_t>(bundle->part1_bytes);
            }
            if (bundle->mtp_experts_base && bundle->mtp_experts_bytes > 0U) {
                eng->b_mtp_experts = bind_zero_copy_ro(bundle->mtp_experts_base, bundle->mtp_experts_bytes, "mtp_experts.strata");
                if (!eng->b_mtp_experts) {
                    dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
                    strata_glm53_gpu_destroy(eng);
                    return nullptr;
                }
                eng->mtp_experts_mapped_bytes = static_cast<uint64_t>(bundle->mtp_experts_bytes);
            }

            // Build deterministic `64 KiB` `E8S-18-V4` `Half8Entry[4096]` codebook & upload to private GPU buffer!
            eng->b_e8s18_lut = sg53_alloc_private_aligned(dev, STRATA_GLM53_E8S18_LUT_BYTES);
            bool lut_ok = false;
            @autoreleasepool {
                std::vector<StrataGlm53E8Half8Entry> host_lut(STRATA_GLM53_E8S18_NUM_ORBITS);
                strata_glm53_e8s18_build_lut(host_lut.data());
                id<MTLBuffer> lut_stage = [dev newBufferWithBytes:host_lut.data()
                                                           length:STRATA_GLM53_E8S18_LUT_BYTES
                                                          options:MTLResourceStorageModeShared];
                if (lut_stage && eng->b_e8s18_lut) {
                    id<MTLCommandBuffer>       cb_lut = [eng->queue commandBuffer];
                    id<MTLBlitCommandEncoder>  bl_lut = [cb_lut blitCommandEncoder];
                    [bl_lut copyFromBuffer:lut_stage sourceOffset:0 toBuffer:eng->b_e8s18_lut destinationOffset:0 size:STRATA_GLM53_E8S18_LUT_BYTES];
                    [bl_lut endEncoding];
                    [cb_lut commit];
                    [cb_lut waitUntilCompleted];
                    lut_ok = ([cb_lut status] != MTLCommandBufferStatusError);
                }
            }
            if (!lut_ok) {
                std::fprintf(stderr, "[strata_glm53_gpu] Failed to allocate/upload 64 KiB E8S-18-V4 codebook LUT\n");
                dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
                strata_glm53_gpu_destroy(eng);
                return nullptr;
            }
        } else {
            const uint64_t rec_bytes   = std::max<uint64_t>(256ULL, eng->hdr.expert_record_bytes);
            const uint32_t tot_experts = eng->hdr.total_routed_experts;
            const uint64_t max_mtl_len = static_cast<uint64_t>([dev maxBufferLength]);
            const uint64_t safe_cap    = (max_mtl_len > (128ULL << 20)) ? (max_mtl_len - (64ULL << 20)) : max_mtl_len;
            const uint64_t rec_ws_all  = static_cast<uint64_t>([dev recommendedMaxWorkingSetSize]);

            if (tot_experts > 0U) {
                float eff_gib = (hot_cache_gib <= 0.0f) ? STRATA_GLM53_DEFAULT_HOT_CACHE_GIB : hot_cache_gib;
                if (hot_cache_gib <= 0.0f && rec_ws_all > (24ULL << 30)) {
                    const float ws_safe_gib = static_cast<float>(rec_ws_all - (22ULL << 30)) / (1024.0f * 1024.0f * 1024.0f);
                    eff_gib = std::min<float>(eff_gib, ws_safe_gib);
                }
                const double   req_b_dbl = static_cast<double>(eff_gib) * (1024.0 * 1024.0 * 1024.0);
                const uint64_t clamped_b = std::min<uint64_t>(safe_cap, static_cast<uint64_t>(std::max(0.0, req_b_dbl)));
                const uint64_t raw_slots = (hot_cache_gib <= 0.0f)
                                               ? static_cast<uint64_t>(std::llround(req_b_dbl / static_cast<double>(rec_bytes)))
                                               : (clamped_b / rec_bytes);
                const uint64_t max_slots = safe_cap / rec_bytes;
                eng->hot_capacity_experts = static_cast<uint32_t>(
                    std::min<uint64_t>(tot_experts, std::min<uint64_t>(raw_slots, max_slots)));
            } else {
                eng->hot_capacity_experts = 0U;
            }
            eng->hot_cache_bytes = static_cast<uint64_t>(eng->hot_capacity_experts) * rec_bytes;

            if (tot_experts > 0U) {
                const uint64_t def_ring_b = static_cast<uint64_t>(
                    std::llround(static_cast<double>(STRATA_GLM53_DEFAULT_RING_DMA_GIB) * (1024.0 * 1024.0 * 1024.0)));
                const uint32_t def_slots  = std::max<uint32_t>(16U, static_cast<uint32_t>((def_ring_b + rec_bytes / 2ULL) / rec_bytes));
                if (eng->hdr.experts_nvme_bytes <= (256ULL << 20)) {
                    eng->ring_dma_slots = std::min<uint32_t>(def_slots, std::max<uint32_t>(16U, tot_experts));
                } else {
                    const uint64_t rec_ws    = static_cast<uint64_t>([dev recommendedMaxWorkingSetSize]);
                    const uint64_t base_used = eng->spine_mapped_bytes + eng->hot_cache_bytes + (8ULL << 30);
                    const uint64_t want_big  = 480ULL * rec_bytes;
                    if (rec_ws > base_used && (rec_ws - base_used) >= want_big) {
                        eng->ring_dma_slots = 480U;
                    } else {
                        eng->ring_dma_slots = def_slots;
                    }
                }
                eng->bg_ring_slots = std::min<uint32_t>(32U, eng->ring_dma_slots / 4U);
                eng->fg_ring_slots = eng->ring_dma_slots - eng->bg_ring_slots;
            } else {
                eng->ring_dma_slots = 1U;
                eng->fg_ring_slots  = 1U;
                eng->bg_ring_slots  = 0U;
            }
            eng->ring_dma_bytes = (tot_experts > 0U) ? (static_cast<uint64_t>(eng->ring_dma_slots) * rec_bytes) : 0ULL;

            eng->b_hot_experts = sg53_alloc_shared_prefaulted(dev, std::max<size_t>(STRATA_GLM53_PAGE_BYTES, eng->hot_cache_bytes));
            eng->b_ring_dma    = sg53_alloc_shared_prefaulted(dev, std::max<size_t>(STRATA_GLM53_PAGE_BYTES, eng->ring_dma_bytes));
            if (!eng->b_hot_experts || !eng->b_ring_dma) {
                std::fprintf(stderr, "[strata_glm53_gpu] Failed to allocate UMA Hot-Expert Cache or Ring_DMA buffer\n");
                dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
                strata_glm53_gpu_destroy(eng);
                return nullptr;
            }
            eng->hot_experts_ptr = static_cast<uint8_t*>([eng->b_hot_experts contents]);
            eng->ring_dma_ptr    = static_cast<uint8_t*>([eng->b_ring_dma contents]);

            const uint32_t num_moe_tot = eng->hdr.num_total_moe_layers;
            const uint32_t E_cnt       = eng->cfg.n_routed_experts;
            eng->layer_hot.resize(num_moe_tot);
            if (num_moe_tot > 0U && E_cnt > 0U) {
                uint32_t mtp_cap    = 0U;
                uint32_t main_cnt   = num_moe_tot;
                uint32_t rem_budget = eng->hot_capacity_experts;
                if (eng->has_mtp && num_moe_tot > 1U && eng->hot_capacity_experts >= E_cnt * 8U) {
                    mtp_cap     = std::min<uint32_t>(E_cnt, rem_budget);
                    rem_budget -= mtp_cap;
                    main_cnt    = num_moe_tot - 1U;
                }
                const uint32_t base_per_l = (main_cnt > 0U) ? (rem_budget / main_cnt) : 0U;
                const uint32_t rem_per_l  = (main_cnt > 0U) ? (rem_budget % main_cnt) : 0U;
                uint32_t cur_base = 0U;
                for (uint32_t ml = 0U; ml < num_moe_tot; ++ml) {
                    const uint32_t rem_cap = (cur_base < eng->hot_capacity_experts) ? (eng->hot_capacity_experts - cur_base) : 0U;
                    const uint32_t want_c  = (mtp_cap > 0U && ml == num_moe_tot - 1U)
                                                 ? mtp_cap
                                                 : std::min<uint32_t>(E_cnt, base_per_l + (ml < rem_per_l ? 1U : 0U));
                    const uint32_t cap_ml  = std::min<uint32_t>(rem_cap, want_c);
                    Sg53LayerHotState& st  = eng->layer_hot[ml];
                    st.base_slot  = cur_base;
                    st.cap_slots  = cap_ml;
                    st.used_slots = 0U;
                    st.clock_hand = 0U;
                    st.slot_of_e.assign(E_cnt, -1);
                    st.e_of_slot.assign(std::max<size_t>(1U, cap_ml), -1);
                    st.freq_of_e.assign(E_cnt, 0U);
                    cur_base += cap_ml;
                }
            }
            eng->phys_hot_gid.assign(std::max<size_t>(1U, eng->hot_capacity_experts), -1);
            eng->fg_slot_of_gid.assign(std::max<size_t>(1U, tot_experts), -1);
            eng->gid_of_fg_slot.assign(std::max<size_t>(1U, eng->fg_ring_slots), -1);
            eng->gid_of_bg_slot.assign(std::max<size_t>(1U, eng->bg_ring_slots), -1);
            eng->bg_slot_ready.assign(std::max<size_t>(1U, eng->bg_ring_slots), 0U);
            eng->bg_slot_pin_epoch.assign(std::max<size_t>(1U, eng->bg_ring_slots), 0ULL);
        }

        // 3. Persistent KDA (`34`L) + Compressed Latent NoPE MLA (`12`L, `1 KiB/tok/layer`!) + Indexer Caches
        const uint32_t D         = eng->cfg.hidden_size;
        const uint32_t H         = std::max<uint32_t>(1U, eng->cfg.hc_mult);
        const uint32_t hc_dim    = H * D;
        const uint32_t V         = eng->cfg.vocab_size;
        const uint32_t num_kda   = std::max<uint32_t>(1U, eng->cfg.num_kda_layers);
        const uint32_t num_dsa   = eng->cfg.num_dsa_layers;
        const uint32_t dsa_slots = std::max<uint32_t>(1U, num_dsa + (eng->has_mtp ? 1U : 0U));
        const uint32_t lin_nh    = std::max<uint32_t>(1U, eng->cfg.linear_num_heads);
        const uint32_t lin_hd    = std::max<uint32_t>(32U, eng->cfg.linear_head_dim);
        const uint32_t qkv_dim   = lin_nh * lin_hd;
        const uint32_t conv_dim  = 3U * qkv_dim;
        const uint32_t mla_nh    = std::max<uint32_t>(1U, eng->cfg.num_attention_heads);
        const uint32_t qr        = std::max<uint32_t>(64U, eng->cfg.q_lora_rank);
        const uint32_t kvr       = std::max<uint32_t>(64U, eng->cfg.kv_lora_rank);
        const uint32_t q_nope    = std::max<uint32_t>(64U, eng->cfg.qk_nope_head_dim);
        const uint32_t v_dim     = std::max<uint32_t>(64U, eng->cfg.v_head_dim);
        const uint32_t idx_nh    = std::max<uint32_t>(1U, eng->cfg.index_n_heads);
        const uint32_t idx_hd    = std::max<uint32_t>(32U, eng->cfg.index_head_dim);
        const uint32_t select_k  = std::max<uint32_t>(1U, eng->cfg.index_topk / kpool);
        const uint32_t E         = std::max<uint32_t>(1U, eng->cfg.n_routed_experts);
        const uint32_t top_k     = std::max<uint32_t>(1U, eng->cfg.num_experts_per_tok);
        const uint32_t d_moe     = std::max<uint32_t>(64U, eng->cfg.moe_intermediate_size);
        const uint32_t d_dense   = std::max<uint32_t>(d_moe, eng->cfg.intermediate_size);

        eng->conv_layer_stride     = 3ULL * conv_dim * sizeof(uint16_t);
        eng->rec_layer_stride      = static_cast<uint64_t>(lin_nh) * lin_hd * lin_hd * sizeof(float);
        eng->kv_stride_tokens      = eng->max_seq_alloc;
        eng->kv_layer_stride       = static_cast<uint64_t>(eng->max_seq_alloc) * kvr * sizeof(uint16_t);
        eng->idx_pool_layer_stride = static_cast<uint64_t>(eng->max_pools_alloc) * idx_hd * sizeof(uint16_t);
        eng->idx_tail_layer_stride = 16ULL * 2ULL * idx_hd * sizeof(float);

        eng->conv_state_bytes = (eng->cfg.num_kda_layers > 0U) ? (static_cast<uint64_t>(num_kda) * eng->conv_layer_stride) : 0ULL;
        eng->rec_state_bytes  = (eng->cfg.num_kda_layers > 0U) ? (static_cast<uint64_t>(num_kda) * eng->rec_layer_stride)  : 0ULL;
        eng->idx_tail_bytes   = static_cast<uint64_t>(dsa_slots) * eng->idx_tail_layer_stride;

        const uint64_t kv_tot   = static_cast<uint64_t>(dsa_slots) * eng->kv_layer_stride;
        const uint64_t pool_tot = static_cast<uint64_t>(dsa_slots) * eng->idx_pool_layer_stride;

        eng->kv_cache_bytes_allocated = kv_tot + pool_tot + eng->idx_tail_bytes;
        eng->kv_cache_bytes_per_token = static_cast<uint64_t>(num_dsa) *
            (static_cast<uint64_t>(kvr) * sizeof(uint16_t) +
             (static_cast<uint64_t>(idx_hd) * sizeof(uint16_t)) / kpool);

        eng->b_conv_state    = sg53_alloc_private_aligned(dev, std::max<size_t>(4096U, eng->conv_state_bytes));
        eng->b_rec_state     = sg53_alloc_private_aligned(dev, std::max<size_t>(4096U, eng->rec_state_bytes));
        eng->b_mla_kv_cache  = sg53_alloc_private_aligned(dev, std::max<size_t>(4096U, kv_tot));
        eng->b_idx_pool_k    = sg53_alloc_private_aligned(dev, std::max<size_t>(4096U, pool_tot));
        eng->b_idx_tail_ring = sg53_alloc_private_aligned(dev, std::max<size_t>(4096U, eng->idx_tail_bytes));
        eng->b_mtp_h_prev    = sg53_alloc_private_aligned(dev, static_cast<size_t>(D) * sizeof(float));

        // 4. Speculative Rollback Snapshot Arenas (`4` boundary slots)
        size_t ws = 0U;
        if (eng->has_mtp && eng->cfg.num_kda_layers > 0U) {
            const size_t r_snaps = STRATA_GLM53_MAX_SPEC_DRAFT_K * eng->rec_state_bytes;
            const size_t c_snaps = STRATA_GLM53_MAX_SPEC_DRAFT_K * eng->conv_state_bytes;
            eng->b_rec_snaps  = sg53_alloc_private_aligned(dev, r_snaps);
            eng->b_conv_snaps = sg53_alloc_private_aligned(dev, c_snaps);
            ws += r_snaps + c_snaps;
        }

        auto track_priv = [&](size_t sz) -> id<MTLBuffer> {
            ws += sz;
            return sg53_alloc_private_aligned(dev, sz);
        };
        auto track_sh = [&](size_t sz) -> id<MTLBuffer> {
            ws += sz;
            return sg53_alloc_shared_prefaulted(dev, sz);
        };

        // 5. Pre-Allocated Workspace Buffers (`kMaxPrefillChunk = 512`, `b_part_m/l/o` scoped to `kMaxSpecBatch = 8`!)
        eng->b_hc             = track_priv(static_cast<size_t>(kMaxPrefillChunk) * hc_dim * sizeof(float));
        eng->b_mtp_x_f32      = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float));
        eng->b_mtp_e_f32      = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float));
        eng->b_mtp_eh_cat_f16 = track_priv(static_cast<size_t>(kMaxPrefillChunk) * 2ULL * D * sizeof(uint16_t));
        eng->b_mtp_h_out_f32  = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float));
        eng->b_h_main_f32     = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float));
        eng->b_mhc_z_f32      = track_priv(static_cast<size_t>(kMaxPrefillChunk) * hc_dim * sizeof(float));
        eng->b_mhc_mix_f32    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * 32ULL * sizeof(float));
        eng->b_mhc_pre_f32    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * H * sizeof(float));
        eng->b_mhc_post_f32   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * H * sizeof(float));
        eng->b_mhc_comb_f32   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * H * H * sizeof(float));
        eng->b_x_in_f16       = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(uint16_t));
        eng->b_x_rot_f16      = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(uint16_t));
        eng->b_x_in_f32       = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float));
        eng->b_block_y_f32    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float));

        eng->b_kda_qkv_f16      = track_priv(static_cast<size_t>(kMaxPrefillChunk) * conv_dim * sizeof(uint16_t));
        eng->b_kda_qkv_conv_f16 = track_priv(static_cast<size_t>(kMaxPrefillChunk) * conv_dim * sizeof(uint16_t));
        eng->b_kda_fbg_f32      = track_priv(static_cast<size_t>(kMaxPrefillChunk) * (2ULL * lin_hd + lin_nh) * sizeof(float));
        eng->b_kda_fa_f16       = track_priv(static_cast<size_t>(kMaxPrefillChunk) * lin_hd * sizeof(uint16_t));
        eng->b_kda_b_f32        = track_priv(static_cast<size_t>(kMaxPrefillChunk) * lin_nh * sizeof(float));
        eng->b_kda_ga_f16       = track_priv(static_cast<size_t>(kMaxPrefillChunk) * lin_hd * sizeof(uint16_t));
        eng->b_kda_a_f32        = track_priv(static_cast<size_t>(kMaxPrefillChunk) * qkv_dim * sizeof(float));
        eng->b_kda_gate_f16     = track_priv(static_cast<size_t>(kMaxPrefillChunk) * qkv_dim * sizeof(uint16_t));
        eng->b_kda_qk_hat_f32   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * 2ULL * qkv_dim * sizeof(float));
        eng->b_kda_decay_f32    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * qkv_dim * sizeof(float));
        eng->b_kda_beta_f32     = track_priv(static_cast<size_t>(kMaxPrefillChunk) * lin_nh * sizeof(float));
        eng->b_kda_raw_out_f32  = track_priv(static_cast<size_t>(kMaxPrefillChunk) * qkv_dim * sizeof(float));
        eng->b_kda_y_f16        = track_priv(static_cast<size_t>(kMaxPrefillChunk) * qkv_dim * sizeof(uint16_t));

        eng->b_mla_qkv_a_f32    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * (qr + kvr) * sizeof(float));
        eng->b_mla_q_resid_f16  = track_priv(static_cast<size_t>(kMaxPrefillChunk) * qr * sizeof(uint16_t));
        eng->b_mla_q_f16        = track_priv(static_cast<size_t>(kMaxPrefillChunk) * mla_nh * q_nope * sizeof(uint16_t));
        eng->b_mla_q_latent_f16 = track_priv(static_cast<size_t>(kMaxPrefillChunk) * mla_nh * kvr * sizeof(uint16_t));
        eng->b_mla_c_out_f16    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * mla_nh * kvr * sizeof(uint16_t));
        eng->b_mla_v_out_f16    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * mla_nh * std::max<uint32_t>(v_dim, q_nope) * sizeof(uint16_t));
        eng->b_idx_wk_f32       = track_priv(static_cast<size_t>(kMaxPrefillChunk) * idx_hd * sizeof(float));
        eng->b_idx_kgate_f32    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * idx_hd * sizeof(float));
        eng->b_idx_cur_kg_f32   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * 2ULL * idx_hd * sizeof(float));
        eng->b_idx_q_f32        = track_priv(static_cast<size_t>(kMaxPrefillChunk) * idx_nh * idx_hd * sizeof(float));
        eng->b_idx_whead_f32    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * std::max<uint32_t>(64U, idx_nh) * sizeof(float));
        eng->b_idx_scores_f32   = track_sh(static_cast<size_t>(kMaxPrefillChunk) * eng->max_pools_alloc * sizeof(float));
        eng->b_idx_sel_bits     = track_priv(static_cast<size_t>(kMaxPrefillChunk) * eng->max_sel_words * sizeof(uint32_t));
        eng->b_idx_sel_pools    = track_sh(static_cast<size_t>(kMaxPrefillChunk) * select_k * sizeof(uint32_t));
        eng->b_part_m           = track_priv(static_cast<size_t>(kMaxSpecBatch) * kMaxDecodeSplits * mla_nh * sizeof(float));
        eng->b_part_l           = track_priv(static_cast<size_t>(kMaxSpecBatch) * kMaxDecodeSplits * mla_nh * sizeof(float));
        eng->b_part_o           = track_priv(static_cast<size_t>(kMaxSpecBatch) * kMaxDecodeSplits * mla_nh * kvr * sizeof(float));

        eng->b_ffn_gate_up_f32  = track_priv(static_cast<size_t>(kMaxPrefillChunk) * 2ULL * d_dense * sizeof(float));
        eng->b_ffn_mid_f16      = track_priv(std::max<size_t>(
            static_cast<size_t>(kMaxPrefillChunk) * d_dense * sizeof(uint16_t),
            static_cast<size_t>(idx_nh) * kMaxPrefillChunk * idx_hd * sizeof(uint16_t)));
        eng->b_shared_out_f32   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float));
        eng->b_moe_mid_f16      = track_priv((static_cast<size_t>(kMaxPrefillChunk) * top_k + 64U) * d_moe * sizeof(uint16_t));
        eng->b_moe_down_part    = track_priv((static_cast<size_t>(kMaxPrefillChunk) * top_k + 64U) * D * sizeof(float));
        eng->b_prefill_x_i8     = track_priv(static_cast<size_t>(kMaxPrefillChunk) * 16384ULL);
        eng->b_prefill_x_rsc    = track_priv(static_cast<size_t>(kMaxPrefillChunk) * sizeof(uint16_t));
        eng->b_prefill_x_sums   = track_priv(static_cast<size_t>(kMaxPrefillChunk) * 256ULL * sizeof(uint16_t));
        eng->b_moe_x_rsc        = track_priv((static_cast<size_t>(kMaxPrefillChunk) * top_k + 64U) * sizeof(uint16_t));
        eng->b_moe_mid_rsc      = track_priv((static_cast<size_t>(kMaxPrefillChunk) * top_k + 64U) * sizeof(uint16_t));
        eng->b_moe_mid_i8       = track_priv((static_cast<size_t>(kMaxPrefillChunk) * top_k + 64U) * d_moe);
        eng->b_sample_stage_v   = track_priv(static_cast<size_t>(kMaxSpecBatch) * 64ULL * 64ULL * sizeof(float));
        eng->b_sample_stage_i   = track_priv(static_cast<size_t>(kMaxSpecBatch) * 64ULL * 64ULL * sizeof(uint32_t));
        eng->b_sample_stage_ms  = track_priv(static_cast<size_t>(kMaxSpecBatch) * 64ULL * 2ULL * sizeof(float));

        eng->b_unit_x_f16       = track_sh(static_cast<size_t>(kAneSpatialAlign) * D * sizeof(uint16_t));
        eng->b_topk_idx         = track_sh(static_cast<size_t>(kMaxPrefillChunk) * top_k * sizeof(uint32_t));
        eng->b_topk_w           = track_sh(static_cast<size_t>(kMaxPrefillChunk) * top_k * sizeof(float));
        eng->b_expert_dispatch  = track_sh(static_cast<size_t>(kMaxPrefillChunk) * top_k * sizeof(StrataGlm53ExpertDispatch));
        eng->b_moe_sorted_pairs = track_sh((static_cast<size_t>(kMaxPrefillChunk) * top_k + 64U) * sizeof(uint32_t));
        eng->b_moe_work_tiles   = track_sh(std::max<size_t>(
            static_cast<size_t>(kMaxPrefillChunk) * top_k + E + 64U,
            3ULL * kMaxMoeBinTiles) * sizeof(Sg53MoeWorkTileHost));
        eng->b_moe_indirect_args= track_priv(4096U);
        eng->b_pen_ids          = track_sh(static_cast<size_t>(V) * sizeof(uint32_t));
        eng->b_pen_counts       = track_sh(static_cast<size_t>(V) * sizeof(uint16_t));
        if (eng->b_pen_counts) {
            std::memset([eng->b_pen_counts contents], 0, static_cast<size_t>(V) * sizeof(uint16_t));
        }
        eng->pen_pos.assign(V, 0xFFFFFFFFU);
        eng->b_token_ids        = track_sh(static_cast<size_t>(kMaxPrefillChunk) * sizeof(uint32_t));
        eng->b_mtp_token_ids    = track_sh(static_cast<size_t>(kMaxPrefillChunk) * sizeof(uint32_t));
        eng->b_mtp_replay_toks  = track_sh(static_cast<size_t>(kMaxSpecBatch + 1U) * sizeof(uint32_t));
        eng->b_logits           = track_sh(static_cast<size_t>(kMaxSpecBatch) * V * sizeof(float));
        eng->b_next_tokens      = track_sh(static_cast<size_t>(kMaxSpecBatch) * sizeof(uint32_t));
        eng->b_draft_pmax       = track_sh(static_cast<size_t>(STRATA_GLM53_MAX_SPEC_DRAFT_K) * sizeof(float));
        eng->b_sample_top_ids   = track_sh(static_cast<size_t>(kMaxSpecBatch) * 64ULL * sizeof(uint32_t));
        eng->b_sample_top_probs = track_sh(static_cast<size_t>(kMaxSpecBatch) * 64ULL * sizeof(float));
        eng->b_sample_keep_k    = track_sh(static_cast<size_t>(kMaxSpecBatch) * sizeof(uint32_t));
        eng->b_draft_top_ids    = track_sh(static_cast<size_t>(STRATA_GLM53_MAX_SPEC_DRAFT_K) * 64ULL * sizeof(uint32_t));
        eng->b_draft_top_probs  = track_sh(static_cast<size_t>(STRATA_GLM53_MAX_SPEC_DRAFT_K) * 64ULL * sizeof(float));
        eng->b_draft_keep_k     = track_sh(static_cast<size_t>(STRATA_GLM53_MAX_SPEC_DRAFT_K) * sizeof(uint32_t));
        eng->b_block_loc_dummy  = track_sh(4096U);
        eng->b_kv_probe_qresid  = track_sh(static_cast<size_t>(std::max<uint32_t>(1U, num_dsa)) * qr * sizeof(uint16_t));
        eng->b_kv_probe_whead   = track_sh(static_cast<size_t>(std::max<uint32_t>(1U, num_dsa)) * idx_nh * sizeof(float));
        eng->b_block_loc        = eng->b_block_loc_dummy;

        // Build 20K SLC-Pinned FR-Spec Draft-Head Shortlist (`45.00 MiB < 48 MiB` SLC, `< 50 ms`, 100% lossless!)
        (void)sg53_build_frspec_shortlist(eng);
        if (eng->b_frspec_lm_head)       ws += static_cast<size_t>([eng->b_frspec_lm_head length]);
        if (eng->b_frspec_short_to_full) ws += static_cast<size_t>([eng->b_frspec_short_to_full length]);
        if (eng->b_frspec_full_to_short) ws += static_cast<size_t>([eng->b_frspec_full_to_short length]);
        if (eng->b_frspec_patch_pairs)   ws += static_cast<size_t>([eng->b_frspec_patch_pairs length]);

        eng->workspace_bytes = ws + (eng->b_e8s18_lut ? STRATA_GLM53_E8S18_LUT_BYTES : 0ULL);

        // Verify all buffers allocated cleanly
        NSMutableArray<id<MTLBuffer>>* all_bufs = [NSMutableArray arrayWithCapacity:110];
        id<MTLBuffer> req_bufs[] = {
            eng->b_spine,
            eng->b_conv_state, eng->b_rec_state, eng->b_mla_kv_cache,
            eng->b_idx_pool_k, eng->b_idx_tail_ring, eng->b_mtp_h_prev,
            eng->b_hc, eng->b_mtp_x_f32, eng->b_mtp_e_f32, eng->b_mtp_eh_cat_f16,
            eng->b_mtp_h_out_f32, eng->b_h_main_f32, eng->b_mhc_z_f32,
            eng->b_mhc_mix_f32, eng->b_mhc_pre_f32, eng->b_mhc_post_f32, eng->b_mhc_comb_f32,
            eng->b_x_in_f16, eng->b_x_rot_f16, eng->b_x_in_f32, eng->b_block_y_f32,
            eng->b_kda_qkv_f16, eng->b_kda_qkv_conv_f16, eng->b_kda_fbg_f32,
            eng->b_kda_fa_f16, eng->b_kda_b_f32, eng->b_kda_ga_f16,
            eng->b_kda_a_f32, eng->b_kda_gate_f16, eng->b_kda_qk_hat_f32,
            eng->b_kda_decay_f32, eng->b_kda_beta_f32, eng->b_kda_raw_out_f32, eng->b_kda_y_f16,
            eng->b_mla_qkv_a_f32, eng->b_mla_q_resid_f16, eng->b_mla_q_f16,
            eng->b_mla_q_latent_f16, eng->b_mla_c_out_f16, eng->b_mla_v_out_f16,
            eng->b_idx_wk_f32, eng->b_idx_kgate_f32, eng->b_idx_cur_kg_f32,
            eng->b_idx_q_f32, eng->b_idx_whead_f32, eng->b_idx_scores_f32,
            eng->b_idx_sel_bits, eng->b_idx_sel_pools, eng->b_part_m, eng->b_part_l, eng->b_part_o,
            eng->b_ffn_gate_up_f32, eng->b_ffn_mid_f16, eng->b_shared_out_f32,
            eng->b_moe_mid_f16, eng->b_moe_down_part,
            eng->b_prefill_x_i8, eng->b_prefill_x_rsc, eng->b_prefill_x_sums,
            eng->b_moe_x_rsc, eng->b_moe_mid_rsc, eng->b_moe_mid_i8,
            eng->b_sample_stage_v, eng->b_sample_stage_i, eng->b_sample_stage_ms,
            eng->b_unit_x_f16, eng->b_topk_idx, eng->b_topk_w, eng->b_expert_dispatch,
            eng->b_moe_sorted_pairs, eng->b_moe_work_tiles, eng->b_moe_indirect_args,
            eng->b_pen_ids, eng->b_pen_counts,
            eng->b_token_ids, eng->b_mtp_token_ids, eng->b_mtp_replay_toks,
            eng->b_logits, eng->b_next_tokens, eng->b_draft_pmax,
            eng->b_sample_top_ids, eng->b_sample_top_probs, eng->b_sample_keep_k,
            eng->b_draft_top_ids, eng->b_draft_top_probs, eng->b_draft_keep_k,
            eng->b_block_loc_dummy, eng->b_kv_probe_qresid, eng->b_kv_probe_whead
        };
        for (id<MTLBuffer> b : req_bufs) {
            if (!b) {
                std::fprintf(stderr, "[strata_glm53_gpu] Failed to allocate GPU buffer\n");
                dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
                strata_glm53_gpu_destroy(eng);
                return nullptr;
            }
            [all_bufs addObject:b];
        }
        if (eng->b_frspec_lm_head)       [all_bufs addObject:eng->b_frspec_lm_head];
        if (eng->b_frspec_short_to_full) [all_bufs addObject:eng->b_frspec_short_to_full];
        if (eng->b_frspec_full_to_short) [all_bufs addObject:eng->b_frspec_full_to_short];
        if (eng->b_frspec_patch_pairs)   [all_bufs addObject:eng->b_frspec_patch_pairs];
        if (eng->is_ram_resident) {
            if (eng->b_experts_part0) [all_bufs addObject:eng->b_experts_part0];
            if (eng->b_experts_part1) [all_bufs addObject:eng->b_experts_part1];
            if (eng->b_mtp_experts)   [all_bufs addObject:eng->b_mtp_experts];
            if (eng->b_e8s18_lut)     [all_bufs addObject:eng->b_e8s18_lut];
        } else {
            if (!eng->b_hot_experts || !eng->b_ring_dma) {
                dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
                strata_glm53_gpu_destroy(eng);
                return nullptr;
            }
            [all_bufs addObject:eng->b_hot_experts];
            [all_bufs addObject:eng->b_ring_dma];
        }
        if (eng->has_mtp && eng->cfg.num_kda_layers > 0U) {
            if (!eng->b_rec_snaps || !eng->b_conv_snaps) {
                dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
                strata_glm53_gpu_destroy(eng);
                return nullptr;
            }
            [all_bufs addObject:eng->b_rec_snaps];
            [all_bufs addObject:eng->b_conv_snaps];
        }

        eng->residency_set = sg53_create_residency_set(dev, eng->queue, all_bufs);

        // 6. Wait for concurrent Metal 4 shader compilation & 16-Core ANE compilation
        dispatch_group_wait(init_grp, DISPATCH_TIME_FOREVER);
        if (!shader_ok) {
            strata_glm53_gpu_destroy(eng);
            return nullptr;
        }

        auto* ds = new StrataGlm53ContextSlot();
        ds->eng              = eng;
        ds->is_default       = true;
        ds->paged            = false;
        ds->max_blocks       = eng->max_pools_alloc;
        ds->n_layers         = dsa_slots;
        ds->kv_stride_tokens = eng->max_seq_alloc;
        ds->kv_layer_stride  = eng->kv_layer_stride;
        ds->b_conv_state     = eng->b_conv_state;
        ds->b_rec_state      = eng->b_rec_state;
        ds->b_mla_kv_cache   = eng->b_mla_kv_cache;
        ds->b_idx_pool_k     = eng->b_idx_pool_k;
        ds->b_idx_tail_ring  = eng->b_idx_tail_ring;
        ds->b_mtp_h_prev     = eng->b_mtp_h_prev;
        ds->b_block_loc      = eng->b_block_loc_dummy;
        ds->stats.mode       = kSg53CompactOff;
        ds->stats.uma_kv_blocks_capacity = eng->max_pools_alloc;
        eng->default_slot    = ds;
        eng->bound_slot      = ds;

        strata_glm53_gpu_reset(eng);

        // Pin main executable's `__TEXT` + `__DATA_CONST` (~2.4 MiB) AFTER wiring the 94.5 GiB GPU residency set so
        // macOS UBC reclaim during residency wiring can NEVER leave cold executable pages un-faulted (`Steady MajFlt == 0`!)
        if (eng->is_ram_resident) {
            const auto* mh = reinterpret_cast<const struct mach_header_64*>(_dyld_get_image_header(0U));
            if (mh) {
                unsigned long text_sz = 0UL;
                const uint8_t* text_ptr = getsegmentdata(mh, "__TEXT", &text_sz);
                if (text_ptr && text_sz > 0UL && text_sz <= (32UL << 20)) {
                    sg53_prefault_pages_ro(text_ptr, static_cast<size_t>(text_sz));
                    if (::mlock(text_ptr, static_cast<size_t>(text_sz)) == 0) {
                        eng->mlocked_text_ptr = text_ptr;
                        eng->mlocked_text_len = static_cast<size_t>(text_sz);
                    }
                }
                unsigned long ro_sz = 0UL;
                const uint8_t* ro_ptr = getsegmentdata(mh, "__DATA_CONST", &ro_sz);
                if (ro_ptr && ro_sz > 0UL && ro_sz <= (16UL << 20)) {
                    sg53_prefault_pages_ro(ro_ptr, static_cast<size_t>(ro_sz));
                    if (::mlock(ro_ptr, static_cast<size_t>(ro_sz)) == 0) {
                        eng->mlocked_rodata_ptr = ro_ptr;
                        eng->mlocked_rodata_len = static_cast<size_t>(ro_sz);
                    }
                }
            }
        }

        const auto t_end  = std::chrono::steady_clock::now();
        eng->load_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        return eng;
    }
}

void strata_glm53_gpu_destroy(StrataGlm53GpuEngine* eng) {
    if (!eng) return;
    @autoreleasepool {
        if (eng->profile_nvme) {
            std::fprintf(stderr,
                         "[strata_glm53_profile] nvme_reads: fg=%llu mtp=%llu fixup=%llu bg_ane=%llu (ane_hits=%llu, budget=%u)\n",
                         static_cast<unsigned long long>(eng->prof_fg_reads.load()),
                         static_cast<unsigned long long>(eng->prof_mtp_reads.load()),
                         static_cast<unsigned long long>(eng->prof_fixup_reads.load()),
                         static_cast<unsigned long long>(eng->prof_bg_reads.load()),
                         static_cast<unsigned long long>(eng->counters.ane_lookahead_hits),
                         eng->ane_prefetch_budget);
        }
        if (eng->bound_slot && !eng->bound_slot->is_default) {
            strata_glm53_bind_slot(eng, nullptr);
        }
        (void)sg53_drain_pending_fix(eng);
        sg53_drain_ane_queue(eng);
        sg53_detach_residency_set(eng->queue, eng->residency_set);
        eng->residency_set = nil;
        if (eng->mlocked_text_ptr && eng->mlocked_text_len > 0U) {
            (void)::munlock(eng->mlocked_text_ptr, eng->mlocked_text_len);
            eng->mlocked_text_ptr = nullptr;
            eng->mlocked_text_len = 0U;
        }
        if (eng->mlocked_rodata_ptr && eng->mlocked_rodata_len > 0U) {
            (void)::munlock(eng->mlocked_rodata_ptr, eng->mlocked_rodata_len);
            eng->mlocked_rodata_ptr = nullptr;
            eng->mlocked_rodata_len = 0U;
        }
        eng->ane_windows.clear();
        delete eng->ane_kv;
        eng->ane_kv = nullptr;
        delete eng->default_slot;
        eng->default_slot = nullptr;
        eng->bound_slot   = nullptr;
        delete eng;
    }
}

void strata_glm53_gpu_reset(StrataGlm53GpuEngine* eng) {
    if (!eng) return;
    @autoreleasepool {
        (void)sg53_drain_pending_fix(eng);
        sg53_drain_ane_queue(eng);
        if (eng->bound_slot) sg53_paged_join_probe(eng->bound_slot);
        eng->seq_pos     = 0U;
        eng->mtp_seq_pos = 0U;
        eng->sample_step = 0ULL;
        eng->paged_error = false;
        sg53_reset_spec_ctl(eng);
        if (eng->bound_slot) {
            eng->bound_slot->seq_pos     = 0U;
            eng->bound_slot->mtp_seq_pos = 0U;
            eng->bound_slot->sample_step = 0ULL;
            if (eng->kv_paged) sg53_paged_reset_tables(eng->bound_slot);
            sg53_ngram_reset(eng->bound_slot);
        }
        strata_glm53_gpu_penalties_reset(eng);

        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        sg53_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_conv_state, [eng->b_conv_state length]);
        sg53_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_rec_state, [eng->b_rec_state length]);
        sg53_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_idx_tail_ring, [eng->b_idx_tail_ring length]);
        sg53_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_mtp_h_prev, [eng->b_mtp_h_prev length]);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

void strata_glm53_gpu_reset_hot_cache(StrataGlm53GpuEngine* eng) {
    if (!eng) return;
    sg53_drain_ane_queue(eng);
    eng->fg_ring_cursor = 0U;
    eng->bg_ring_cursor = 0U;
    for (auto& st : eng->layer_hot) {
        st.used_slots = 0U;
        st.clock_hand = 0U;
        std::fill(st.slot_of_e.begin(), st.slot_of_e.end(), -1);
        std::fill(st.e_of_slot.begin(), st.e_of_slot.end(), -1);
        std::fill(st.freq_of_e.begin(), st.freq_of_e.end(), 0U);
    }
    std::fill(eng->phys_hot_gid.begin(), eng->phys_hot_gid.end(), -1);
    std::fill(eng->fg_slot_of_gid.begin(), eng->fg_slot_of_gid.end(), -1);
    std::fill(eng->gid_of_fg_slot.begin(), eng->gid_of_fg_slot.end(), -1);
    {
        Sg53ScopedLock blk(&eng->bg_ring_lock);
        std::fill(eng->gid_of_bg_slot.begin(), eng->gid_of_bg_slot.end(), -1);
        std::fill(eng->bg_slot_ready.begin(), eng->bg_slot_ready.end(), 0U);
        std::fill(eng->bg_slot_pin_epoch.begin(), eng->bg_slot_pin_epoch.end(), 0ULL);
    }
}

uint32_t strata_glm53_gpu_seq_pos(const StrataGlm53GpuEngine* eng) {
    return eng ? eng->seq_pos : 0U;
}

void strata_glm53_gpu_set_exec_mode(
    StrataGlm53GpuEngine* eng,
    StrataGlm53ExecMode   exec_mode
) {
    if (!eng) return;
    eng->exec_mode = exec_mode;
}

StrataGlm53ExecMode strata_glm53_gpu_get_exec_mode(const StrataGlm53GpuEngine* eng) {
    return eng ? eng->exec_mode : STRATA_GLM53_MODE_LOSSLESS;
}

void strata_glm53_gpu_set_adaptive_params(
    StrataGlm53GpuEngine* eng,
    float                 gate_tau,
    uint32_t              cold_cap,
    bool                  renorm_selected
) {
    if (!eng) return;
    eng->adaptive_gate_tau = (gate_tau <= 0.0f) ? STRATA_GLM53_DEFAULT_ADAPTIVE_TAU : gate_tau;
    eng->adaptive_cold_cap = cold_cap;
    eng->adaptive_renorm   = renorm_selected;
}

void strata_glm53_gpu_set_int8_prefill(StrataGlm53GpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->int8_prefill_enabled = enable;
}

void strata_glm53_gpu_set_adaptive_k(StrataGlm53GpuEngine* eng, bool enable) {
    if (!eng) return;
    eng->adaptive_k_enabled = enable;
}

void strata_glm53_gpu_set_frspec(StrataGlm53GpuEngine* eng, bool enable) {
    if (!eng) return;
    (void)sg53_drain_pending_fix(eng);
    eng->frspec_enabled      = enable;
    eng->frspec_widened      = false;
    eng->shortlist_miss_mask = 0U;
}

void strata_glm53_gpu_get_info(
    const StrataGlm53GpuEngine* eng,
    StrataGlm53GpuInfo*         info_out
) {
    if (!info_out) return;
    std::memset(info_out, 0, sizeof(*info_out));
    if (!eng) return;

    const uint64_t kda_bytes = eng->conv_state_bytes + eng->rec_state_bytes;
    std::memcpy(info_out->device_name, eng->device_name, sizeof(info_out->device_name));
    info_out->zero_copy_spine             = eng->zero_copy_spine;
    info_out->ane_verified_physical       = eng->ane_verified_physical;
    info_out->ane_lookahead_horizon       = eng->ane_lookahead_horizon;
    info_out->num_nvme_fds                = eng->is_ram_resident ? 0U : (eng->bundle ? eng->bundle->num_experts_fds : 0U);
    info_out->exec_mode                   = eng->exec_mode;
    info_out->adaptive_gate_tau           = eng->adaptive_gate_tau;
    info_out->adaptive_cold_cap           = eng->adaptive_cold_cap;
    info_out->load_time_ms                = eng->load_time_ms;
    info_out->max_seq_len                 = eng->max_seq_len;
    info_out->spine_mapped_bytes          = eng->spine_mapped_bytes;
    info_out->hot_cache_bytes             = eng->hot_cache_bytes;
    info_out->hot_cache_capacity_experts  = eng->hot_capacity_experts;
    info_out->hot_cache_resident_experts  = eng->is_ram_resident ? 0U : sg53_total_hot_resident(eng);
    info_out->total_routed_experts        = eng->hdr.total_routed_experts;
    info_out->has_mtp                     = eng->has_mtp ? 1U : 0U;
    info_out->ring_dma_bytes              = eng->ring_dma_bytes;
    info_out->ring_dma_slots              = eng->ring_dma_slots;
    info_out->experts_nvme_file_bytes     = eng->is_ram_resident ? 0ULL : (eng->bundle ? eng->bundle->experts_nvme_bytes : eng->hdr.experts_nvme_bytes);
    info_out->kda_state_bytes             = kda_bytes;
    info_out->kv_cache_bytes_allocated    = eng->kv_cache_bytes_allocated;
    info_out->kv_cache_bytes_per_token    = eng->kv_cache_bytes_per_token;
    info_out->workspace_bytes             = eng->workspace_bytes;
    info_out->is_ram_resident             = eng->is_ram_resident ? 1U : 0U;
    info_out->main_expert_dtype           = eng->is_ram_resident ? eng->hdr.main_expert_dtype : STRATA_GLM53_EXPERT_DTYPE_Q4_A;
    info_out->uma_part0_mapped_bytes      = eng->uma_part0_mapped_bytes;
    info_out->uma_part1_mapped_bytes      = eng->uma_part1_mapped_bytes;
    info_out->mtp_experts_mapped_bytes    = eng->mtp_experts_mapped_bytes;
    info_out->total_uma_wired_bytes       = eng->spine_mapped_bytes +
                                            eng->uma_part0_mapped_bytes + eng->uma_part1_mapped_bytes +
                                            eng->mtp_experts_mapped_bytes +
                                            eng->hot_cache_bytes + eng->ring_dma_bytes +
                                            kda_bytes + eng->kv_cache_bytes_allocated + eng->workspace_bytes;
    info_out->active_spine_bytes_per_token= eng->hdr.active_spine_bytes_per_tok;
    info_out->active_bytes_per_token      = eng->hdr.active_bytes_per_tok;
    info_out->mtp_active_bytes_per_draft  = eng->hdr.mtp_active_bytes_per_draft;
    info_out->int8_prefill_enabled        = eng->int8_prefill_enabled ? 1U : 0U;
    info_out->adaptive_k_enabled          = eng->adaptive_k_enabled ? 1U : 0U;
    info_out->frspec_enabled              = eng->frspec_enabled ? 1U : 0U;
    info_out->frspec_shortlist_size       = eng->frspec_shortlist_size;
    info_out->metal_cache_hit             = eng->metal_cache_hit;
    info_out->pso_load_ms                 = eng->pso_load_ms;
}

void strata_glm53_gpu_penalties_reset(StrataGlm53GpuEngine* eng) {
    if (!eng) return;
    sg53_penalties_clear_active(eng);
    eng->pen_hist.clear();
    eng->pen_window_start = 0U;
    eng->pen_window_end   = 0U;
}

void strata_glm53_gpu_penalties_record(
    StrataGlm53GpuEngine* eng,
    const uint32_t*       tokens,
    uint32_t              num_tokens
) {
    if (!eng || !tokens || num_tokens == 0U) return;
    const uint32_t V = eng->cfg.vocab_size;
    eng->pen_hist.reserve(eng->pen_hist.size() + num_tokens);
    for (uint32_t i = 0U; i < num_tokens; ++i) {
        if (tokens[i] < V) {
            eng->pen_hist.push_back(tokens[i]);
        }
    }
}

void strata_glm53_gpu_get_counters(
    const StrataGlm53GpuEngine* eng,
    StrataGlm53PerfCounters*    counters_out
) {
    if (!counters_out) return;
    std::memset(counters_out, 0, sizeof(*counters_out));
    if (!eng) return;
    sg53_drain_ane_queue(eng);
    Sg53ScopedLock lk(const_cast<os_unfair_lock*>(&eng->telemetry_lock));
    *counters_out = eng->counters;
}

void strata_glm53_gpu_reset_counters(StrataGlm53GpuEngine* eng) {
    if (!eng) return;
    sg53_drain_ane_queue(eng);
    Sg53ScopedLock lk(&eng->telemetry_lock);
    std::memset(&eng->counters, 0, sizeof(eng->counters));
}

void strata_glm53_gpu_set_prefill_progress_cb(
    StrataGlm53GpuEngine*        eng,
    StrataGlm53PrefillProgressFn cb,
    void*                        user_ctx
) {
    if (!eng) return;
    eng->progress_cb       = cb;
    eng->progress_user_ctx = user_ctx;
}

static int sg53_prefill_impl(
    StrataGlm53GpuEngine*            eng,
    const uint32_t*                  token_ids,
    uint32_t                         num_tokens,
    const StrataGlm53SamplingParams* sampling,
    uint32_t*                        next_token_out,
    float*                           last_logits_out,
    bool                             compute_final_logits
) {
    if (!eng || !token_ids || num_tokens == 0U) return -1;
    if (eng->seq_pos + num_tokens > eng->max_seq_len) return -2;

    @autoreleasepool {
        if (!sg53_drain_pending_fix(eng)) return -4;
        if (compute_final_logits) {
            strata_glm53_gpu_penalties_reset(eng);
            sg53_reset_spec_ctl(eng);
        }
        const uint32_t D          = eng->cfg.hidden_size;
        const uint32_t H          = std::max<uint32_t>(1U, eng->cfg.hc_mult);
        const uint32_t V          = eng->cfg.vocab_size;
        const uint32_t num_layers = eng->cfg.num_hidden_layers;

        /* Paged KV: one sparse group pins its write-range blocks plus up to `select_k` selected pools, and a group
         * cannot be split below a single query. Keep `write blocks + select_k` inside the UMA pool (same guard as
         * Qwen-Flash): a 512-token chunk at an unaligned position needs 129 + 512 = 641 blocks, one more than the
         * 640-block minimum pool, which used to end in `paged_error`. The minimum pool caps the chunk at 448. */
        uint32_t chunk_cap = kMaxPrefillChunk;
        if (eng->kv_paged && eng->bound_slot && eng->bound_slot->pool_blocks < eng->bound_slot->max_blocks) {
            const uint32_t kpool     = STRATA_GLM53_KV_BLOCK_TOKENS;
            const uint32_t select_k  = std::max<uint32_t>(1U, eng->cfg.index_topk / kpool);
            const uint32_t pool_blks = eng->bound_slot->pool_blocks;
            const uint32_t room_blks = (pool_blks > select_k + 16U) ? (pool_blks - select_k - 16U) : 0U;
            chunk_cap = std::min<uint32_t>(chunk_cap, std::max<uint32_t>(64U, (room_blks * kpool) & ~63U));
        }

        for (uint32_t chunk_off = 0U; chunk_off < num_tokens; chunk_off += chunk_cap) {
            const uint32_t M             = std::min<uint32_t>(chunk_cap, num_tokens - chunk_off);
            const bool     is_last_chunk = (chunk_off + M == num_tokens);
            const bool     want_head     = is_last_chunk && compute_final_logits;
            const uint32_t cur_pos       = eng->seq_pos;
            const bool     force_gemv    = !eng->has_nax;
            const uint32_t M_padded      = force_gemv ? M : ((M + 63U) & ~63U);

            if (eng->kv_paged) sg53_paged_step_begin(eng, cur_pos + M - 1U);
            std::memcpy([eng->b_token_ids contents], token_ids + chunk_off, static_cast<size_t>(M) * sizeof(uint32_t));

            id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            // If `cur_pos > 0` and MTP has pending `(b_mtp_h_prev, token_ids[chunk_off])`, sync MTP at `cur_pos - 1`!
            if (eng->has_mtp && cur_pos > 0U && eng->mtp_seq_pos == cur_pos - 1U) {
                const int mrc = sg53_execute_mtp_pass(
                    eng, eng->b_mtp_h_prev, 0,
                    eng->b_token_ids, 0,
                    cur_pos - 1U, 1U, /*run_head=*/false, force_gemv, /*is_speculative=*/false,
                    &cb, &enc
                );
                if (mrc != 0) {
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    return mrc;
                }
                eng->mtp_seq_pos = cur_pos;
            }

            // 1. Embed + 4-Stream Tiling -> `b_hc [M, 4 * D]`
            {
                Sg53EmbedParamsHost ep{M, D, V, H, cur_pos, 0U, 0U, 0U};
                [enc setComputePipelineState:eng->pso_embed_q4];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.weight_off) atIndex:0];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.scales_off) atIndex:1];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.biases_off) atIndex:2];
                [enc setBuffer:eng->b_token_ids offset:0 atIndex:3];
                [enc setBuffer:eng->b_hc offset:0 atIndex:4];
                [enc setBytes:&ep length:sizeof(ep) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
            }

            // 2. Execute all `45` Main Hybrid Layers (`0 .. num_layers - 1`)
            uint32_t kda_slot = 0U;
            uint32_t dsa_slot = 0U;
            for (uint32_t l = 0U; l < num_layers; ++l) {
                const StrataGlm53LayerDesc& ld = eng->hdr.layers[l];
                const int lrc = sg53_execute_decoder_layer(
                    eng, ld, eng->b_hc, kda_slot, dsa_slot,
                    cur_pos, M, M_padded, 0U, force_gemv,
                    /*is_speculative=*/false, /*allow_ane_lookahead=*/(M <= 16U),
                    &cb, &enc, nullptr
                );
                if (lrc != 0) {
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    return lrc;
                }
                if (ld.attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) ++kda_slot;
                else ++dsa_slot;
            }

            // 3. `Glm5NextTextHyperHead` (`0.25 * sum_s H[s]` -> `final_norm`) -> `b_h_main_f32 [M, D]`!
            {
                Sg53HyperHeadParamsHost hp{
                    M, M_padded, D, eng->cfg.rms_norm_eps,
                    want_head ? 1U : 0U, 1U /* write b_h_main_f32 */, 0U, 0U
                };
                [enc setComputePipelineState:eng->pso_hyper_head_norm];
                [enc setBuffer:eng->b_hc offset:0 atIndex:0];
                [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.final_norm.offset) atIndex:1];
                [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
                [enc setBuffer:eng->b_h_main_f32 offset:0 atIndex:3];
                [enc setBytes:&hp length:sizeof(hp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, M_padded, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            // If final chunk & logits requested, compute `lm_head` + Sampler on `[M - 1]` BEFORE MTP overwrites `b_x_in_f16`!
            if (want_head) {
                const NSUInteger last_f16_off = static_cast<NSUInteger>(M - 1U) * D * sizeof(uint16_t);
                sg53_encode_q4_linear(enc, eng, eng->hdr.lm_head,
                                      eng->b_x_in_f16, last_f16_off, nil, 0, eng->b_logits, 0,
                                      1U, 1U, kOutF32, true);
                eng->sample_step = static_cast<uint64_t>(cur_pos + M) << 8U;
                sg53_encode_sample_batch(enc, eng, sampling, 1U, eng->b_next_tokens, 0, nil, 0, nil, 0, nil, 0);
            }

            // 4. Synchronize Native Layer-45 `MTP` over intra-chunk positions `cur_pos .. cur_pos + M - 2`
            if (eng->has_mtp) {
                if (M > 1U && eng->mtp_seq_pos == cur_pos) {
                    std::memcpy([eng->b_mtp_token_ids contents], token_ids + chunk_off + 1U,
                                static_cast<size_t>(M - 1U) * sizeof(uint32_t));
                    const int mrc = sg53_execute_mtp_pass(
                        eng, eng->b_h_main_f32, 0,
                        eng->b_mtp_token_ids, 0,
                        cur_pos, M - 1U, /*run_head=*/false, force_gemv,
                        /*is_speculative=*/false, &cb, &enc
                    );
                    if (mrc != 0) {
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        return mrc;
                    }
                    eng->mtp_seq_pos = cur_pos + M - 1U;
                }
                const NSUInteger last_h_off = static_cast<NSUInteger>(M - 1U) * D * sizeof(float);
                sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                        eng->b_h_main_f32, last_h_off,
                                        eng->b_mtp_h_prev, 0,
                                        static_cast<size_t>(D) * sizeof(float));
                // Self-healing FR-Spec: pin unique prompt tokens into 1,024-slot dynamic ring (< 10 us GPU copy!)
                if (eng->frspec_enabled && !eng->frspec_widened) {
                    sg53_frspec_encode_patch_tokens(enc, eng, token_ids + chunk_off, M);
                }
            }

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb status] == MTLCommandBufferStatusError || eng->paged_error) {
                NSError* err = [cb error];
                std::fprintf(stderr, "[strata_glm53_gpu] GPU prefill CB error at pos=%u: %s\n",
                             cur_pos, err ? [[err localizedDescription] UTF8String] : "paged_error");
                return -4;
            }
            ++eng->cb_epoch;

            eng->seq_pos += M;
            sg53_ngram_append(eng->bound_slot, token_ids + chunk_off, M);
            {
                Sg53ScopedLock tlk(&eng->telemetry_lock);
                eng->counters.prefill_tokens += M;
            }
            if (eng->progress_cb) {
                eng->progress_cb(chunk_off + M, num_tokens, eng->seq_pos, eng->progress_user_ctx);
            }
        }

        if (!eng->is_ram_resident) sg53_drain_ane_queue(eng);

        if (compute_final_logits) {
            const uint32_t first_tok = *reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents]);
            strata_glm53_gpu_penalties_record(eng, &first_tok, 1U);
            if (next_token_out) {
                *next_token_out = first_tok;
            }
            if (last_logits_out) {
                std::memcpy(last_logits_out, [eng->b_logits contents], static_cast<size_t>(V) * sizeof(float));
            }
        }
        return 0;
    }
}

int strata_glm53_gpu_prefill(
    StrataGlm53GpuEngine*            eng,
    const uint32_t*                  token_ids,
    uint32_t                         num_tokens,
    const StrataGlm53SamplingParams* sampling,
    uint32_t*                        next_token_out,
    float*                           last_logits_out
) {
    return sg53_prefill_impl(eng, token_ids, num_tokens, sampling, next_token_out, last_logits_out, true);
}

int strata_glm53_gpu_prefill_slice(
    StrataGlm53GpuEngine* eng,
    const uint32_t*       token_ids,
    uint32_t              slice_tokens,
    bool                  compute_final_logits,
    uint32_t*             next_token_out
) {
    if (!eng || !token_ids || slice_tokens == 0U || slice_tokens > kMaxPrefillChunk) return -1;
    StrataGlm53SamplingParams greedy{};
    greedy.greedy = true;
    return sg53_prefill_impl(eng, token_ids, slice_tokens, &greedy, next_token_out, nullptr, compute_final_logits);
}

int strata_glm53_gpu_decode_step(
    StrataGlm53GpuEngine*            eng,
    uint32_t                         token_id,
    const StrataGlm53SamplingParams* sampling,
    uint32_t*                        next_token_out,
    float*                           logits_out
) {
    if (!eng) return -1;
    if (eng->seq_pos >= eng->max_seq_len) return -2;

    @autoreleasepool {
        if (!sg53_drain_pending_fix(eng)) return -4;
        const uint32_t D          = eng->cfg.hidden_size;
        const uint32_t H          = std::max<uint32_t>(1U, eng->cfg.hc_mult);
        const uint32_t V          = eng->cfg.vocab_size;
        const uint32_t num_layers = eng->cfg.num_hidden_layers;
        const uint32_t cur_pos    = eng->seq_pos;

        if (eng->kv_paged) sg53_paged_step_begin(eng, cur_pos);
        *reinterpret_cast<uint32_t*>([eng->b_token_ids contents]) = token_id;

        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // 1. Embed + 4-Stream Tiling -> `b_hc [1, 4 * D]`
        {
            Sg53EmbedParamsHost ep{1U, D, V, H, cur_pos, 0U, 0U, 0U};
            [enc setComputePipelineState:eng->pso_embed_q4];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.weight_off) atIndex:0];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.scales_off) atIndex:1];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.biases_off) atIndex:2];
            [enc setBuffer:eng->b_token_ids offset:0 atIndex:3];
            [enc setBuffer:eng->b_hc offset:0 atIndex:4];
            [enc setBytes:&ep length:sizeof(ep) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
        }

        // 2. Execute all `45` Main Hybrid Layers (`336` active experts lossless, overlapped w/ 16c ANE + 8-FD NVMe!)
        uint32_t kda_slot = 0U;
        uint32_t dsa_slot = 0U;
        for (uint32_t l = 0U; l < num_layers; ++l) {
            const StrataGlm53LayerDesc& ld = eng->hdr.layers[l];
            const int lrc = sg53_execute_decoder_layer(
                eng, ld, eng->b_hc, kda_slot, dsa_slot,
                cur_pos, 1U, 1U, 0U, /*force_gemv=*/true,
                /*is_speculative=*/false, /*allow_ane_lookahead=*/true,
                &cb, &enc, nullptr
            );
            if (lrc != 0) {
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                sg53_drain_ane_queue(eng);
                return lrc;
            }
            if (ld.attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) ++kda_slot;
            else ++dsa_slot;
        }

        // 3. `Glm5NextTextHyperHead` -> `b_x_in_f16` (for `lm_head`) & `b_mtp_h_prev` (`H_main[cur_pos]`)!
        {
            Sg53HyperHeadParamsHost hp{
                1U, 1U, D, eng->cfg.rms_norm_eps,
                1U, eng->has_mtp ? 1U : 0U, 0U, 0U
            };
            [enc setComputePipelineState:eng->pso_hyper_head_norm];
            [enc setBuffer:eng->b_hc offset:0 atIndex:0];
            [enc setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.final_norm.offset) atIndex:1];
            [enc setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
            [enc setBuffer:eng->b_mtp_h_prev offset:0 atIndex:3];
            [enc setBytes:&hp length:sizeof(hp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        sg53_encode_q4_linear(enc, eng, eng->hdr.lm_head,
                              eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                              1U, 1U, kOutF32, true);
        sg53_encode_logit_penalties(enc, eng, sampling, 1U, eng->b_token_ids, 0U, 0U);
        eng->sample_step = static_cast<uint64_t>(cur_pos + 1U) << 8U;
        sg53_encode_sample_batch(enc, eng, sampling, 1U, eng->b_next_tokens, 0, nil, 0, nil, 0, nil, 0);

        [enc endEncoding];
        [cb commit];
        if (!eng->is_ram_resident) eng->ane_step_epoch.fetch_add(1ULL, std::memory_order_release);
        [cb waitUntilCompleted];
        if (!eng->is_ram_resident) sg53_drain_ane_queue(eng);

        if ([cb status] == MTLCommandBufferStatusError) {
            NSError* err = [cb error];
            std::fprintf(stderr, "[strata_glm53_gpu] GPU decode CB error: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return -4;
        }
        ++eng->cb_epoch;

        eng->seq_pos += 1U;
        sg53_ngram_append(eng->bound_slot, &token_id, 1U);
        const uint32_t sampled_tok = *reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents]);
        strata_glm53_gpu_penalties_record(eng, &sampled_tok, 1U);
        {
            Sg53ScopedLock tlk(&eng->telemetry_lock);
            eng->counters.decode_steps += 1ULL;
            const uint32_t kpool     = std::max<uint32_t>(1U, eng->cfg.index_kpool);
            const uint64_t num_comp  = static_cast<uint64_t>((cur_pos + 1U) / kpool);
            const uint64_t sel_k     = std::max<uint64_t>(1ULL, eng->cfg.index_topk / kpool);
            const uint64_t idx_reads = (num_comp > sel_k)
                ? (num_comp * eng->cfg.index_head_dim * sizeof(uint16_t) * eng->cfg.num_dsa_layers)
                : 0ULL;
            const uint64_t eff_kv = std::min<uint64_t>(cur_pos + 1U, eng->cfg.index_topk + kpool);
            eng->counters.physical_dram_bytes_read +=
                eng->hdr.active_bytes_per_tok +
                2ULL * eng->rec_state_bytes +
                eff_kv * eng->kv_cache_bytes_per_token +
                idx_reads;
        }

        if (next_token_out) {
            *next_token_out = sampled_tok;
        }
        if (logits_out) {
            std::memcpy(logits_out, [eng->b_logits contents], static_cast<size_t>(V) * sizeof(float));
        }
        return 0;
    }
}

int strata_glm53_gpu_spec_decode_step(
    StrataGlm53GpuEngine*            eng,
    uint32_t                         pending_token_id,
    uint32_t                         draft_k,
    const StrataGlm53SamplingParams* sampling,
    uint32_t*                        emitted_tokens_out,
    uint32_t*                        num_emitted_out,
    uint32_t*                        num_draft_accepted_out
) {
    if (!eng || !emitted_tokens_out || !num_emitted_out) return -1;

    if (draft_k == 0U || !eng->has_mtp || eng->seq_pos == 0U) {
        uint32_t next_tok = 0U;
        int rc = strata_glm53_gpu_decode_step(eng, pending_token_id, sampling, &next_tok, nullptr);
        if (rc != 0) return rc;
        emitted_tokens_out[0] = next_tok;
        *num_emitted_out      = 1U;
        if (num_draft_accepted_out) *num_draft_accepted_out = 0U;
        return 0;
    }

    if (draft_k > STRATA_GLM53_MAX_SPEC_DRAFT_K) {
        draft_k = STRATA_GLM53_MAX_SPEC_DRAFT_K;
    }
    if (eng->seq_pos + 1U + draft_k > eng->max_seq_len) {
        uint32_t next_tok = 0U;
        int rc = strata_glm53_gpu_decode_step(eng, pending_token_id, sampling, &next_tok, nullptr);
        if (rc != 0) return rc;
        emitted_tokens_out[0] = next_tok;
        *num_emitted_out      = 1U;
        if (num_draft_accepted_out) *num_draft_accepted_out = 0U;
        return 0;
    }

    @autoreleasepool {
        const uint32_t D          = eng->cfg.hidden_size;
        const uint32_t H          = std::max<uint32_t>(1U, eng->cfg.hc_mult);
        const uint32_t V          = eng->cfg.vocab_size;
        const uint32_t num_layers = eng->cfg.num_hidden_layers;
        const uint32_t cur_pos    = eng->seq_pos; // P >= 1
        const bool     is_greedy  = (!sampling) || sampling->greedy || (sampling->temperature <= 0.0f);
        const uint32_t eff_max_k  = draft_k;
        const uint32_t mtp_base_pos = cur_pos - 1U;
        const bool     can_fuse_cb  = eng->is_ram_resident && !eng->kv_paged;

        // 1. 3-State SpecCtl Cooldown Check + Longest-Match-First 5-Gram -> 4-Gram Suffix Drafter (0.00 ms GPU!)
        const bool in_cooldown = eng->adaptive_k_enabled &&
                                 (eng->spec_state == StrataGlm53GpuEngine::kSpecCooldownK0) &&
                                 (eng->spec_cooldown_rem > 0U);
        uint32_t ng_drafts[kMaxSpecBatch] = {};
        uint32_t eff_k      = eff_max_k;
        bool     used_ngram = false;
        if (!in_cooldown && eng->ngram_draft_enabled && eng->bound_slot) {
            const uint32_t ng_k = sg53_ngram_propose(eng->bound_slot, pending_token_id, eff_max_k, ng_drafts);
            if (ng_k > 0U) {
                used_ngram = true;
                eff_k      = ng_k;
            } else if (eng->bound_slot->ngram_cooldown > 0U) {
                --eng->bound_slot->ngram_cooldown;
            }
        }
        const bool used_cooldown_k0 = (!used_ngram && in_cooldown);

        // 2. 20K SLC-Pinned FR-Spec Draft-Head Shortlist (`45.00 MiB < 48 MiB` SLC, auto-widens on >= 8/32 misses!)
        const bool use_frspec = (!used_ngram && !used_cooldown_k0) &&
                                eng->frspec_enabled &&
                                !eng->frspec_widened &&
                                (eng->b_frspec_lm_head != nil) &&
                                (eng->frspec_shortlist_size > 0U);

        auto* mtp_toks_host = reinterpret_cast<uint32_t*>([eng->b_mtp_token_ids contents]);
        uint32_t verify_tokens[kMaxSpecBatch] = {};
        verify_tokens[0] = pending_token_id;

        id<MTLBuffer> verify_tok_buf    = eng->b_token_ids;
        uint32_t      M                 = 1U + eff_k;
        bool          used_fused_cb     = false;
        bool          used_adaptive_cb  = false;
        bool          used_high_fused   = false;
        double        measured_draft_ms = 0.0;
        double        step_draft_est    = 0.0;
        double        probe_best_tps    = 0.0;
        double        probe_k0_tps      = 0.0;
        double        draft_pmax_saved[STRATA_GLM53_MAX_SPEC_DRAFT_K]{};

        const uint32_t mtp_ml = static_cast<uint32_t>(eng->hdr.mtp.layer.moe_layer_idx);
        const bool mtp_full_hot = eng->is_ram_resident ||
                                  (mtp_ml < eng->layer_hot.size() &&
                                   eng->layer_hot[mtp_ml].cap_slots >= eng->cfg.n_routed_experts);

        auto encode_draft_loop = [&](id<MTLCommandBuffer> __strong* io_cb,
                                     id<MTLComputeCommandEncoder> __strong* io_enc,
                                     uint32_t k_cnt,
                                     bool want_pmax) -> int {
            eng->sample_step = (static_cast<uint64_t>(cur_pos + 1U) << 8U) | 0x80ULL;
            for (uint32_t k = 0U; k < k_cnt; ++k) {
                id<MTLBuffer> h_in = (k == 0U) ? eng->b_mtp_h_prev : eng->b_mtp_h_out_f32;
                const int drc = sg53_execute_mtp_pass(
                    eng, h_in, 0,
                    eng->b_mtp_token_ids, static_cast<NSUInteger>(k) * sizeof(uint32_t),
                    mtp_base_pos + k, 1U, /*run_head=*/true, /*force_gemv=*/true,
                    /*is_speculative=*/(k > 0U && !mtp_full_hot), io_cb, io_enc, use_frspec
                );
                if (drc != 0) return drc;

                if (use_frspec) {
                    sg53_encode_logit_penalties_shortlist(*io_enc, eng, sampling, eng->b_mtp_token_ids, k);
                } else {
                    sg53_encode_logit_penalties(*io_enc, eng, sampling, 1U, eng->b_mtp_token_ids, 1U, k);
                }

                sg53_encode_draft_sample_step(
                    *io_enc, eng, sampling, use_frspec, want_pmax,
                    eng->b_mtp_token_ids, static_cast<NSUInteger>(k + 1U) * sizeof(uint32_t),
                    eng->b_draft_pmax,    static_cast<NSUInteger>(k) * sizeof(float),
                    is_greedy ? nil : eng->b_draft_top_ids,
                    static_cast<NSUInteger>(k) * 64ULL * sizeof(uint32_t),
                    is_greedy ? nil : eng->b_draft_top_probs,
                    static_cast<NSUInteger>(k) * 64ULL * sizeof(float),
                    is_greedy ? nil : eng->b_draft_keep_k,
                    static_cast<NSUInteger>(k) * sizeof(uint32_t)
                );
            }
            return 0;
        };

        if (used_ngram) {
            for (uint32_t k = 0U; k < eff_k; ++k) verify_tokens[1U + k] = ng_drafts[k];
            std::memcpy([eng->b_token_ids contents], verify_tokens, static_cast<size_t>(M) * sizeof(uint32_t));
            verify_tok_buf = eng->b_token_ids;
        } else if (used_cooldown_k0) {
            // State A (`CooldownK0`): skip `cb_draft` completely (`0` draft cost + `0` CPU sync!),
            // run pure `K = 0` (`M = 1`) verify + async 1-token MTP sync in `cb_fix`!
            eff_k = 0U;
            M     = 1U;
            *reinterpret_cast<uint32_t*>([eng->b_token_ids contents]) = pending_token_id;
            verify_tok_buf = eng->b_token_ids;
        } else if (eng->adaptive_k_enabled &&
                   eng->spec_state == StrataGlm53GpuEngine::kSpecHighConfFused) {
            // State B (`HighConfFused`): Zero-Sync Geometric-MLE Fused `cb_draft + cb_ver`!
            const double p_acc   = std::clamp<double>(eng->ema_geom_p, 0.0, 0.99);
            const double d_per_k = eng->ema_draft_ms / std::max<double>(1.0, static_cast<double>(eff_max_k));
            uint32_t best_k   = std::min<uint32_t>(eff_max_k, 2U);
            double   best_tps = 0.0;
            double   cum_p    = 1.0;
            double   exp_toks = 1.0;
            for (uint32_t k = 1U; k <= eff_max_k; ++k) {
                cum_p    *= p_acc;
                exp_toks += cum_p;
                if (k >= 2U || eff_max_k == 1U) {
                    const double d_ms = d_per_k * static_cast<double>(k);
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
            eff_k           = best_k;
            M               = 1U + eff_k;
            step_draft_est  = d_per_k * static_cast<double>(eff_k);
            mtp_toks_host[0]= pending_token_id;
            used_high_fused = true;
            if (can_fuse_cb) {
                used_fused_cb  = true;
                verify_tok_buf = eng->b_mtp_token_ids;
            } else {
                if (eng->kv_paged) sg53_paged_step_begin(eng, cur_pos + M - 1U);
                id<MTLCommandBuffer>         cb_d  = [eng->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc_d = [cb_d computeCommandEncoder];
                const int drc = encode_draft_loop(&cb_d, &enc_d, eff_k, /*want_pmax=*/false);
                [enc_d endEncoding];
                [cb_d commit];
                [cb_d waitUntilCompleted];
                if (drc != 0 || [cb_d status] == MTLCommandBufferStatusError) return (drc != 0) ? drc : -4;
                ++eng->cb_epoch;
                for (uint32_t m = 0U; m < M; ++m) verify_tokens[m] = mtp_toks_host[m];
                std::memcpy([eng->b_token_ids contents], verify_tokens, static_cast<size_t>(M) * sizeof(uint32_t));
                verify_tok_buf = eng->b_token_ids;
            }
        } else if (eng->adaptive_k_enabled) {
            // State C (`ActiveProbe`): draft `eff_max_k` w/ exact `p_max`, pick optimal `K in 0..eff_max_k`!
            mtp_toks_host[0] = pending_token_id;
            if (eng->kv_paged) sg53_paged_step_begin(eng, cur_pos + eff_max_k);
            id<MTLCommandBuffer>         cb_d  = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_d = [cb_d computeCommandEncoder];
            const int drc = encode_draft_loop(&cb_d, &enc_d, eff_max_k, /*want_pmax=*/true);
            [enc_d endEncoding];
            [cb_d commit];
            [cb_d waitUntilCompleted];
            if (drc != 0 || [cb_d status] == MTLCommandBufferStatusError) return (drc != 0) ? drc : -4;
            if (eng->pending_fix_cb) {
                if ([eng->pending_fix_cb status] == MTLCommandBufferStatusError) return -4;
                eng->pending_fix_cb = nil;
            }
            ++eng->cb_epoch;

            const double g_dt = ([cb_d GPUEndTime] - [cb_d GPUStartTime]) * 1e3;
            if (is_greedy && g_dt > 0.05 && g_dt < 100.0) {
                measured_draft_ms = g_dt;
                eng->ema_draft_ms = 0.85 * eng->ema_draft_ms + 0.15 * g_dt;
            } else {
                measured_draft_ms = eng->ema_draft_ms;
            }

            const float* pmax = reinterpret_cast<const float*>([eng->b_draft_pmax contents]);
            const double cal  = eng->ema_acc_cal;

            uint32_t best_k    = 0U;
            double   best_tps  = 1.0 / std::max(measured_draft_ms + eng->ema_ver_ms[0], 0.1);
            double   cum_prob  = 1.0;
            double   exp_toks  = 1.0;
            double   best_etok = 1.0;
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
            probe_best_tps   = best_etok / std::max(eng->ema_draft_ms + eng->ema_ver_ms[best_k], 0.1);
            probe_k0_tps     = 1.0 / std::max(eng->ema_ver_ms[0], 0.1);
            eff_k            = best_k;
            M                = 1U + eff_k;
            used_adaptive_cb = true;
            for (uint32_t k = 0U; k < eff_k; ++k) verify_tokens[1U + k] = mtp_toks_host[1U + k];
            verify_tok_buf = eng->b_mtp_token_ids;
        } else {
            // Fixed-K Mode (`--no-adaptive-k`): zero-sync fused CB on v2 `!kv_paged`, 2-CB on v1/paged!
            mtp_toks_host[0] = pending_token_id;
            step_draft_est   = eng->ema_draft_ms;
            if (can_fuse_cb) {
                used_fused_cb  = true;
                verify_tok_buf = eng->b_mtp_token_ids;
            } else {
                if (eng->kv_paged) sg53_paged_step_begin(eng, cur_pos + M - 1U);
                id<MTLCommandBuffer>         cb_d  = [eng->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc_d = [cb_d computeCommandEncoder];
                const int drc = encode_draft_loop(&cb_d, &enc_d, eff_k, /*want_pmax=*/false);
                [enc_d endEncoding];
                [cb_d commit];
                [cb_d waitUntilCompleted];
                if (drc != 0 || [cb_d status] == MTLCommandBufferStatusError) return (drc != 0) ? drc : -4;
                ++eng->cb_epoch;
                for (uint32_t m = 0U; m < M; ++m) verify_tokens[m] = mtp_toks_host[m];
                std::memcpy([eng->b_token_ids contents], verify_tokens, static_cast<size_t>(M) * sizeof(uint32_t));
                verify_tok_buf = eng->b_token_ids;
            }
        }

        if (eng->kv_paged && (used_ngram || used_cooldown_k0)) {
            sg53_paged_step_begin(eng, cur_pos + M - 1U);
        }

        // 3. Single Batched 45-Layer Main Model Verification (`M = 1 + eff_k`, `force_gemv = true`
        //    so every IEEE-754 op is 100.0000% bit-identical to `K = 0`! 1 SINGLE FUSED CB on v2!)
        const uint32_t snap_mask = (eff_k > 0U) ? ((1U << eff_k) - 1U) : 0U;
        std::vector<Sg53SpecLayerSnap> spec_snaps;
        if (!eng->is_ram_resident) spec_snaps.reserve(eng->hdr.num_main_moe_layers);

        id<MTLCommandBuffer>         cb_v  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc_v = [cb_v computeCommandEncoder];

        if (used_fused_cb) {
            const int drc = encode_draft_loop(&cb_v, &enc_v, eff_k, /*want_pmax=*/false);
            if (drc != 0) {
                [enc_v endEncoding];
                [cb_v commit];
                [cb_v waitUntilCompleted];
                return drc;
            }
        }

        {
            Sg53EmbedParamsHost ep{M, D, V, H, cur_pos, 0U, 0U, 0U};
            [enc_v setComputePipelineState:eng->pso_embed_q4];
            [enc_v setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.weight_off) atIndex:0];
            [enc_v setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.scales_off) atIndex:1];
            [enc_v setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.biases_off) atIndex:2];
            [enc_v setBuffer:verify_tok_buf offset:0 atIndex:3];
            [enc_v setBuffer:eng->b_hc offset:0 atIndex:4];
            [enc_v setBytes:&ep length:sizeof(ep) atIndex:5];
            [enc_v dispatchThreadgroups:MTLSizeMake(M, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256U, std::max<uint32_t>(32U, D / 32U)), 1, 1)];
        }

        uint32_t kda_slot = 0U;
        uint32_t dsa_slot = 0U;
        for (uint32_t l = 0U; l < num_layers; ++l) {
            const StrataGlm53LayerDesc& ld = eng->hdr.layers[l];
            const int lrc = sg53_execute_decoder_layer(
                eng, ld, eng->b_hc, kda_slot, dsa_slot,
                cur_pos, M, M, snap_mask, /*force_gemv=*/true,
                /*is_speculative=*/(eff_k > 0U), /*allow_ane_lookahead=*/!eng->is_ram_resident,
                &cb_v, &enc_v, (eng->is_ram_resident || eff_k == 0U) ? nullptr : &spec_snaps
            );
            if (lrc != 0) {
                [enc_v endEncoding];
                [cb_v commit];
                [cb_v waitUntilCompleted];
                if (!eng->is_ram_resident) sg53_drain_ane_queue(eng);
                return lrc;
            }
            if (ld.attn_type == STRATA_GLM53_ATTN_KDA_LINEAR) ++kda_slot;
            else ++dsa_slot;
        }

        // `Glm5NextTextHyperHead` over all `M` tokens -> `b_x_in_f16 [M, D]` & `b_h_main_f32 [M, D]` (`H_verify`)!
        {
            Sg53HyperHeadParamsHost hp{M, M, D, eng->cfg.rms_norm_eps, 1U, 1U, 0U, 0U};
            [enc_v setComputePipelineState:eng->pso_hyper_head_norm];
            [enc_v setBuffer:eng->b_hc offset:0 atIndex:0];
            [enc_v setBuffer:eng->b_spine offset:static_cast<NSUInteger>(eng->hdr.final_norm.offset) atIndex:1];
            [enc_v setBuffer:eng->b_x_in_f16 offset:0 atIndex:2];
            [enc_v setBuffer:eng->b_h_main_f32 offset:0 atIndex:3];
            [enc_v setBytes:&hp length:sizeof(hp) atIndex:4];
            [enc_v dispatchThreadgroups:MTLSizeMake(1, M, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // Main model verification ALWAYS runs full 155K vocab -> 100.0000% lossless (`K>0 == K=0` bit-exact!)
        sg53_encode_q4_linear(enc_v, eng, eng->hdr.lm_head,
                              eng->b_x_in_f16, 0, nil, 0, eng->b_logits, 0,
                              M, M, kOutF32, true);
        sg53_encode_logit_penalties(enc_v, eng, sampling, M, verify_tok_buf, 0U, eff_k);
        eng->sample_step = static_cast<uint64_t>(cur_pos + 1U) << 8U;
        sg53_encode_sample_batch(
            enc_v, eng, sampling, M,
            eng->b_next_tokens, 0,
            is_greedy ? nil : eng->b_sample_top_ids, 0,
            is_greedy ? nil : eng->b_sample_top_probs, 0,
            is_greedy ? nil : eng->b_sample_keep_k, 0
        );

        [enc_v endEncoding];
        [cb_v commit];
        if (!eng->is_ram_resident) eng->ane_step_epoch.fetch_add(1ULL, std::memory_order_release);
        [cb_v waitUntilCompleted];
        if ([cb_v status] == MTLCommandBufferStatusError) {
            if (!eng->is_ram_resident) sg53_drain_ane_queue(eng);
            return -4;
        }
        if (eng->pending_fix_cb) {
            if ([eng->pending_fix_cb status] == MTLCommandBufferStatusError) return -4;
            eng->pending_fix_cb = nil; // Guaranteed completed before cb_v on same serial queue!
        }
        ++eng->cb_epoch;

        // Live-calibrate `ema_ver_ms[0..4]` from exact GPU timestamps on greedy runs
        // (on stochastic runs keep analytical `ema_ver_ms` so `eff_k` is 100% seed-deterministic across runs!)
        if (is_greedy) {
            const double raw_dt = ([cb_v GPUEndTime] - [cb_v GPUStartTime]) * 1e3;
            const double v_dt   = used_fused_cb ? (raw_dt - step_draft_est) : raw_dt;
            if (v_dt > 0.2 && v_dt < 500.0 && (M - 1U) < kMaxSpecBatch) {
                const double old_m        = eng->ema_ver_ms[M - 1U];
                const double ratio        = std::clamp<double>(v_dt / std::max(old_m, 0.5), 0.7, 1.4);
                const double global_nudge = 1.0 + 0.08 * (ratio - 1.0);
                for (uint32_t j = 0U; j < kMaxSpecBatch; ++j) eng->ema_ver_ms[j] *= global_nudge;
                eng->ema_ver_ms[M - 1U] = 0.78 * eng->ema_ver_ms[M - 1U] + 0.22 * v_dt;
                for (uint32_t j = 1U; j < kMaxSpecBatch; ++j) {
                    if (eng->ema_ver_ms[j] < eng->ema_ver_ms[j - 1U] + 0.5) {
                        eng->ema_ver_ms[j] = eng->ema_ver_ms[j - 1U] + 0.5;
                    }
                }
            }
        }

        if (used_fused_cb) {
            for (uint32_t k = 0U; k < eff_k; ++k) verify_tokens[1U + k] = mtp_toks_host[1U + k];
        }

        // 4. Exact Leviathan Greedy / Stochastic Verification on CPU (`< 200 ns`!)
        const auto* target_toks = reinterpret_cast<const uint32_t*>([eng->b_next_tokens contents]);
        uint32_t accepted  = 0U;
        uint32_t bonus_tok = target_toks[0];

        if (is_greedy || eff_k == 0U) {
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
            const auto* d_ids   = reinterpret_cast<const uint32_t*>([eng->b_draft_top_ids contents]);
            const auto* d_probs = reinterpret_cast<const float*>([eng->b_draft_top_probs contents]);
            const auto* d_keep  = reinterpret_cast<const uint32_t*>([eng->b_draft_keep_k contents]);

            const auto* p_ids   = reinterpret_cast<const uint32_t*>([eng->b_sample_top_ids contents]);
            const auto* p_probs = reinterpret_cast<const float*>([eng->b_sample_top_probs contents]);
            const auto* p_keep  = reinterpret_cast<const uint32_t*>([eng->b_sample_keep_k contents]);

            const uint64_t rng_base = sampling ? sampling->rng_seed : 12345ULL;
            eng->sample_step = (static_cast<uint64_t>(cur_pos + 1U) << 8U) | 0x40ULL;
            bool rejected = false;

            for (uint32_t k = 0U; k < eff_k; ++k) {
                const uint32_t drafted = verify_tokens[k + 1U];
                const float q_val = used_ngram
                    ? 1.0f
                    : std::max<float>(1e-12f, sg53_lookup_sparse_prob(drafted, d_ids + k * 64U, d_probs + k * 64U, d_keep[k]));
                const float p_val = sg53_lookup_sparse_prob(
                    drafted, p_ids + k * 64U, p_probs + k * 64U, p_keep[k]);

                const float u_acc = sg53_host_rand_u01(rng_base, eng->sample_step++);
                if (p_val >= q_val || u_acc < (p_val / q_val)) {
                    emitted_tokens_out[accepted++] = drafted;
                } else {
                    const uint32_t kp = p_keep[k];
                    float res_p[64];
                    float res_sum = 0.0f;
                    for (uint32_t i = 0U; i < kp; ++i) {
                        const uint32_t cand = p_ids[k * 64U + i];
                        const float    pv   = p_probs[k * 64U + i];
                        const float    qv   = used_ngram
                            ? ((cand == drafted) ? 1.0f : 0.0f)
                            : sg53_lookup_sparse_prob(cand, d_ids + k * 64U, d_probs + k * 64U, d_keep[k]);
                        const float    rv   = std::max<float>(0.0f, pv - qv);
                        res_p[i] = rv;
                        res_sum += rv;
                    }
                    if (res_sum > 0.0f) {
                        const float u_res = sg53_host_rand_u01(rng_base, eng->sample_step++) * res_sum;
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

        const uint32_t n_commit = 1U + accepted;

        // 5. Update 3-State `SpecCtl` Calibration, Geometric MLE & State Transitions!
        if (used_adaptive_cb && eff_k > 0U) {
            const uint32_t n_tested = std::min<uint32_t>(accepted + 1U, eff_k);
            double sum_p = 0.0;
            for (uint32_t j = 0U; j < n_tested; ++j) sum_p += draft_pmax_saved[j];
            eng->ema_cal_num = 0.88 * eng->ema_cal_num + static_cast<double>(accepted);
            eng->ema_cal_den = 0.88 * eng->ema_cal_den + sum_p;
            eng->ema_acc_cal = std::clamp<double>(eng->ema_cal_num / std::max(eng->ema_cal_den, 1e-3), 0.65, 1.02);
        }
        if (eng->adaptive_k_enabled && !used_ngram && eff_k > 0U) {
            const uint32_t n_tested = std::min<uint32_t>(accepted + 1U, eff_k);
            eng->ema_geom_num = 0.78 * eng->ema_geom_num + static_cast<double>(accepted);
            eng->ema_geom_den = 0.78 * eng->ema_geom_den + static_cast<double>(n_tested);
            eng->ema_geom_p   = std::clamp<double>(eng->ema_geom_num / std::max(eng->ema_geom_den, 1e-3), 0.0, 0.99);
        }
        if (eng->adaptive_k_enabled && !used_ngram) {
            if (used_cooldown_k0) {
                if (eng->spec_cooldown_rem > 0U) --eng->spec_cooldown_rem;
                if (eng->spec_cooldown_rem == 0U) {
                    eng->spec_state = StrataGlm53GpuEngine::kSpecActiveProbe;
                }
            } else if (used_high_fused) {
                if (accepted == eff_k && eff_k >= 2U) {
                    ++eng->spec_high_streak;
                    if (eng->spec_high_streak >= 2U) {
                        eng->spec_locked_k = std::min<uint32_t>(eff_max_k, eff_k + 1U);
                    }
                } else {
                    eng->spec_high_streak = 0U;
                    eng->spec_locked_k    = 0U;
                }
                if (eng->ema_geom_p < 0.58 || accepted == 0U || (eff_k >= 3U && accepted <= 1U)) {
                    eng->spec_state       = StrataGlm53GpuEngine::kSpecActiveProbe;
                    eng->spec_high_streak = 0U;
                    eng->spec_locked_k    = 0U;
                }
            } else if (used_adaptive_cb) {
                const bool spec_loses_to_k0 = (eff_k == 0U) ||
                                              (probe_best_tps < probe_k0_tps * 1.02 && accepted == 0U) ||
                                              (probe_best_tps < probe_k0_tps * 0.96);
                if (spec_loses_to_k0) {
                    ++eng->spec_low_streak;
                    eng->spec_high_streak = 0U;
                    if (eng->spec_low_streak >= 2U) {
                        eng->spec_state        = StrataGlm53GpuEngine::kSpecCooldownK0;
                        eng->spec_cooldown_rem = (eng->spec_low_streak >= 3U) ? 8U : 6U;
                    }
                } else {
                    eng->spec_low_streak = 0U;
                    if (eff_k >= 2U && accepted * 100U >= eff_k * 65U) {
                        ++eng->spec_high_streak;
                        if (eng->spec_high_streak >= 2U || (eff_k >= 3U && accepted == eff_k && eng->ema_geom_p >= 0.70)) {
                            eng->spec_state       = StrataGlm53GpuEngine::kSpecHighConfFused;
                            eng->spec_high_streak = (accepted == eff_k) ? 1U : 0U;
                            eng->spec_locked_k    = (accepted == eff_k) ? std::min<uint32_t>(eff_max_k, eff_k + 1U) : eff_k;
                        }
                    } else {
                        eng->spec_high_streak = 0U;
                    }
                }
            }
        }

        // 6. FR-Spec Auto-Widen Tracker (`>= 4/32` misses after self-healing -> widen draft head back to full 155K!)
        uint32_t frspec_miss_toks[kMaxSpecBatch + 1U] = {};
        uint32_t frspec_miss_cnt = 0U;
        if (eng->frspec_enabled && !eng->frspec_widened && eng->b_frspec_full_to_short != nil) {
            const uint32_t* f2s = reinterpret_cast<const uint32_t*>([eng->b_frspec_full_to_short contents]);
            for (uint32_t i = 0U; i < n_commit; ++i) {
                const uint32_t tok  = emitted_tokens_out[i];
                const uint32_t miss = (tok >= V || f2s[tok] == 0xFFFFFFFFU) ? 1U : 0U;
                if (use_frspec) {
                    eng->shortlist_miss_mask = (eng->shortlist_miss_mask << 1U) | miss;
                }
                if (miss != 0U && tok < V && frspec_miss_cnt <= kMaxSpecBatch) {
                    frspec_miss_toks[frspec_miss_cnt++] = tok;
                }
            }
            if (use_frspec && __builtin_popcount(eng->shortlist_miss_mask) >= 4) {
                eng->frspec_widened = true;
            }
        }

        // v1 Regime II only: commit UMA Hot-Expert LFU promotions for accepted tokens `m = 0 .. accepted`!
        if (!eng->is_ram_resident && eff_k > 0U) {
            const int cm_rc = sg53_commit_spec_snaps_post_verify(eng, spec_snaps, accepted);
            if (cm_rc != 0) {
                sg53_drain_ane_queue(eng);
                return cm_rc;
            }
        }

        // 7. Roll back rejected KDA states (`accepted < eff_k`), replay MTP state (using dedicated `b_mtp_replay_toks`
        //    so async `pending_fix_cb` on v2 `!kv_paged` is 100% race-free!), and save `b_mtp_h_prev`!
        {
            auto* replay_host = reinterpret_cast<uint32_t*>([eng->b_mtp_replay_toks contents]);
            replay_host[0] = pending_token_id;
            if (accepted > 0U) {
                std::memcpy(replay_host + 1U, emitted_tokens_out, static_cast<size_t>(accepted) * sizeof(uint32_t));
            }

            id<MTLCommandBuffer>         cb_f  = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_f = [cb_f computeCommandEncoder];

            if (accepted < eff_k && eng->cfg.num_kda_layers > 0U) {
                sg53_encode_copy_buffer(
                    enc_f, eng->pso_copy_u32,
                    eng->b_rec_snaps, static_cast<NSUInteger>(static_cast<uint64_t>(accepted) * eng->rec_state_bytes),
                    eng->b_rec_state, 0, eng->rec_state_bytes);
                sg53_encode_copy_buffer(
                    enc_f, eng->pso_copy_u32,
                    eng->b_conv_snaps, static_cast<NSUInteger>(static_cast<uint64_t>(accepted) * eng->conv_state_bytes),
                    eng->b_conv_state, 0, eng->conv_state_bytes);
            }

            // If N-gram or CooldownK0 skipped `cb_draft`, replay `(old b_mtp_h_prev, pending_token_id)` at `mtp_base_pos`!
            if (used_ngram || used_cooldown_k0) {
                const int nrc = sg53_execute_mtp_pass(
                    eng, eng->b_mtp_h_prev, 0,
                    eng->b_mtp_replay_toks, 0,
                    mtp_base_pos, 1U, /*run_head=*/false, /*force_gemv=*/true,
                    /*is_speculative=*/false, &cb_f, &enc_f, /*use_frspec=*/false
                );
                if (nrc != 0) {
                    [enc_f endEncoding];
                    [cb_f commit];
                    [cb_f waitUntilCompleted];
                    if (!eng->is_ram_resident) sg53_drain_ane_queue(eng);
                    return nrc;
                }
            }

            // Replay true `(emitted_tokens_out[0 .. accepted - 1], b_h_main_f32[0 .. accepted - 1])` at `mtp_base_pos + 1 .. + accepted`!
            if (accepted > 0U) {
                const int src_rc = sg53_execute_mtp_pass(
                    eng, eng->b_h_main_f32, 0,
                    eng->b_mtp_replay_toks, sizeof(uint32_t),
                    mtp_base_pos + 1U, accepted, /*run_head=*/false, /*force_gemv=*/true,
                    /*is_speculative=*/false, &cb_f, &enc_f, /*use_frspec=*/false
                );
                if (src_rc != 0) {
                    [enc_f endEncoding];
                    [cb_f commit];
                    [cb_f waitUntilCompleted];
                    if (!eng->is_ram_resident) sg53_drain_ane_queue(eng);
                    return src_rc;
                }
            }

            const NSUInteger h_acc_off = static_cast<NSUInteger>(accepted) * D * sizeof(float);
            sg53_encode_copy_buffer(enc_f, eng->pso_copy_u32,
                                    eng->b_h_main_f32, h_acc_off,
                                    eng->b_mtp_h_prev, 0,
                                    static_cast<size_t>(D) * sizeof(float));

            // Self-healing FR-Spec: patch any missed emitted tokens into dynamic ring inside `cb_fix` (< 1 us!)
            if (frspec_miss_cnt > 0U && !eng->frspec_widened) {
                sg53_frspec_encode_patch_tokens(enc_f, eng, frspec_miss_toks, frspec_miss_cnt);
            }

            [enc_f endEncoding];
            [cb_f commit];
            if (can_fuse_cb) {
                // Async pipelined `cb_fix` on v2 `!kv_paged` (serial queue orders strictly before next step!)
                eng->pending_fix_cb = cb_f;
            } else {
                [cb_f waitUntilCompleted];
                if (!eng->is_ram_resident) sg53_drain_ane_queue(eng);
                if ([cb_f status] == MTLCommandBufferStatusError) return -4;
            }
            ++eng->cb_epoch;
        }

        eng->seq_pos     = cur_pos + n_commit;
        eng->mtp_seq_pos = mtp_base_pos + n_commit;
        strata_glm53_gpu_penalties_record(eng, emitted_tokens_out, n_commit);
        if (eng->bound_slot) {
            if (used_ngram) {
                eng->bound_slot->ngram_cooldown = (accepted < 2U) ? 2U : 0U;
            }
            sg53_ngram_append(eng->bound_slot, verify_tokens, n_commit);
        }

        *num_emitted_out = n_commit;
        if (num_draft_accepted_out) *num_draft_accepted_out = accepted;

        {
            Sg53ScopedLock tlk(&eng->telemetry_lock);
            eng->counters.spec_verify_steps  += 1ULL;
            eng->counters.mtp_drafted_tokens += eff_k;
            eng->counters.mtp_accepted_tokens+= accepted;

            const uint32_t kpool     = std::max<uint32_t>(1U, eng->cfg.index_kpool);
            const uint64_t num_comp  = static_cast<uint64_t>((cur_pos + M) / kpool);
            const uint64_t sel_k     = std::max<uint64_t>(1ULL, eng->cfg.index_topk / kpool);
            const uint32_t mtp_ran   = (used_ngram || used_cooldown_k0) ? 0U : (used_adaptive_cb ? eff_max_k : eff_k);
            const uint64_t idx_reads = (num_comp > sel_k)
                ? (num_comp * eng->cfg.index_head_dim * sizeof(uint16_t) *
                   (static_cast<uint64_t>(M) * eng->cfg.num_dsa_layers + mtp_ran))
                : 0ULL;
            const uint64_t eff_kv = std::min<uint64_t>(cur_pos + M, eng->cfg.index_topk + kpool);
            eng->counters.physical_dram_bytes_read +=
                static_cast<uint64_t>(mtp_ran) * eng->hdr.mtp_active_bytes_per_draft +
                eng->hdr.active_bytes_per_tok +
                2ULL * eng->rec_state_bytes +
                eff_kv * eng->kv_cache_bytes_per_token +
                idx_reads;
        }
        return 0;
    }
}

/* ============================================================================
 * O(1) Turn-Boundary GPU Checkpoints (`~141 MiB` GPU->GPU Blit, `< 0.5 ms`)
 * ============================================================================ */

struct StrataGlm53Checkpoint {
    id<MTLResidencySet> residency_set;
    id<MTLBuffer>       b_arena;
    uint64_t            conv_off;
    uint64_t            conv_len;
    uint64_t            rec_off;
    uint64_t            rec_len;
    uint64_t            tail_off;
    uint64_t            tail_len;
    uint64_t            mtp_off;
    uint64_t            mtp_len;
    uint64_t            total_bytes;
    uint32_t            seq_pos;
    uint32_t            mtp_seq_pos;
    uint64_t            sample_step;
    bool                valid;
};

StrataGlm53Checkpoint* strata_glm53_checkpoint_create(const StrataGlm53GpuEngine* eng) {
    if (!eng || !eng->device) return nullptr;
    @autoreleasepool {
        auto* ckpt = new StrataGlm53Checkpoint();
        ckpt->residency_set = eng->residency_set;
        ckpt->seq_pos       = 0U;
        ckpt->mtp_seq_pos   = 0U;
        ckpt->sample_step   = 0ULL;
        ckpt->valid         = false;

        uint64_t cur = 0ULL;
        ckpt->conv_off = cur;
        ckpt->conv_len = eng->b_conv_state ? static_cast<uint64_t>([eng->b_conv_state length]) : 0ULL;
        cur = sg53_align_up_u64(cur + ckpt->conv_len, 256ULL);

        ckpt->rec_off = cur;
        ckpt->rec_len = eng->b_rec_state ? static_cast<uint64_t>([eng->b_rec_state length]) : 0ULL;
        cur = sg53_align_up_u64(cur + ckpt->rec_len, 256ULL);

        ckpt->tail_off = cur;
        ckpt->tail_len = eng->b_idx_tail_ring ? static_cast<uint64_t>([eng->b_idx_tail_ring length]) : 0ULL;
        cur = sg53_align_up_u64(cur + ckpt->tail_len, 256ULL);

        ckpt->mtp_off = cur;
        ckpt->mtp_len = eng->b_mtp_h_prev ? static_cast<uint64_t>([eng->b_mtp_h_prev length]) : 0ULL;
        cur = sg53_align_up_u64(cur + ckpt->mtp_len, 4096ULL);

        ckpt->total_bytes = std::max<uint64_t>(4096ULL, cur);
        ckpt->b_arena     = sg53_alloc_private_aligned(eng->device, static_cast<size_t>(ckpt->total_bytes));
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

void strata_glm53_checkpoint_destroy(StrataGlm53Checkpoint* ckpt) {
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

int strata_glm53_checkpoint_save(StrataGlm53GpuEngine* eng, StrataGlm53Checkpoint* ckpt) {
    if (!eng || !ckpt || !ckpt->b_arena) return -1;
    @autoreleasepool {
        if (!sg53_drain_pending_fix(eng)) return -4;
        sg53_drain_ane_queue(eng);
        ckpt->seq_pos     = eng->seq_pos;
        ckpt->mtp_seq_pos = eng->mtp_seq_pos;
        ckpt->sample_step = eng->sample_step;

        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (ckpt->conv_len > 0ULL && eng->b_conv_state) {
            sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    eng->b_conv_state, 0,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->conv_off),
                                    ckpt->conv_len);
        }
        if (ckpt->rec_len > 0ULL && eng->b_rec_state) {
            sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    eng->b_rec_state, 0,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->rec_off),
                                    ckpt->rec_len);
        }
        if (ckpt->tail_len > 0ULL && eng->b_idx_tail_ring) {
            sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    eng->b_idx_tail_ring, 0,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->tail_off),
                                    ckpt->tail_len);
        }
        if (ckpt->mtp_len > 0ULL && eng->b_mtp_h_prev) {
            sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    eng->b_mtp_h_prev, 0,
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

int strata_glm53_checkpoint_restore(StrataGlm53GpuEngine* eng, const StrataGlm53Checkpoint* ckpt) {
    if (!eng || !ckpt || !ckpt->b_arena || !ckpt->valid) return -1;
    @autoreleasepool {
        (void)sg53_drain_pending_fix(eng);
        sg53_drain_ane_queue(eng);
        sg53_reset_spec_ctl(eng);
        eng->seq_pos     = ckpt->seq_pos;
        eng->mtp_seq_pos = ckpt->mtp_seq_pos;
        eng->sample_step = ckpt->sample_step;
        if (eng->kv_paged && eng->bound_slot) {
            sg53_paged_on_rewind(eng->bound_slot, eng->seq_pos, eng->mtp_seq_pos);
        }
        if (eng->bound_slot) {
            sg53_ngram_rewind(eng->bound_slot, eng->seq_pos);
        }
        strata_glm53_gpu_penalties_reset(eng);

        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (ckpt->conv_len > 0ULL && eng->b_conv_state) {
            sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->conv_off),
                                    eng->b_conv_state, 0,
                                    ckpt->conv_len);
        }
        if (ckpt->rec_len > 0ULL && eng->b_rec_state) {
            sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->rec_off),
                                    eng->b_rec_state, 0,
                                    ckpt->rec_len);
        }
        if (ckpt->tail_len > 0ULL && eng->b_idx_tail_ring) {
            sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->tail_off),
                                    eng->b_idx_tail_ring, 0,
                                    ckpt->tail_len);
        }
        if (ckpt->mtp_len > 0ULL && eng->b_mtp_h_prev) {
            sg53_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->mtp_off),
                                    eng->b_mtp_h_prev, 0,
                                    ckpt->mtp_len);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return 0;
    }
}

uint64_t strata_glm53_checkpoint_bytes(const StrataGlm53Checkpoint* ckpt) {
    return ckpt ? ckpt->total_bytes : 0ULL;
}

/* ============================================================================
 * Step 4 Public Slot ABI (`GLM-5.3-Flash` Tier 1 O(1) RCU + Tier 2 Paged KV + Save/Load)
 * ============================================================================ */

StrataGlm53ContextSlot* strata_glm53_slot_create(
    StrataGlm53GpuEngine* eng,
    uint32_t              uma_kv_pool_tokens,
    const char*           nvme_swap_path
) {
    if (!eng || !eng->default_slot) return nullptr;
    if (uma_kv_pool_tokens != 0U && uma_kv_pool_tokens < STRATA_GLM53_COMPACT_MIN_POOL_TOKENS) {
        std::fprintf(stderr, "[strata_glm53_gpu] slot pool %u < minimum %u tokens\n",
                     uma_kv_pool_tokens, STRATA_GLM53_COMPACT_MIN_POOL_TOKENS);
        return nullptr;
    }
    @autoreleasepool {
        id<MTLDevice> dev = eng->device;
        const StrataGlm53ContextSlot* ds = eng->default_slot;
        const uint32_t dsa_slots = std::max<uint32_t>(1U, eng->cfg.num_dsa_layers + (eng->has_mtp ? 1U : 0U));
        const uint32_t kvr       = std::max<uint32_t>(64U, eng->cfg.kv_lora_rank);

        auto* s = new StrataGlm53ContextSlot();
        s->eng        = eng;
        s->n_layers   = dsa_slots;
        s->max_blocks = eng->max_pools_alloc;

        s->b_conv_state    = sg53_alloc_private_aligned(dev, [ds->b_conv_state length]);
        s->b_rec_state     = sg53_alloc_private_aligned(dev, [ds->b_rec_state length]);
        s->b_idx_tail_ring = sg53_alloc_private_aligned(dev, [ds->b_idx_tail_ring length]);
        s->b_mtp_h_prev    = sg53_alloc_private_aligned(dev, [ds->b_mtp_h_prev length]);
        s->bytes = [ds->b_conv_state length] + [ds->b_rec_state length] +
                   [ds->b_idx_tail_ring length] + [ds->b_mtp_h_prev length];

        if (uma_kv_pool_tokens == 0U) {
            s->paged            = false;
            s->kv_stride_tokens = eng->max_seq_alloc;
            s->kv_layer_stride  = ds->kv_layer_stride;
            s->b_mla_kv_cache   = sg53_alloc_private_aligned(dev, [ds->b_mla_kv_cache length]);
            s->b_idx_pool_k     = sg53_alloc_private_aligned(dev, [ds->b_idx_pool_k length]);
            s->b_block_loc      = eng->b_block_loc_dummy;
            s->bytes += [ds->b_mla_kv_cache length] + [ds->b_idx_pool_k length];
        } else {
            uint32_t pool_tok = (uma_kv_pool_tokens / 64U) * 64U;
            pool_tok          = std::min<uint32_t>(pool_tok, eng->max_seq_alloc);
            s->paged            = true;
            s->pool_tokens      = pool_tok;
            s->pool_blocks      = pool_tok / STRATA_GLM53_KV_BLOCK_TOKENS;
            s->kv_stride_tokens = pool_tok;
            s->kv_layer_stride  = static_cast<uint64_t>(pool_tok) * kvr * sizeof(uint16_t);
            const size_t kv_bytes  = static_cast<size_t>(dsa_slots) * s->kv_layer_stride;
            const size_t loc_bytes = static_cast<size_t>(dsa_slots) * s->max_blocks * sizeof(uint32_t);
            s->b_mla_kv_cache = sg53_alloc_shared_prefaulted(dev, kv_bytes);
            s->b_idx_pool_k   = sg53_alloc_shared_prefaulted(dev, [ds->b_idx_pool_k length]);
            s->b_block_loc    = sg53_alloc_shared_prefaulted(dev, loc_bytes);
            s->bytes += kv_bytes + [ds->b_idx_pool_k length] + loc_bytes;

            s->layers.resize(dsa_slots);
            for (auto& L : s->layers) {
                L.st.assign(s->max_blocks, kSg53BlkUnalloc);
                L.on_nvme.assign(s->max_blocks, 0U);
                L.phys_owner.assign(s->pool_blocks, kSg53PhysFree);
                L.u.assign(s->max_blocks, 0.0f);
                L.pin_epoch.assign(s->max_blocks, 0U);
                L.pf_epoch.assign(s->max_blocks, 0U);
                L.free_list.reserve(static_cast<size_t>(s->pool_blocks) * 2U);
            }

            std::string path;
            if (nvme_swap_path && nvme_swap_path[0] != '\0') {
                path = nvme_swap_path;
                s->swap_fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
            } else {
                std::string tmpl = std::string([NSTemporaryDirectory() UTF8String]) + "strata_glm53_kv_swap_XXXXXX";
                std::vector<char> buf(tmpl.begin(), tmpl.end());
                buf.push_back('\0');
                s->swap_fd = ::mkstemp(buf.data());
                if (s->swap_fd >= 0) path.assign(buf.data());
            }
            std::snprintf(s->swap_path, sizeof(s->swap_path), "%s", path.c_str());
            if (s->swap_fd < 0) {
                delete s;
                return nullptr;
            }
            (void)fcntl(s->swap_fd, F_NOCACHE, 1);
            (void)::unlink(path.c_str());
            s->probe_grp = dispatch_group_create();
            std::snprintf(s->stats.nvme_swap_path, sizeof(s->stats.nvme_swap_path), "%s (F_NOCACHE, unlinked)", s->swap_path);

            const size_t blk_bytes = static_cast<size_t>(STRATA_GLM53_KV_BLOCK_TOKENS) * kvr * sizeof(uint16_t);
            const size_t io_bytes  = sg53_align_up_u64(static_cast<uint64_t>(kSg53MaxIoBatchBlks) * blk_bytes, STRATA_GLM53_PAGE_BYTES);
            const size_t ane_bytes = sg53_align_up_u64(static_cast<uint64_t>(dsa_slots) * kSg53AneMaxPrefetchLyr * blk_bytes, STRATA_GLM53_PAGE_BYTES);
            void* raw_io  = nullptr;
            void* raw_ane = nullptr;
            if (posix_memalign(&raw_io, STRATA_GLM53_PAGE_BYTES, io_bytes) != 0 ||
                posix_memalign(&raw_ane, STRATA_GLM53_PAGE_BYTES, ane_bytes) != 0) {
                if (raw_io) std::free(raw_io);
                if (raw_ane) std::free(raw_ane);
                ::close(s->swap_fd);
                delete s;
                return nullptr;
            }
            std::memset(raw_io, 0, io_bytes);
            std::memset(raw_ane, 0, ane_bytes);
            s->io_stage  = static_cast<uint8_t*>(raw_io);
            s->ane_stage = static_cast<uint8_t*>(raw_ane);
        }

        if (!s->b_conv_state || !s->b_rec_state || !s->b_mla_kv_cache ||
            !s->b_idx_pool_k || !s->b_idx_tail_ring || !s->b_mtp_h_prev || !s->b_block_loc) {
            if (s->io_stage) std::free(s->io_stage);
            if (s->ane_stage) std::free(s->ane_stage);
            if (s->swap_fd >= 0) ::close(s->swap_fd);
            delete s;
            return nullptr;
        }

        NSMutableArray<id<MTLBuffer>>* bufs = [NSMutableArray arrayWithObjects:
            s->b_conv_state, s->b_rec_state, s->b_mla_kv_cache, s->b_idx_pool_k, s->b_idx_tail_ring, s->b_mtp_h_prev, nil];
        if (s->paged) [bufs addObject:s->b_block_loc];
        s->rset = sg53_create_residency_set(dev, eng->queue, bufs);

        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        sg53_encode_zero_buffer(enc, eng->pso_zero_u32, s->b_conv_state, [s->b_conv_state length]);
        sg53_encode_zero_buffer(enc, eng->pso_zero_u32, s->b_rec_state, [s->b_rec_state length]);
        sg53_encode_zero_buffer(enc, eng->pso_zero_u32, s->b_idx_tail_ring, [s->b_idx_tail_ring length]);
        sg53_encode_zero_buffer(enc, eng->pso_zero_u32, s->b_mtp_h_prev, [s->b_mtp_h_prev length]);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        s->seq_pos     = 0U;
        s->mtp_seq_pos = 0U;
        s->sample_step = 0ULL;
        if (s->paged) {
            sg53_paged_reset_tables(s);
            sg53_ane_kv_probe_init(eng);
        }
        s->stats.mode                   = s->paged ? kSg53CompactPaged : kSg53CompactOff;
        s->stats.uma_kv_blocks_capacity = s->paged ? s->pool_blocks : eng->max_pools_alloc;
        return s;
    }
}

void strata_glm53_bind_slot(StrataGlm53GpuEngine* eng, StrataGlm53ContextSlot* slot) {
    if (!eng || !eng->default_slot) return;
    StrataGlm53ContextSlot* tgt = slot ? slot : eng->default_slot;
    if (tgt->eng != eng) return;
    StrataGlm53ContextSlot* cur = eng->bound_slot;
    if (cur == tgt) return;
    (void)sg53_drain_pending_fix(eng);
    sg53_drain_ane_queue(eng);
    sg53_reset_spec_ctl(eng);
    if (cur) {
        if (cur->probe_pending) sg53_paged_join_probe(cur);
        cur->seq_pos     = eng->seq_pos;
        cur->mtp_seq_pos = eng->mtp_seq_pos;
        cur->sample_step = eng->sample_step;
    }
    eng->b_conv_state     = tgt->b_conv_state;
    eng->b_rec_state      = tgt->b_rec_state;
    eng->b_mla_kv_cache   = tgt->b_mla_kv_cache;
    eng->b_idx_pool_k     = tgt->b_idx_pool_k;
    eng->b_idx_tail_ring  = tgt->b_idx_tail_ring;
    eng->b_mtp_h_prev     = tgt->b_mtp_h_prev;
    eng->b_block_loc      = tgt->b_block_loc;
    eng->kv_stride_tokens = tgt->kv_stride_tokens;
    eng->kv_layer_stride  = tgt->kv_layer_stride;
    eng->kv_paged         = tgt->paged;
    eng->seq_pos          = tgt->seq_pos;
    eng->mtp_seq_pos      = tgt->mtp_seq_pos;
    eng->sample_step      = tgt->sample_step;
    eng->bound_slot       = tgt;
}

StrataGlm53ContextSlot* strata_glm53_bound_slot(const StrataGlm53GpuEngine* eng) {
    if (!eng || !eng->bound_slot || eng->bound_slot->is_default) return nullptr;
    return eng->bound_slot;
}

void strata_glm53_slot_destroy(StrataGlm53ContextSlot* slot) {
    if (!slot || slot->is_default) return;
    @autoreleasepool {
        StrataGlm53GpuEngine* eng = slot->eng;
        if (eng && eng->bound_slot == slot) {
            strata_glm53_bind_slot(eng, nullptr);
        }
        sg53_paged_join_probe(slot);
        if (slot->io_stage)  { std::free(slot->io_stage);  slot->io_stage  = nullptr; }
        if (slot->ane_stage) { std::free(slot->ane_stage); slot->ane_stage = nullptr; }
        if (slot->swap_fd >= 0) {
            ::close(slot->swap_fd);
            slot->swap_fd = -1;
        }
        if (eng) sg53_detach_residency_set(eng->queue, slot->rset);
        slot->rset = nil;
        delete slot;
    }
}

uint32_t strata_glm53_slot_seq_pos(const StrataGlm53ContextSlot* slot) {
    if (!slot) return 0U;
    return (slot->eng && slot->eng->bound_slot == slot) ? slot->eng->seq_pos : slot->seq_pos;
}

bool strata_glm53_slot_is_paged(const StrataGlm53ContextSlot* slot) {
    return slot ? slot->paged : false;
}

uint64_t strata_glm53_slot_bytes(const StrataGlm53ContextSlot* slot) {
    return slot ? slot->bytes : 0ULL;
}

void strata_glm53_slot_get_stats(const StrataGlm53ContextSlot* slot, StrataGlm53CompactionStats* out) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    if (!slot) return;
    auto* s = const_cast<StrataGlm53ContextSlot*>(slot);
    sg53_paged_join_probe(s);
    *out = s->stats;
    out->episodes_distilled  = 0U;
    out->shadow_seq_pos      = 0U;
    out->shadow_dual_synced  = false;
    out->rcu_swaps_completed = 0U;
    out->last_switchover_ms  = 0.0;
    out->max_user_stall_ms   = 0.0;
    const uint32_t pos       = strata_glm53_slot_seq_pos(slot);
    out->logical_seq_pos     = pos;
    out->logical_blocks      = (pos + STRATA_GLM53_KV_BLOCK_TOKENS - 1U) / STRATA_GLM53_KV_BLOCK_TOKENS;
    if (s->paged) {
        uint32_t max_res   = 0U;
        uint32_t paged_now = 0U;
        for (const auto& L : s->layers) {
            max_res = std::max<uint32_t>(max_res, L.resident);
            for (uint32_t b = 0U; b < s->max_blocks; ++b) paged_now += (L.st[b] == kSg53BlkPaged) ? 1U : 0U;
        }
        out->uma_kv_blocks_used    = max_res;
        out->nvme_blocks_now_paged = paged_now;
    } else {
        out->uma_kv_blocks_used = out->logical_blocks;
    }
    out->ane_probe_us_avg = (s->stats.ane_probe_calls > 0ULL)
        ? (s->ane_us_total / static_cast<double>(s->stats.ane_probe_calls)) : 0.0;
}

void strata_glm53_slot_reset_stats(StrataGlm53ContextSlot* slot) {
    if (!slot) return;
    sg53_paged_join_probe(slot);
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

int strata_glm53_ane_kv_probe_ready(const StrataGlm53GpuEngine* eng, char* note_out, size_t note_cap) {
    if (!eng) return 0;
    const char* note = eng->ane_kv ? eng->ane_kv->note.c_str()
                                   : "ANE KV lookahead probe not built yet (created with the first paged slot)";
    if (note_out && note_cap > 0U) std::snprintf(note_out, note_cap, "%s", note);
    return (eng->ane_kv && eng->ane_kv->ready) ? 1 : 0;
}

namespace {

bool sg53_io_all(int fd, void* buf, size_t n, bool write) {
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

bool sg53_blit_copy(StrataGlm53GpuEngine* eng, id<MTLBuffer> src, NSUInteger src_off,
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

bool sg53_blit_buf_io(StrataGlm53GpuEngine* eng, id<MTLBuffer> gpu_buf, NSUInteger off, size_t len,
                      id<MTLBuffer> stage, int fd, bool save) {
    if (len == 0U) return true;
    const size_t cap = [stage length];
    uint8_t* sp = static_cast<uint8_t*>([stage contents]);
    size_t done = 0U;
    while (done < len) {
        const size_t n = std::min<size_t>(cap, len - done);
        if (!save && !sg53_io_all(fd, sp, n, false)) return false;
        if (save) {
            if (!sg53_blit_copy(eng, gpu_buf, off + done, stage, 0U, n)) return false;
            if (!sg53_io_all(fd, sp, n, true)) return false;
        } else {
            if (!sg53_blit_copy(eng, stage, 0U, gpu_buf, off + done, n)) return false;
        }
        done += n;
    }
    return true;
}

struct Sg53StateHdr {
    uint64_t magic;
    uint32_t seq_pos;
    uint32_t mtp_seq_pos;
    uint32_t dsa_slots;
    uint32_t kvr;
    uint32_t idx_hd;
    uint32_t pad0;
    uint64_t sample_step;
    uint64_t conv_bytes;
    uint64_t rec_bytes;
    uint64_t tail_bytes;
    uint64_t mtp_bytes;
};
constexpr uint64_t kSg53StateMagic = 0x3156533335474D53ULL; /* "SMG53SV1" */

} // namespace

int strata_glm53_state_save_fd(StrataGlm53GpuEngine* eng, int fd, uint64_t* bytes_written_out) {
    if (!eng || !eng->bound_slot || fd < 0) return -1;
    @autoreleasepool {
        if (!sg53_drain_pending_fix(eng)) return -4;
        sg53_drain_ane_queue(eng);
        StrataGlm53ContextSlot* s = eng->bound_slot;
        sg53_paged_join_probe(s);

        const uint32_t dsa_slots = std::max<uint32_t>(1U, eng->cfg.num_dsa_layers + (eng->has_mtp ? 1U : 0U));
        const uint32_t kvr       = std::max<uint32_t>(64U, eng->cfg.kv_lora_rank);
        const uint32_t idx_hd    = std::max<uint32_t>(32U, eng->cfg.index_head_dim);
        const size_t   blk_bytes = static_cast<size_t>(STRATA_GLM53_KV_BLOCK_TOKENS) * kvr * sizeof(uint16_t);

        constexpr size_t kStageBytes = 16ULL << 20;
        id<MTLBuffer> stage = sg53_alloc_shared_prefaulted(eng->device, kStageBytes);
        if (!stage) return -2;

        Sg53StateHdr sh{
            kSg53StateMagic,
            eng->seq_pos,
            eng->mtp_seq_pos,
            dsa_slots,
            kvr,
            idx_hd,
            0U,
            eng->sample_step,
            static_cast<uint64_t>([eng->b_conv_state length]),
            static_cast<uint64_t>([eng->b_rec_state length]),
            static_cast<uint64_t>([eng->b_idx_tail_ring length]),
            static_cast<uint64_t>([eng->b_mtp_h_prev length])
        };
        if (!sg53_io_all(fd, &sh, sizeof(sh), true)) return -3;
        uint64_t tot = sizeof(sh);

        if (!sg53_blit_buf_io(eng, eng->b_conv_state, 0U, sh.conv_bytes, stage, fd, true) ||
            !sg53_blit_buf_io(eng, eng->b_rec_state, 0U, sh.rec_bytes, stage, fd, true) ||
            !sg53_blit_buf_io(eng, eng->b_idx_tail_ring, 0U, sh.tail_bytes, stage, fd, true) ||
            !sg53_blit_buf_io(eng, eng->b_mtp_h_prev, 0U, sh.mtp_bytes, stage, fd, true)) {
            return -4;
        }
        tot += sh.conv_bytes + sh.rec_bytes + sh.tail_bytes + sh.mtp_bytes;

        std::vector<uint8_t> pack_buf(256U * blk_bytes);
        for (uint32_t ls = 0U; ls < dsa_slots; ++ls) {
            const uint32_t valid_pos = (ls + 1U == dsa_slots && eng->has_mtp) ? eng->mtp_seq_pos : eng->seq_pos;
            const uint32_t num_blks  = (valid_pos + 3U) / 4U;
            const uint32_t comp_blks = valid_pos / 4U;

            const size_t pool_k_b = static_cast<size_t>(comp_blks) * idx_hd * sizeof(uint16_t);
            if (!sg53_blit_buf_io(eng, eng->b_idx_pool_k, static_cast<NSUInteger>(ls * eng->idx_pool_layer_stride),
                                  pool_k_b, stage, fd, true)) {
                return -5;
            }
            tot += pool_k_b;
            if (num_blks == 0U) continue;

            if (!s->paged) {
                const size_t kv_b = static_cast<size_t>(num_blks) * blk_bytes;
                if (!sg53_blit_buf_io(eng, eng->b_mla_kv_cache, static_cast<NSUInteger>(ls * eng->kv_layer_stride),
                                      kv_b, stage, fd, true)) {
                    return -6;
                }
                tot += kv_b;
            } else {
                const Sg53PagedLayerState& L = s->layers[ls];
                const uint32_t* loc = sg53_loc_row(s, ls);
                constexpr uint32_t kBatchBlks = 256U;
                for (uint32_t b0 = 0U; b0 < num_blks; b0 += kBatchBlks) {
                    const uint32_t cnt = std::min<uint32_t>(kBatchBlks, num_blks - b0);
                    for (uint32_t bi = 0U; bi < cnt; ++bi) {
                        const uint32_t b   = b0 + bi;
                        uint8_t*       dst = pack_buf.data() + static_cast<size_t>(bi) * blk_bytes;
                        if (L.st[b] == kSg53BlkResident) {
                            const uint32_t phys = loc[b] & STRATA_GLM53_KV_LOC_IDX_MASK;
                            std::memcpy(dst, sg53_pool_blk_cptr(s, ls, phys), blk_bytes);
                        } else if (L.st[b] == kSg53BlkPaged && L.on_nvme[b]) {
                            const off_t foff = static_cast<off_t>(
                                (static_cast<uint64_t>(ls) * s->max_blocks + b) * blk_bytes);
                            if (::pread(s->swap_fd, dst, blk_bytes, foff) != static_cast<ssize_t>(blk_bytes)) return -7;
                        } else {
                            std::memset(dst, 0, blk_bytes);
                        }
                    }
                    const size_t out_b = static_cast<size_t>(cnt) * blk_bytes;
                    if (!sg53_io_all(fd, pack_buf.data(), out_b, true)) return -7;
                    tot += out_b;
                }
            }
        }
        if (bytes_written_out) *bytes_written_out = tot;
        return 0;
    }
}

int strata_glm53_state_load_fd(StrataGlm53GpuEngine* eng, int fd, uint64_t* bytes_read_out) {
    if (!eng || !eng->bound_slot || fd < 0) return -1;
    @autoreleasepool {
        Sg53StateHdr sh{};
        if (!sg53_io_all(fd, &sh, sizeof(sh), false)) return -2;

        const uint32_t dsa_slots = std::max<uint32_t>(1U, eng->cfg.num_dsa_layers + (eng->has_mtp ? 1U : 0U));
        const uint32_t kvr       = std::max<uint32_t>(64U, eng->cfg.kv_lora_rank);
        const uint32_t idx_hd    = std::max<uint32_t>(32U, eng->cfg.index_head_dim);
        if (sh.magic != kSg53StateMagic || sh.dsa_slots != dsa_slots || sh.kvr != kvr ||
            sh.idx_hd != idx_hd || sh.seq_pos > eng->max_seq_len || sh.mtp_seq_pos > eng->max_seq_len ||
            sh.conv_bytes != [eng->b_conv_state length] || sh.rec_bytes != [eng->b_rec_state length] ||
            sh.tail_bytes != [eng->b_idx_tail_ring length] || sh.mtp_bytes != [eng->b_mtp_h_prev length]) {
            return -3;
        }

        constexpr size_t kStageBytes = 16ULL << 20;
        id<MTLBuffer> stage = sg53_alloc_shared_prefaulted(eng->device, kStageBytes);
        if (!stage) return -4;

        strata_glm53_gpu_reset(eng);
        StrataGlm53ContextSlot* s = eng->bound_slot;
        uint64_t tot = sizeof(sh);

        if (!sg53_blit_buf_io(eng, eng->b_conv_state, 0U, sh.conv_bytes, stage, fd, false) ||
            !sg53_blit_buf_io(eng, eng->b_rec_state, 0U, sh.rec_bytes, stage, fd, false) ||
            !sg53_blit_buf_io(eng, eng->b_idx_tail_ring, 0U, sh.tail_bytes, stage, fd, false) ||
            !sg53_blit_buf_io(eng, eng->b_mtp_h_prev, 0U, sh.mtp_bytes, stage, fd, false)) {
            return -5;
        }
        tot += sh.conv_bytes + sh.rec_bytes + sh.tail_bytes + sh.mtp_bytes;

        const size_t blk_bytes = static_cast<size_t>(STRATA_GLM53_KV_BLOCK_TOKENS) * kvr * sizeof(uint16_t);
        std::vector<uint8_t> pack_buf(256U * blk_bytes);

        for (uint32_t ls = 0U; ls < dsa_slots; ++ls) {
            const uint32_t valid_pos = (ls + 1U == dsa_slots && eng->has_mtp) ? sh.mtp_seq_pos : sh.seq_pos;
            const uint32_t num_blks  = (valid_pos + 3U) / 4U;
            const uint32_t comp_blks = valid_pos / 4U;

            const size_t pool_k_b = static_cast<size_t>(comp_blks) * idx_hd * sizeof(uint16_t);
            if (!sg53_blit_buf_io(eng, eng->b_idx_pool_k, static_cast<NSUInteger>(ls * eng->idx_pool_layer_stride),
                                  pool_k_b, stage, fd, false)) {
                return -6;
            }
            tot += pool_k_b;
            if (num_blks == 0U) continue;

            if (!s->paged) {
                const size_t kv_b = static_cast<size_t>(num_blks) * blk_bytes;
                if (!sg53_blit_buf_io(eng, eng->b_mla_kv_cache, static_cast<NSUInteger>(ls * eng->kv_layer_stride),
                                      kv_b, stage, fd, false)) {
                    return -7;
                }
                tot += kv_b;
            } else {
                Sg53PagedLayerState& L = s->layers[ls];
                uint32_t* loc = sg53_loc_row(s, ls);
                const uint32_t sink_keep = (num_blks <= s->pool_blocks)
                    ? num_blks
                    : std::min<uint32_t>(STRATA_GLM53_COMPACT_SINK_BLOCKS, s->pool_blocks / 4U);
                const uint32_t tail_keep = (num_blks <= s->pool_blocks) ? 0U : (s->pool_blocks - sink_keep);
                uint32_t phys_cur = 0U;

                constexpr uint32_t kBatchBlks = 256U;
                for (uint32_t b0 = 0U; b0 < num_blks; b0 += kBatchBlks) {
                    const uint32_t cnt  = std::min<uint32_t>(kBatchBlks, num_blks - b0);
                    const size_t   in_b = static_cast<size_t>(cnt) * blk_bytes;
                    if (!sg53_io_all(fd, pack_buf.data(), in_b, false)) return -8;
                    tot += in_b;
                    for (uint32_t bi = 0U; bi < cnt; ++bi) {
                        const uint32_t b   = b0 + bi;
                        const uint8_t* src = pack_buf.data() + static_cast<size_t>(bi) * blk_bytes;
                        const bool in_uma  = (num_blks <= s->pool_blocks) || (b < sink_keep) || (b >= num_blks - tail_keep);
                        if (in_uma && phys_cur < s->pool_blocks) {
                            const uint32_t phys = phys_cur++;
                            std::memcpy(sg53_pool_blk_ptr(s, ls, phys), src, blk_bytes);
                            L.st[b]            = kSg53BlkResident;
                            L.on_nvme[b]       = 0U;
                            L.phys_owner[phys] = b;
                            loc[b]             = STRATA_GLM53_KV_LOC_UMA_BIT | phys;
                            ++L.resident;
                        } else {
                            const off_t foff = static_cast<off_t>(
                                (static_cast<uint64_t>(ls) * s->max_blocks + b) * blk_bytes);
                            if (::pwrite(s->swap_fd, src, blk_bytes, foff) != static_cast<ssize_t>(blk_bytes)) return -8;
                            L.st[b]      = kSg53BlkPaged;
                            L.on_nvme[b] = 1U;
                            loc[b]       = b & STRATA_GLM53_KV_LOC_IDX_MASK;
                        }
                    }
                }
                L.free_list.clear();
                for (uint32_t p = s->pool_blocks; p-- > phys_cur;) {
                    L.free_list.push_back(p);
                }
            }
        }

        eng->seq_pos     = sh.seq_pos;
        eng->mtp_seq_pos = sh.mtp_seq_pos;
        eng->sample_step = sh.sample_step;
        s->seq_pos       = sh.seq_pos;
        s->mtp_seq_pos   = sh.mtp_seq_pos;
        s->sample_step   = sh.sample_step;
        if (bytes_read_out) *bytes_read_out = tot;
        return 0;
    }
}

} /* extern "C" */

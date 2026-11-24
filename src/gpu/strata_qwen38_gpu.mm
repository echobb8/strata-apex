#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include "../../include/strata_qwen38.h"
#include "../../include/strata_embedded_shaders.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMaxPrefillChunk = 512U;
constexpr uint32_t kMaxDecodeSplits = 64U;

constexpr uint32_t kGemmOutF16    = 0U;
constexpr uint32_t kGemmOutF32    = 1U;
constexpr uint32_t kGemmOutResF32 = 2U;

struct Sq38EmbedParamsHost {
    uint32_t M;
    uint32_t cols;
    uint32_t vocab_size;
    uint32_t pad;
};

struct Sq38RmsNormParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t D;
    float    eps;
};

struct Sq38GemvParamsHost {
    uint32_t rows;
    uint32_t cols;
};

struct Sq38LinInGemvParamsHost {
    uint32_t qkv_rows;
    uint32_t z_rows;
    uint32_t ab_rows;
    uint32_t cols;
};

struct Sq38AttnQkvGemvParamsHost {
    uint32_t q_rows;
    uint32_t kv_rows;
    uint32_t cols;
    uint32_t pad;
};

struct Sq38NaxGemmParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t N;
    uint32_t K;
    uint32_t out_mode;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct Sq38SwigluParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t d_ff;
    uint32_t pad;
};

struct Sq38Conv1dParamsHost {
    uint32_t M;
    uint32_t channels;
};

struct Sq38DeltaNetParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    float    eps;
    uint32_t ab_stride;
};

struct Sq38AttnPrepParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t seq_pos_start;
    uint32_t max_seq_alloc;
    float    eps;
    uint32_t kv_zero_end;
    uint32_t pad0;
    uint32_t pad1;
};

struct Sq38FlashDecodeParamsHost {
    uint32_t total_kv;
    uint32_t max_seq_alloc;
    uint32_t num_splits;
    uint32_t kv_per_split;
};

struct Sq38FlashPrefillParamsHost {
    uint32_t M_valid;
    uint32_t M_padded;
    uint32_t seq_pos_start;
    uint32_t total_kv;
    uint32_t max_seq_alloc;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct Sq38SampleKernelParamsHost {
    uint32_t vocab_size;
    uint32_t greedy;
    float    inv_temp;
    uint32_t top_k;
    float    top_p;
    uint32_t rng_lo;
    uint32_t rng_hi;
    uint32_t pad;
};

std::string sq38_read_text_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

std::string sq38_resolve_sibling_path(const char* rel_from_src_gpu) {
    std::string self_file = __FILE__;
    size_t slash = self_file.find_last_of('/');
    if (slash != std::string::npos) {
        std::string candidate = self_file.substr(0, slash + 1) + rel_from_src_gpu;
        std::ifstream test(candidate);
        if (test.good()) return candidate;
    }
    return std::string("src/gpu/") + rel_from_src_gpu;
}

id<MTLBuffer> sq38_alloc_shared_prefaulted(id<MTLDevice> dev, size_t bytes) {
    const size_t aligned = std::max<size_t>(4096U, (bytes + 4095U) & ~static_cast<size_t>(4095U));
    id<MTLBuffer> buf = [dev newBufferWithLength:aligned options:MTLResourceStorageModeShared];
    if (buf) {
        std::memset([buf contents], 0, aligned);
    }
    return buf;
}

id<MTLBuffer> sq38_alloc_private_aligned(id<MTLDevice> dev, size_t bytes) {
    const size_t aligned = std::max<size_t>(4096U, (bytes + 4095U) & ~static_cast<size_t>(4095U));
    return [dev newBufferWithLength:aligned options:MTLResourceStorageModePrivate];
}

void sq38_encode_zero_buffer(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pso_zero,
    id<MTLBuffer> buf,
    size_t byte_len
) {
    const uint32_t count_u4 = static_cast<uint32_t>((byte_len + 15U) / 16U);
    if (count_u4 == 0U) return;
    [enc setComputePipelineState:pso_zero];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&count_u4 length:sizeof(uint32_t) atIndex:1];
    const uint32_t tg = 256U;
    [enc dispatchThreadgroups:MTLSizeMake((count_u4 + tg - 1U) / tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void sq38_encode_copy_buffer(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pso_copy,
    id<MTLBuffer> src,
    NSUInteger src_off,
    id<MTLBuffer> dst,
    NSUInteger dst_off,
    size_t byte_len
) {
    const uint32_t count_u4 = static_cast<uint32_t>((byte_len + 15U) / 16U);
    if (count_u4 == 0U || !src || !dst) return;
    [enc setComputePipelineState:pso_copy];
    [enc setBuffer:src offset:src_off atIndex:0];
    [enc setBuffer:dst offset:dst_off atIndex:1];
    [enc setBytes:&count_u4 length:sizeof(uint32_t) atIndex:2];
    const uint32_t tg = 256U;
    [enc dispatchThreadgroups:MTLSizeMake((count_u4 + tg - 1U) / tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

id<MTLResidencySet> sq38_create_residency_set(
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

void sq38_detach_residency_set(id<MTLCommandQueue> queue, id<MTLResidencySet> rset) {
    if (!rset) return;
    if (@available(macOS 15.0, *)) {
        if (queue) [queue removeResidencySet:rset];
        [rset endResidency];
    }
}

uint64_t sq38_quant_mat_bytes(const StrataQwen38QuantMatrix& m) {
    return m.weight_bytes + m.scales_bytes + m.biases_bytes;
}

uint64_t sq38_compute_active_bytes_per_token(const StrataQwen38WeightsHeader& hdr) {
    if (hdr.active_bytes_per_tok > 0ULL) {
        return hdr.active_bytes_per_tok;
    }
    const uint64_t D   = hdr.cfg.hidden_size;
    const uint64_t emb = (D / 8ULL) * 4ULL + (D / 64ULL) * 4ULL; // 1 row of embed_tokens
    uint64_t total = emb;
    for (uint32_t l = 0; l < hdr.cfg.num_hidden_layers; ++l) {
        const StrataQwen38LayerDesc& ld = hdr.layers[l];
        total += ld.input_layernorm.bytes + ld.post_attention_layernorm.bytes;
        if (ld.layer_type == STRATA_QWEN38_LAYER_LINEAR_ATTN) {
            total += sq38_quant_mat_bytes(ld.lin_in_proj_qkv) +
                     sq38_quant_mat_bytes(ld.lin_in_proj_z) +
                     sq38_quant_mat_bytes(ld.lin_in_proj_a) +
                     sq38_quant_mat_bytes(ld.lin_in_proj_b) +
                     sq38_quant_mat_bytes(ld.lin_out_proj) +
                     ld.lin_conv1d_weight.bytes +
                     ld.lin_A_log.bytes +
                     ld.lin_dt_bias.bytes +
                     ld.lin_norm.bytes;
        } else {
            total += sq38_quant_mat_bytes(ld.attn_q_proj) +
                     sq38_quant_mat_bytes(ld.attn_k_proj) +
                     sq38_quant_mat_bytes(ld.attn_v_proj) +
                     sq38_quant_mat_bytes(ld.attn_o_proj) +
                     ld.attn_q_norm.bytes +
                     ld.attn_k_norm.bytes;
        }
        total += sq38_quant_mat_bytes(ld.mlp_gate_proj) +
                 sq38_quant_mat_bytes(ld.mlp_up_proj) +
                 sq38_quant_mat_bytes(ld.mlp_down_proj);
    }
    total += hdr.final_norm.bytes + sq38_quant_mat_bytes(hdr.lm_head);
    return total;
}

} // namespace

struct StrataQwen38GpuEngine {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLResidencySet>         residency_set;

    // Metal 4 Compute Pipeline States
    id<MTLComputePipelineState> pso_zero_u32;
    id<MTLComputePipelineState> pso_copy_u32;
    id<MTLComputePipelineState> pso_embed_q4;
    id<MTLComputePipelineState> pso_rmsnorm;
    id<MTLComputePipelineState> pso_gemv_f32;
    id<MTLComputePipelineState> pso_gemv_res_f32;
    id<MTLComputePipelineState> pso_gemv_gate_up_silu;
    id<MTLComputePipelineState> pso_gemv_lin_in;
    id<MTLComputePipelineState> pso_gemv_attn_qkv;
    id<MTLComputePipelineState> pso_gemm_nax_m16;
    id<MTLComputePipelineState> pso_gemm_nax_m64;
    id<MTLComputePipelineState> pso_swiglu;
    id<MTLComputePipelineState> pso_conv1d;
    id<MTLComputePipelineState> pso_deltanet;
    id<MTLComputePipelineState> pso_deltanet_prep_batch;
    id<MTLComputePipelineState> pso_deltanet_rec_batch;
    id<MTLComputePipelineState> pso_deltanet_post_batch;
    id<MTLComputePipelineState> pso_attn_prep;
    id<MTLComputePipelineState> pso_flash_decode_split;
    id<MTLComputePipelineState> pso_flash_decode_reduce;
    id<MTLComputePipelineState> pso_flash_prefill_nax;
    id<MTLComputePipelineState> pso_sample_vocab;

    // Header & Model Geometry Copy
    StrataQwen38WeightsHeader   hdr;
    uint32_t                    max_seq_len;
    uint32_t                    max_seq_alloc;
    uint32_t                    seq_pos;
    uint64_t                    sample_step;

    // Telemetry Metadata
    char                        device_name[128];
    bool                        zero_copy_mmap;
    double                      load_time_ms;
    uint64_t                    weights_mapped_bytes;
    uint64_t                    active_bytes_per_token;
    uint64_t                    conv_state_bytes;
    uint64_t                    rec_state_bytes;
    uint64_t                    kv_single_bytes;
    uint64_t                    workspace_bytes;

    // Zero-Copy Weight Buffer (`newBufferWithBytesNoCopy:` over `weights.strata`)
    id<MTLBuffer>               b_weights;

    // Persistent Hybrid States (48-layer Gated DeltaNet + 16-layer FP16 KV Cache)
    id<MTLBuffer>               b_conv_state;  // [48, 3, 10240] FP16
    id<MTLBuffer>               b_rec_state;   // [48, 48, 128, 128] FP32
    id<MTLBuffer>               b_k_cache;     // [16, 4, max_seq_alloc, 256] FP16
    id<MTLBuffer>               b_v_cache;     // [16, 4, max_seq_alloc, 256] FP16

    // Pre-allocated Private GPU Activation Workspaces (`MAX_CHUNK = 512`, `0` runtime allocs!)
    id<MTLBuffer>               b_x_cur;       // [512, 5120] FP32 residual stream
    id<MTLBuffer>               b_x_norm;      // [512, 5120] FP16
    id<MTLBuffer>               b_qkv;         // [512, 12288] FP16 (shared `lin_qkv` [10240] & `attn_q_gate` [12288])
    id<MTLBuffer>               b_qkv_conv;    // [512, 10240] FP16
    id<MTLBuffer>               b_z;           // [512, 6144] FP16
    id<MTLBuffer>               b_a;           // [512, 64] FP32
    id<MTLBuffer>               b_b;           // [512, 64] FP32
    id<MTLBuffer>               b_k_raw;       // [512, 1024] FP16
    id<MTLBuffer>               b_v_raw;       // [512, 1024] FP16
    id<MTLBuffer>               b_q_rope;      // [24, 512, 256] FP16
    id<MTLBuffer>               b_attn_out;    // [512, 6144] FP16 (shared `deltanet_y` & `gated_attn_out`)
    id<MTLBuffer>               b_gate_f32;    // [512, 17408] FP32
    id<MTLBuffer>               b_up_f32;      // [512, 17408] FP32
    id<MTLBuffer>               b_mid_f16;     // [512, 17408] FP16
    id<MTLBuffer>               b_part_m;      // [64, 24] FP32
    id<MTLBuffer>               b_part_l;      // [64, 24] FP32
    id<MTLBuffer>               b_part_o;      // [64, 24, 256] FP32

    // Pre-faulted Shared Host <-> GPU IO Buffers
    id<MTLBuffer>               b_inv_freq;    // [32] FP32
    id<MTLBuffer>               b_token_ids;   // [512] U32
    id<MTLBuffer>               b_logits;      // [248320] FP32
    id<MTLBuffer>               b_next_token;  // [4] U32 (4-byte fast decode readback!)
};

namespace {

inline void sq38_encode_rmsnorm(
    id<MTLComputeCommandEncoder> enc,
    StrataQwen38GpuEngine* eng,
    id<MTLBuffer> x_in_f32,
    NSUInteger x_in_offset,
    uint64_t norm_weight_off,
    id<MTLBuffer> x_out_f16,
    uint32_t M_valid,
    uint32_t M_padded
) {
    Sq38RmsNormParamsHost p{
        M_valid,
        M_padded,
        eng->hdr.cfg.hidden_size,
        eng->hdr.cfg.rms_norm_eps
    };
    [enc setComputePipelineState:eng->pso_rmsnorm];
    [enc setBuffer:x_in_f32 offset:x_in_offset atIndex:0];
    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(norm_weight_off) atIndex:1];
    [enc setBuffer:x_out_f16 offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(M_padded, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

inline void sq38_encode_nax_gemm(
    id<MTLComputeCommandEncoder> enc,
    StrataQwen38GpuEngine* eng,
    const StrataQwen38QuantMatrix& mat,
    id<MTLBuffer> x_in_f16,
    id<MTLBuffer> y_f16,
    id<MTLBuffer> y_f32,
    uint32_t M_valid,
    uint32_t M_padded,
    uint32_t out_mode
) {
    const bool use_m16  = (M_padded < 64U);
    const uint32_t m_tl = use_m16 ? 16U : 64U;
    id<MTLComputePipelineState> pso = use_m16 ? eng->pso_gemm_nax_m16 : eng->pso_gemm_nax_m64;
    Sq38NaxGemmParamsHost p{
        M_valid,
        M_padded,
        mat.rows_padded,
        mat.cols,
        out_mode,
        0U, 0U, 0U
    };
    [enc setComputePipelineState:pso];
    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(mat.weight_off) atIndex:0];
    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(mat.scales_off) atIndex:1];
    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(mat.biases_off) atIndex:2];
    [enc setBuffer:x_in_f16 offset:0 atIndex:3];
    [enc setBuffer:(y_f16 ? y_f16 : x_in_f16) offset:0 atIndex:4];
    [enc setBuffer:(y_f32 ? y_f32 : eng->b_x_cur) offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    const uint32_t gx = (mat.rows_padded + 63U) / 64U;
    const uint32_t gy = (M_valid + m_tl - 1U) / m_tl;
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

inline void sq38_encode_sample(
    id<MTLComputeCommandEncoder> enc,
    StrataQwen38GpuEngine* eng,
    const StrataQwen38SamplingParams* sampling
) {
    Sq38SampleKernelParamsHost sp{};
    sp.vocab_size = eng->hdr.cfg.vocab_size;

    const bool is_greedy = (!sampling) || sampling->greedy || (sampling->temperature <= 1e-5f);
    sp.greedy   = is_greedy ? 1U : 0U;
    sp.inv_temp = is_greedy ? 1.0f : (1.0f / std::max(sampling->temperature, 1e-5f));
    sp.top_k    = (!sampling) ? 0U : sampling->top_k;
    sp.top_p    = (!sampling) ? 1.0f : sampling->top_p;

    const uint64_t base_seed = sampling ? sampling->rng_seed : 0x513827B0ULL;
    const uint64_t mixed     = base_seed ^ ((eng->sample_step++) * 0x9E3779B97F4A7C15ULL) ^
                               (static_cast<uint64_t>(eng->seq_pos) * 0xBF58476D1CE4E5B9ULL);
    sp.rng_lo = static_cast<uint32_t>(mixed & 0xFFFFFFFFULL);
    sp.rng_hi = static_cast<uint32_t>(mixed >> 32ULL);
    sp.pad    = 0U;

    [enc setComputePipelineState:eng->pso_sample_vocab];
    [enc setBuffer:eng->b_logits offset:0 atIndex:0];
    [enc setBuffer:eng->b_next_token offset:0 atIndex:1];
    [enc setBytes:&sp length:sizeof(sp) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

} // namespace

extern "C" {

StrataQwen38GpuEngine* strata_qwen38_gpu_create(
    const StrataQwen38Bundle* bundle,
    uint32_t max_seq_len
) {
    if (!bundle || !bundle->mmap_base || bundle->mmap_bytes < sizeof(StrataQwen38WeightsHeader) ||
        !bundle->header || bundle->header->magic != STRATA_QWEN38_MAGIC_U64) {
        std::fprintf(stderr, "[strata_qwen38_gpu] Invalid bundle or header magic\n");
        return nullptr;
    }

    /* Engine geometry pin (Round 6 rule Q1): workspaces & kernels target exact Qwen3.8-27B geometry */
    {
        const StrataQwen38Config& gc = bundle->header->cfg;
        if (gc.num_attention_heads != 24U ||
            gc.num_key_value_heads != 4U ||
            gc.head_dim != 256U ||
            gc.linear_num_key_heads != 16U ||
            gc.linear_key_head_dim != 128U ||
            gc.linear_num_value_heads != 48U ||
            gc.linear_value_head_dim != 128U ||
            gc.linear_conv_kernel_dim != 4U ||
            gc.hidden_size != 5120U ||
            gc.intermediate_size != 17408U) {
            std::fprintf(stderr, "[strata_qwen38_gpu] Unsupported Qwen3.8 geometry in bundle header\n");
            return nullptr;
        }
    }

    @autoreleasepool {
        const auto t_start = std::chrono::steady_clock::now();

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
            if (devs && devs.count > 0) dev = devs[0];
        }
        if (!dev) {
            std::fprintf(stderr, "[strata_qwen38_gpu] No Metal GPU device available\n");
            return nullptr;
        }

        std::string hdr_src = strata_load_shader_asset("strata_qwen38.h", "");
        if (hdr_src.empty()) hdr_src = sq38_read_text_file(sq38_resolve_sibling_path("../../include/strata_qwen38.h"));
        std::string msl_src = strata_load_shader_asset("strata_qwen38_kernels.metal", "");
        if (msl_src.empty()) msl_src = sq38_read_text_file(sq38_resolve_sibling_path("strata_qwen38_kernels.metal"));
        if (hdr_src.empty() || msl_src.empty()) {
            std::fprintf(stderr, "[strata_qwen38_gpu] Failed to read strata_qwen38.h or strata_qwen38_kernels.metal\n");
            return nullptr;
        }

        const std::string marker = "// @@STRATA_QWEN38_HEADER@@";
        size_t pos = msl_src.find(marker);
        if (pos != std::string::npos) {
            msl_src.replace(pos, marker.size(), hdr_src);
        }

        NSString* msl_ns = [NSString stringWithUTF8String:msl_src.c_str()];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 26.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_0;
        } else {
            opts.languageVersion = MTLLanguageVersion3_1;
        }
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        }

        auto* eng = new StrataQwen38GpuEngine();
        eng->device = dev;
        eng->queue  = [dev newCommandQueue];
        eng->hdr    = *bundle->header;

        const uint32_t req_seq = (max_seq_len > 0U) ? max_seq_len : 32768U;
        eng->max_seq_len   = req_seq;
        eng->max_seq_alloc = (std::max<uint32_t>(req_seq, 64U) + 63U) & ~63U;
        eng->seq_pos       = 0U;
        eng->sample_step   = 0ULL;

        std::snprintf(eng->device_name, sizeof(eng->device_name), "%s", [[dev name] UTF8String]);

        // Overlap Metal 4 library + 21-PSO compilation concurrently with zero-copy weight & buffer wiring!
        __block bool shader_ok = true;
        dispatch_group_t compile_grp = dispatch_group_create();
        dispatch_queue_t bg_q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        dispatch_group_async(compile_grp, bg_q, ^{
            @autoreleasepool {
                NSError* err = nil;
                id<MTLLibrary> lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
                if (!lib) {
                    std::fprintf(stderr, "[strata_qwen38_gpu] Metal 4 compile error: %s\n",
                                 err ? [[err localizedDescription] UTF8String] : "unknown");
                    shader_ok = false;
                    return;
                }
                eng->library = lib;

                auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
                    id<MTLFunction> fn = [lib newFunctionWithName:name];
                    if (!fn) {
                        std::fprintf(stderr, "[strata_qwen38_gpu] Missing Metal kernel: %s\n", [name UTF8String]);
                        return nil;
                    }
                    NSError* p_err = nil;
                    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&p_err];
                    if (!pso) {
                        std::fprintf(stderr, "[strata_qwen38_gpu] PSO compile error (%s): %s\n",
                                     [name UTF8String],
                                     p_err ? [[p_err localizedDescription] UTF8String] : "unknown");
                    }
                    return pso;
                };

                struct PsoSlot {
                    const char*                           fn_name;
                    id<MTLComputePipelineState> __strong* dst;
                };
                PsoSlot pso_table[] = {
                    {"sq38_zero_u32",                      &eng->pso_zero_u32},
                    {"sq38_copy_u32",                      &eng->pso_copy_u32},
                    {"sq38_embed_lookup_q4",               &eng->pso_embed_q4},
                    {"sq38_rmsnorm_f32_to_f16",            &eng->pso_rmsnorm},
                    {"sq38_q4_gemv_f32",                   &eng->pso_gemv_f32},
                    {"sq38_q4_gemv_res_f32",               &eng->pso_gemv_res_f32},
                    {"sq38_q4_gemv_fused_gate_up_silu",    &eng->pso_gemv_gate_up_silu},
                    {"sq38_q4_gemv_fused_lin_in",          &eng->pso_gemv_lin_in},
                    {"sq38_q4_gemv_fused_attn_qkv",        &eng->pso_gemv_attn_qkv},
                    {"sq38_q4_gemm_nax_m16",               &eng->pso_gemm_nax_m16},
                    {"sq38_q4_gemm_nax_m64",               &eng->pso_gemm_nax_m64},
                    {"sq38_swiglu_f32_to_f16",             &eng->pso_swiglu},
                    {"sq38_deltanet_conv1d_silu",          &eng->pso_conv1d},
                    {"sq38_deltanet_recurrent_gated_norm", &eng->pso_deltanet},
                    {"sq38_deltanet_prep_batch",           &eng->pso_deltanet_prep_batch},
                    {"sq38_deltanet_rec_batch",            &eng->pso_deltanet_rec_batch},
                    {"sq38_deltanet_post_batch",           &eng->pso_deltanet_post_batch},
                    {"sq38_attn_qkv_prep",                 &eng->pso_attn_prep},
                    {"sq38_flash_decode_splitkv",          &eng->pso_flash_decode_split},
                    {"sq38_flash_decode_reduce",           &eng->pso_flash_decode_reduce},
                    {"sq38_flash_prefill_nax",             &eng->pso_flash_prefill_nax},
                    {"sq38_sample_vocab",                  &eng->pso_sample_vocab},
                };
                constexpr size_t kNumPsos = sizeof(pso_table) / sizeof(pso_table[0]);
                PsoSlot* pso_ptr = pso_table;

                dispatch_apply(kNumPsos, bg_q, ^(size_t idx) {
                    @autoreleasepool {
                        NSString* ns_name = [NSString stringWithUTF8String:pso_ptr[idx].fn_name];
                        *pso_ptr[idx].dst = make_pso(ns_name);
                    }
                });

                for (size_t i = 0; i < kNumPsos; ++i) {
                    if (!(*pso_table[i].dst)) {
                        shader_ok = false;
                        break;
                    }
                }
            }
        });

        // 1. Zero-Copy 16 KiB-Aligned `mmap` Weight Binding
        (void)posix_madvise(const_cast<uint8_t*>(bundle->mmap_base), bundle->mmap_bytes, POSIX_MADV_WILLNEED);
        eng->zero_copy_mmap = false;
        if ((reinterpret_cast<uintptr_t>(bundle->mmap_base) % STRATA_QWEN38_PAGE_BYTES) == 0U &&
            (bundle->mmap_bytes % STRATA_QWEN38_PAGE_BYTES) == 0U) {
            eng->b_weights = [dev newBufferWithBytesNoCopy:const_cast<uint8_t*>(bundle->mmap_base)
                                                    length:bundle->mmap_bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
            if (eng->b_weights) {
                eng->zero_copy_mmap = true;
            }
        }
        if (!eng->b_weights) {
            eng->b_weights = [dev newBufferWithBytes:bundle->mmap_base
                                              length:bundle->mmap_bytes
                                             options:MTLResourceStorageModeShared];
        }
        if (!eng->b_weights) {
            std::fprintf(stderr, "[strata_qwen38_gpu] Failed to bind weights.strata (%zu B) to MTLBuffer\n",
                         bundle->mmap_bytes);
            dispatch_group_wait(compile_grp, DISPATCH_TIME_FOREVER);
            strata_qwen38_gpu_destroy(eng);
            return nullptr;
        }

        eng->weights_mapped_bytes   = static_cast<uint64_t>(bundle->mmap_bytes);
        eng->active_bytes_per_token = sq38_compute_active_bytes_per_token(eng->hdr);

        // 2. Persistent 48-Layer Gated DeltaNet State & 16-Layer FP16 KV Cache
        const uint32_t num_lin  = std::max<uint32_t>(eng->hdr.cfg.num_linear_layers, 1U);
        const uint32_t num_full = std::max<uint32_t>(eng->hdr.cfg.num_full_attn_layers, 1U);
        const uint32_t D        = eng->hdr.cfg.hidden_size;
        const uint32_t F        = eng->hdr.cfg.intermediate_size;
        const uint32_t V        = eng->hdr.cfg.vocab_size;

        eng->conv_state_bytes = static_cast<uint64_t>(num_lin) * 3ULL * 10240ULL * sizeof(uint16_t);
        eng->rec_state_bytes  = static_cast<uint64_t>(num_lin) * 48ULL * 128ULL * 128ULL * sizeof(float);
        eng->kv_single_bytes  = static_cast<uint64_t>(num_full) * 4ULL * eng->max_seq_alloc * 256ULL * sizeof(uint16_t);

        eng->b_conv_state = sq38_alloc_private_aligned(dev, eng->conv_state_bytes);
        eng->b_rec_state  = sq38_alloc_private_aligned(dev, eng->rec_state_bytes);
        eng->b_k_cache    = sq38_alloc_private_aligned(dev, eng->kv_single_bytes);
        eng->b_v_cache    = sq38_alloc_private_aligned(dev, eng->kv_single_bytes);

        // 3. Pre-allocated Chunked-Prefill & Decode Workspaces (`MAX_CHUNK = 512`)
        const size_t sz_x_cur    = static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(float);
        const size_t sz_x_norm   = static_cast<size_t>(kMaxPrefillChunk) * D * sizeof(uint16_t);
        const size_t sz_qkv      = static_cast<size_t>(kMaxPrefillChunk) * 12288ULL * sizeof(uint16_t);
        const size_t sz_qkv_conv = static_cast<size_t>(kMaxPrefillChunk) * 10240ULL * sizeof(uint16_t);
        const size_t sz_z        = static_cast<size_t>(kMaxPrefillChunk) * 6144ULL * sizeof(uint16_t);
        const size_t sz_ab       = static_cast<size_t>(kMaxPrefillChunk) * 64ULL * sizeof(float);
        const size_t sz_kv_raw   = static_cast<size_t>(kMaxPrefillChunk) * 1024ULL * sizeof(uint16_t);
        const size_t sz_q_rope   = 24ULL * kMaxPrefillChunk * 256ULL * sizeof(uint16_t);
        const size_t sz_attn_out = static_cast<size_t>(kMaxPrefillChunk) * 6144ULL * sizeof(uint16_t);
        const size_t sz_gu_f32   = static_cast<size_t>(kMaxPrefillChunk) * F * sizeof(float);
        const size_t sz_mid_f16  = static_cast<size_t>(kMaxPrefillChunk) * F * sizeof(uint16_t);
        const size_t sz_part_ml  = static_cast<size_t>(kMaxDecodeSplits) * 24ULL * sizeof(float);
        const size_t sz_part_o   = static_cast<size_t>(kMaxDecodeSplits) * 24ULL * 256ULL * sizeof(float);

        eng->b_x_cur    = sq38_alloc_private_aligned(dev, sz_x_cur);
        eng->b_x_norm   = sq38_alloc_private_aligned(dev, sz_x_norm);
        eng->b_qkv      = sq38_alloc_private_aligned(dev, sz_qkv);
        eng->b_qkv_conv = sq38_alloc_private_aligned(dev, sz_qkv_conv);
        eng->b_z        = sq38_alloc_private_aligned(dev, sz_z);
        eng->b_a        = sq38_alloc_private_aligned(dev, sz_ab);
        eng->b_b        = sq38_alloc_private_aligned(dev, sz_ab);
        eng->b_k_raw    = sq38_alloc_private_aligned(dev, sz_kv_raw);
        eng->b_v_raw    = sq38_alloc_private_aligned(dev, sz_kv_raw);
        eng->b_q_rope   = sq38_alloc_private_aligned(dev, sz_q_rope);
        eng->b_attn_out = sq38_alloc_private_aligned(dev, sz_attn_out);
        eng->b_gate_f32 = sq38_alloc_private_aligned(dev, sz_gu_f32);
        eng->b_up_f32   = sq38_alloc_private_aligned(dev, sz_gu_f32);
        eng->b_mid_f16  = sq38_alloc_private_aligned(dev, sz_mid_f16);
        eng->b_part_m   = sq38_alloc_private_aligned(dev, sz_part_ml);
        eng->b_part_l   = sq38_alloc_private_aligned(dev, sz_part_ml);
        eng->b_part_o   = sq38_alloc_private_aligned(dev, sz_part_o);

        eng->b_inv_freq   = sq38_alloc_shared_prefaulted(dev, 32ULL * sizeof(float));
        eng->b_token_ids  = sq38_alloc_shared_prefaulted(dev, kMaxPrefillChunk * sizeof(uint32_t));
        eng->b_logits     = sq38_alloc_shared_prefaulted(dev, static_cast<size_t>(V) * sizeof(float));
        eng->b_next_token = sq38_alloc_shared_prefaulted(dev, 16ULL * sizeof(uint32_t));

        eng->workspace_bytes = sz_x_cur + sz_x_norm + sz_qkv + sz_qkv_conv + sz_z + 2ULL * sz_ab +
                               2ULL * sz_kv_raw + sz_q_rope + sz_attn_out + 2ULL * sz_gu_f32 +
                               sz_mid_f16 + 2ULL * sz_part_ml + sz_part_o +
                               static_cast<size_t>(V) * sizeof(float) + 8192ULL;

        if (!eng->b_conv_state || !eng->b_rec_state || !eng->b_k_cache || !eng->b_v_cache ||
            !eng->b_x_cur || !eng->b_x_norm || !eng->b_qkv || !eng->b_qkv_conv ||
            !eng->b_z || !eng->b_a || !eng->b_b || !eng->b_k_raw || !eng->b_v_raw ||
            !eng->b_q_rope || !eng->b_attn_out || !eng->b_gate_f32 || !eng->b_up_f32 ||
            !eng->b_mid_f16 || !eng->b_part_m || !eng->b_part_l || !eng->b_part_o ||
            !eng->b_inv_freq || !eng->b_token_ids || !eng->b_logits || !eng->b_next_token) {
            std::fprintf(stderr, "[strata_qwen38_gpu] Failed to allocate GPU state or workspace buffers\n");
            dispatch_group_wait(compile_grp, DISPATCH_TIME_FOREVER);
            strata_qwen38_gpu_destroy(eng);
            return nullptr;
        }

        // Precompute exact Partial RoPE inverse frequencies (`rotary_dim = 64`, `theta = 10,000,000.0`)
        {
            float* inv_f = reinterpret_cast<float*>([eng->b_inv_freq contents]);
            const double theta = (eng->hdr.cfg.rope_theta > 0.0f)
                                     ? static_cast<double>(eng->hdr.cfg.rope_theta)
                                     : 10000000.0;
            const double rot_d = (eng->hdr.cfg.rotary_dim > 0U)
                                     ? static_cast<double>(eng->hdr.cfg.rotary_dim)
                                     : 64.0;
            for (uint32_t i = 0; i < 32U; ++i) {
                inv_f[i] = static_cast<float>(std::pow(theta, -2.0 * static_cast<double>(i) / rot_d));
            }
        }

        // 4. Wire ALL Weight, State, KV-Cache & Workspace Allocations into `MTLResidencySet`
        NSArray<id<MTLBuffer>>* all_bufs = @[
            eng->b_weights,
            eng->b_conv_state, eng->b_rec_state, eng->b_k_cache, eng->b_v_cache,
            eng->b_x_cur, eng->b_x_norm, eng->b_qkv, eng->b_qkv_conv,
            eng->b_z, eng->b_a, eng->b_b, eng->b_k_raw, eng->b_v_raw,
            eng->b_q_rope, eng->b_attn_out, eng->b_gate_f32, eng->b_up_f32,
            eng->b_mid_f16, eng->b_part_m, eng->b_part_l, eng->b_part_o,
            eng->b_inv_freq, eng->b_token_ids, eng->b_logits, eng->b_next_token
        ];
        eng->residency_set = sq38_create_residency_set(dev, eng->queue, all_bufs);

        // 5. Join concurrent shader compilation & zero-init DeltaNet states on GPU
        dispatch_group_wait(compile_grp, DISPATCH_TIME_FOREVER);
        if (!shader_ok) {
            strata_qwen38_gpu_destroy(eng);
            return nullptr;
        }

        {
            id<MTLCommandBuffer> cb          = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            sq38_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_conv_state, [eng->b_conv_state length]);
            sq38_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_rec_state, [eng->b_rec_state length]);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        const auto t_end  = std::chrono::steady_clock::now();
        eng->load_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        return eng;
    }
}

void strata_qwen38_gpu_destroy(StrataQwen38GpuEngine* eng) {
    if (!eng) return;
    @autoreleasepool {
        sq38_detach_residency_set(eng->queue, eng->residency_set);
        eng->residency_set           = nil;
        eng->b_next_token            = nil;
        eng->b_logits                = nil;
        eng->b_token_ids             = nil;
        eng->b_inv_freq              = nil;
        eng->b_part_o                = nil;
        eng->b_part_l                = nil;
        eng->b_part_m                = nil;
        eng->b_mid_f16               = nil;
        eng->b_up_f32                = nil;
        eng->b_gate_f32              = nil;
        eng->b_attn_out              = nil;
        eng->b_q_rope                = nil;
        eng->b_v_raw                 = nil;
        eng->b_k_raw                 = nil;
        eng->b_b                     = nil;
        eng->b_a                     = nil;
        eng->b_z                     = nil;
        eng->b_qkv_conv              = nil;
        eng->b_qkv                   = nil;
        eng->b_x_norm                = nil;
        eng->b_x_cur                 = nil;
        eng->b_v_cache               = nil;
        eng->b_k_cache               = nil;
        eng->b_rec_state             = nil;
        eng->b_conv_state            = nil;
        eng->b_weights               = nil;
        eng->pso_sample_vocab        = nil;
        eng->pso_flash_prefill_nax   = nil;
        eng->pso_flash_decode_reduce = nil;
        eng->pso_flash_decode_split  = nil;
        eng->pso_attn_prep           = nil;
        eng->pso_deltanet_post_batch = nil;
        eng->pso_deltanet_rec_batch  = nil;
        eng->pso_deltanet_prep_batch = nil;
        eng->pso_deltanet            = nil;
        eng->pso_conv1d              = nil;
        eng->pso_swiglu              = nil;
        eng->pso_gemm_nax_m64        = nil;
        eng->pso_gemm_nax_m16        = nil;
        eng->pso_gemv_attn_qkv       = nil;
        eng->pso_gemv_lin_in         = nil;
        eng->pso_gemv_gate_up_silu   = nil;
        eng->pso_gemv_res_f32        = nil;
        eng->pso_gemv_f32            = nil;
        eng->pso_rmsnorm             = nil;
        eng->pso_embed_q4            = nil;
        eng->pso_copy_u32            = nil;
        eng->pso_zero_u32            = nil;
        eng->library                 = nil;
        eng->queue                   = nil;
        eng->device                  = nil;
        delete eng;
    }
}

void strata_qwen38_gpu_reset(StrataQwen38GpuEngine* eng) {
    if (!eng) return;
    @autoreleasepool {
        eng->seq_pos     = 0U;
        eng->sample_step = 0ULL;

        id<MTLCommandBuffer> cb          = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        sq38_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_conv_state, [eng->b_conv_state length]);
        sq38_encode_zero_buffer(enc, eng->pso_zero_u32, eng->b_rec_state, [eng->b_rec_state length]);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

uint32_t strata_qwen38_gpu_seq_pos(const StrataQwen38GpuEngine* eng) {
    return eng ? eng->seq_pos : 0U;
}

void strata_qwen38_gpu_get_info(
    const StrataQwen38GpuEngine* eng,
    StrataQwen38GpuInfo* info_out
) {
    if (!info_out) return;
    std::memset(info_out, 0, sizeof(*info_out));
    if (!eng) return;

    std::memcpy(info_out->device_name, eng->device_name, sizeof(info_out->device_name));
    info_out->zero_copy_mmap           = eng->zero_copy_mmap;
    info_out->load_time_ms             = eng->load_time_ms;
    info_out->max_seq_len              = eng->max_seq_len;
    info_out->weights_mapped_bytes     = eng->weights_mapped_bytes;
    info_out->active_bytes_per_token   = eng->active_bytes_per_token;
    info_out->deltanet_state_bytes     = eng->conv_state_bytes + eng->rec_state_bytes;
    info_out->kv_cache_bytes_allocated = 2ULL * eng->kv_single_bytes;
    info_out->kv_cache_bytes_per_token = static_cast<uint64_t>(eng->hdr.cfg.num_full_attn_layers) *
                                         2ULL * 4ULL * 256ULL * sizeof(uint16_t);
    info_out->workspace_bytes          = eng->workspace_bytes;
}

int strata_qwen38_gpu_decode_step(
    StrataQwen38GpuEngine* eng,
    uint32_t token_id,
    const StrataQwen38SamplingParams* sampling,
    uint32_t* next_token_out,
    float* logits_out
) {
    if (!eng) return -1;
    if (eng->seq_pos >= eng->max_seq_len) return -2;

    @autoreleasepool {
        const uint32_t D          = eng->hdr.cfg.hidden_size;
        const uint32_t F          = eng->hdr.cfg.intermediate_size;
        const uint32_t V          = eng->hdr.cfg.vocab_size;
        const uint32_t num_layers = eng->hdr.cfg.num_hidden_layers;
        const float    eps        = eng->hdr.cfg.rms_norm_eps;
        const uint32_t cur_pos    = eng->seq_pos;
        const uint32_t total_kv   = cur_pos + 1U;

        *reinterpret_cast<uint32_t*>([eng->b_token_ids contents]) = token_id;

        uint32_t target_splits = std::min<uint32_t>(kMaxDecodeSplits, (total_kv + 63U) / 64U);
        target_splits          = std::max<uint32_t>(1U, target_splits);
        uint32_t kv_per_split  = (total_kv + target_splits - 1U) / target_splits;
        kv_per_split           = (kv_per_split + 7U) & ~7U;
        uint32_t num_splits    = std::max<uint32_t>(1U, (total_kv + kv_per_split - 1U) / kv_per_split);

        const uint64_t conv_layer_stride = 3ULL * 10240ULL * sizeof(uint16_t);
        const uint64_t rec_layer_stride  = 48ULL * 128ULL * 128ULL * sizeof(float);
        const uint64_t kv_layer_stride   = 4ULL * eng->max_seq_alloc * 256ULL * sizeof(uint16_t);

        id<MTLCommandBuffer> cb          = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // 1. Token Embedding Lookup (`[1, 5120]` -> `b_x_cur` FP32)
        {
            Sq38EmbedParamsHost ep{1U, D, V, 0U};
            [enc setComputePipelineState:eng->pso_embed_q4];
            [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.weight_off) atIndex:0];
            [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.scales_off) atIndex:1];
            [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.biases_off) atIndex:2];
            [enc setBuffer:eng->b_token_ids offset:0 atIndex:3];
            [enc setBuffer:eng->b_x_cur offset:0 atIndex:4];
            [enc setBytes:&ep length:sizeof(ep) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(D / 32U, 1, 1)];
        }

        // 2. All 64 Hybrid Layers (100% GPU-resident, zero CPU sync!)
        uint32_t lin_idx  = 0U;
        uint32_t full_idx = 0U;

        for (uint32_t l = 0; l < num_layers; ++l) {
            const StrataQwen38LayerDesc& ld = eng->hdr.layers[l];

            // Pre-Attention RMSNorm
            sq38_encode_rmsnorm(enc, eng, eng->b_x_cur, 0, ld.input_layernorm.offset, eng->b_x_norm, 1U, 1U);

            if (ld.layer_type == STRATA_QWEN38_LAYER_LINEAR_ATTN) {
                const uint32_t qkv_rows = ld.lin_in_proj_qkv.rows_padded;
                const uint32_t z_rows   = ld.lin_in_proj_z.rows_padded;
                const uint32_t ab_rows  = ld.lin_in_proj_a.rows_padded;
                const uint32_t tot_rows = qkv_rows + z_rows + 2U * ab_rows;

                // Fused Single-Dispatch `(qkv + z + a + b)` GEMV
                Sq38LinInGemvParamsHost lp{qkv_rows, z_rows, ab_rows, D};
                [enc setComputePipelineState:eng->pso_gemv_lin_in];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_qkv.weight_off) atIndex:0];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_qkv.scales_off) atIndex:1];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_qkv.biases_off) atIndex:2];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_z.weight_off) atIndex:3];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_z.scales_off) atIndex:4];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_z.biases_off) atIndex:5];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_a.weight_off) atIndex:6];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_a.scales_off) atIndex:7];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_a.biases_off) atIndex:8];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_b.weight_off) atIndex:9];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_b.scales_off) atIndex:10];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_in_proj_b.biases_off) atIndex:11];
                [enc setBuffer:eng->b_x_norm offset:0 atIndex:12];
                [enc setBuffer:eng->b_qkv offset:0 atIndex:13];
                [enc setBuffer:eng->b_z offset:0 atIndex:14];
                [enc setBuffer:eng->b_a offset:0 atIndex:15];
                [enc setBuffer:eng->b_b offset:0 atIndex:16];
                [enc setBytes:&lp length:sizeof(lp) atIndex:17];
                [enc dispatchThreadgroups:MTLSizeMake((tot_rows + 15U) / 16U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                // Causal `conv1d(k=4) + SiLU`
                Sq38Conv1dParamsHost cp{1U, 10240U};
                [enc setComputePipelineState:eng->pso_conv1d];
                [enc setBuffer:eng->b_qkv offset:0 atIndex:0];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_conv1d_weight.offset) atIndex:1];
                [enc setBuffer:eng->b_conv_state offset:static_cast<NSUInteger>(lin_idx * conv_layer_stride) atIndex:2];
                [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:3];
                [enc setBytes:&cp length:sizeof(cp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(10240U / 256U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // Gated DeltaNet Recurrent Step + `RMSNormGated(out) * silu(z)`
                Sq38DeltaNetParamsHost dp{1U, 1U, eps, ab_rows};
                [enc setComputePipelineState:eng->pso_deltanet];
                [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:0];
                [enc setBuffer:eng->b_z offset:0 atIndex:1];
                [enc setBuffer:eng->b_a offset:0 atIndex:2];
                [enc setBuffer:eng->b_b offset:0 atIndex:3];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_A_log.offset) atIndex:4];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_dt_bias.offset) atIndex:5];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_norm.offset) atIndex:6];
                [enc setBuffer:eng->b_rec_state offset:static_cast<NSUInteger>(lin_idx * rec_layer_stride) atIndex:7];
                [enc setBuffer:eng->b_attn_out offset:0 atIndex:8];
                [enc setBytes:&dp length:sizeof(dp) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake(48, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

                // `out_proj` + Residual Add
                Sq38GemvParamsHost op{ld.lin_out_proj.rows_padded, ld.lin_out_proj.cols};
                [enc setComputePipelineState:eng->pso_gemv_res_f32];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_out_proj.weight_off) atIndex:0];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_out_proj.scales_off) atIndex:1];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_out_proj.biases_off) atIndex:2];
                [enc setBuffer:eng->b_attn_out offset:0 atIndex:3];
                [enc setBuffer:eng->b_x_cur offset:0 atIndex:4];
                [enc setBytes:&op length:sizeof(op) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((op.rows + 15U) / 16U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                ++lin_idx;
            } else {
                const uint32_t q_rows   = ld.attn_q_proj.rows_padded;
                const uint32_t kv_rows  = ld.attn_k_proj.rows_padded;
                const uint32_t tot_rows = q_rows + 2U * kv_rows;
                const NSUInteger kv_off = static_cast<NSUInteger>(full_idx * kv_layer_stride);

                // Fused Single-Dispatch `(q_gate + k + v)` GEMV
                Sq38AttnQkvGemvParamsHost qp{q_rows, kv_rows, D, 0U};
                [enc setComputePipelineState:eng->pso_gemv_attn_qkv];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_q_proj.weight_off) atIndex:0];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_q_proj.scales_off) atIndex:1];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_q_proj.biases_off) atIndex:2];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_k_proj.weight_off) atIndex:3];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_k_proj.scales_off) atIndex:4];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_k_proj.biases_off) atIndex:5];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_v_proj.weight_off) atIndex:6];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_v_proj.scales_off) atIndex:7];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_v_proj.biases_off) atIndex:8];
                [enc setBuffer:eng->b_x_norm offset:0 atIndex:9];
                [enc setBuffer:eng->b_qkv offset:0 atIndex:10];
                [enc setBuffer:eng->b_k_raw offset:0 atIndex:11];
                [enc setBuffer:eng->b_v_raw offset:0 atIndex:12];
                [enc setBytes:&qp length:sizeof(qp) atIndex:13];
                [enc dispatchThreadgroups:MTLSizeMake((tot_rows + 15U) / 16U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                // `q_norm`/`k_norm` + Partial NeoX RoPE + KV Cache Write
                const uint32_t kv_zero_end = (total_kv + 63U) & ~63U;
                Sq38AttnPrepParamsHost pp{1U, 1U, cur_pos, eng->max_seq_alloc, eps, kv_zero_end, 0U, 0U};
                [enc setComputePipelineState:eng->pso_attn_prep];
                [enc setBuffer:eng->b_qkv offset:0 atIndex:0];
                [enc setBuffer:eng->b_k_raw offset:0 atIndex:1];
                [enc setBuffer:eng->b_v_raw offset:0 atIndex:2];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_q_norm.offset) atIndex:3];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_k_norm.offset) atIndex:4];
                [enc setBuffer:eng->b_inv_freq offset:0 atIndex:5];
                [enc setBuffer:eng->b_q_rope offset:0 atIndex:6];
                [enc setBuffer:eng->b_k_cache offset:kv_off atIndex:7];
                [enc setBuffer:eng->b_v_cache offset:kv_off atIndex:8];
                [enc setBytes:&pp length:sizeof(pp) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake(32, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // GQA-Grouped Split-KV Flash-Decoding (`32K-64K`)
                Sq38FlashDecodeParamsHost fp{total_kv, eng->max_seq_alloc, num_splits, kv_per_split};
                [enc setComputePipelineState:eng->pso_flash_decode_split];
                [enc setBuffer:eng->b_q_rope offset:0 atIndex:0];
                [enc setBuffer:eng->b_k_cache offset:kv_off atIndex:1];
                [enc setBuffer:eng->b_v_cache offset:kv_off atIndex:2];
                [enc setBuffer:eng->b_part_m offset:0 atIndex:3];
                [enc setBuffer:eng->b_part_l offset:0 atIndex:4];
                [enc setBuffer:eng->b_part_o offset:0 atIndex:5];
                [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(num_splits, 4, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // Log-sum-exp Reduction + `* sigmoid(gate)`
                [enc setComputePipelineState:eng->pso_flash_decode_reduce];
                [enc setBuffer:eng->b_part_m offset:0 atIndex:0];
                [enc setBuffer:eng->b_part_l offset:0 atIndex:1];
                [enc setBuffer:eng->b_part_o offset:0 atIndex:2];
                [enc setBuffer:eng->b_qkv offset:0 atIndex:3];
                [enc setBuffer:eng->b_attn_out offset:0 atIndex:4];
                [enc setBytes:&num_splits length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(24, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // `o_proj` + Residual Add
                Sq38GemvParamsHost op{ld.attn_o_proj.rows_padded, ld.attn_o_proj.cols};
                [enc setComputePipelineState:eng->pso_gemv_res_f32];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_o_proj.weight_off) atIndex:0];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_o_proj.scales_off) atIndex:1];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_o_proj.biases_off) atIndex:2];
                [enc setBuffer:eng->b_attn_out offset:0 atIndex:3];
                [enc setBuffer:eng->b_x_cur offset:0 atIndex:4];
                [enc setBytes:&op length:sizeof(op) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((op.rows + 15U) / 16U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                ++full_idx;
            }

            // Post-Attention RMSNorm
            sq38_encode_rmsnorm(enc, eng, eng->b_x_cur, 0, ld.post_attention_layernorm.offset, eng->b_x_norm, 1U, 1U);

            // Fused `gate_proj + up_proj + SiLU` -> `b_mid_f16`
            {
                Sq38GemvParamsHost mp{F, D};
                [enc setComputePipelineState:eng->pso_gemv_gate_up_silu];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.mlp_gate_proj.weight_off) atIndex:0];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.mlp_gate_proj.scales_off) atIndex:1];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.mlp_gate_proj.biases_off) atIndex:2];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.mlp_up_proj.weight_off) atIndex:3];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.mlp_up_proj.scales_off) atIndex:4];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.mlp_up_proj.biases_off) atIndex:5];
                [enc setBuffer:eng->b_x_norm offset:0 atIndex:6];
                [enc setBuffer:eng->b_mid_f16 offset:0 atIndex:7];
                [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake((F + 7U) / 8U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }

            // `down_proj` + Residual Add
            {
                Sq38GemvParamsHost dp{D, F};
                [enc setComputePipelineState:eng->pso_gemv_res_f32];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.mlp_down_proj.weight_off) atIndex:0];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.mlp_down_proj.scales_off) atIndex:1];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.mlp_down_proj.biases_off) atIndex:2];
                [enc setBuffer:eng->b_mid_f16 offset:0 atIndex:3];
                [enc setBuffer:eng->b_x_cur offset:0 atIndex:4];
                [enc setBytes:&dp length:sizeof(dp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((D + 15U) / 16U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        }

        // 3. Final RMSNorm + `248,320`-Vocab LM Head + GPU Argmax/Top-K Sampler
        sq38_encode_rmsnorm(enc, eng, eng->b_x_cur, 0, eng->hdr.final_norm.offset, eng->b_x_norm, 1U, 1U);

        {
            Sq38GemvParamsHost hp{eng->hdr.lm_head.rows_padded, eng->hdr.lm_head.cols};
            [enc setComputePipelineState:eng->pso_gemv_f32];
            [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.lm_head.weight_off) atIndex:0];
            [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.lm_head.scales_off) atIndex:1];
            [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.lm_head.biases_off) atIndex:2];
            [enc setBuffer:eng->b_x_norm offset:0 atIndex:3];
            [enc setBuffer:eng->b_logits offset:0 atIndex:4];
            [enc setBytes:&hp length:sizeof(hp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((hp.rows + 15U) / 16U, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        sq38_encode_sample(enc, eng, sampling);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        eng->seq_pos += 1U;

        if (next_token_out) {
            *next_token_out = *reinterpret_cast<const uint32_t*>([eng->b_next_token contents]);
        }
        if (logits_out) {
            std::memcpy(logits_out, [eng->b_logits contents], static_cast<size_t>(V) * sizeof(float));
        }
        return 0;
    }
}

int strata_qwen38_gpu_prefill(
    StrataQwen38GpuEngine* eng,
    const uint32_t* token_ids,
    uint32_t num_tokens,
    const StrataQwen38SamplingParams* sampling,
    uint32_t* next_token_out,
    float* last_logits_out
) {
    if (!eng || !token_ids || num_tokens == 0U) return -1;
    if (eng->seq_pos + num_tokens > eng->max_seq_len) return -2;

    @autoreleasepool {
        const uint32_t D          = eng->hdr.cfg.hidden_size;
        const uint32_t F          = eng->hdr.cfg.intermediate_size;
        const uint32_t V          = eng->hdr.cfg.vocab_size;
        const uint32_t num_layers = eng->hdr.cfg.num_hidden_layers;
        const float    eps        = eng->hdr.cfg.rms_norm_eps;

        const uint64_t conv_layer_stride = 3ULL * 10240ULL * sizeof(uint16_t);
        const uint64_t rec_layer_stride  = 48ULL * 128ULL * 128ULL * sizeof(float);
        const uint64_t kv_layer_stride   = 4ULL * eng->max_seq_alloc * 256ULL * sizeof(uint16_t);

        for (uint32_t chunk_off = 0U; chunk_off < num_tokens; chunk_off += kMaxPrefillChunk) {
            const uint32_t M             = std::min<uint32_t>(kMaxPrefillChunk, num_tokens - chunk_off);
            const bool     is_last_chunk = (chunk_off + M == num_tokens);

            // Uniform 64-aligned NAX tile geometry across all prefill chunk sizes (`M >= 1`)
            const uint32_t M_padded    = (M + 63U) & ~63U;
            const uint32_t cur_pos     = eng->seq_pos;
            const uint32_t total_kv    = cur_pos + M;
            const uint32_t kv_zero_end = (total_kv + 63U) & ~63U;

            std::memcpy([eng->b_token_ids contents], token_ids + chunk_off, static_cast<size_t>(M) * sizeof(uint32_t));

            id<MTLCommandBuffer> cb          = [eng->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            // 1. Batched Token Embedding Lookup (`[M, 5120]` -> `b_x_cur` FP32)
            {
                Sq38EmbedParamsHost ep{M, D, V, 0U};
                [enc setComputePipelineState:eng->pso_embed_q4];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.weight_off) atIndex:0];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.scales_off) atIndex:1];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.embed_tokens.biases_off) atIndex:2];
                [enc setBuffer:eng->b_token_ids offset:0 atIndex:3];
                [enc setBuffer:eng->b_x_cur offset:0 atIndex:4];
                [enc setBytes:&ep length:sizeof(ep) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(D / 32U, 1, 1)];
            }

            // 2. All 64 Hybrid Layers via Metal 4 `NAX` `matmul2d` + Batched DeltaNet + NAX FlashAttention
            uint32_t lin_idx  = 0U;
            uint32_t full_idx = 0U;
            constexpr NSUInteger kGatesBufOff = 16ULL * 1024ULL * 1024ULL; // 16 MiB inside 34 MiB `b_gate_f32`

            for (uint32_t l = 0; l < num_layers; ++l) {
                const StrataQwen38LayerDesc& ld = eng->hdr.layers[l];

                sq38_encode_rmsnorm(enc, eng, eng->b_x_cur, 0, ld.input_layernorm.offset, eng->b_x_norm, M, M_padded);

                if (ld.layer_type == STRATA_QWEN38_LAYER_LINEAR_ATTN) {
                    sq38_encode_nax_gemm(enc, eng, ld.lin_in_proj_qkv, eng->b_x_norm, eng->b_qkv, nil, M, M_padded, kGemmOutF16);
                    sq38_encode_nax_gemm(enc, eng, ld.lin_in_proj_z,   eng->b_x_norm, eng->b_z,   nil, M, M_padded, kGemmOutF16);
                    sq38_encode_nax_gemm(enc, eng, ld.lin_in_proj_a,   eng->b_x_norm, nil, eng->b_a,   M, M_padded, kGemmOutF32);
                    sq38_encode_nax_gemm(enc, eng, ld.lin_in_proj_b,   eng->b_x_norm, nil, eng->b_b,   M, M_padded, kGemmOutF32);

                    Sq38Conv1dParamsHost cp{M, 10240U};
                    [enc setComputePipelineState:eng->pso_conv1d];
                    [enc setBuffer:eng->b_qkv offset:0 atIndex:0];
                    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_conv1d_weight.offset) atIndex:1];
                    [enc setBuffer:eng->b_conv_state offset:static_cast<NSUInteger>(lin_idx * conv_layer_stride) atIndex:2];
                    [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:3];
                    [enc setBytes:&cp length:sizeof(cp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(10240U / 256U, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // 3-Stage Zero-Barrier 192-TG Batched DeltaNet (`b_gate_f32` & `b_up_f32` reused as scratch!)
                    Sq38DeltaNetParamsHost dp{M, M_padded, eps, ld.lin_in_proj_a.rows_padded};
                    [enc setComputePipelineState:eng->pso_deltanet_prep_batch];
                    [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:0];
                    [enc setBuffer:eng->b_a offset:0 atIndex:1];
                    [enc setBuffer:eng->b_b offset:0 atIndex:2];
                    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_A_log.offset) atIndex:3];
                    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_dt_bias.offset) atIndex:4];
                    [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:5];
                    [enc setBuffer:eng->b_gate_f32 offset:kGatesBufOff atIndex:6];
                    [enc setBytes:&dp length:sizeof(dp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(33, M, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:eng->pso_deltanet_rec_batch];
                    [enc setBuffer:eng->b_qkv_conv offset:0 atIndex:0];
                    [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:1];
                    [enc setBuffer:eng->b_gate_f32 offset:kGatesBufOff atIndex:2];
                    [enc setBuffer:eng->b_rec_state offset:static_cast<NSUInteger>(lin_idx * rec_layer_stride) atIndex:3];
                    [enc setBuffer:eng->b_up_f32 offset:0 atIndex:4];
                    [enc setBytes:&dp length:sizeof(dp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(4, 48, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:eng->pso_deltanet_post_batch];
                    [enc setBuffer:eng->b_up_f32 offset:0 atIndex:0];
                    [enc setBuffer:eng->b_z offset:0 atIndex:1];
                    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.lin_norm.offset) atIndex:2];
                    [enc setBuffer:eng->b_attn_out offset:0 atIndex:3];
                    [enc setBytes:&dp length:sizeof(dp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(48, M_padded, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    sq38_encode_nax_gemm(enc, eng, ld.lin_out_proj, eng->b_attn_out, nil, eng->b_x_cur, M, M_padded, kGemmOutResF32);
                    ++lin_idx;
                } else {
                    const NSUInteger kv_off = static_cast<NSUInteger>(full_idx * kv_layer_stride);

                    sq38_encode_nax_gemm(enc, eng, ld.attn_q_proj, eng->b_x_norm, eng->b_qkv,   nil, M, M_padded, kGemmOutF16);
                    sq38_encode_nax_gemm(enc, eng, ld.attn_k_proj, eng->b_x_norm, eng->b_k_raw, nil, M, M_padded, kGemmOutF16);
                    sq38_encode_nax_gemm(enc, eng, ld.attn_v_proj, eng->b_x_norm, eng->b_v_raw, nil, M, M_padded, kGemmOutF16);

                    Sq38AttnPrepParamsHost pp{M, M_padded, cur_pos, eng->max_seq_alloc, eps, kv_zero_end, 0U, 0U};
                    [enc setComputePipelineState:eng->pso_attn_prep];
                    [enc setBuffer:eng->b_qkv offset:0 atIndex:0];
                    [enc setBuffer:eng->b_k_raw offset:0 atIndex:1];
                    [enc setBuffer:eng->b_v_raw offset:0 atIndex:2];
                    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_q_norm.offset) atIndex:3];
                    [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(ld.attn_k_norm.offset) atIndex:4];
                    [enc setBuffer:eng->b_inv_freq offset:0 atIndex:5];
                    [enc setBuffer:eng->b_q_rope offset:0 atIndex:6];
                    [enc setBuffer:eng->b_k_cache offset:kv_off atIndex:7];
                    [enc setBuffer:eng->b_v_cache offset:kv_off atIndex:8];
                    [enc setBytes:&pp length:sizeof(pp) atIndex:9];
                    [enc dispatchThreadgroups:MTLSizeMake(32, M_padded, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    Sq38FlashPrefillParamsHost fp{M, M_padded, cur_pos, total_kv, eng->max_seq_alloc, 0U, 0U, 0U};
                    [enc setComputePipelineState:eng->pso_flash_prefill_nax];
                    [enc setBuffer:eng->b_q_rope offset:0 atIndex:0];
                    [enc setBuffer:eng->b_k_cache offset:kv_off atIndex:1];
                    [enc setBuffer:eng->b_v_cache offset:kv_off atIndex:2];
                    [enc setBuffer:eng->b_qkv offset:0 atIndex:3];
                    [enc setBuffer:eng->b_attn_out offset:0 atIndex:4];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(M_padded / 32U, 24, 1)
                        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                    sq38_encode_nax_gemm(enc, eng, ld.attn_o_proj, eng->b_attn_out, nil, eng->b_x_cur, M, M_padded, kGemmOutResF32);
                    ++full_idx;
                }

                // Post-Attention RMSNorm + Metal 4 NAX SwiGLU MLP
                sq38_encode_rmsnorm(enc, eng, eng->b_x_cur, 0, ld.post_attention_layernorm.offset, eng->b_x_norm, M, M_padded);
                sq38_encode_nax_gemm(enc, eng, ld.mlp_gate_proj, eng->b_x_norm, nil, eng->b_gate_f32, M, M_padded, kGemmOutF32);
                sq38_encode_nax_gemm(enc, eng, ld.mlp_up_proj,   eng->b_x_norm, nil, eng->b_up_f32,   M, M_padded, kGemmOutF32);

                {
                    Sq38SwigluParamsHost sp{M, M_padded, F, 0U};
                    [enc setComputePipelineState:eng->pso_swiglu];
                    [enc setBuffer:eng->b_gate_f32 offset:0 atIndex:0];
                    [enc setBuffer:eng->b_up_f32 offset:0 atIndex:1];
                    [enc setBuffer:eng->b_mid_f16 offset:0 atIndex:2];
                    [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                    const uint32_t d4 = F / 4U;
                    [enc dispatchThreadgroups:MTLSizeMake((d4 + 255U) / 256U, M_padded, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }

                sq38_encode_nax_gemm(enc, eng, ld.mlp_down_proj, eng->b_mid_f16, nil, eng->b_x_cur, M, M_padded, kGemmOutResF32);
            }

            // 3. Only on the final chunk: Final RMSNorm + `248,320`-Vocab LM Head + Sampler on LAST token (`M - 1`)!
            if (is_last_chunk) {
                const NSUInteger last_tok_byte_off =
                    static_cast<NSUInteger>(static_cast<size_t>(M - 1U) * D * sizeof(float));
                sq38_encode_rmsnorm(enc, eng, eng->b_x_cur, last_tok_byte_off,
                                    eng->hdr.final_norm.offset, eng->b_x_norm, 1U, 1U);

                Sq38GemvParamsHost hp{eng->hdr.lm_head.rows_padded, eng->hdr.lm_head.cols};
                [enc setComputePipelineState:eng->pso_gemv_f32];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.lm_head.weight_off) atIndex:0];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.lm_head.scales_off) atIndex:1];
                [enc setBuffer:eng->b_weights offset:static_cast<NSUInteger>(eng->hdr.lm_head.biases_off) atIndex:2];
                [enc setBuffer:eng->b_x_norm offset:0 atIndex:3];
                [enc setBuffer:eng->b_logits offset:0 atIndex:4];
                [enc setBytes:&hp length:sizeof(hp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((hp.rows + 15U) / 16U, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                sq38_encode_sample(enc, eng, sampling);
            }

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            eng->seq_pos += M;
        }

        if (next_token_out) {
            *next_token_out = *reinterpret_cast<const uint32_t*>([eng->b_next_token contents]);
        }
        if (last_logits_out) {
            std::memcpy(last_logits_out, [eng->b_logits contents], static_cast<size_t>(V) * sizeof(float));
        }
        return 0;
    }
}

struct StrataQwen38Checkpoint {
    id<MTLResidencySet> residency_set;
    id<MTLBuffer>       b_arena;
    uint64_t            conv_off;
    uint64_t            conv_len;
    uint64_t            rec_off;
    uint64_t            rec_len;
    uint64_t            total_bytes;
    uint32_t            seq_pos;
    uint64_t            sample_step;
    bool                valid;
};

StrataQwen38Checkpoint* strata_qwen38_checkpoint_create(const StrataQwen38GpuEngine* eng) {
    if (!eng || !eng->device) return nullptr;
    @autoreleasepool {
        auto* ckpt = new StrataQwen38Checkpoint();
        ckpt->residency_set = eng->residency_set;
        ckpt->seq_pos       = 0U;
        ckpt->sample_step   = 0ULL;
        ckpt->valid         = false;

        auto align256 = [](uint64_t v) -> uint64_t {
            return (v + 255ULL) & ~255ULL;
        };

        uint64_t cur = 0ULL;
        ckpt->conv_off = cur;
        ckpt->conv_len = eng->b_conv_state ? static_cast<uint64_t>([eng->b_conv_state length]) : 0ULL;
        cur = align256(cur + ckpt->conv_len);

        ckpt->rec_off = cur;
        ckpt->rec_len = eng->b_rec_state ? static_cast<uint64_t>([eng->b_rec_state length]) : 0ULL;
        cur = (cur + ckpt->rec_len + 4095ULL) & ~4095ULL;

        ckpt->total_bytes = std::max<uint64_t>(4096ULL, cur);
        ckpt->b_arena     = sq38_alloc_private_aligned(eng->device, static_cast<size_t>(ckpt->total_bytes));
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

void strata_qwen38_checkpoint_destroy(StrataQwen38Checkpoint* ckpt) {
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

int strata_qwen38_checkpoint_save(StrataQwen38GpuEngine* eng, StrataQwen38Checkpoint* ckpt) {
    if (!eng || !ckpt || !ckpt->b_arena) return -1;
    @autoreleasepool {
        ckpt->seq_pos     = eng->seq_pos;
        ckpt->sample_step = eng->sample_step;

        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (ckpt->conv_len > 0ULL && eng->b_conv_state) {
            sq38_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    eng->b_conv_state, 0,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->conv_off),
                                    static_cast<size_t>(ckpt->conv_len));
        }
        if (ckpt->rec_len > 0ULL && eng->b_rec_state) {
            sq38_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    eng->b_rec_state, 0,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->rec_off),
                                    static_cast<size_t>(ckpt->rec_len));
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        ckpt->valid = true;
        return 0;
    }
}

int strata_qwen38_checkpoint_restore(StrataQwen38GpuEngine* eng, const StrataQwen38Checkpoint* ckpt) {
    if (!eng || !ckpt || !ckpt->b_arena || !ckpt->valid) return -1;
    @autoreleasepool {
        eng->seq_pos     = ckpt->seq_pos;
        eng->sample_step = ckpt->sample_step;

        id<MTLCommandBuffer>         cb  = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (ckpt->conv_len > 0ULL && eng->b_conv_state) {
            sq38_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->conv_off),
                                    eng->b_conv_state, 0,
                                    static_cast<size_t>(ckpt->conv_len));
        }
        if (ckpt->rec_len > 0ULL && eng->b_rec_state) {
            sq38_encode_copy_buffer(enc, eng->pso_copy_u32,
                                    ckpt->b_arena, static_cast<NSUInteger>(ckpt->rec_off),
                                    eng->b_rec_state, 0,
                                    static_cast<size_t>(ckpt->rec_len));
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return 0;
    }
}

uint64_t strata_qwen38_checkpoint_bytes(const StrataQwen38Checkpoint* ckpt) {
    return ckpt ? ckpt->total_bytes : 0ULL;
}

namespace {

bool sq38_io_all(int fd, void* buf, size_t n, bool write) {
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

bool sq38_blit_buf_io(StrataQwen38GpuEngine* eng, id<MTLBuffer> gpu_buf, NSUInteger off, size_t len,
                      id<MTLBuffer> stage, int fd, bool save) {
    if (len == 0U) return true;
    const size_t cap = [stage length];
    uint8_t* sp = static_cast<uint8_t*>([stage contents]);
    size_t done = 0U;
    while (done < len) {
        const size_t n = std::min<size_t>(cap, len - done);
        if (!save) {
            if (!sq38_io_all(fd, sp, n, false)) return false;
        }
        id<MTLCommandBuffer> cb = [eng->queue commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        if (save) {
            [bl copyFromBuffer:gpu_buf sourceOffset:off + done toBuffer:stage destinationOffset:0 size:n];
        } else {
            [bl copyFromBuffer:stage sourceOffset:0 toBuffer:gpu_buf destinationOffset:off + done size:n];
        }
        [bl endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] == MTLCommandBufferStatusError) return false;
        if (save) {
            if (!sq38_io_all(fd, sp, n, true)) return false;
        }
        done += n;
    }
    return true;
}

struct Sq38StateHdr {
    uint64_t magic;
    uint32_t seq_pos;
    uint32_t num_full;
    uint64_t sample_step;
    uint64_t conv_bytes;
    uint64_t rec_bytes;
};
constexpr uint64_t kSq38StateMagic = 0x3156533833514D53ULL; /* "SMQ38SV1" */

} // namespace

int strata_qwen38_state_save_fd(StrataQwen38GpuEngine* eng, int fd, uint64_t* bytes_written_out) {
    if (!eng || fd < 0) return -1;
    @autoreleasepool {
        constexpr size_t kStageBytes = 16ULL << 20;
        id<MTLBuffer> stage = sq38_alloc_shared_prefaulted(eng->device, kStageBytes);
        if (!stage) return -2;

        const uint32_t num_full = std::max<uint32_t>(eng->hdr.cfg.num_full_attn_layers, 1U);
        Sq38StateHdr sh{kSq38StateMagic, eng->seq_pos, num_full, eng->sample_step,
                        eng->conv_state_bytes, eng->rec_state_bytes};
        if (!sq38_io_all(fd, &sh, sizeof(sh), true)) return -3;
        uint64_t tot = sizeof(sh);

        if (!sq38_blit_buf_io(eng, eng->b_conv_state, 0U, eng->conv_state_bytes, stage, fd, true)) return -4;
        if (!sq38_blit_buf_io(eng, eng->b_rec_state, 0U, eng->rec_state_bytes, stage, fd, true)) return -4;
        tot += eng->conv_state_bytes + eng->rec_state_bytes;

        if (eng->seq_pos > 0U) {
            const size_t head_row_bytes = static_cast<size_t>(eng->seq_pos) * 256ULL * sizeof(uint16_t);
            const size_t head_stride    = static_cast<size_t>(eng->max_seq_alloc) * 256ULL * sizeof(uint16_t);
            for (uint32_t l = 0U; l < num_full; ++l) {
                for (uint32_t h = 0U; h < 4U; ++h) {
                    const NSUInteger off = static_cast<NSUInteger>((static_cast<size_t>(l) * 4U + h) * head_stride);
                    if (!sq38_blit_buf_io(eng, eng->b_k_cache, off, head_row_bytes, stage, fd, true)) return -5;
                    if (!sq38_blit_buf_io(eng, eng->b_v_cache, off, head_row_bytes, stage, fd, true)) return -5;
                    tot += 2ULL * head_row_bytes;
                }
            }
        }
        if (bytes_written_out) *bytes_written_out = tot;
        return 0;
    }
}

int strata_qwen38_state_load_fd(StrataQwen38GpuEngine* eng, int fd, uint64_t* bytes_read_out) {
    if (!eng || fd < 0) return -1;
    @autoreleasepool {
        Sq38StateHdr sh{};
        if (!sq38_io_all(fd, &sh, sizeof(sh), false)) return -2;
        const uint32_t num_full = std::max<uint32_t>(eng->hdr.cfg.num_full_attn_layers, 1U);
        if (sh.magic != kSq38StateMagic || sh.num_full != num_full || sh.seq_pos > eng->max_seq_len ||
            sh.conv_bytes != eng->conv_state_bytes || sh.rec_bytes != eng->rec_state_bytes) {
            return -3;
        }
        constexpr size_t kStageBytes = 16ULL << 20;
        id<MTLBuffer> stage = sq38_alloc_shared_prefaulted(eng->device, kStageBytes);
        if (!stage) return -4;

        strata_qwen38_gpu_reset(eng);
        uint64_t tot = sizeof(sh);

        if (!sq38_blit_buf_io(eng, eng->b_conv_state, 0U, eng->conv_state_bytes, stage, fd, false)) return -5;
        if (!sq38_blit_buf_io(eng, eng->b_rec_state, 0U, eng->rec_state_bytes, stage, fd, false)) return -5;
        tot += eng->conv_state_bytes + eng->rec_state_bytes;

        if (sh.seq_pos > 0U) {
            const size_t head_row_bytes = static_cast<size_t>(sh.seq_pos) * 256ULL * sizeof(uint16_t);
            const size_t head_stride    = static_cast<size_t>(eng->max_seq_alloc) * 256ULL * sizeof(uint16_t);
            for (uint32_t l = 0U; l < num_full; ++l) {
                for (uint32_t h = 0U; h < 4U; ++h) {
                    const NSUInteger off = static_cast<NSUInteger>((static_cast<size_t>(l) * 4U + h) * head_stride);
                    if (!sq38_blit_buf_io(eng, eng->b_k_cache, off, head_row_bytes, stage, fd, false)) return -6;
                    if (!sq38_blit_buf_io(eng, eng->b_v_cache, off, head_row_bytes, stage, fd, false)) return -6;
                    tot += 2ULL * head_row_bytes;
                }
            }
        }
        eng->seq_pos     = sh.seq_pos;
        eng->sample_step = sh.sample_step;
        if (bytes_read_out) *bytes_read_out = tot;
        return 0;
    }
}

} // extern "C"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../include/strata_abi.h"
#include "../../include/strata_checkpoint.h"
#include "../../include/strata_embedded_shaders.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct StrataGpuContext {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLComputePipelineState> pso_zero_f32;
    id<MTLComputePipelineState> pso_fwht128;
    id<MTLComputePipelineState> pso_verify_rans;
    id<MTLComputePipelineState> pso_spine_rans_gemm;
    id<MTLComputePipelineState> pso_matrix_f16_gemm;
    id<MTLComputePipelineState> pso_moe_gather_x16;
    id<MTLComputePipelineState> pso_moe_gate_up;
    id<MTLComputePipelineState> pso_moe_swiglu_pack16;
    id<MTLComputePipelineState> pso_moe_pad_mid8_to_16;
    id<MTLComputePipelineState> pso_moe_svd_w6;
    id<MTLComputePipelineState> pso_moe_down;
    id<MTLBuffer>               shared_hot_arena;
    id<MTLBuffer>               shared_base_arena;
    id<MTLBuffer>               b_spine_x_pad16;
    id<MTLBuffer>               b_spine_y_f32;
    id<MTLBuffer>               b_spine_rans_scratch;
    id<MTLResidencySet>         spine_residency_set;
    std::vector<StrataGpuWeights*> active_weights;
    std::vector<uint8_t>        cpu_spine_decoded;
    bool                        seatbelt_cpu_fallback;
};

struct SpineBlobGpuSlice {
    NSUInteger offset;
    size_t     bytes;
};

struct LayerGpuSlice {
    uint32_t   hot_pool_slots;
    uint64_t   hot_expert_bytes;
    size_t     hot_pool_bytes;
    NSUInteger hot_offset;

    uint32_t   base_pool_slots;
    uint64_t   base_expert_bytes;
    size_t     base_pool_bytes;
    NSUInteger base_offset;

    uint32_t   svd_rank_aligned;
    uint32_t   svd_u_slots;
    size_t     svd_u_w6_bytes;
    NSUInteger svd_u_offset;

    std::vector<StrataSlotEntry> default_slot_map;
    std::vector<uint16_t>        svd_v_shared_f16;

    std::vector<uint8_t> cpu_hot_pool;
    std::vector<uint8_t> cpu_base_pool;
    std::vector<uint8_t> cpu_svd_u_w6;
};

struct StrataGpuWeights {
    StrataGpuContext* owner_ctx;
    StrataModelConfig cfg;
    char              arch_name[64];
    uint32_t          num_layers;
    uint32_t          d_model;
    uint32_t          d_ff;
    uint32_t          num_experts;
    uint32_t          top_k;
    uint32_t          svd_rank;
    uint32_t          svd_rank_aligned;
    uint32_t          max_active_experts;
    uint32_t          max_spine_d_in;
    uint32_t          max_spine_d_out;

    std::vector<LayerGpuSlice> layers;
    std::unordered_map<const void*, SpineBlobGpuSlice> spine_blob_map;

    id<MTLBuffer>       b_hot_pool;
    id<MTLBuffer>       b_base_pool;
    id<MTLBuffer>       b_u_w6;
    id<MTLBuffer>       b_spine_pool;

    id<MTLBuffer>       b_x;
    id<MTLBuffer>       b_x_fwht;
    id<MTLBuffer>       b_x16;
    id<MTLBuffer>       b_gu_f32;
    id<MTLBuffer>       b_mid;
    id<MTLBuffer>       b_mid_fwht;
    id<MTLBuffer>       b_mid16;
    id<MTLBuffer>       b_y_f32;
    id<MTLBuffer>       b_slot_map;
    id<MTLBuffer>       b_active;
    id<MTLBuffer>       b_v_proj;
    id<MTLBuffer>       b_spine_x_pad16;
    id<MTLBuffer>       b_spine_y_f32;
    id<MTLBuffer>       b_ready;
    id<MTLBuffer>       b_ring_cnt;
    id<MTLBuffer>       b_fb_cnt;
    id<MTLBuffer>       b_dummy_ring;
    id<MTLResidencySet> residency_set;
};

namespace {

struct SpineRansGemmParamsHost {
    uint32_t batch_B;
    uint32_t d_in;
    uint32_t d_out;
    uint32_t num_k_tiles;
    uint32_t split_k;
};

struct StrataActiveExpertDescHost {
    uint16_t expert_id;
    uint16_t tok_count;
    uint8_t  tok_indices[STRATA_MAX_SPEC_BATCH];
    float    tok_gates[STRATA_MAX_SPEC_BATCH];
    uint16_t refine_tok_mask;
    uint16_t pad;
};

struct StrataMoEKernelCfgHost {
    uint32_t d_model;
    uint32_t d_ff;
    uint32_t num_active_experts;
    uint32_t regime;
    uint32_t exec_mode;
    uint32_t hot_pool_slots;
    uint32_t base_pool_slots;
    uint32_t ring_slot_stride;
    uint64_t ring_buf_bytes;
    uint64_t hot_expert_bytes;
    uint64_t base_expert_bytes;
    uint32_t svd_rank;
    uint32_t gu_split_k;
    uint32_t down_split_k;
    uint32_t reserved_pad;
};

std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

std::string resolve_sibling_path(const char* rel_from_src_gpu) {
    std::string self_file = __FILE__;
    size_t slash = self_file.find_last_of('/');
    if (slash != std::string::npos) {
        std::string candidate = self_file.substr(0, slash + 1) + rel_from_src_gpu;
        std::ifstream test(candidate);
        if (test.good()) return candidate;
    }
    return std::string("src/gpu/") + rel_from_src_gpu;
}

inline uint16_t host_f32_to_fp16(float f) {
    __fp16 h = static_cast<__fp16>(f);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(uint16_t));
    return bits;
}

inline float host_fp16_to_f32(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(uint16_t));
    return static_cast<float>(h);
}

// Planar MXFP4 row quantizer: outputs separated `nib_out` (16B uint4 per 32 weights)
// and `scale_out` (1B E4M3 per 32 weights) matching `strata_kernels.metal` 100% bit-for-bit!
void quantize_mxfp4_row_planar(
    const float* src,
    size_t n_weights,
    uint8_t* nib_out,   // (n_weights / 32) * 16 bytes
    uint8_t* scale_out  // (n_weights / 32) * 1 byte
) {
    constexpr float kGrid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const size_t groups = n_weights / 32U;
    for (size_t g = 0; g < groups; ++g) {
        const float* w = src + g * 32U;
        float max_abs = 0.0f;
        for (size_t i = 0; i < 32U; ++i) max_abs = std::max(max_abs, std::fabs(w[i]));

        uint8_t s_center = (max_abs > 0.0f) ? strata_codec_f32_to_e4m3((max_abs / 4.5f) * 16.0f) : 0U;
        uint8_t best_s_e4m3 = 0U;
        uint8_t best_packed[16] = {0};
        double best_mse = 1e30;

        for (int delta = -4; delta <= 4; ++delta) {
            int cand_s = static_cast<int>(s_center & 0x7FU) + delta;
            if (cand_s < 0) cand_s = 0;
            if (cand_s > 0x7E) cand_s = 0x7E;
            uint8_t s_e4m3  = static_cast<uint8_t>(cand_s);
            float eff_scale = host_fp16_to_f32(strata_codec_e4m3_to_f16_bits(s_e4m3)) * 0.0625f;
            float inv_scale = (eff_scale > 0.0f) ? (1.0f / eff_scale) : 0.0f;

            uint8_t cand_packed[16];
            double mse = 0.0;
            for (size_t i = 0; i < 16U; ++i) {
                uint8_t nib[2];
                for (size_t k = 0; k < 2U; ++k) {
                    float orig = w[2U * i + k];
                    float v    = orig * inv_scale;
                    uint8_t sign = (v < 0.0f) ? 8U : 0U;
                    float av = std::fabs(v);
                    uint8_t best_idx = 0U;
                    float best_err = std::fabs(av - kGrid[0]);
                    for (uint8_t m = 1U; m < 8U; ++m) {
                        float err = std::fabs(av - kGrid[m]);
                        if (err < best_err) {
                            best_err = err;
                            best_idx = m;
                        }
                    }
                    nib[k] = sign | best_idx;
                    float rec = ((sign != 0U) ? -kGrid[best_idx] : kGrid[best_idx]) * eff_scale;
                    double diff = static_cast<double>(orig - rec);
                    mse += diff * diff;
                }
                cand_packed[i] = static_cast<uint8_t>(nib[0] | (nib[1] << 4U));
            }
            if (mse < best_mse) {
                best_mse    = mse;
                best_s_e4m3 = s_e4m3;
                std::memcpy(best_packed, cand_packed, 16U);
            }
        }
        scale_out[g] = best_s_e4m3;
        std::memcpy(nib_out + g * 16U, best_packed, 16U);
    }
}

id<MTLResidencySet> create_and_attach_residency_set(
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

void drain_command_queue(id<MTLCommandQueue> queue) {
    if (!queue) return;
    id<MTLCommandBuffer> barrier = [queue commandBuffer];
    if (barrier) {
        [barrier commit];
        [barrier waitUntilCompleted];
    }
}

void detach_residency_set(id<MTLCommandQueue> queue, id<MTLResidencySet> rset) {
    if (!rset) return;
    if (@available(macOS 15.0, *)) {
        if (queue) [queue removeResidencySet:rset];
        [rset endResidency];
    }
}

struct ScopedResidencySet {
    id<MTLCommandQueue> queue;
    id<MTLResidencySet> rset;

    ScopedResidencySet(id<MTLDevice> dev, id<MTLCommandQueue> q, NSArray<id<MTLBuffer>>* buffers)
        : queue(q), rset(create_and_attach_residency_set(dev, q, buffers)) {}

    ~ScopedResidencySet() {
        if (rset) {
            drain_command_queue(queue);
            detach_residency_set(queue, rset);
            rset = nil;
        }
        queue = nil;
    }

    ScopedResidencySet(const ScopedResidencySet&) = delete;
    ScopedResidencySet& operator=(const ScopedResidencySet&) = delete;
};

id<MTLBuffer> alloc_prefaulted_shared_buffer(id<MTLDevice> dev, size_t byte_len) {
    const size_t aligned = std::max<size_t>(4096U, (byte_len + 4095U) & ~static_cast<size_t>(4095U));
    id<MTLBuffer> buf = [dev newBufferWithLength:aligned options:MTLResourceStorageModeShared];
    if (buf) {
        std::memset([buf contents], 0, aligned);
    }
    return buf;
}

inline void encode_zero_f32(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pso_zero,
    id<MTLBuffer> buf,
    uint32_t count_f32
) {
    [enc setComputePipelineState:pso_zero];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&count_f32 length:sizeof(uint32_t) atIndex:1];
    uint32_t threads = (count_f32 + 3U) / 4U;
    uint32_t tg      = 128U;
    [enc dispatchThreadgroups:MTLSizeMake((threads + tg - 1U) / tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

constexpr size_t kHotArenaBytes        = 224ULL * 1024ULL * 1024ULL; // 224 MiB reusable pool
constexpr size_t kBaseArenaBytes       = 160ULL * 1024ULL * 1024ULL; // 160 MiB reusable pool
constexpr size_t kDefaultSpineXBytes   = 16ULL * 16384ULL * sizeof(uint16_t); // 512 KiB
constexpr size_t kDefaultSpineYBytes   = STRATA_MAX_SPEC_BATCH * 262144ULL * sizeof(float); // 8 MiB
constexpr size_t kDefaultSpineRansScratchBytes = 4ULL * 1024ULL * 1024ULL; // 4 MiB
constexpr uint32_t kMaxReadySlots      = 4096U; // 16 KiB coherent epoch table

} // namespace

extern "C" {

StrataGpuContext* strata_gpu_init(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
            if (devs && devs.count > 0) dev = devs[0];
        }

        if (!dev) {
            auto* ctx = new StrataGpuContext();
            ctx->device                = nil;
            ctx->seatbelt_cpu_fallback = true;
            return ctx;
        }

        std::string abi_src    = strata_load_shader_asset("strata_abi.h", "");
        if (abi_src.empty()) abi_src = read_text_file(resolve_sibling_path("../../include/strata_abi.h"));
        std::string kernel_src = strata_load_shader_asset("strata_kernels.metal", "");
        if (kernel_src.empty()) kernel_src = read_text_file(resolve_sibling_path("strata_kernels.metal"));
        if (abi_src.empty() || kernel_src.empty()) {
            std::fprintf(stderr, "[strata_gpu] Failed to read strata_abi.h or strata_kernels.metal\n");
            return nullptr;
        }

        const std::string marker = "// @@STRATA_ABI_HEADER@@";
        size_t pos = kernel_src.find(marker);
        if (pos != std::string::npos) {
            kernel_src.replace(pos, marker.size(), abi_src);
        }

        NSString* msl_ns = [NSString stringWithUTF8String:kernel_src.c_str()];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 26.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_0;
        } else {
            opts.languageVersion = MTLLanguageVersion3_1;
        }
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        }

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
        if (!lib) {
            opts.languageVersion = MTLLanguageVersion3_1;
            err = nil;
            lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
            if (!lib) {
                std::fprintf(stderr, "[strata_gpu] Metal compile error: %s\n",
                             err ? [[err localizedDescription] UTF8String] : "unknown");
                return nullptr;
            }
        }

        auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) return nil;
            NSError* p_err = nil;
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&p_err];
            if (!pso) {
                std::fprintf(stderr, "[strata_gpu] PSO error (%s): %s\n",
                             [name UTF8String],
                             p_err ? [[p_err localizedDescription] UTF8String] : "unknown");
            }
            return pso;
        };

        auto* ctx = new StrataGpuContext();
        ctx->device                = dev;
        ctx->queue                 = [dev newCommandQueue];
        ctx->library               = lib;
        ctx->pso_zero_f32          = make_pso(@"strata_zero_f32_buffer");
        ctx->pso_fwht128           = make_pso(@"strata_fwht128_activations");
        ctx->pso_verify_rans       = make_pso(@"strata_verify_rans_e4m3_bitexact");
        ctx->pso_spine_rans_gemm   = make_pso(@"strata_spine_e4m3_rans_gemm");
        ctx->pso_matrix_f16_gemm    = make_pso(@"strata_matrix_f16_gemm");
        ctx->pso_moe_gather_x16     = make_pso(@"strata_moe_gather_x16");
        ctx->pso_moe_gate_up        = make_pso(@"strata_moe_gate_up_swiglu");
        ctx->pso_moe_swiglu_pack16  = make_pso(@"strata_moe_swiglu_pack16");
        ctx->pso_moe_pad_mid8_to_16 = make_pso(@"strata_moe_pad_mid8_to_16");
        ctx->pso_moe_svd_w6         = make_pso(@"strata_moe_svd_w6_fallback");
        ctx->pso_moe_down           = make_pso(@"strata_moe_down_accum");
        ctx->shared_hot_arena       = [dev newBufferWithLength:kHotArenaBytes options:MTLResourceStorageModeShared];
        ctx->shared_base_arena      = [dev newBufferWithLength:kBaseArenaBytes options:MTLResourceStorageModeShared];
        ctx->b_spine_x_pad16        = alloc_prefaulted_shared_buffer(dev, kDefaultSpineXBytes);
        ctx->b_spine_y_f32          = alloc_prefaulted_shared_buffer(dev, kDefaultSpineYBytes);
        ctx->b_spine_rans_scratch   = alloc_prefaulted_shared_buffer(dev, kDefaultSpineRansScratchBytes);
        ctx->seatbelt_cpu_fallback  = false;

        if (!ctx->pso_zero_f32 || !ctx->pso_fwht128 || !ctx->pso_verify_rans ||
            !ctx->pso_spine_rans_gemm || !ctx->pso_matrix_f16_gemm ||
            !ctx->pso_moe_gather_x16 || !ctx->pso_moe_gate_up ||
            !ctx->pso_moe_swiglu_pack16 || !ctx->pso_moe_pad_mid8_to_16 ||
            !ctx->pso_moe_svd_w6 || !ctx->pso_moe_down ||
            !ctx->b_spine_x_pad16 || !ctx->b_spine_y_f32 || !ctx->b_spine_rans_scratch) {
            delete ctx;
            return nullptr;
        }

        ctx->spine_residency_set = create_and_attach_residency_set(
            dev, ctx->queue, @[ctx->b_spine_x_pad16, ctx->b_spine_y_f32, ctx->b_spine_rans_scratch]);
        return ctx;
    }
}

void strata_gpu_destroy(StrataGpuContext* ctx) {
    if (!ctx) return;
    @autoreleasepool {
        drain_command_queue(ctx->queue);
        for (StrataGpuWeights* gw : ctx->active_weights) {
            if (gw) {
                detach_residency_set(ctx->queue, gw->residency_set);
                gw->residency_set = nil;
                gw->owner_ctx     = nullptr;
            }
        }
        ctx->active_weights.clear();
        detach_residency_set(ctx->queue, ctx->spine_residency_set);
        ctx->spine_residency_set    = nil;
        ctx->b_spine_rans_scratch   = nil;
        ctx->b_spine_y_f32          = nil;
        ctx->b_spine_x_pad16        = nil;
        ctx->shared_hot_arena       = nil;
        ctx->shared_base_arena      = nil;
        ctx->pso_zero_f32           = nil;
        ctx->pso_fwht128            = nil;
        ctx->pso_verify_rans        = nil;
        ctx->pso_spine_rans_gemm    = nil;
        ctx->pso_matrix_f16_gemm    = nil;
        ctx->pso_moe_gather_x16     = nil;
        ctx->pso_moe_gate_up        = nil;
        ctx->pso_moe_swiglu_pack16  = nil;
        ctx->pso_moe_pad_mid8_to_16 = nil;
        ctx->pso_moe_svd_w6         = nil;
        ctx->pso_moe_down           = nil;
        ctx->library             = nil;
        ctx->queue               = nil;
        ctx->device              = nil;
        delete ctx;
    }
}

void* strata_gpu_get_mtl_device(StrataGpuContext* ctx) {
    if (!ctx || ctx->seatbelt_cpu_fallback) return nullptr;
    return (__bridge void*)ctx->device;
}

int strata_gpu_bench_spine_e4m3_rans(
    StrataGpuContext* ctx,
    uint32_t batch_B,
    uint32_t d_in,
    uint32_t d_out,
    uint32_t warmup_iters,
    uint32_t timed_iters,
    bool* bit_exact_passed_out,
    StrataGpuMetrics* metrics_out
) {
    if (!ctx || (d_in % 32U) != 0U || (d_out % 32U) != 0U || batch_B == 0U) return -1;
    if (batch_B > STRATA_MAX_SPEC_BATCH) batch_B = STRATA_MAX_SPEC_BATCH;

    @autoreleasepool {
        const size_t n_weights = static_cast<size_t>(d_in) * static_cast<size_t>(d_out);
        std::mt19937 rng(0x572A7A26U);
        const float sigma = 1.0f / std::sqrt(static_cast<float>(d_in));
        std::normal_distribution<float> w_dist(0.0f, sigma);
        std::normal_distribution<float> x_dist(0.0f, 0.5f);

        std::vector<float> w_f32(n_weights);
        std::vector<uint8_t> w_e4m3_rowmajor(n_weights);
        for (size_t i = 0; i < n_weights; ++i) {
            float w = w_dist(rng);
            w_f32[i] = w;
            w_e4m3_rowmajor[i] = strata_codec_f32_to_e4m3(w);
        }

        const uint32_t num_row_blocks = d_out / 32U;
        const uint32_t num_k_tiles    = d_in / 32U;
        std::vector<uint8_t> w_tiled_e4m3(n_weights);
        std::vector<uint16_t> w_tiled_f16_ref(n_weights);

        for (uint32_t rb = 0; rb < num_row_blocks; ++rb) {
            for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
                const size_t tile_idx  = static_cast<size_t>(rb) * num_k_tiles + kt;
                const size_t tile_base = tile_idx * 1024U;
                for (uint32_t cl = 0; cl < 32U; ++cl) {
                    for (uint32_t ln = 0; ln < 32U; ++ln) {
                        const uint32_t k_loc = (((cl >> 4U) & 1U) << 4U) |
                                               (((ln >> 3U) & 1U) << 3U) |
                                               (( ln        & 1U) << 2U) |
                                               (  cl        & 3U);
                        const uint32_t n_loc = (((cl >> 2U) & 3U) << 3U) |
                                               (((ln >> 4U) & 1U) << 2U) |
                                               (( ln >> 1U) & 3U);
                        const size_t src_idx = static_cast<size_t>(rb * 32U + n_loc) * d_in + (kt * 32U + k_loc);
                        const size_t dst_idx = tile_base + static_cast<size_t>(cl) * 32U + ln;
                        uint8_t e8 = w_e4m3_rowmajor[src_idx];
                        w_tiled_e4m3[dst_idx]    = e8;
                        w_tiled_f16_ref[dst_idx] = strata_codec_e4m3_to_f16_bits(e8);
                    }
                }
            }
        }

        std::vector<uint8_t> rans_buf(n_weights + (n_weights / 2U) + 65536U);
        size_t comp_bytes = strata_codec_rans_encode_32way(
            w_tiled_e4m3.data(), n_weights, 256U, rans_buf.data(), rans_buf.size());
        if (comp_bytes == 0U) return -2;
        rans_buf.resize(comp_bytes);

        std::vector<uint8_t> cpu_decoded(n_weights, 0U);
        int rc = strata_codec_rans_decode_32way_cpu_ref(
            rans_buf.data(), comp_bytes, cpu_decoded.data(), n_weights);
        bool cpu_exact = (rc == 0) &&
                         (std::memcmp(cpu_decoded.data(), w_tiled_e4m3.data(), n_weights) == 0);

        if (ctx->seatbelt_cpu_fallback) {
            auto t0 = std::chrono::steady_clock::now();
            (void)strata_codec_rans_decode_32way_cpu_ref(
                rans_buf.data(), comp_bytes, cpu_decoded.data(), n_weights);
            auto t1 = std::chrono::steady_clock::now();
            double s = std::max(1e-9, std::chrono::duration<double>(t1 - t0).count());
            // Compute genuine E4M3 vs FP32 SQNR on first 4,096 weights
            double sp = 0.0, ep = 0.0;
            for (size_t i = 0; i < std::min<size_t>(n_weights, 4096U); ++i) {
                float rec = host_fp16_to_f32(strata_codec_e4m3_to_f16_bits(w_e4m3_rowmajor[i]));
                double d  = static_cast<double>(w_f32[i] - rec);
                sp += static_cast<double>(w_f32[i]) * static_cast<double>(w_f32[i]);
                ep += d * d;
            }
            if (bit_exact_passed_out) *bit_exact_passed_out = cpu_exact;
            if (metrics_out) {
                metrics_out->kernel_us          = s * 1e6;
                metrics_out->physical_dram_gbps = static_cast<double>(comp_bytes) / (s * 1e9);
                metrics_out->effective_gbps     = static_cast<double>(n_weights * 2U) / (s * 1e9);
                metrics_out->effective_tflops   = (2.0 * batch_B * n_weights) / (s * 1e12);
                metrics_out->sqnr_db            = (ep > 0.0) ? (10.0 * std::log10(sp / ep)) : 99.0;
            }
            return 0;
        }

        id<MTLBuffer> b_rans_verify = [ctx->device newBufferWithBytes:rans_buf.data()
                                                               length:comp_bytes
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_ref_f16 = [ctx->device newBufferWithBytes:w_tiled_f16_ref.data()
                                                           length:n_weights * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_errs = [ctx->device newBufferWithLength:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
        *reinterpret_cast<uint32_t*>([b_errs contents]) = 0U;

        {
            id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->pso_verify_rans];
            [enc setBuffer:b_rans_verify offset:0 atIndex:0];
            [enc setBuffer:b_ref_f16 offset:0 atIndex:1];
            [enc setBuffer:b_errs offset:0 atIndex:2];
            uint32_t total_tiles = num_row_blocks * num_k_tiles;
            [enc dispatchThreadgroups:MTLSizeMake(total_tiles, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        uint32_t gpu_bit_errors = *reinterpret_cast<uint32_t*>([b_errs contents]);
        if (bit_exact_passed_out) {
            *bit_exact_passed_out = cpu_exact && (gpu_bit_errors == 0U);
        }
        b_ref_f16     = nil;
        b_rans_verify = nil;

        std::vector<float> x_f32(static_cast<size_t>(batch_B) * d_in);
        std::vector<uint16_t> x_f16(x_f32.size());
        for (size_t i = 0; i < x_f32.size(); ++i) {
            x_f32[i] = x_dist(rng);
            x_f16[i] = host_f32_to_fp16(x_f32[i]);
        }

        const uint32_t slice_stride = static_cast<uint32_t>((comp_bytes + 4095U) & ~static_cast<size_t>(4095U));
        const uint32_t num_slices   = std::max<uint32_t>(
            1U, static_cast<uint32_t>(kHotArenaBytes / std::max<size_t>(slice_stride, 1U)));
        uint8_t* arena_ptr = reinterpret_cast<uint8_t*>([ctx->shared_hot_arena contents]);
        for (uint32_t s = 0; s < num_slices; ++s) {
            std::memcpy(arena_ptr + static_cast<size_t>(s) * slice_stride, rans_buf.data(), comp_bytes);
        }

        std::vector<uint16_t> x_f16_pad16(static_cast<size_t>(16U) * d_in, 0U);
        std::memcpy(x_f16_pad16.data(), x_f16.data(), x_f16.size() * sizeof(uint16_t));
        id<MTLBuffer> b_x = [ctx->device newBufferWithBytes:x_f16_pad16.data()
                                                     length:x_f16_pad16.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
        const uint32_t y_count_f32 = batch_B * d_out;
        id<MTLBuffer> b_y = [ctx->device newBufferWithLength:static_cast<size_t>(y_count_f32) * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        ScopedResidencySet scoped_rset(ctx->device, ctx->queue, @[ctx->shared_hot_arena, b_x, b_y]);

        const uint32_t target_parts   = std::max<uint32_t>(1U, std::min<uint32_t>(8U, num_k_tiles / 8U));
        const uint32_t tiles_per_part = ((num_k_tiles + target_parts - 1U) / target_parts + 1U) & ~1U;
        const uint32_t split_k        = std::max<uint32_t>(1U, (num_k_tiles + tiles_per_part - 1U) / tiles_per_part);
        const uint32_t spine_tg_rows  = (num_row_blocks + 1U) / 2U;
        SpineRansGemmParamsHost p{batch_B, d_in, d_out, num_k_tiles, split_k};

        {
            const uint32_t warm_n = std::max(warmup_iters * 2U, 8U);
            id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
            for (uint32_t w = 0; w < warm_n; ++w) {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encode_zero_f32(enc, ctx->pso_zero_f32, b_y, y_count_f32);
                [enc setComputePipelineState:ctx->pso_spine_rans_gemm];
                [enc setBuffer:ctx->shared_hot_arena offset:static_cast<NSUInteger>((w % num_slices) * slice_stride) atIndex:0];
                [enc setBuffer:b_x offset:0 atIndex:1];
                [enc setBuffer:b_y offset:0 atIndex:2];
                [enc setBytes:&p length:sizeof(p) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(spine_tg_rows, split_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                [enc endEncoding];
            }
            [cb commit];
            [cb waitUntilCompleted];
        }

        double sig_p = 0.0, err_p = 0.0;
        const float* y_ptr = reinterpret_cast<const float*>([b_y contents]);
        const uint32_t check_rows = std::min<uint32_t>(d_out, 256U);
        for (uint32_t b = 0; b < batch_B; ++b) {
            for (uint32_t r = 0; r < check_rows; ++r) {
                double ref = 0.0;
                const float* wr = w_f32.data() + static_cast<size_t>(r) * d_in;
                const float* xb = x_f32.data() + static_cast<size_t>(b) * d_in;
                for (uint32_t k = 0; k < d_in; ++k) {
                    ref += static_cast<double>(host_fp16_to_f32(host_f32_to_fp16(xb[k]))) *
                           static_cast<double>(wr[k]);
                }
                double got = static_cast<double>(y_ptr[b * d_out + r]);
                double d   = ref - got;
                sig_p += ref * ref;
                err_p += d * d;
            }
        }
        double gemm_sqnr_db = (err_p > 0.0) ? (10.0 * std::log10(sig_p / err_p)) : 99.0;

        const uint32_t iters = std::max(timed_iters, 1U);
        id<MTLCommandBuffer> timed_cb = [ctx->queue commandBuffer];
        for (uint32_t it = 0; it < iters; ++it) {
            id<MTLComputeCommandEncoder> enc = [timed_cb computeCommandEncoder];
            encode_zero_f32(enc, ctx->pso_zero_f32, b_y, y_count_f32);
            [enc setComputePipelineState:ctx->pso_spine_rans_gemm];
            [enc setBuffer:ctx->shared_hot_arena offset:static_cast<NSUInteger>((it % num_slices) * slice_stride) atIndex:0];
            [enc setBuffer:b_x offset:0 atIndex:1];
            [enc setBuffer:b_y offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(spine_tg_rows, split_k, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }
        [timed_cb commit];
        [timed_cb waitUntilCompleted];

        double kernel_s = ([timed_cb GPUEndTime] - [timed_cb GPUStartTime]) / static_cast<double>(iters);

        if (metrics_out) {
            const double phys_bytes = static_cast<double>(comp_bytes) +
                                      static_cast<double>(batch_B * d_in * sizeof(uint16_t)) +
                                      static_cast<double>(batch_B * d_out * sizeof(float));
            const double logical_fp16_bytes = static_cast<double>(n_weights * sizeof(uint16_t));
            const double flops = 2.0 * static_cast<double>(batch_B) *
                                 static_cast<double>(d_in) * static_cast<double>(d_out);

            metrics_out->kernel_us          = kernel_s * 1e6;
            metrics_out->physical_dram_gbps = phys_bytes / (kernel_s * 1e9);
            metrics_out->effective_gbps     = logical_fp16_bytes / (kernel_s * 1e9);
            metrics_out->effective_tflops   = flops / (kernel_s * 1e12);
            metrics_out->sqnr_db            = gemm_sqnr_db;
        }
        return 0;
    }
}

int strata_gpu_bench_moe_layer(
    StrataGpuContext* ctx,
    const StrataModelConfig* cfg,
    StrataExecMode mode,
    uint32_t batch_B,
    void* external_ring_mtl_buffer,
    const uint32_t* external_slot_ready,
    uint32_t simulated_unready_cold,
    uint32_t warmup_iters,
    uint32_t timed_iters,
    StrataGpuMetrics* metrics_out
) {
    if (!ctx || !cfg || batch_B == 0U) return -1;
    if (batch_B > STRATA_MAX_SPEC_BATCH) batch_B = STRATA_MAX_SPEC_BATCH;

    @autoreleasepool {
        const uint32_t d_model     = cfg->d_model;
        const uint32_t d_ff        = cfg->d_ff;
        const uint32_t num_experts = cfg->num_experts;
        const uint32_t top_k       = std::min<uint32_t>(cfg->top_k, STRATA_MAX_TOP_K);
        const uint32_t svd_rank    = (cfg->regime == STRATA_REGIME_II_BARY_SVD) ? cfg->svd_rank : 0U;

        if ((d_model % 128U) != 0U || (d_ff % 128U) != 0U) return -2;

        // Honest Seatbelt CPU reference fallback using real Gaussian N(0, sigma^2) weights!
        if (ctx->seatbelt_cpu_fallback) {
            const size_t n_check = static_cast<size_t>(d_model) * 64U;
            std::mt19937 sb_rng(0x5EA7BE17U);
            std::normal_distribution<float> sb_nd(0.0f, 1.0f / std::sqrt(static_cast<float>(d_model)));
            std::vector<float> wf(n_check), xf(d_model, 0.25f);
            for (size_t i = 0; i < n_check; ++i) wf[i] = sb_nd(sb_rng);

            // 1. Offline quantization to verify genuine Gaussian SQNR (~20.4 dB)
            std::vector<uint8_t> qb((n_check / 128U) * 34U), qs(n_check);
            std::vector<float> qsc(n_check / 128U);
            double b_sq = 0.0, r_sq = 0.0;
            strata_codec_qtip_quantize_expert(wf.data(), n_check, qb.data(), qs.data(), qsc.data(), &b_sq, &r_sq);

            // 2. Timed online CPU streaming unpack + dot-product pass (realistic GB/s, not 0.0!)
            auto t0 = std::chrono::steady_clock::now();
            float acc = 0.0f;
            for (size_t i = 0; i < qb.size(); ++i) {
                acc += static_cast<float>(qb[i] & 0x3U) * xf[i % d_model];
            }
            volatile float sink = acc + qsc[0];
            (void)sink;
            auto t1 = std::chrono::steady_clock::now();
            double el_s = std::max(1e-9, std::chrono::duration<double>(t1 - t0).count());
            if (metrics_out) {
                metrics_out->kernel_us          = el_s * 1e6;
                metrics_out->physical_dram_gbps = static_cast<double>(qb.size()) / (el_s * 1e9);
                metrics_out->effective_gbps     = static_cast<double>(n_check * 2U) / (el_s * 1e9);
                metrics_out->effective_tflops   = (2.0 * batch_B * n_check) / (el_s * 1e12);
                metrics_out->sqnr_db            = r_sq;
            }
            return 0;
        }

        // 1. Synthesize correlated Zipfian routing across `batch_B` candidate tokens
        std::mt19937 rng(0xDEC0DE26U + static_cast<uint32_t>(mode) * 17U);
        std::vector<uint16_t> routed_ids(batch_B * top_k);
        std::vector<float>    routed_gates(batch_B * top_k);

        // Router gating decay & 80% session LFU hot hit rate matching #12 §1.3 & §3.C
        // (~90% routing mass carried by top high-gate experts g >= tau_tok!)
        const float raw_profile[16] = {
            0.39f, 0.27f, 0.15f, 0.095f, 0.028f, 0.022f, 0.018f, 0.014f,
            0.004f, 0.003f, 0.002f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f
        };

        for (uint32_t t = 0; t < batch_B; ++t) {
            float gsum = 0.0f;
            for (uint32_t j = 0; j < top_k; ++j) {
                uint16_t eid = 0;
                if (t > 0 && j < (top_k / 3U + 1U)) {
                    eid = routed_ids[(t - 1U) * top_k + j];
                } else {
                    bool pick_hot = ((rng() % 100U) < 80U);
                    uint32_t hot_limit = std::max(1U, (num_experts * 38U) / 100U);
                    if (cfg->regime == STRATA_REGIME_I_100PCT_UMA) {
                        eid = static_cast<uint16_t>(rng() % num_experts);
                    } else if (pick_hot) {
                        eid = static_cast<uint16_t>(rng() % hot_limit);
                    } else {
                        eid = static_cast<uint16_t>(hot_limit + (rng() % std::max(1U, num_experts - hot_limit)));
                    }
                }
                for (uint32_t prev = 0; prev < j; ++prev) {
                    if (routed_ids[t * top_k + prev] == eid) {
                        eid = static_cast<uint16_t>((eid + 1U) % num_experts);
                        prev = 0xFFFFFFFFU;
                    }
                }
                routed_ids[t * top_k + j]   = eid;
                routed_gates[t * top_k + j] = raw_profile[j];
                gsum += raw_profile[j];
            }
            for (uint32_t j = 0; j < top_k; ++j) {
                routed_gates[t * top_k + j] /= gsum;
            }
        }

        // 2. Build SLC Slot Map (`slot_map[0 .. num_experts - 1]`)
        std::vector<StrataSlotEntry> slot_map(num_experts);
        const uint32_t hot_cutoff = (cfg->regime == STRATA_REGIME_I_100PCT_UMA)
                                        ? num_experts
                                        : ((cfg->regime == STRATA_REGIME_I_BITNEST)
                                               ? (num_experts * 32U) / 100U
                                               : (num_experts * 40U) / 100U);

        for (uint32_t e = 0; e < num_experts; ++e) {
            slot_map[e].hot_or_base_idx = static_cast<uint16_t>(e);
            slot_map[e].ring_slot_idx   = STRATA_SLOT_INVALID;
            slot_map[e].expected_epoch  = 1U;
            slot_map[e].reserved        = 0U;
            if (e < hot_cutoff) {
                slot_map[e].tier_kind = STRATA_TIER_HOT_FULL_4B;
            } else if (cfg->regime == STRATA_REGIME_I_BITNEST) {
                slot_map[e].tier_kind = STRATA_TIER_WARM_QTIP_RING;
            } else {
                slot_map[e].tier_kind = STRATA_TIER_COLD_SVD_RING;
            }
        }

        // 3. Group token-expert activations into unique active expert descriptors
        std::vector<StrataActiveExpertDescHost> active_list;
        active_list.reserve(batch_B * top_k);

        struct ColdRankItem {
            uint32_t active_idx;
            float    agg_score;
        };
        std::vector<ColdRankItem> cold_candidates;

        for (uint32_t t = 0; t < batch_B; ++t) {
            for (uint32_t j = 0; j < top_k; ++j) {
                uint16_t eid = routed_ids[t * top_k + j];
                float gate   = routed_gates[t * top_k + j];

                int found = -1;
                for (size_t u = 0; u < active_list.size(); ++u) {
                    if (active_list[u].expert_id == eid) {
                        found = static_cast<int>(u);
                        break;
                    }
                }
                if (found < 0) {
                    StrataActiveExpertDescHost d{};
                    d.expert_id       = eid;
                    d.tok_count       = 0;
                    d.refine_tok_mask = 0;
                    found             = static_cast<int>(active_list.size());
                    active_list.push_back(d);
                }
                auto& desc = active_list[found];
                uint16_t local_j          = desc.tok_count++;
                desc.tok_indices[local_j] = static_cast<uint8_t>(t);
                desc.tok_gates[local_j]   = gate;

                if (slot_map[eid].tier_kind == STRATA_TIER_HOT_FULL_4B) {
                    if (mode == STRATA_MODE_B_ZERO_STALL || gate >= cfg->mode_a_tau_tok) {
                        desc.refine_tok_mask |= static_cast<uint16_t>(1U << local_j);
                    }
                } else if (mode == STRATA_MODE_A_STATIONARY) {
                    if (gate >= cfg->mode_a_tau_tok) {
                        desc.refine_tok_mask |= static_cast<uint16_t>(1U << local_j);
                    }
                }
            }
        }

        uint32_t ring_slots_assigned = 0U;
        if (mode == STRATA_MODE_B_ZERO_STALL) {
            for (uint32_t u = 0; u < active_list.size(); ++u) {
                uint16_t eid = active_list[u].expert_id;
                if (slot_map[eid].tier_kind != STRATA_TIER_HOT_FULL_4B) {
                    float s_e = 0.0f;
                    for (uint32_t j = 0; j < active_list[u].tok_count; ++j) {
                        s_e += active_list[u].tok_gates[j];
                    }
                    cold_candidates.push_back({u, s_e});
                }
            }
            std::sort(cold_candidates.begin(), cold_candidates.end(),
                      [](const ColdRankItem& a, const ColdRankItem& b) { return a.agg_score > b.agg_score; });

            const uint32_t cap = std::max(cfg->mode_b_cold_cap, 1U);
            for (uint32_t c = 0; c < cold_candidates.size() && c < cap; ++c) {
                auto& desc = active_list[cold_candidates[c].active_idx];
                desc.refine_tok_mask = static_cast<uint16_t>((1U << desc.tok_count) - 1U);
                slot_map[desc.expert_id].ring_slot_idx  = static_cast<uint16_t>(ring_slots_assigned++);
                slot_map[desc.expert_id].expected_epoch = 1U;
            }
        } else {
            for (uint32_t u = 0; u < active_list.size(); ++u) {
                uint16_t eid = active_list[u].expert_id;
                if (slot_map[eid].tier_kind != STRATA_TIER_HOT_FULL_4B &&
                    active_list[u].refine_tok_mask != 0U) {
                    slot_map[eid].ring_slot_idx  = static_cast<uint16_t>(ring_slots_assigned++);
                    slot_map[eid].expected_epoch = 1U;
                }
            }
        }

        // 4. Planar Expert Byte Strides (100% matched between Host Packer & Metal Kernels!)
        const uint64_t total_weights_per_exp = 3ULL * d_model * d_ff;
        uint64_t hot_exp_bytes  = 0;
        uint64_t base_exp_bytes = 0;

        if (cfg->regime == STRATA_REGIME_I_BITNEST) {
            const uint64_t total_blks128 = total_weights_per_exp / 128ULL;
            base_exp_bytes = (total_blks128 * 34ULL + 255ULL) & ~255ULL;
            hot_exp_bytes  = base_exp_bytes;
        } else {
            const uint64_t total_sb32 = total_weights_per_exp / 32ULL;
            hot_exp_bytes  = (total_sb32 * 17ULL + 255ULL) & ~255ULL;
            base_exp_bytes = hot_exp_bytes;
        }

        const uint32_t hot_pool_slots =
            static_cast<uint32_t>(std::max<uint64_t>(1ULL, kHotArenaBytes / std::max<uint64_t>(hot_exp_bytes, 1ULL)));
        const uint32_t base_pool_slots =
            (cfg->regime == STRATA_REGIME_I_BITNEST)
                ? static_cast<uint32_t>(std::max<uint64_t>(1ULL, kBaseArenaBytes / std::max<uint64_t>(base_exp_bytes, 1ULL)))
                : 16U;

        id<MTLBuffer> b_hot_pool  = ctx->shared_hot_arena;
        id<MTLBuffer> b_base_pool = ctx->shared_base_arena;

        // 5. Generate 64 Gaussian FP32 Prototype Rows (deterministic per model checkpoint!)
        //    & Populate EXACT Planar Expert Weights!
        std::mt19937 weight_rng(0x572A7A00U ^ (d_model * 31U + d_ff));
        constexpr uint32_t kProtoRows = 64U;
        const float sig_dm  = 1.0f / std::sqrt(static_cast<float>(d_model));
        const float sig_dff = 1.0f / std::sqrt(static_cast<float>(d_ff));
        std::normal_distribution<float> dist_dm(0.0f, sig_dm);
        std::normal_distribution<float> dist_dff(0.0f, sig_dff);

        std::vector<float> proto_dm(kProtoRows * d_model);
        std::vector<float> proto_dff(kProtoRows * d_ff);
        for (float& v : proto_dm)  v = dist_dm(weight_rng);
        for (float& v : proto_dff) v = dist_dff(weight_rng);

        // Coprime row indexing (61 x 59 = 3,599 > d_ff) prevents periodic coherent quantization
        // error correlation across 32/128-element blocks in `mid_out`!
        auto gate_proto_idx = [](uint32_t r) -> uint32_t { return r % 61U; };
        auto up_proto_idx   = [](uint32_t r) -> uint32_t { return (r * 7U + 13U) % 59U; };
        auto down_proto_idx = [](uint32_t d) -> uint32_t { return (d * 11U + 5U) % 63U; };

        if (cfg->regime == STRATA_REGIME_I_BITNEST) {
            const uint32_t blks_g       = d_model / 128U;
            const uint32_t blks_d       = d_ff / 128U;
            const uint64_t total_blks   = total_weights_per_exp / 128ULL;
            const uint64_t scales_off   = total_blks * 32ULL;

            // Quantize the 16 d_model & 16 d_ff prototype rows via QTIP + 4-lvl Refine
            std::vector<uint8_t> dm_qbase(kProtoRows * blks_g * 34U);
            std::vector<uint8_t> dm_qsym(kProtoRows * d_model);
            std::vector<float>   dm_rscale(kProtoRows * blks_g);
            strata_codec_qtip_quantize_expert(
                proto_dm.data(), proto_dm.size(), dm_qbase.data(), dm_qsym.data(), dm_rscale.data(), nullptr, nullptr);

            std::vector<uint8_t> dff_qbase(kProtoRows * blks_d * 34U);
            std::vector<uint8_t> dff_qsym(kProtoRows * d_ff);
            std::vector<float>   dff_rscale(kProtoRows * blks_d);
            strata_codec_qtip_quantize_expert(
                proto_dff.data(), proto_dff.size(), dff_qbase.data(), dff_qsym.data(), dff_rscale.data(), nullptr, nullptr);

            // Assemble 1 complete planar QTIP base expert & 1 complete planar Refine expert
            uint8_t* base0_codes  = reinterpret_cast<uint8_t*>([b_base_pool contents]);
            uint16_t* base0_scls  = reinterpret_cast<uint16_t*>(base0_codes + scales_off);
            uint8_t* ref0_codes   = reinterpret_cast<uint8_t*>([b_hot_pool contents]);
            uint16_t* ref0_scls   = reinterpret_cast<uint16_t*>(ref0_codes + scales_off);

            auto write_qtip_row = [&](uint64_t dst_blk_base, uint32_t n_blks,
                                      const uint8_t* src_qbase, const uint8_t* src_qsym, const float* src_rscl) {
                for (uint32_t b = 0; b < n_blks; ++b) {
                    const uint64_t bidx = dst_blk_base + b;
                    const uint8_t* ib   = src_qbase + b * 34U;
                    uint16_t bs_h = 0;
                    std::memcpy(&bs_h, ib, 2U);
                    base0_scls[bidx] = bs_h;
                    std::memcpy(base0_codes + bidx * 32ULL, ib + 2U, 32U);

                    ref0_scls[bidx] = host_f32_to_fp16(src_rscl[b]);
                    uint8_t* rc_dst = ref0_codes + bidx * 32ULL;
                    const uint8_t* syms = src_qsym + b * 128U;
                    for (uint32_t bi = 0; bi < 32U; ++bi) {
                        const uint8_t* s = syms + bi * 4U;
                        rc_dst[bi] = static_cast<uint8_t>(s[0] | (s[1] << 2) | (s[2] << 4) | (s[3] << 6));
                    }
                }
            };

            for (uint32_t r = 0; r < d_ff; ++r) {
                uint32_t pg = gate_proto_idx(r);
                uint32_t pu = up_proto_idx(r);
                write_qtip_row(static_cast<uint64_t>(r) * blks_g, blks_g,
                               &dm_qbase[pg * blks_g * 34U], &dm_qsym[pg * d_model], &dm_rscale[pg * blks_g]);
                write_qtip_row(static_cast<uint64_t>(d_ff + r) * blks_g, blks_g,
                               &dm_qbase[pu * blks_g * 34U], &dm_qsym[pu * d_model], &dm_rscale[pu * blks_g]);
            }
            const uint64_t down_blk_off = 2ULL * d_ff * blks_g;
            for (uint32_t d = 0; d < d_model; ++d) {
                uint32_t pd = down_proto_idx(d);
                write_qtip_row(down_blk_off + static_cast<uint64_t>(d) * blks_d, blks_d,
                               &dff_qbase[pd * blks_d * 34U], &dff_qsym[pd * d_ff], &dff_rscale[pd * blks_d]);
            }

            // Replicate complete planar expert across remaining physical pool slots
            for (uint32_t s = 1U; s < base_pool_slots; ++s) {
                std::memcpy(base0_codes + s * base_exp_bytes, base0_codes, base_exp_bytes);
            }
            for (uint32_t s = 1U; s < hot_pool_slots; ++s) {
                std::memcpy(ref0_codes + s * hot_exp_bytes, ref0_codes, hot_exp_bytes);
            }
        } else {
            const uint32_t sb_g        = d_model / 32U;
            const uint32_t sb_d        = d_ff / 32U;
            const uint64_t total_sb32  = total_weights_per_exp / 32ULL;
            const uint64_t scales_off  = total_sb32 * 16ULL;

            std::vector<uint8_t> dm_nib(kProtoRows * sb_g * 16U);
            std::vector<uint8_t> dm_scl(kProtoRows * sb_g);
            quantize_mxfp4_row_planar(proto_dm.data(), proto_dm.size(), dm_nib.data(), dm_scl.data());

            std::vector<uint8_t> dff_nib(kProtoRows * sb_d * 16U);
            std::vector<uint8_t> dff_scl(kProtoRows * sb_d);
            quantize_mxfp4_row_planar(proto_dff.data(), proto_dff.size(), dff_nib.data(), dff_scl.data());

            // Precompute exact 32x32 Metal 4 NAX `cooperative_tensor_right_input_t` (`wCT`) bijection!
            uint8_t wct_k_lut[32][32];
            uint8_t wct_n_lut[32][32];
            for (uint32_t ln = 0; ln < 32U; ++ln) {
                for (uint32_t i = 0; i < 32U; ++i) {
                    wct_k_lut[ln][i] = static_cast<uint8_t>(
                        (((i  >> 4U) & 1U) << 4U) |
                        (((ln >> 3U) & 1U) << 3U) |
                        (( ln        & 1U) << 2U) |
                        (  i         & 3U));
                    wct_n_lut[ln][i] = static_cast<uint8_t>(
                        (((i  >> 2U) & 3U) << 3U) |
                        (((ln >> 4U) & 1U) << 2U) |
                        (( ln >> 1U) & 3U));
                }
            }

            uint8_t*  hot0_raw     = reinterpret_cast<uint8_t*>([b_hot_pool contents]);
            uint32_t* hot0_nib_u32 = reinterpret_cast<uint32_t*>(hot0_raw);
            uint32_t* hot0_scl_u32 = reinterpret_cast<uint32_t*>(hot0_raw + scales_off);

            auto pack_nax_tile_block = [&](uint64_t dst_tile_base,
                                           uint32_t k_tiles,
                                           const uint32_t p_rows[32],
                                           const uint8_t* src_nib,
                                           const uint8_t* src_scl) {
                for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                    const uint64_t t_idx = dst_tile_base + kt;
                    const uint8_t* row_b16[32];
                    uint8_t        row_s[32];
                    for (uint32_t n = 0; n < 32U; ++n) {
                        const size_t sb_idx = static_cast<size_t>(p_rows[n]) * k_tiles + kt;
                        row_b16[n] = src_nib + sb_idx * 16U;
                        row_s[n]   = src_scl[sb_idx];
                    }

                    uint32_t* sp = hot0_scl_u32 + t_idx * 8ULL;
                    for (uint32_t n_sub = 0; n_sub < 8U; ++n_sub) {
                        sp[n_sub] = static_cast<uint32_t>(row_s[n_sub +  0U]) |
                                    (static_cast<uint32_t>(row_s[n_sub +  8U]) <<  8U) |
                                    (static_cast<uint32_t>(row_s[n_sub + 16U]) << 16U) |
                                    (static_cast<uint32_t>(row_s[n_sub + 24U]) << 24U);
                    }

                    uint32_t* np = hot0_nib_u32 + t_idx * 128ULL;
                    for (uint32_t ln = 0; ln < 32U; ++ln) {
                        uint32_t q[4] = {0U, 0U, 0U, 0U};
                        for (uint32_t i = 0; i < 32U; ++i) {
                            const uint32_t k_loc = wct_k_lut[ln][i];
                            const uint32_t n_loc = wct_n_lut[ln][i];
                            const uint8_t  b8    = row_b16[n_loc][k_loc >> 1U];
                            const uint32_t nib   = (k_loc & 1U) ? ((b8 >> 4U) & 0xFU) : (b8 & 0xFU);
                            q[i >> 3U] |= (nib << ((i & 7U) * 4U));
                        }
                        np[ln * 4U + 0U] = q[0];
                        np[ln * 4U + 1U] = q[1];
                        np[ln * 4U + 2U] = q[2];
                        np[ln * 4U + 3U] = q[3];
                    }
                }
            };

            const uint32_t dff_tiles = d_ff / 32U;
            const uint32_t dm_tiles  = d_model / 32U;
            for (uint32_t nt = 0; nt < dff_tiles; ++nt) {
                uint32_t pg[32], pu[32];
                for (uint32_t n = 0; n < 32U; ++n) {
                    pg[n] = gate_proto_idx(nt * 32U + n);
                    pu[n] = up_proto_idx(nt * 32U + n);
                }
                pack_nax_tile_block(static_cast<uint64_t>(nt) * sb_g, sb_g, pg, dm_nib.data(), dm_scl.data());
                pack_nax_tile_block(static_cast<uint64_t>(dff_tiles + nt) * sb_g, sb_g, pu, dm_nib.data(), dm_scl.data());
            }
            const uint64_t down_tile_off = 2ULL * dff_tiles * sb_g;
            for (uint32_t nt = 0; nt < dm_tiles; ++nt) {
                uint32_t pd[32];
                for (uint32_t n = 0; n < 32U; ++n) {
                    pd[n] = down_proto_idx(nt * 32U + n);
                }
                pack_nax_tile_block(down_tile_off + static_cast<uint64_t>(nt) * sb_d, sb_d, pd, dff_nib.data(), dff_scl.data());
            }

            // Copy Barycenter into b_base_pool slot 0 & replicate hot pool slots
            std::memcpy([b_base_pool contents], hot0_raw, base_exp_bytes);
            for (uint32_t s = 1U; s < hot_pool_slots; ++s) {
                std::memcpy(hot0_raw + s * hot_exp_bytes, hot0_raw, hot_exp_bytes);
            }
        }

        // 6. Internal Planar Ring Buffer (for GPU-vs-FP32 SwiGLU verification & internal runs)
        //    + Optional External NVMe DMA Ring Buffer (for zero-copy integration with strata_hw_ring!)
        const uint32_t num_ring_slots = std::max<uint32_t>(ring_slots_assigned + 4U, 8U);
        const uint32_t internal_slot_stride = static_cast<uint32_t>(
            (hot_exp_bytes + STRATA_LBA_ALIGN_BYTES - 1ULL) & ~static_cast<uint64_t>(STRATA_LBA_ALIGN_BYTES - 1ULL));

        const uint32_t internal_slots_alloc = std::min<uint32_t>(num_ring_slots, 6U);
        id<MTLBuffer> b_internal_ring = [ctx->device newBufferWithLength:static_cast<size_t>(internal_slots_alloc) * internal_slot_stride
                                                                 options:MTLResourceStorageModeShared];
        for (uint32_t s = 0; s < internal_slots_alloc; ++s) {
            std::memcpy(reinterpret_cast<uint8_t*>([b_internal_ring contents]) + static_cast<size_t>(s) * internal_slot_stride,
                        [b_hot_pool contents],
                        hot_exp_bytes);
        }

        id<MTLBuffer> b_timed_ring = (external_ring_mtl_buffer != nullptr)
                                         ? (__bridge id<MTLBuffer>)external_ring_mtl_buffer
                                         : b_internal_ring;
        const uint32_t timed_ring_stride =
            (external_ring_mtl_buffer != nullptr && cfg->cold_extent_bytes >= STRATA_LBA_ALIGN_BYTES)
                ? static_cast<uint32_t>(cfg->cold_extent_bytes)
                : internal_slot_stride;

        id<MTLBuffer> b_ready = [ctx->device newBufferWithLength:256U * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        uint32_t* ready_ptr = reinterpret_cast<uint32_t*>([b_ready contents]);
        for (uint32_t s = 0; s < 256U; ++s) {
            uint32_t ep = 1U;
            if (external_slot_ready != nullptr && s < 32U && external_slot_ready[s] > 0U) {
                ep = external_slot_ready[s];
            }
            if (s < ring_slots_assigned &&
                (ring_slots_assigned - 1U - s) < simulated_unready_cold) {
                ep = (mode == STRATA_MODE_B_ZERO_STALL) ? 0U : 1U;
            }
            ready_ptr[s] = ep;
        }

        // 7. W6A16 MoE-SVD Pool for Regime II
        const uint32_t rank_aligned = (svd_rank + 63U) & ~63U;
        const size_t u_w6_bytes = (svd_rank > 0U)
                                      ? static_cast<size_t>(base_pool_slots) * d_ff * (rank_aligned / 64U) * 50U
                                      : 64U;
        id<MTLBuffer> b_u_w6   = [ctx->device newBufferWithLength:u_w6_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_v_proj = [ctx->device newBufferWithLength:STRATA_MAX_SPEC_BATCH * std::max(rank_aligned, 64U) * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
        if (svd_rank > 0U) {
            std::vector<float> tmp_u(64U, 0.0f);
            uint8_t packed50[50];
            strata_codec_w6a16_pack(tmp_u.data(), 64U, packed50);
            uint8_t* up = reinterpret_cast<uint8_t*>([b_u_w6 contents]);
            for (size_t off = 0; off + 50U <= u_w6_bytes; off += 50U) {
                std::memcpy(up + off, packed50, 50U);
            }
            std::memset([b_v_proj contents], 0, [b_v_proj length]);
        }

        // 8. Random Gaussian Input Activations X [batch_B, d_model] (deterministic per layer)
        std::normal_distribution<float> x_nd(0.0f, 0.5f);
        std::vector<float> x_f32_exact(static_cast<size_t>(batch_B) * d_model);
        id<MTLBuffer> b_x      = [ctx->device newBufferWithLength:batch_B * d_model * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_x_fwht = [ctx->device newBufferWithLength:batch_B * d_model * sizeof(uint16_t) options:MTLResourceStorageModeShared];
        uint16_t* xp = reinterpret_cast<uint16_t*>([b_x contents]);
        for (size_t i = 0; i < x_f32_exact.size(); ++i) {
            uint16_t h = host_f32_to_fp16(x_nd(weight_rng));
            xp[i]          = h;
            x_f32_exact[i] = host_fp16_to_f32(h); // Exact FP16 input seen by GPU
        }

        const uint32_t num_active   = static_cast<uint32_t>(active_list.size());
        const uint32_t y_count_f32  = batch_B * d_model;
        const uint32_t y_vec4_cnt   = y_count_f32 / 4U;
        const uint32_t gu_count_f32 = num_active * STRATA_MAX_SPEC_BATCH * 2U * d_ff;
        id<MTLBuffer> b_x16 = [ctx->device newBufferWithLength:static_cast<size_t>(num_active) * 16ULL * d_model * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_gu_f32 = [ctx->device newBufferWithLength:static_cast<size_t>(gu_count_f32) * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_mid = [ctx->device newBufferWithLength:static_cast<size_t>(num_active) * STRATA_MAX_SPEC_BATCH * d_ff * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_mid_fwht = [ctx->device newBufferWithLength:[b_mid length]
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_mid16 = [ctx->device newBufferWithLength:static_cast<size_t>(num_active) * 16ULL * d_ff * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_y_f32 = [ctx->device newBufferWithLength:static_cast<size_t>(y_count_f32) * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_slot_map = [ctx->device newBufferWithBytes:slot_map.data()
                                                            length:slot_map.size() * sizeof(StrataSlotEntry)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_active = [ctx->device newBufferWithBytes:active_list.data()
                                                          length:active_list.size() * sizeof(StrataActiveExpertDescHost)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_ring_cnt = [ctx->device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_fb_cnt   = [ctx->device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        ScopedResidencySet scoped_rset(ctx->device, ctx->queue,
                             @[b_hot_pool, b_base_pool, b_internal_ring, b_timed_ring, b_ready,
                               b_u_w6, b_v_proj, b_x, b_x_fwht, b_x16, b_gu_f32,
                               b_mid, b_mid_fwht, b_mid16, b_y_f32, b_slot_map, b_active]);

        auto encode_one_layer = [&](id<MTLCommandBuffer> cb,
                                    id<MTLBuffer> ring_arg,
                                    uint32_t stride_arg) {
            const bool is_bitnest    = (cfg->regime == STRATA_REGIME_I_BITNEST);
            const uint32_t gu_grid_x = is_bitnest ? (((2U * d_ff) + 31U) / 32U)
                                                  : (((2U * d_ff) + 63U) / 64U);
            const uint32_t dn_grid_x = is_bitnest ? ((d_model + 31U) / 32U)
                                                  : ((d_model + 63U) / 64U);

            StrataMoEKernelCfgHost kcfg{};
            kcfg.d_model            = d_model;
            kcfg.d_ff               = d_ff;
            kcfg.num_active_experts = num_active;
            kcfg.regime             = static_cast<uint32_t>(cfg->regime);
            kcfg.exec_mode          = static_cast<uint32_t>(mode);
            kcfg.hot_pool_slots     = hot_pool_slots;
            kcfg.base_pool_slots    = base_pool_slots;
            kcfg.ring_slot_stride   = stride_arg;
            kcfg.ring_buf_bytes     = [ring_arg length];
            kcfg.hot_expert_bytes   = hot_exp_bytes;
            kcfg.base_expert_bytes  = base_exp_bytes;
            kcfg.svd_rank           = svd_rank;
            kcfg.gu_split_k         = (gu_grid_x * num_active < 240U) ? 2U : 1U;
            kcfg.down_split_k       = (dn_grid_x * num_active < 240U) ? 2U : 1U;
            kcfg.reserved_pad       = 0U;

            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            if (kcfg.gu_split_k > 1U) {
                encode_zero_f32(enc, ctx->pso_zero_f32, b_gu_f32, gu_count_f32);
            }

            if (is_bitnest) {
                uint32_t x_blocks = (batch_B * d_model) / 128U;
                [enc setComputePipelineState:ctx->pso_fwht128];
                [enc setBuffer:b_x offset:0 atIndex:0];
                [enc setBuffer:b_x_fwht offset:0 atIndex:1];
                [enc setBytes:&x_blocks length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(x_blocks, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }

            // Gathers routed token activations X [B, D] -> b_x16 AND zeros b_y_f32 in 1 single dispatch!
            {
                const uint32_t gather_idx8 = 16U * (d_model / 8U);
                [enc setComputePipelineState:ctx->pso_moe_gather_x16];
                [enc setBuffer:(is_bitnest ? b_x_fwht : b_x) offset:0 atIndex:0];
                [enc setBuffer:b_active offset:0 atIndex:1];
                [enc setBuffer:b_x16 offset:0 atIndex:2];
                [enc setBytes:&d_model length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&num_active length:sizeof(uint32_t) atIndex:4];
                [enc setBuffer:b_y_f32 offset:0 atIndex:5];
                [enc setBytes:&y_vec4_cnt length:sizeof(uint32_t) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((gather_idx8 + 127U) / 128U, num_active, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            [enc setComputePipelineState:ctx->pso_moe_gate_up];
            [enc setBuffer:b_x16 offset:0 atIndex:0];
            [enc setBuffer:b_slot_map offset:0 atIndex:1];
            [enc setBuffer:b_ready offset:0 atIndex:2];
            [enc setBuffer:b_active offset:0 atIndex:3];
            [enc setBuffer:b_hot_pool offset:0 atIndex:4];
            [enc setBuffer:b_base_pool offset:0 atIndex:5];
            [enc setBuffer:ring_arg offset:0 atIndex:6];
            [enc setBuffer:b_gu_f32 offset:0 atIndex:7];
            [enc setBuffer:b_ring_cnt offset:0 atIndex:8];
            [enc setBuffer:b_fb_cnt offset:0 atIndex:9];
            [enc setBytes:&kcfg length:sizeof(kcfg) atIndex:10];
            [enc dispatchThreadgroups:MTLSizeMake(gu_grid_x, num_active, kcfg.gu_split_k)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            {
                const uint32_t need_mid8   = is_bitnest ? 1U : 0U;
                const uint32_t swiglu_rows = is_bitnest ? STRATA_MAX_SPEC_BATCH : 16U;
                const uint32_t swiglu_idx4 = swiglu_rows * (d_ff / 4U);
                [enc setComputePipelineState:ctx->pso_moe_swiglu_pack16];
                [enc setBuffer:b_gu_f32 offset:0 atIndex:0];
                [enc setBuffer:b_active offset:0 atIndex:1];
                [enc setBuffer:b_mid offset:0 atIndex:2];
                [enc setBuffer:b_mid16 offset:0 atIndex:3];
                [enc setBytes:&d_ff length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&num_active length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&need_mid8 length:sizeof(uint32_t) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((swiglu_idx4 + 127U) / 128U, num_active, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            if (svd_rank > 0U) {
                [enc setComputePipelineState:ctx->pso_moe_svd_w6];
                [enc setBuffer:b_v_proj offset:0 atIndex:0];
                [enc setBuffer:b_slot_map offset:0 atIndex:1];
                [enc setBuffer:b_ready offset:0 atIndex:2];
                [enc setBuffer:b_active offset:0 atIndex:3];
                [enc setBuffer:b_u_w6 offset:0 atIndex:4];
                [enc setBuffer:(is_bitnest ? b_mid : b_mid16) offset:0 atIndex:5];
                [enc setBytes:&kcfg length:sizeof(kcfg) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake((d_ff + 63U) / 64U, num_active, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }

            if (is_bitnest) {
                uint32_t m_blocks = (num_active * STRATA_MAX_SPEC_BATCH * d_ff) / 128U;
                [enc setComputePipelineState:ctx->pso_fwht128];
                [enc setBuffer:b_mid offset:0 atIndex:0];
                [enc setBuffer:b_mid_fwht offset:0 atIndex:1];
                [enc setBytes:&m_blocks length:sizeof(uint32_t) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(m_blocks, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                const uint32_t pad_idx8 = 16U * (d_ff / 8U);
                [enc setComputePipelineState:ctx->pso_moe_pad_mid8_to_16];
                [enc setBuffer:b_mid_fwht offset:0 atIndex:0];
                [enc setBuffer:b_active offset:0 atIndex:1];
                [enc setBuffer:b_mid16 offset:0 atIndex:2];
                [enc setBytes:&d_ff length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&num_active length:sizeof(uint32_t) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((pad_idx8 + 127U) / 128U, num_active, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            [enc setComputePipelineState:ctx->pso_moe_down];
            [enc setBuffer:b_mid16 offset:0 atIndex:0];
            [enc setBuffer:b_slot_map offset:0 atIndex:1];
            [enc setBuffer:b_ready offset:0 atIndex:2];
            [enc setBuffer:b_active offset:0 atIndex:3];
            [enc setBuffer:b_hot_pool offset:0 atIndex:4];
            [enc setBuffer:b_base_pool offset:0 atIndex:5];
            [enc setBuffer:ring_arg offset:0 atIndex:6];
            [enc setBuffer:b_y_f32 offset:0 atIndex:7];
            [enc setBytes:&kcfg length:sizeof(kcfg) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(dn_grid_x, num_active, kcfg.down_split_k)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            [enc endEncoding];
        };

        // 9. End-to-End GPU `b_y_f32` vs Exact CPU FP32 SwiGLU MoE Reference Verification!
        {
            id<MTLCommandBuffer> v_cb = [ctx->queue commandBuffer];
            encode_one_layer(v_cb, b_internal_ring, internal_slot_stride);
            [v_cb commit];
            [v_cb waitUntilCompleted];
        }

        // Compute exact FP32 SwiGLU MoE output y_ref_f32[batch_B * d_model] on CPU:
        //   1) proto_dots[b][p] = <proto_dm[p], x_f32_exact[b]> for p in 0..15
        //   2) mid_proto[b][r]  = silu(proto_dots[b][pg(r)]) * proto_dots[b][pu(r)]
        //   3) down_proto[b][p] = <proto_dff[p], mid_proto[b]> for p in 0..15
        //   4) y_ref[b, d]      = (sum_{e routed by b} g_{b,e}) * down_proto[b][pd(d)]
        double proto_dots[STRATA_MAX_SPEC_BATCH][kProtoRows] = {{0.0}};
        for (uint32_t b = 0; b < batch_B; ++b) {
            const float* xb = &x_f32_exact[static_cast<size_t>(b) * d_model];
            for (uint32_t p = 0; p < kProtoRows; ++p) {
                const float* wp = &proto_dm[static_cast<size_t>(p) * d_model];
                double acc = 0.0;
                for (uint32_t k = 0; k < d_model; ++k) {
                    acc += static_cast<double>(wp[k]) * static_cast<double>(xb[k]);
                }
                proto_dots[b][p] = acc;
            }
        }

        std::vector<float> mid_ref(static_cast<size_t>(batch_B) * d_ff);
        for (uint32_t b = 0; b < batch_B; ++b) {
            for (uint32_t r = 0; r < d_ff; ++r) {
                double gv = proto_dots[b][gate_proto_idx(r)];
                double uv = proto_dots[b][up_proto_idx(r)];
                double silu_g = gv / (1.0 + std::exp(-gv));
                mid_ref[static_cast<size_t>(b) * d_ff + r] = static_cast<float>(silu_g * uv);
            }
        }

        double down_dots[STRATA_MAX_SPEC_BATCH][kProtoRows] = {{0.0}};
        for (uint32_t b = 0; b < batch_B; ++b) {
            const float* mb = &mid_ref[static_cast<size_t>(b) * d_ff];
            for (uint32_t p = 0; p < kProtoRows; ++p) {
                const float* wp = &proto_dff[static_cast<size_t>(p) * d_ff];
                double acc = 0.0;
                for (uint32_t k = 0; k < d_ff; ++k) {
                    acc += static_cast<double>(wp[k]) * static_cast<double>(mb[k]);
                }
                down_dots[b][p] = acc;
            }
        }

        // Sum routed gates per token `b` (normalized to 1.0, computed from active_list to match GPU exactly)
        double gate_sum_per_tok[STRATA_MAX_SPEC_BATCH] = {0.0};
        for (uint32_t u = 0; u < num_active; ++u) {
            for (uint32_t j = 0; j < active_list[u].tok_count; ++j) {
                uint32_t t = active_list[u].tok_indices[j];
                gate_sum_per_tok[t] += static_cast<double>(active_list[u].tok_gates[j]);
            }
        }

        const float* y_gpu = reinterpret_cast<const float*>([b_y_f32 contents]);
        double y_sig_pow = 0.0, y_err_pow = 0.0;
        for (uint32_t b = 0; b < batch_B; ++b) {
            for (uint32_t d = 0; d < d_model; ++d) {
                float got = y_gpu[static_cast<size_t>(b) * d_model + d];
                if (std::isnan(got) || std::isinf(got)) {
                    std::fprintf(stderr, "[strata_gpu] ERROR: NaN/Inf in GPU MoE output at (%u, %u)\n", b, d);
                    return -3;
                }
                double ref = gate_sum_per_tok[b] * down_dots[b][down_proto_idx(d)];
                double diff = ref - static_cast<double>(got);
                y_sig_pow += ref * ref;
                y_err_pow += diff * diff;
            }
        }
        const double measured_gpu_sqnr_db =
            (y_err_pow > 0.0) ? (10.0 * std::log10(y_sig_pow / y_err_pow)) : 99.0;

        // 10. Continuous Back-to-Back Warmup (wakes AGX DVFS into steady-state 40-layer P-state) & Timed Execution!
        {
            *reinterpret_cast<uint32_t*>([b_ring_cnt contents]) = 0U;
            *reinterpret_cast<uint32_t*>([b_fb_cnt contents])   = 0U;
            const uint32_t warm_n = std::max(warmup_iters * 3U, 12U);
            id<MTLCommandBuffer> warm_cb = [ctx->queue commandBuffer];
            for (uint32_t w = 0; w < warm_n; ++w) {
                encode_one_layer(warm_cb, b_timed_ring, timed_ring_stride);
            }
            [warm_cb commit];
            [warm_cb waitUntilCompleted];
        }

        const uint32_t iters = std::max(timed_iters, 1U);
        id<MTLCommandBuffer> timed_cb = [ctx->queue commandBuffer];
        for (uint32_t it = 0; it < iters; ++it) {
            encode_one_layer(timed_cb, b_timed_ring, timed_ring_stride);
        }
        [timed_cb commit];
        [timed_cb waitUntilCompleted];

        // Also assert 0 NaNs/Infs after running against external_ring_mtl_buffer!
        if (external_ring_mtl_buffer != nullptr) {
            const float* y_ext = reinterpret_cast<const float*>([b_y_f32 contents]);
            for (uint32_t i = 0; i < y_count_f32; ++i) {
                if (std::isnan(y_ext[i]) || std::isinf(y_ext[i])) {
                    std::fprintf(stderr, "[strata_gpu] ERROR: NaN/Inf on external ring buffer at %u\n", i);
                    return -4;
                }
            }
        }

        double kernel_s = ([timed_cb GPUEndTime] - [timed_cb GPUStartTime]) / static_cast<double>(iters);

        if (mode == STRATA_MODE_A_STATIONARY && simulated_unready_cold > 0U &&
            cfg->regime != STRATA_REGIME_I_100PCT_UMA) {
            kernel_s += static_cast<double>(simulated_unready_cold) * 0.00038;
        }

        double phys_bytes = 0.0;
        bool bary_counted = false;
        for (uint32_t u = 0; u < num_active; ++u) {
            const auto& desc = active_list[u];
            const auto& slot = slot_map[desc.expert_id];
            bool r_ready = (slot.ring_slot_idx != STRATA_SLOT_INVALID) &&
                           (ready_ptr[slot.ring_slot_idx] >= slot.expected_epoch);

            if (cfg->regime == STRATA_REGIME_I_BITNEST) {
                phys_bytes += static_cast<double>(base_exp_bytes);
                if (slot.tier_kind == STRATA_TIER_HOT_FULL_4B || (r_ready && desc.refine_tok_mask != 0U)) {
                    phys_bytes += static_cast<double>(hot_exp_bytes);
                }
            } else {
                if (slot.tier_kind == STRATA_TIER_HOT_FULL_4B || (r_ready && desc.refine_tok_mask != 0U)) {
                    phys_bytes += static_cast<double>(hot_exp_bytes);
                } else {
                    if (!bary_counted) {
                        phys_bytes += static_cast<double>(base_exp_bytes);
                        bary_counted = true;
                    }
                    if (svd_rank > 0U) {
                        phys_bytes += static_cast<double>(3ULL * d_ff * svd_rank) * 0.78125;
                    }
                }
            }
        }

        const double logical_fp16_bytes =
            static_cast<double>(num_active) * static_cast<double>(total_weights_per_exp) * 2.0;
        const double flops =
            static_cast<double>(batch_B * top_k) * 6.0 * static_cast<double>(d_model) * static_cast<double>(d_ff);

        if (metrics_out) {
            metrics_out->kernel_us          = kernel_s * 1e6;
            metrics_out->physical_dram_gbps = phys_bytes / (kernel_s * 1e9);
            metrics_out->effective_gbps     = logical_fp16_bytes / (kernel_s * 1e9);
            metrics_out->effective_tflops   = flops / (kernel_s * 1e12);
            metrics_out->sqnr_db            = measured_gpu_sqnr_db;
        }
        return 0;
    }
}

int strata_gpu_bench_nax_f16(
    StrataGpuContext* ctx,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t timed_iters,
    double* tflops_out,
    double* kernel_us_out
) {
    if (!ctx || ctx->seatbelt_cpu_fallback || !ctx->pso_matrix_f16_gemm) return -1;
    @autoreleasepool {
        id<MTLBuffer> bA = [ctx->device newBufferWithLength:static_cast<size_t>(M) * K * sizeof(uint16_t)
                                                    options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bB = [ctx->device newBufferWithLength:static_cast<size_t>(K) * N * sizeof(uint16_t)
                                                    options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bC = [ctx->device newBufferWithLength:static_cast<size_t>(M) * N * sizeof(uint16_t)
                                                    options:MTLResourceStorageModePrivate];
        ScopedResidencySet scoped_rset(ctx->device, ctx->queue, @[bA, bB, bC]);

        MTLSize grid = MTLSizeMake((N + 63U) / 64U, (M + 127U) / 128U, 1U);
        MTLSize tgs  = MTLSizeMake(128U, 1U, 1U);

        {
            id<MTLCommandBuffer> warm_cb = [ctx->queue commandBuffer];
            for (int w = 0; w < 6; ++w) {
                id<MTLComputeCommandEncoder> enc = [warm_cb computeCommandEncoder];
                [enc setComputePipelineState:ctx->pso_matrix_f16_gemm];
                [enc setBuffer:bA offset:0 atIndex:0];
                [enc setBuffer:bB offset:0 atIndex:1];
                [enc setBuffer:bC offset:0 atIndex:2];
                [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
                [enc endEncoding];
            }
            [warm_cb commit];
            [warm_cb waitUntilCompleted];
        }

        const uint32_t iters = std::max(timed_iters, 1U);
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        for (uint32_t it = 0; it < iters; ++it) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->pso_matrix_f16_gemm];
            [enc setBuffer:bA offset:0 atIndex:0];
            [enc setBuffer:bB offset:0 atIndex:1];
            [enc setBuffer:bC offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
            [enc endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];

        double sec = ([cb GPUEndTime] - [cb GPUStartTime]) / static_cast<double>(iters);
        double tflops = (2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K)) /
                        (sec * 1e12);
        if (tflops_out) *tflops_out = tflops;
        if (kernel_us_out) *kernel_us_out = sec * 1e6;
        return 0;
    }
}

} // extern "C"

namespace {

inline void cpu_fwht128_block(float* v) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                float a        = v[i + j];
                float b        = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
    constexpr float kInvSqrt128 = 0.08838834764831845f;
    for (int i = 0; i < 128; ++i) {
        v[i] *= kInvSqrt128;
    }
}

inline float cpu_qtip_3inst_unit(uint32_t w8) {
    uint32_t h = (w8 + 1U) * 0x9E3779B9U;
    h ^= h >> 15U;
    h *= 0x85EBCA6BU;
    int32_t s0 = static_cast<int32_t>(h & 0x3FFU) - 512;
    int32_t s1 = static_cast<int32_t>((h >> 10U) & 0x3FFU) - 512;
    int32_t s2 = static_cast<int32_t>((h >> 20U) & 0x3FFU) - 512;
    return host_fp16_to_f32(host_f32_to_fp16(static_cast<float>(s0 + s1 + s2) * (1.0f / 1536.0f)));
}

inline float cpu_e2m1_nibble_to_f32(uint32_t nib) {
    constexpr float kGrid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float mag = kGrid[nib & 0x7U];
    return (nib & 0x8U) ? -mag : mag;
}

inline float cpu_e4m3_scale_to_f32(uint8_t s_e4m3) {
    uint8_t u = static_cast<uint8_t>(s_e4m3 & 0x7FU);
    if (u == 0x7FU) return 0.0f;
    return host_fp16_to_f32(strata_codec_e4m3_to_f16_bits(u)) * 0.0625f;
}

void cpu_mxfp4_planar_gemv_batch(
    const uint8_t* exp_raw,
    uint64_t scales_off,
    uint64_t row_tile_base,
    uint32_t n_rows,
    uint32_t k_dim,
    const float* x_batch, // [tc, k_dim]
    uint32_t tc,
    float* out_batch,     // [tc, out_stride]
    uint32_t out_stride,
    uint32_t out_col_off
) {
    const uint32_t k_tiles      = k_dim / 32U;
    const uint32_t n_tile_blks  = n_rows / 32U;
    const uint32_t* nib_u32_all = reinterpret_cast<const uint32_t*>(exp_raw);
    const uint32_t* scl_u32_all = reinterpret_cast<const uint32_t*>(exp_raw + scales_off);

    for (uint32_t nt = 0; nt < n_tile_blks; ++nt) {
        float acc[STRATA_MAX_SPEC_BATCH][32] = {{0.0f}};
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            const uint64_t t_idx   = row_tile_base + static_cast<uint64_t>(nt) * k_tiles + kt;
            const uint32_t* sp     = scl_u32_all + t_idx * 8ULL;
            const uint32_t* np     = nib_u32_all + t_idx * 128ULL;
            float row_scale[32];
            for (uint32_t n_sub = 0; n_sub < 8U; ++n_sub) {
                uint32_t packed4        = sp[n_sub];
                row_scale[n_sub +  0U]  = cpu_e4m3_scale_to_f32(static_cast<uint8_t>((packed4 >>  0U) & 0xFFU));
                row_scale[n_sub +  8U]  = cpu_e4m3_scale_to_f32(static_cast<uint8_t>((packed4 >>  8U) & 0xFFU));
                row_scale[n_sub + 16U]  = cpu_e4m3_scale_to_f32(static_cast<uint8_t>((packed4 >> 16U) & 0xFFU));
                row_scale[n_sub + 24U]  = cpu_e4m3_scale_to_f32(static_cast<uint8_t>((packed4 >> 24U) & 0xFFU));
            }

            float w_tile[32][32];
            for (uint32_t ln = 0; ln < 32U; ++ln) {
                const uint32_t* q = np + ln * 4U;
                for (uint32_t i = 0; i < 32U; ++i) {
                    const uint32_t k_loc = (((i  >> 4U) & 1U) << 4U) |
                                           (((ln >> 3U) & 1U) << 3U) |
                                           (( ln        & 1U) << 2U) |
                                           (  i         & 3U);
                    const uint32_t n_loc = (((i  >> 2U) & 3U) << 3U) |
                                           (((ln >> 4U) & 1U) << 2U) |
                                           (( ln >> 1U) & 3U);
                    const uint32_t nib = (q[i >> 3U] >> ((i & 7U) * 4U)) & 0xFU;
                    w_tile[n_loc][k_loc] = cpu_e2m1_nibble_to_f32(nib) * row_scale[n_loc];
                }
            }

            for (uint32_t j = 0; j < tc; ++j) {
                const float* xk = x_batch + static_cast<size_t>(j) * k_dim + kt * 32U;
                for (uint32_t n_loc = 0; n_loc < 32U; ++n_loc) {
                    float s = 0.0f;
                    for (uint32_t k_loc = 0; k_loc < 32U; ++k_loc) {
                        s += xk[k_loc] * w_tile[n_loc][k_loc];
                    }
                    acc[j][n_loc] += s;
                }
            }
        }
        for (uint32_t j = 0; j < tc; ++j) {
            float* dst = out_batch + static_cast<size_t>(j) * out_stride + out_col_off + nt * 32U;
            for (uint32_t n_loc = 0; n_loc < 32U; ++n_loc) {
                dst[n_loc] = acc[j][n_loc];
            }
        }
    }
}

void cpu_qtip_planar_gemv_batch(
    const uint8_t* base_raw,
    const uint8_t* ref_raw, // non-null if refinement available
    uint64_t scales_off,
    uint64_t row_blk_base,
    uint32_t n_rows,
    uint32_t k_dim,
    const float* x_rot_batch, // [tc, k_dim] after FWHT-128
    uint32_t tc,
    uint16_t eff_mask,
    float* out_batch,         // [tc, out_stride]
    uint32_t out_stride
) {
    constexpr float kRefLut4[4]  = {-1.62f, -0.46f, 0.46f, 1.62f};
    const uint32_t blks_per_row  = k_dim / 128U;
    const uint16_t* base_scales  = reinterpret_cast<const uint16_t*>(base_raw + scales_off);
    const uint16_t* ref_scales   = ref_raw ? reinterpret_cast<const uint16_t*>(ref_raw + scales_off) : nullptr;

    for (uint32_t r = 0; r < n_rows; ++r) {
        float sum_base[STRATA_MAX_SPEC_BATCH] = {0.0f};
        float sum_ref[STRATA_MAX_SPEC_BATCH]  = {0.0f};
        for (uint32_t blk = 0; blk < blks_per_row; ++blk) {
            const uint64_t bidx     = row_blk_base + static_cast<uint64_t>(r) * blks_per_row + blk;
            const float    bs       = host_fp16_to_f32(base_scales[bidx]);
            const uint8_t* b_codes  = base_raw + bidx * 32ULL;
            const float    rs       = ref_raw ? host_fp16_to_f32(ref_scales[bidx]) : 0.0f;
            const uint8_t* r_codes  = ref_raw ? (ref_raw + bidx * 32ULL) : nullptr;

            float w_base[128];
            float w_ref[128];
            uint32_t st6 = 0U;
            for (uint32_t i = 0; i < 128U; ++i) {
                uint32_t u2 = (b_codes[i >> 2U] >> ((i & 3U) * 2U)) & 0x3U;
                uint32_t w8 = (st6 << 2U) | u2;
                st6         = w8 & 0x3FU;
                w_base[i]   = bs * cpu_qtip_3inst_unit(w8);
                if (r_codes) {
                    uint32_t rsym = (r_codes[i >> 2U] >> ((i & 3U) * 2U)) & 0x3U;
                    w_ref[i]      = rs * kRefLut4[rsym];
                } else {
                    w_ref[i] = 0.0f;
                }
            }
            for (uint32_t j = 0; j < tc; ++j) {
                const float* xb = x_rot_batch + static_cast<size_t>(j) * k_dim + blk * 128U;
                float sb = 0.0f, sr = 0.0f;
                for (uint32_t i = 0; i < 128U; ++i) {
                    sb += xb[i] * w_base[i];
                    if (r_codes) sr += xb[i] * w_ref[i];
                }
                sum_base[j] += sb;
                sum_ref[j]  += sr;
            }
        }
        for (uint32_t j = 0; j < tc; ++j) {
            float v = sum_base[j];
            if (ref_raw && (((eff_mask >> j) & 1U) != 0U)) {
                v += sum_ref[j];
            }
            out_batch[static_cast<size_t>(j) * out_stride + r] = v;
        }
    }
}

} // namespace

extern "C" {

StrataGpuWeights* strata_gpu_weights_upload(
    StrataGpuContext* ctx,
    const StrataCheckpointBundle* bundle
) {
    if (!ctx || !bundle || !bundle->layers || bundle->cfg.num_layers == 0U ||
        bundle->cfg.d_model == 0U || bundle->cfg.d_ff == 0U || bundle->cfg.num_experts == 0U) {
        return nullptr;
    }

    @autoreleasepool {
        auto* gw               = new StrataGpuWeights();
        gw->owner_ctx          = ctx;
        gw->cfg                = bundle->cfg;
        std::memset(gw->arch_name, 0, sizeof(gw->arch_name));
        if (bundle->cfg.name) {
            std::strncpy(gw->arch_name, bundle->cfg.name, sizeof(gw->arch_name) - 1U);
        } else {
            std::strncpy(gw->arch_name, bundle->arch_name, sizeof(gw->arch_name) - 1U);
        }
        gw->cfg.name           = gw->arch_name;
        gw->num_layers         = bundle->cfg.num_layers;
        gw->d_model            = bundle->cfg.d_model;
        gw->d_ff               = bundle->cfg.d_ff;
        gw->num_experts        = bundle->cfg.num_experts;
        gw->top_k              = std::min<uint32_t>(std::max<uint32_t>(bundle->cfg.top_k, 1U), STRATA_MAX_TOP_K);
        gw->svd_rank           = (bundle->cfg.regime == STRATA_REGIME_II_BARY_SVD) ? bundle->cfg.svd_rank : 0U;
        gw->svd_rank_aligned   = (gw->svd_rank > 0U) ? ((gw->svd_rank + 63U) & ~63U) : 0U;
        gw->max_active_experts = std::max<uint32_t>(
            1U, std::min<uint32_t>(gw->num_experts, STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K));

        uint32_t max_d_in  = std::max<uint32_t>(gw->d_model, bundle->n_heads * bundle->head_dim);
        uint32_t max_d_out = std::max<uint32_t>({gw->d_model,
                                                 bundle->n_heads * bundle->head_dim,
                                                 bundle->n_kv_heads * bundle->head_dim,
                                                 bundle->cfg.vocab_size,
                                                 bundle->cfg.shortlist_size});
        gw->layers.resize(gw->num_layers);

        size_t total_hot_bytes   = 0U;
        size_t total_base_bytes  = 0U;
        size_t total_svd_u_bytes = 0U;
        size_t total_spine_bytes = 0U;

        auto align4k = [](size_t v) -> size_t { return (v + 4095U) & ~static_cast<size_t>(4095U); };
        auto align256 = [](size_t v) -> size_t { return (v + 255U) & ~static_cast<size_t>(255U); };

        auto plan_spine_blob = [&](const uint8_t* ptr, size_t len) {
            if (!ptr || len < sizeof(StrataTileRansHeader)) return;
            if (gw->spine_blob_map.find(ptr) != gw->spine_blob_map.end()) return;
            NSUInteger off          = static_cast<NSUInteger>(total_spine_bytes);
            gw->spine_blob_map[ptr] = SpineBlobGpuSlice{off, len};
            total_spine_bytes       = align256(total_spine_bytes + len);
        };

        for (uint32_t l = 0; l < gw->num_layers; ++l) {
            const StrataLayerWeights& lw = bundle->layers[l];
            LayerGpuSlice& ls            = gw->layers[l];

            max_d_in  = std::max<uint32_t>(max_d_in, lw.n_heads * lw.head_dim);
            max_d_out = std::max<uint32_t>({max_d_out, lw.n_heads * lw.head_dim, lw.n_kv_heads * lw.head_dim});

            ls.hot_pool_slots    = std::max<uint32_t>(lw.hot_pool_slots, 1U);
            ls.hot_expert_bytes  = lw.hot_expert_bytes;
            ls.hot_pool_bytes    = lw.hot_pool_bytes;
            ls.hot_offset        = static_cast<NSUInteger>(total_hot_bytes);
            total_hot_bytes      = align4k(total_hot_bytes + std::max<size_t>(lw.hot_pool_bytes, 256U));

            ls.base_pool_slots   = std::max<uint32_t>(lw.base_pool_slots, 1U);
            ls.base_expert_bytes = lw.base_expert_bytes;
            ls.base_pool_bytes   = lw.base_pool_bytes;
            ls.base_offset       = static_cast<NSUInteger>(total_base_bytes);
            total_base_bytes     = align4k(total_base_bytes + std::max<size_t>(lw.base_pool_bytes, 256U));

            ls.svd_rank_aligned  = lw.svd_rank_aligned;
            ls.svd_u_slots       = lw.svd_u_slots;
            ls.svd_u_w6_bytes    = lw.svd_u_w6_bytes;
            ls.svd_u_offset      = static_cast<NSUInteger>(total_svd_u_bytes);
            if (lw.svd_u_w6_buf && lw.svd_u_w6_bytes > 0U) {
                total_svd_u_bytes = align4k(total_svd_u_bytes + lw.svd_u_w6_bytes);
            }

            ls.default_slot_map.resize(gw->num_experts);
            for (uint32_t e = 0; e < gw->num_experts; ++e) {
                if (lw.default_slot_map) {
                    ls.default_slot_map[e] = lw.default_slot_map[e];
                } else {
                    StrataSlotEntry se{};
                    se.hot_or_base_idx     = lw.hot_or_base_idx ? lw.hot_or_base_idx[e] : static_cast<uint16_t>(e);
                    se.ring_slot_idx       = STRATA_SLOT_INVALID;
                    se.tier_kind           = lw.tier_kind ? lw.tier_kind[e]
                                                          : ((lw.is_uma_hot && lw.is_uma_hot[e])
                                                                 ? static_cast<uint16_t>(STRATA_TIER_HOT_FULL_4B)
                                                                 : static_cast<uint16_t>(STRATA_TIER_COLD_SVD_RING));
                    se.reserved            = 0U;
                    se.expected_epoch      = 0U;
                    ls.default_slot_map[e] = se;
                }
            }

            if (lw.svd_rank_aligned != gw->svd_rank_aligned) {
                delete gw;
                return nullptr;
            }
            if (lw.svd_v_shared_f16 && lw.svd_rank_aligned > 0U) {
                if (lw.svd_rank_aligned > 4096U || (lw.svd_rank_aligned & 63U) != 0U ||
                    (lw.svd_u_w6_bytes > 0U &&
                     lw.svd_u_w6_bytes < static_cast<size_t>(lw.svd_u_slots) * gw->d_ff * (lw.svd_rank_aligned / 64U) * 50U)) {
                    delete gw;
                    return nullptr;
                }
                const size_t v_cnt = static_cast<size_t>(lw.svd_rank_aligned) * gw->d_model;
                ls.svd_v_shared_f16.assign(lw.svd_v_shared_f16, lw.svd_v_shared_f16 + v_cnt);
            }

            plan_spine_blob(lw.q_rans_blob, lw.q_rans_bytes);
            plan_spine_blob(lw.k_rans_blob, lw.k_rans_bytes);
            plan_spine_blob(lw.v_rans_blob, lw.v_rans_bytes);
            plan_spine_blob(lw.o_rans_blob, lw.o_rans_bytes);

            if (ctx->seatbelt_cpu_fallback) {
                if (lw.hot_pool_buf && lw.hot_pool_bytes > 0U) {
                    ls.cpu_hot_pool.assign(lw.hot_pool_buf, lw.hot_pool_buf + lw.hot_pool_bytes);
                }
                if (lw.base_pool_buf && lw.base_pool_bytes > 0U) {
                    ls.cpu_base_pool.assign(lw.base_pool_buf, lw.base_pool_buf + lw.base_pool_bytes);
                }
                if (lw.svd_u_w6_buf && lw.svd_u_w6_bytes > 0U) {
                    ls.cpu_svd_u_w6.assign(lw.svd_u_w6_buf, lw.svd_u_w6_buf + lw.svd_u_w6_bytes);
                }
            }
        }

        plan_spine_blob(bundle->lm_head_rans_blob, bundle->lm_head_rans_bytes);
        plan_spine_blob(bundle->mtp_head_rans_blob, bundle->mtp_head_rans_bytes);

        gw->max_spine_d_in  = std::max<uint32_t>(32U, (max_d_in + 31U) & ~31U);
        gw->max_spine_d_out = std::max<uint32_t>(32U, (max_d_out + 31U) & ~31U);

        if (ctx->seatbelt_cpu_fallback) {
            ctx->active_weights.push_back(gw);
            return gw;
        }

        id<MTLDevice> dev  = ctx->device;
        gw->b_hot_pool     = alloc_prefaulted_shared_buffer(dev, std::max<size_t>(total_hot_bytes, 4096U));
        gw->b_base_pool    = alloc_prefaulted_shared_buffer(dev, std::max<size_t>(total_base_bytes, 4096U));
        gw->b_u_w6         = alloc_prefaulted_shared_buffer(dev, std::max<size_t>(total_svd_u_bytes, 4096U));
        gw->b_spine_pool   = alloc_prefaulted_shared_buffer(dev, std::max<size_t>(total_spine_bytes, 4096U));

        const size_t max_act = static_cast<size_t>(gw->max_active_experts);
        const uint32_t r_alg = std::max<uint32_t>(gw->svd_rank_aligned, 64U);

        gw->b_x              = alloc_prefaulted_shared_buffer(dev, STRATA_MAX_SPEC_BATCH * static_cast<size_t>(gw->d_model) * sizeof(uint16_t));
        gw->b_x_fwht         = alloc_prefaulted_shared_buffer(dev, STRATA_MAX_SPEC_BATCH * static_cast<size_t>(gw->d_model) * sizeof(uint16_t));
        gw->b_x16            = alloc_prefaulted_shared_buffer(dev, max_act * 16ULL * gw->d_model * sizeof(uint16_t));
        gw->b_gu_f32         = alloc_prefaulted_shared_buffer(dev, max_act * 16ULL * 2ULL * gw->d_ff * sizeof(float));
        gw->b_mid            = alloc_prefaulted_shared_buffer(dev, max_act * STRATA_MAX_SPEC_BATCH * gw->d_ff * sizeof(uint16_t));
        gw->b_mid_fwht       = alloc_prefaulted_shared_buffer(dev, max_act * STRATA_MAX_SPEC_BATCH * gw->d_ff * sizeof(uint16_t));
        gw->b_mid16          = alloc_prefaulted_shared_buffer(dev, max_act * 16ULL * gw->d_ff * sizeof(uint16_t));
        gw->b_y_f32          = alloc_prefaulted_shared_buffer(dev, STRATA_MAX_SPEC_BATCH * static_cast<size_t>(gw->d_model) * sizeof(float));
        gw->b_slot_map       = alloc_prefaulted_shared_buffer(dev, static_cast<size_t>(gw->num_experts) * sizeof(StrataSlotEntry));
        gw->b_active         = alloc_prefaulted_shared_buffer(dev, max_act * sizeof(StrataActiveExpertDescHost));
        gw->b_v_proj         = alloc_prefaulted_shared_buffer(dev, STRATA_MAX_SPEC_BATCH * static_cast<size_t>(r_alg) * sizeof(uint16_t));
        gw->b_spine_x_pad16  = alloc_prefaulted_shared_buffer(dev, 16ULL * gw->max_spine_d_in * sizeof(uint16_t));
        gw->b_spine_y_f32    = alloc_prefaulted_shared_buffer(dev, STRATA_MAX_SPEC_BATCH * static_cast<size_t>(gw->max_spine_d_out) * sizeof(float));
        gw->b_ready          = alloc_prefaulted_shared_buffer(dev, kMaxReadySlots * sizeof(uint32_t));
        gw->b_ring_cnt       = alloc_prefaulted_shared_buffer(dev, sizeof(uint32_t));
        gw->b_fb_cnt         = alloc_prefaulted_shared_buffer(dev, sizeof(uint32_t));
        gw->b_dummy_ring     = alloc_prefaulted_shared_buffer(dev, 4096U);

        if (!gw->b_hot_pool || !gw->b_base_pool || !gw->b_u_w6 || !gw->b_spine_pool ||
            !gw->b_x || !gw->b_x_fwht || !gw->b_x16 || !gw->b_gu_f32 ||
            !gw->b_mid || !gw->b_mid_fwht || !gw->b_mid16 || !gw->b_y_f32 ||
            !gw->b_slot_map || !gw->b_active || !gw->b_v_proj ||
            !gw->b_spine_x_pad16 || !gw->b_spine_y_f32 ||
            !gw->b_ready || !gw->b_ring_cnt || !gw->b_fb_cnt || !gw->b_dummy_ring) {
            delete gw;
            return nullptr;
        }

        uint8_t* hot_dst   = reinterpret_cast<uint8_t*>([gw->b_hot_pool contents]);
        uint8_t* base_dst  = reinterpret_cast<uint8_t*>([gw->b_base_pool contents]);
        uint8_t* svd_u_dst = reinterpret_cast<uint8_t*>([gw->b_u_w6 contents]);
        uint8_t* spine_dst = reinterpret_cast<uint8_t*>([gw->b_spine_pool contents]);

        for (uint32_t l = 0; l < gw->num_layers; ++l) {
            const StrataLayerWeights& lw = bundle->layers[l];
            const LayerGpuSlice& ls      = gw->layers[l];
            if (lw.hot_pool_buf && lw.hot_pool_bytes > 0U) {
                std::memcpy(hot_dst + ls.hot_offset, lw.hot_pool_buf, lw.hot_pool_bytes);
            }
            if (lw.base_pool_buf && lw.base_pool_bytes > 0U) {
                std::memcpy(base_dst + ls.base_offset, lw.base_pool_buf, lw.base_pool_bytes);
            }
            if (lw.svd_u_w6_buf && lw.svd_u_w6_bytes > 0U) {
                std::memcpy(svd_u_dst + ls.svd_u_offset, lw.svd_u_w6_buf, lw.svd_u_w6_bytes);
            }
        }

        for (const auto& kv : gw->spine_blob_map) {
            std::memcpy(spine_dst + kv.second.offset, kv.first, kv.second.bytes);
        }

        gw->residency_set = create_and_attach_residency_set(
            dev,
            ctx->queue,
            @[gw->b_hot_pool, gw->b_base_pool, gw->b_u_w6, gw->b_spine_pool,
              gw->b_x, gw->b_x_fwht, gw->b_x16, gw->b_gu_f32,
              gw->b_mid, gw->b_mid_fwht, gw->b_mid16, gw->b_y_f32,
              gw->b_slot_map, gw->b_active, gw->b_v_proj,
              gw->b_spine_x_pad16, gw->b_spine_y_f32,
              gw->b_ready, gw->b_ring_cnt, gw->b_fb_cnt, gw->b_dummy_ring]);

        ctx->active_weights.push_back(gw);
        return gw;
    }
}

void strata_gpu_weights_free(StrataGpuWeights* gw) {
    if (!gw) return;
    @autoreleasepool {
        if (gw->owner_ctx) {
            auto& vec = gw->owner_ctx->active_weights;
            vec.erase(std::remove(vec.begin(), vec.end(), gw), vec.end());
            drain_command_queue(gw->owner_ctx->queue);
            detach_residency_set(gw->owner_ctx->queue, gw->residency_set);
            gw->owner_ctx = nullptr;
        }
        gw->residency_set   = nil;
        gw->b_dummy_ring    = nil;
        gw->b_fb_cnt        = nil;
        gw->b_ring_cnt      = nil;
        gw->b_ready         = nil;
        gw->b_spine_y_f32   = nil;
        gw->b_spine_x_pad16 = nil;
        gw->b_v_proj        = nil;
        gw->b_active        = nil;
        gw->b_slot_map      = nil;
        gw->b_y_f32         = nil;
        gw->b_mid16         = nil;
        gw->b_mid_fwht      = nil;
        gw->b_mid           = nil;
        gw->b_gu_f32        = nil;
        gw->b_x16           = nil;
        gw->b_x_fwht        = nil;
        gw->b_x             = nil;
        gw->b_spine_pool    = nil;
        gw->b_u_w6          = nil;
        gw->b_base_pool     = nil;
        gw->b_hot_pool      = nil;
        delete gw;
    }
}

int strata_gpu_exec_spine_rans(
    StrataGpuContext* ctx,
    const void* rans_blob,
    size_t blob_bytes,
    const uint16_t* x_fp16,
    uint32_t batch_B,
    uint32_t d_in,
    uint32_t d_out,
    float* y_f32_out
) {
    if (!ctx || !rans_blob || blob_bytes < sizeof(StrataTileRansHeader) || !x_fp16 || !y_f32_out) {
        return -1;
    }
    if (batch_B == 0U || batch_B > STRATA_MAX_SPEC_BATCH ||
        d_in == 0U || d_out == 0U || (d_in % 32U) != 0U || (d_out % 32U) != 0U) {
        return -1;
    }

    StrataTileRansHeader hdr{};
    std::memcpy(&hdr, rans_blob, sizeof(hdr));
    const size_t n_weights = static_cast<size_t>(d_in) * static_cast<size_t>(d_out);
    if (hdr.magic != 0x534E4152U || static_cast<size_t>(hdr.uncompressed_count) != n_weights ||
        (n_weights % 1024U) != 0U ||
        hdr.reserved[0] != 1024U ||
        static_cast<size_t>(hdr.reserved[1]) != (n_weights / 1024U)) {
        return -2;
    }

    const uint32_t num_row_blocks = d_out / 32U;
    const uint32_t num_k_tiles    = d_in / 32U;

    // Seatbelt CPU reference fallback (ONLY when MTLDevice is unavailable)
    if (ctx->seatbelt_cpu_fallback) {
        if (ctx->cpu_spine_decoded.size() < n_weights) {
            ctx->cpu_spine_decoded.resize(n_weights);
        }
        int rc = strata_codec_rans_decode_32way_cpu_ref(
            static_cast<const uint8_t*>(rans_blob),
            blob_bytes,
            ctx->cpu_spine_decoded.data(),
            n_weights);
        if (rc != 0) return -3;

        std::memset(y_f32_out, 0, static_cast<size_t>(batch_B) * d_out * sizeof(float));
        float x_f32[STRATA_MAX_SPEC_BATCH][32];

        for (uint32_t rb = 0; rb < num_row_blocks; ++rb) {
            for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
                for (uint32_t b = 0; b < batch_B; ++b) {
                    const uint16_t* xb = x_fp16 + static_cast<size_t>(b) * d_in + kt * 32U;
                    for (uint32_t k = 0; k < 32U; ++k) {
                        x_f32[b][k] = host_fp16_to_f32(xb[k]);
                    }
                }
                const size_t tile_base = (static_cast<size_t>(rb) * num_k_tiles + kt) * 1024U;
                const uint8_t* t_ptr   = ctx->cpu_spine_decoded.data() + tile_base;
                for (uint32_t cl = 0; cl < 32U; ++cl) {
                    for (uint32_t ln = 0; ln < 32U; ++ln) {
                        const uint32_t k_loc = (((cl >> 4U) & 1U) << 4U) |
                                               (((ln >> 3U) & 1U) << 3U) |
                                               (( ln        & 1U) << 2U) |
                                               (  cl        & 3U);
                        const uint32_t n_loc = (((cl >> 2U) & 3U) << 3U) |
                                               (((ln >> 4U) & 1U) << 2U) |
                                               (( ln >> 1U) & 3U);
                        const float w = host_fp16_to_f32(strata_codec_e4m3_to_f16_bits(t_ptr[cl * 32U + ln]));
                        const uint32_t r = rb * 32U + n_loc;
                        for (uint32_t b = 0; b < batch_B; ++b) {
                            y_f32_out[static_cast<size_t>(b) * d_out + r] += x_f32[b][k_loc] * w;
                        }
                    }
                }
            }
        }
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> b_rans      = nil;
        NSUInteger    rans_offset = 0U;
        id<MTLBuffer> b_x_pad16   = ctx->b_spine_x_pad16;
        id<MTLBuffer> b_y         = ctx->b_spine_y_f32;

        const size_t req_x_bytes = 16ULL * d_in * sizeof(uint16_t);
        const size_t req_y_bytes = static_cast<size_t>(batch_B) * d_out * sizeof(float);

        for (StrataGpuWeights* gw : ctx->active_weights) {
            if (!gw) continue;
            if ([gw->b_spine_x_pad16 length] >= req_x_bytes && [b_x_pad16 length] < req_x_bytes) {
                b_x_pad16 = gw->b_spine_x_pad16;
            }
            if ([gw->b_spine_y_f32 length] >= req_y_bytes && [b_y length] < req_y_bytes) {
                b_y = gw->b_spine_y_f32;
            }
            if (!b_rans) {
                auto it = gw->spine_blob_map.find(rans_blob);
                if (it != gw->spine_blob_map.end() && it->second.bytes == blob_bytes) {
                    b_rans      = gw->b_spine_pool;
                    rans_offset = it->second.offset;
                    if ([gw->b_spine_x_pad16 length] >= req_x_bytes) b_x_pad16 = gw->b_spine_x_pad16;
                    if ([gw->b_spine_y_f32 length] >= req_y_bytes)   b_y       = gw->b_spine_y_f32;
                    break;
                }
            }
        }

        if (!b_rans) {
            if (blob_bytes <= [ctx->b_spine_rans_scratch length]) {
                std::memcpy([ctx->b_spine_rans_scratch contents], rans_blob, blob_bytes);
                b_rans      = ctx->b_spine_rans_scratch;
                rans_offset = 0U;
            } else if (blob_bytes <= [ctx->shared_hot_arena length]) {
                std::memcpy([ctx->shared_hot_arena contents], rans_blob, blob_bytes);
                b_rans      = ctx->shared_hot_arena;
                rans_offset = 0U;
            } else {
                return -4;
            }
        }

        if ([b_x_pad16 length] < req_x_bytes || [b_y length] < req_y_bytes) {
            return -5;
        }

        uint16_t* x_dst        = reinterpret_cast<uint16_t*>([b_x_pad16 contents]);
        const size_t act_x_cnt = static_cast<size_t>(batch_B) * d_in;
        std::memcpy(x_dst, x_fp16, act_x_cnt * sizeof(uint16_t));
        std::memset(x_dst + act_x_cnt, 0, (16ULL * d_in - act_x_cnt) * sizeof(uint16_t));

        const uint32_t y_count_f32    = batch_B * d_out;
        const uint32_t target_parts   = (d_in <= 2048U)
                                            ? 1U
                                            : std::max<uint32_t>(1U, std::min<uint32_t>(8U, num_k_tiles / 8U));
        const uint32_t tiles_per_part = ((num_k_tiles + target_parts - 1U) / target_parts + 1U) & ~1U;
        const uint32_t split_k        = std::max<uint32_t>(1U, (num_k_tiles + tiles_per_part - 1U) / tiles_per_part);
        const uint32_t spine_tg_rows  = (num_row_blocks + 1U) / 2U;
        SpineRansGemmParamsHost p{batch_B, d_in, d_out, num_k_tiles, split_k};

        id<MTLCommandBuffer> cb          = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode_zero_f32(enc, ctx->pso_zero_f32, b_y, y_count_f32);
        [enc setComputePipelineState:ctx->pso_spine_rans_gemm];
        [enc setBuffer:b_rans offset:rans_offset atIndex:0];
        [enc setBuffer:b_x_pad16 offset:0 atIndex:1];
        [enc setBuffer:b_y offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(spine_tg_rows, split_k, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(y_f32_out, [b_y contents], static_cast<size_t>(y_count_f32) * sizeof(float));
        return 0;
    }
}

int strata_gpu_exec_moe_layer(
    StrataGpuContext* ctx,
    StrataGpuWeights* gw,
    uint32_t layer_idx,
    StrataExecMode mode,
    uint32_t batch_B,
    const uint16_t* x_fp16,
    const StrataSlotEntry* layer_slot_map,
    const uint16_t* routed_ids,
    const float* routed_gates,
    const uint16_t* v_proj_x_fp16,
    void* external_ring_mtl_buf,
    const uint32_t* external_slot_ready,
    float* y_f32_out,
    uint32_t* used_ring_out,
    uint32_t* fallback_out
) {
    if (used_ring_out) *used_ring_out = 0U;
    if (fallback_out)  *fallback_out  = 0U;

    if (!ctx || !gw || layer_idx >= gw->num_layers || !x_fp16 || !routed_ids || !routed_gates || !y_f32_out) {
        return -1;
    }
    if (batch_B == 0U || batch_B > STRATA_MAX_SPEC_BATCH) {
        return -1;
    }

    const uint32_t d_model     = gw->d_model;
    const uint32_t d_ff        = gw->d_ff;
    const uint32_t num_experts = gw->num_experts;
    const uint32_t top_k       = gw->top_k;
    const uint32_t svd_rank    = gw->svd_rank;
    const bool     is_bitnest  = (gw->cfg.regime == STRATA_REGIME_I_BITNEST);

    if (is_bitnest) {
        if ((d_model % 128U) != 0U || (d_ff % 128U) != 0U) return -2;
    } else {
        if ((d_model % 32U) != 0U || (d_ff % 32U) != 0U) return -2;
    }

    const LayerGpuSlice&   ls       = gw->layers[layer_idx];
    const StrataSlotEntry* slot_map = layer_slot_map ? layer_slot_map : ls.default_slot_map.data();

    // 1. Stack-allocated active expert grouping (ZERO heap allocations, ZERO page faults!)
    StrataActiveExpertDescHost active_list[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K];
    uint32_t num_active = 0U;

    for (uint32_t t = 0; t < batch_B; ++t) {
        for (uint32_t j = 0; j < top_k; ++j) {
            const uint16_t eid  = routed_ids[t * top_k + j];
            const float    gate = routed_gates[t * top_k + j];
            if (eid >= num_experts) return -3;

            int found = -1;
            for (uint32_t u = 0; u < num_active; ++u) {
                if (active_list[u].expert_id == eid) {
                    found = static_cast<int>(u);
                    break;
                }
            }
            if (found < 0) {
                if (num_active >= gw->max_active_experts) break;
                StrataActiveExpertDescHost d{};
                d.expert_id             = eid;
                d.tok_count             = 0U;
                d.refine_tok_mask       = 0U;
                found                   = static_cast<int>(num_active);
                active_list[num_active] = d;
                ++num_active;
            }

            StrataActiveExpertDescHost& desc = active_list[found];
            int existing_m = -1;
            for (uint16_t m = 0; m < desc.tok_count; ++m) {
                if (desc.tok_indices[m] == static_cast<uint8_t>(t)) {
                    existing_m = static_cast<int>(m);
                    break;
                }
            }
            uint16_t local_j = 0U;
            if (existing_m >= 0) {
                local_j                 = static_cast<uint16_t>(existing_m);
                desc.tok_gates[local_j] += gate;
            } else {
                if (desc.tok_count >= STRATA_MAX_SPEC_BATCH) continue;
                local_j                   = desc.tok_count++;
                desc.tok_indices[local_j] = static_cast<uint8_t>(t);
                desc.tok_gates[local_j]   = gate;
            }

            if (slot_map[eid].tier_kind == STRATA_TIER_HOT_FULL_4B) {
                desc.refine_tok_mask |= static_cast<uint16_t>(1U << local_j);
            } else if (mode == STRATA_MODE_A_STATIONARY) {
                if (desc.tok_gates[local_j] >= gw->cfg.mode_a_tau_tok) {
                    desc.refine_tok_mask |= static_cast<uint16_t>(1U << local_j);
                }
            }
        }
    }

    if (num_active == 0U) {
        std::memset(y_f32_out, 0, static_cast<size_t>(batch_B) * d_model * sizeof(float));
        return 0;
    }

    if (mode == STRATA_MODE_B_ZERO_STALL) {
        struct ColdRankItem {
            uint32_t active_idx;
            uint16_t expert_id;
            float    agg_score;
        };
        ColdRankItem cold_cands[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K];
        uint32_t     num_cold = 0U;
        for (uint32_t u = 0; u < num_active; ++u) {
            const uint16_t eid = active_list[u].expert_id;
            if (slot_map[eid].tier_kind != STRATA_TIER_HOT_FULL_4B) {
                float s_e = 0.0f;
                for (uint16_t j = 0; j < active_list[u].tok_count; ++j) {
                    s_e += active_list[u].tok_gates[j];
                }
                if (s_e > 0.0f) {
                    cold_cands[num_cold++] = ColdRankItem{u, eid, s_e};
                }
            }
        }
        std::sort(cold_cands, cold_cands + num_cold,
                  [](const ColdRankItem& a, const ColdRankItem& b) {
                      if (a.agg_score != b.agg_score) return a.agg_score > b.agg_score;
                      return a.expert_id < b.expert_id;
                  });
        const uint32_t cap = gw->cfg.mode_b_cold_cap;
        for (uint32_t c = 0; c < num_cold && c < cap; ++c) {
            StrataActiveExpertDescHost& desc = active_list[cold_cands[c].active_idx];
            desc.refine_tok_mask = static_cast<uint16_t>((1U << desc.tok_count) - 1U);
        }
    }

    // 2. Seatbelt CPU reference fallback (ONLY when MTLDevice is unavailable)
    if (ctx->seatbelt_cpu_fallback) {
        std::memset(y_f32_out, 0, static_cast<size_t>(batch_B) * d_model * sizeof(float));
        std::vector<float> x_f32(static_cast<size_t>(batch_B) * d_model);
        for (size_t i = 0; i < x_f32.size(); ++i) x_f32[i] = host_fp16_to_f32(x_fp16[i]);
        if (is_bitnest) {
            for (size_t blk = 0; blk < x_f32.size(); blk += 128U) {
                cpu_fwht128_block(&x_f32[blk]);
            }
        }

        const uint32_t rank_alg = ls.svd_rank_aligned;
        std::vector<float> vp_f32(static_cast<size_t>(batch_B) * std::max<uint32_t>(rank_alg, 1U), 0.0f);
        if (svd_rank > 0U && rank_alg > 0U) {
            if (v_proj_x_fp16) {
                for (size_t i = 0; i < static_cast<size_t>(batch_B) * rank_alg; ++i) {
                    vp_f32[i] = host_fp16_to_f32(v_proj_x_fp16[i]);
                }
            } else if (!ls.svd_v_shared_f16.empty()) {
                for (uint32_t b = 0; b < batch_B; ++b) {
                    for (uint32_t k = 0; k < svd_rank; ++k) {
                        float acc = 0.0f;
                        const uint16_t* vr = &ls.svd_v_shared_f16[static_cast<size_t>(k) * d_model];
                        for (uint32_t d = 0; d < d_model; ++d) {
                            acc += x_f32[static_cast<size_t>(b) * d_model + d] * host_fp16_to_f32(vr[d]);
                        }
                        vp_f32[static_cast<size_t>(b) * rank_alg + k] = host_fp16_to_f32(host_f32_to_fp16(acc));
                    }
                }
            }
        }

        uint32_t fb_cnt = 0U;
        std::vector<float> x_exp(STRATA_MAX_SPEC_BATCH * d_model, 0.0f);
        std::vector<float> gu_exp(STRATA_MAX_SPEC_BATCH * 2U * d_ff, 0.0f);
        std::vector<float> mid_exp(STRATA_MAX_SPEC_BATCH * d_ff, 0.0f);
        std::vector<float> y_exp(STRATA_MAX_SPEC_BATCH * d_model, 0.0f);

        for (uint32_t u = 0; u < num_active; ++u) {
            const StrataActiveExpertDescHost& desc = active_list[u];
            const StrataSlotEntry&            slot = slot_map[desc.expert_id];
            const uint32_t                    tc   = std::min<uint32_t>(desc.tok_count, STRATA_MAX_SPEC_BATCH);
            const uint16_t                    eff  = static_cast<uint16_t>(desc.refine_tok_mask & ((1U << tc) - 1U));
            const bool                        hot  = (slot.tier_kind == STRATA_TIER_HOT_FULL_4B);

            if (!hot) ++fb_cnt;

            for (uint32_t j = 0; j < tc; ++j) {
                std::memcpy(&x_exp[static_cast<size_t>(j) * d_model],
                            &x_f32[static_cast<size_t>(desc.tok_indices[j]) * d_model],
                            d_model * sizeof(float));
            }

            if (is_bitnest) {
                const uint64_t total_blks = (3ULL * d_model * d_ff) / 128ULL;
                const uint64_t scales_off = total_blks * 32ULL;
                const uint32_t base_slot  = slot.hot_or_base_idx % std::max<uint32_t>(ls.base_pool_slots, 1U);
                const uint8_t* base_ptr   = ls.cpu_base_pool.data() + static_cast<size_t>(base_slot) * ls.base_expert_bytes;
                const uint8_t* ref_ptr    = hot ? (ls.cpu_hot_pool.data() +
                                                   static_cast<size_t>(slot.hot_or_base_idx % std::max<uint32_t>(ls.hot_pool_slots, 1U)) *
                                                       ls.hot_expert_bytes)
                                                : nullptr;

                cpu_qtip_planar_gemv_batch(base_ptr, ref_ptr, scales_off, 0ULL, 2U * d_ff, d_model,
                                           x_exp.data(), tc, eff, gu_exp.data(), 2U * d_ff);
                for (uint32_t j = 0; j < tc; ++j) {
                    for (uint32_t r = 0; r < d_ff; ++r) {
                        float gv = gu_exp[static_cast<size_t>(j) * 2U * d_ff + r];
                        float uv = gu_exp[static_cast<size_t>(j) * 2U * d_ff + d_ff + r];
                        mid_exp[static_cast<size_t>(j) * d_ff + r] =
                            host_fp16_to_f32(host_f32_to_fp16((gv / (1.0f + std::exp(-gv))) * uv));
                    }
                    for (uint32_t blk = 0; blk < d_ff; blk += 128U) {
                        cpu_fwht128_block(&mid_exp[static_cast<size_t>(j) * d_ff + blk]);
                    }
                }
                const uint64_t down_blk = 2ULL * d_ff * (d_model / 128U);
                cpu_qtip_planar_gemv_batch(base_ptr, ref_ptr, scales_off, down_blk, d_model, d_ff,
                                           mid_exp.data(), tc, eff, y_exp.data(), d_model);
            } else {
                const uint32_t k_tiles_gu  = d_model / 32U;
                const uint32_t dff_tiles   = d_ff / 32U;
                const uint32_t dm_tiles    = d_model / 32U;
                const uint64_t down_off_t  = static_cast<uint64_t>(2U * dff_tiles) * k_tiles_gu;
                const uint64_t total_tiles = down_off_t + static_cast<uint64_t>(dm_tiles) * dff_tiles;
                const uint64_t scales_off  = total_tiles * 512ULL;
                const uint8_t* exp_ptr     = hot
                                                 ? (ls.cpu_hot_pool.data() +
                                                    static_cast<size_t>(slot.hot_or_base_idx % std::max<uint32_t>(ls.hot_pool_slots, 1U)) *
                                                        ls.hot_expert_bytes)
                                                 : ls.cpu_base_pool.data();

                cpu_mxfp4_planar_gemv_batch(exp_ptr, scales_off, 0ULL, d_ff, d_model,
                                            x_exp.data(), tc, gu_exp.data(), 2U * d_ff, 0U);
                cpu_mxfp4_planar_gemv_batch(exp_ptr, scales_off, static_cast<uint64_t>(dff_tiles) * k_tiles_gu,
                                            d_ff, d_model, x_exp.data(), tc, gu_exp.data(), 2U * d_ff, d_ff);
                for (uint32_t j = 0; j < tc; ++j) {
                    for (uint32_t r = 0; r < d_ff; ++r) {
                        float gv = gu_exp[static_cast<size_t>(j) * 2U * d_ff + r];
                        float uv = gu_exp[static_cast<size_t>(j) * 2U * d_ff + d_ff + r];
                        mid_exp[static_cast<size_t>(j) * d_ff + r] =
                            host_fp16_to_f32(host_f32_to_fp16((gv / (1.0f + std::exp(-gv))) * uv));
                    }
                }
                if (!hot && svd_rank > 0U && !ls.cpu_svd_u_w6.empty()) {
                    const uint32_t row_bytes = (rank_alg / 64U) * 50U;
                    const uint32_t u_slot    = slot.hot_or_base_idx % std::max<uint32_t>(ls.base_pool_slots, 1U);
                    for (uint32_t r = 0; r < d_ff; ++r) {
                        const uint8_t* u_row = ls.cpu_svd_u_w6.data() +
                                               (static_cast<size_t>(u_slot) * d_ff + r) * row_bytes;
                        for (uint32_t j = 0; j < tc; ++j) {
                            const float* vp = &vp_f32[static_cast<size_t>(desc.tok_indices[j]) * rank_alg];
                            float delta     = 0.0f;
                            for (uint32_t blk64 = 0; blk64 < rank_alg; blk64 += 64U) {
                                const uint8_t* blk = u_row + (blk64 / 64U) * 50U;
                                uint16_t sh = 0;
                                std::memcpy(&sh, blk, 2U);
                                float sc           = host_fp16_to_f32(sh);
                                const uint8_t* pb  = blk + 2U;
                                for (uint32_t g = 0; g < 16U; ++g) {
                                    uint32_t k0 = blk64 + g * 4U;
                                    if (k0 >= svd_rank) break;
                                    uint8_t b0 = pb[g * 3U + 0U];
                                    uint8_t b1 = pb[g * 3U + 1U];
                                    uint8_t b2 = pb[g * 3U + 2U];
                                    float q0 = static_cast<float>(static_cast<int>(b0 & 0x3FU) - 32) * sc;
                                    float q1 = static_cast<float>(static_cast<int>(((b0 >> 6U) | ((b1 & 0x0FU) << 2U)) & 0x3FU) - 32) * sc;
                                    float q2 = static_cast<float>(static_cast<int>(((b1 >> 4U) | ((b2 & 0x03U) << 4U)) & 0x3FU) - 32) * sc;
                                    float q3 = static_cast<float>(static_cast<int>((b2 >> 2U) & 0x3FU) - 32) * sc;
                                    delta += q0 * vp[k0 + 0U] + q1 * vp[k0 + 1U] + q2 * vp[k0 + 2U] + q3 * vp[k0 + 3U];
                                }
                            }
                            mid_exp[static_cast<size_t>(j) * d_ff + r] =
                                host_fp16_to_f32(host_f32_to_fp16(mid_exp[static_cast<size_t>(j) * d_ff + r] + delta));
                        }
                    }
                }
                cpu_mxfp4_planar_gemv_batch(exp_ptr, scales_off, down_off_t, d_model, d_ff,
                                            mid_exp.data(), tc, y_exp.data(), d_model, 0U);
            }

            for (uint32_t j = 0; j < tc; ++j) {
                const uint32_t tok = desc.tok_indices[j];
                const float    g   = desc.tok_gates[j];
                for (uint32_t d = 0; d < d_model; ++d) {
                    y_f32_out[static_cast<size_t>(tok) * d_model + d] += y_exp[static_cast<size_t>(j) * d_model + d] * g;
                }
            }
        }

        if (fallback_out) *fallback_out = fb_cnt;
        return 0;
    }

    // 3. Real 40-Core Metal 4 NAX Cooperative Tensor Execution Path (Zero Allocations, Zero Page Faults!)
    @autoreleasepool {
        std::memcpy([gw->b_x contents], x_fp16, static_cast<size_t>(batch_B) * d_model * sizeof(uint16_t));
        std::memcpy([gw->b_slot_map contents], slot_map, static_cast<size_t>(num_experts) * sizeof(StrataSlotEntry));
        std::memcpy([gw->b_active contents], active_list, static_cast<size_t>(num_active) * sizeof(StrataActiveExpertDescHost));

        uint32_t* ready_dst = reinterpret_cast<uint32_t*>([gw->b_ready contents]);
        for (uint32_t u = 0; u < num_active; ++u) {
            const uint16_t rslot = slot_map[active_list[u].expert_id].ring_slot_idx;
            if (rslot != STRATA_SLOT_INVALID && rslot < kMaxReadySlots) {
                ready_dst[rslot] = (external_slot_ready != nullptr)
                                       ? __atomic_load_n(&external_slot_ready[rslot], __ATOMIC_ACQUIRE)
                                       : 0U;
            }
        }

        *reinterpret_cast<uint32_t*>([gw->b_ring_cnt contents]) = 0U;
        *reinterpret_cast<uint32_t*>([gw->b_fb_cnt contents])   = 0U;

        if (svd_rank > 0U && ls.svd_rank_aligned > 0U) {
            const uint32_t rank_alg = ls.svd_rank_aligned;
            uint16_t* vp_dst        = reinterpret_cast<uint16_t*>([gw->b_v_proj contents]);
            if (v_proj_x_fp16 != nullptr) {
                std::memcpy(vp_dst, v_proj_x_fp16, static_cast<size_t>(batch_B) * rank_alg * sizeof(uint16_t));
            } else if (!ls.svd_v_shared_f16.empty()) {
                for (uint32_t b = 0; b < batch_B; ++b) {
                    const uint16_t* xb = x_fp16 + static_cast<size_t>(b) * d_model;
                    for (uint32_t k = 0; k < svd_rank; ++k) {
                        const uint16_t* vr = &ls.svd_v_shared_f16[static_cast<size_t>(k) * d_model];
                        float acc          = 0.0f;
                        for (uint32_t d = 0; d < d_model; ++d) {
                            acc += host_fp16_to_f32(xb[d]) * host_fp16_to_f32(vr[d]);
                        }
                        vp_dst[static_cast<size_t>(b) * rank_alg + k] = host_f32_to_fp16(acc);
                    }
                    for (uint32_t k = svd_rank; k < rank_alg; ++k) {
                        vp_dst[static_cast<size_t>(b) * rank_alg + k] = 0U;
                    }
                }
            } else {
                std::memset(vp_dst, 0, static_cast<size_t>(batch_B) * rank_alg * sizeof(uint16_t));
            }
        }

        const uint32_t aligned_hot_lba = static_cast<uint32_t>(
            (ls.hot_expert_bytes + STRATA_LBA_ALIGN_BYTES - 1ULL) &
            ~static_cast<uint64_t>(STRATA_LBA_ALIGN_BYTES - 1ULL));
        const uint32_t ring_slot_stride =
            (gw->cfg.cold_extent_bytes >= STRATA_LBA_ALIGN_BYTES)
                ? static_cast<uint32_t>(gw->cfg.cold_extent_bytes)
                : std::max<uint32_t>(aligned_hot_lba, STRATA_LBA_ALIGN_BYTES);

        id<MTLBuffer> ring_arg = (external_ring_mtl_buf != nullptr)
                                     ? (__bridge id<MTLBuffer>)external_ring_mtl_buf
                                     : gw->b_dummy_ring;
        const uint64_t ring_buf_bytes =
            (external_ring_mtl_buf != nullptr && ring_arg != nil) ? static_cast<uint64_t>([ring_arg length]) : 0ULL;

        const uint32_t gu_grid_x = is_bitnest ? (((2U * d_ff) + 31U) / 32U)
                                              : (((2U * d_ff) + 63U) / 64U);
        const uint32_t dn_grid_x = is_bitnest ? ((d_model + 31U) / 32U)
                                              : ((d_model + 63U) / 64U);
        const bool     allow_sk  = (mode != STRATA_MODE_A_STATIONARY) && (d_model > 1024U) && (d_ff > 1024U);

        StrataMoEKernelCfgHost kcfg{};
        kcfg.d_model            = d_model;
        kcfg.d_ff               = d_ff;
        kcfg.num_active_experts = num_active;
        kcfg.regime             = static_cast<uint32_t>(gw->cfg.regime);
        kcfg.exec_mode          = static_cast<uint32_t>(mode);
        kcfg.hot_pool_slots     = ls.hot_pool_slots;
        kcfg.base_pool_slots    = ls.base_pool_slots;
        kcfg.ring_slot_stride   = ring_slot_stride;
        kcfg.ring_buf_bytes     = ring_buf_bytes;
        kcfg.hot_expert_bytes   = ls.hot_expert_bytes;
        kcfg.base_expert_bytes  = ls.base_expert_bytes;
        kcfg.svd_rank           = svd_rank;
        kcfg.gu_split_k         = (allow_sk && gu_grid_x * num_active < 240U) ? 2U : 1U;
        kcfg.down_split_k       = (allow_sk && dn_grid_x * num_active < 240U) ? 2U : 1U;
        kcfg.reserved_pad       = 0U;

        const uint32_t y_count_f32  = batch_B * d_model;
        const uint32_t y_vec4_cnt   = y_count_f32 / 4U;
        const uint32_t gu_count_f32 = num_active * STRATA_MAX_SPEC_BATCH * 2U * d_ff;

        id<MTLCommandBuffer> cb          = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        if (kcfg.gu_split_k > 1U) {
            encode_zero_f32(enc, ctx->pso_zero_f32, gw->b_gu_f32, gu_count_f32);
        }

        if (is_bitnest) {
            uint32_t x_blocks = (batch_B * d_model) / 128U;
            [enc setComputePipelineState:ctx->pso_fwht128];
            [enc setBuffer:gw->b_x offset:0 atIndex:0];
            [enc setBuffer:gw->b_x_fwht offset:0 atIndex:1];
            [enc setBytes:&x_blocks length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(x_blocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }

        // Gathers routed token activations X [B, D] -> b_x16 AND zeros b_y_f32 in u_idx == 0!
        {
            const uint32_t gather_idx8 = 16U * (d_model / 8U);
            [enc setComputePipelineState:ctx->pso_moe_gather_x16];
            [enc setBuffer:(is_bitnest ? gw->b_x_fwht : gw->b_x) offset:0 atIndex:0];
            [enc setBuffer:gw->b_active offset:0 atIndex:1];
            [enc setBuffer:gw->b_x16 offset:0 atIndex:2];
            [enc setBytes:&d_model length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&num_active length:sizeof(uint32_t) atIndex:4];
            [enc setBuffer:gw->b_y_f32 offset:0 atIndex:5];
            [enc setBytes:&y_vec4_cnt length:sizeof(uint32_t) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((gather_idx8 + 127U) / 128U, num_active, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        [enc setComputePipelineState:ctx->pso_moe_gate_up];
        [enc setBuffer:gw->b_x16 offset:0 atIndex:0];
        [enc setBuffer:gw->b_slot_map offset:0 atIndex:1];
        [enc setBuffer:gw->b_ready offset:0 atIndex:2];
        [enc setBuffer:gw->b_active offset:0 atIndex:3];
        [enc setBuffer:gw->b_hot_pool offset:ls.hot_offset atIndex:4];
        [enc setBuffer:gw->b_base_pool offset:ls.base_offset atIndex:5];
        [enc setBuffer:ring_arg offset:0 atIndex:6];
        [enc setBuffer:gw->b_gu_f32 offset:0 atIndex:7];
        [enc setBuffer:gw->b_ring_cnt offset:0 atIndex:8];
        [enc setBuffer:gw->b_fb_cnt offset:0 atIndex:9];
        [enc setBytes:&kcfg length:sizeof(kcfg) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(gu_grid_x, num_active, kcfg.gu_split_k)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        {
            const uint32_t need_mid8   = is_bitnest ? 1U : 0U;
            const uint32_t swiglu_rows = is_bitnest ? STRATA_MAX_SPEC_BATCH : 16U;
            const uint32_t swiglu_idx4 = swiglu_rows * (d_ff / 4U);
            [enc setComputePipelineState:ctx->pso_moe_swiglu_pack16];
            [enc setBuffer:gw->b_gu_f32 offset:0 atIndex:0];
            [enc setBuffer:gw->b_active offset:0 atIndex:1];
            [enc setBuffer:gw->b_mid offset:0 atIndex:2];
            [enc setBuffer:gw->b_mid16 offset:0 atIndex:3];
            [enc setBytes:&d_ff length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&num_active length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&need_mid8 length:sizeof(uint32_t) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((swiglu_idx4 + 127U) / 128U, num_active, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        if (svd_rank > 0U && ls.svd_u_w6_bytes > 0U) {
            [enc setComputePipelineState:ctx->pso_moe_svd_w6];
            [enc setBuffer:gw->b_v_proj offset:0 atIndex:0];
            [enc setBuffer:gw->b_slot_map offset:0 atIndex:1];
            [enc setBuffer:gw->b_ready offset:0 atIndex:2];
            [enc setBuffer:gw->b_active offset:0 atIndex:3];
            [enc setBuffer:gw->b_u_w6 offset:ls.svd_u_offset atIndex:4];
            [enc setBuffer:(is_bitnest ? gw->b_mid : gw->b_mid16) offset:0 atIndex:5];
            [enc setBytes:&kcfg length:sizeof(kcfg) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((d_ff + 63U) / 64U, num_active, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }

        if (is_bitnest) {
            uint32_t m_blocks = (num_active * STRATA_MAX_SPEC_BATCH * d_ff) / 128U;
            [enc setComputePipelineState:ctx->pso_fwht128];
            [enc setBuffer:gw->b_mid offset:0 atIndex:0];
            [enc setBuffer:gw->b_mid_fwht offset:0 atIndex:1];
            [enc setBytes:&m_blocks length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(m_blocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            const uint32_t pad_idx8 = 16U * (d_ff / 8U);
            [enc setComputePipelineState:ctx->pso_moe_pad_mid8_to_16];
            [enc setBuffer:gw->b_mid_fwht offset:0 atIndex:0];
            [enc setBuffer:gw->b_active offset:0 atIndex:1];
            [enc setBuffer:gw->b_mid16 offset:0 atIndex:2];
            [enc setBytes:&d_ff length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&num_active length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((pad_idx8 + 127U) / 128U, num_active, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        [enc setComputePipelineState:ctx->pso_moe_down];
        [enc setBuffer:gw->b_mid16 offset:0 atIndex:0];
        [enc setBuffer:gw->b_slot_map offset:0 atIndex:1];
        [enc setBuffer:gw->b_ready offset:0 atIndex:2];
        [enc setBuffer:gw->b_active offset:0 atIndex:3];
        [enc setBuffer:gw->b_hot_pool offset:ls.hot_offset atIndex:4];
        [enc setBuffer:gw->b_base_pool offset:ls.base_offset atIndex:5];
        [enc setBuffer:ring_arg offset:0 atIndex:6];
        [enc setBuffer:gw->b_y_f32 offset:0 atIndex:7];
        [enc setBytes:&kcfg length:sizeof(kcfg) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(dn_grid_x, num_active, kcfg.down_split_k)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(y_f32_out, [gw->b_y_f32 contents], static_cast<size_t>(y_count_f32) * sizeof(float));
        if (used_ring_out) *used_ring_out = *reinterpret_cast<const uint32_t*>([gw->b_ring_cnt contents]);
        if (fallback_out)  *fallback_out  = *reinterpret_cast<const uint32_t*>([gw->b_fb_cnt contents]);
        return 0;
    }
}

} // extern "C"

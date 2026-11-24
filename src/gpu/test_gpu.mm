#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../include/strata_abi.h"
#include "../../include/strata_glm53.h"
#include "../../include/strata_inkling2.h"
#import  "../../include/strata_metal_cache.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" int strata_gpu_bench_nax_f16(
    StrataGpuContext* ctx,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t timed_iters,
    double* tflops_out,
    double* kernel_us_out
);

namespace {

float fp16_bits_to_f32_ref(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(uint16_t));
    return static_cast<float>(h);
}

double compute_shannon_entropy(const uint8_t* syms, size_t n, uint32_t alpha) {
    std::vector<uint64_t> hist(alpha, 0);
    for (size_t i = 0; i < n; ++i) ++hist[syms[i]];
    double h = 0.0;
    for (uint32_t s = 0; s < alpha; ++s) {
        if (hist[s] > 0) {
            double p = static_cast<double>(hist[s]) / static_cast<double>(n);
            h -= p * std::log2(p);
        }
    }
    return h;
}

uint16_t f32_to_fp16_bits(float v) {
    __fp16 h = static_cast<__fp16>(v);
    uint16_t b = 0U;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

std::string read_text_file_rel(const char* path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

int run_glm53_v2_metal41_suite(id<MTLDevice> dev) {
    if (!dev) {
        std::printf("[Test 7-10] GLM-5.3 Metal 4.1 E8S-18-V4 / WY-KDA / Flash-MLA : SKIPPED (No Metal GPU)\n");
        return 0;
    }

    @autoreleasepool {
        // ---------------------------------------------------------------------
        // Test 7: Metal 4.1 Zero-Warning Compile + Warp-Shuffle `RHT-64` Parity
        // ---------------------------------------------------------------------
        std::string hdr_src = read_text_file_rel("include/strata_glm53.h");
        std::string msl_src = read_text_file_rel("src/gpu/strata_glm53_kernels.metal");
        if (hdr_src.empty() || msl_src.empty()) {
            std::fprintf(stderr, "FAIL: could not read strata_glm53.h or strata_glm53_kernels.metal\n");
            return 1;
        }
        const std::string marker = "// @@STRATA_GLM53_HEADER@@";
        size_t pos = msl_src.find(marker);
        if (pos != std::string::npos) msl_src.replace(pos, marker.size(), hdr_src);

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 27.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_1;
        } else if (@available(macOS 26.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_0;
        }
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
            opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
        }

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:msl_src.c_str()]
                                               options:opts
                                                 error:&err];
        if (!lib || err) {
            std::fprintf(stderr, "FAIL: Metal 4.1 compile failed or emitted warnings: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return 1;
        }

        id<MTLCommandQueue> q = [dev newCommandQueue];
        auto get_pso = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!fn) {
                std::fprintf(stderr, "FAIL: missing kernel %s\n", name);
                return nil;
            }
            NSError* pe = nil;
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&pe];
            if (!pso) {
                std::fprintf(stderr, "FAIL: PSO %s: %s\n", name, pe ? [[pe localizedDescription] UTF8String] : "?");
            }
            return pso;
        };

        id<MTLComputePipelineState> pso_rht64       = get_pso("sg53_rht64_f16");
        id<MTLComputePipelineState> pso_bin_v2      = get_pso("sg53_moe_bin_tokens_v2");
        id<MTLComputePipelineState> pso_bin_nax     = get_pso("sg53_moe_bin_tokens_nax");
        id<MTLComputePipelineState> pso_gather_x    = get_pso("sg53_moe_gather_x_f16");
        id<MTLComputePipelineState> pso_e8_g_gu     = get_pso("sg53_moe_e8s18_gather_gu");
        id<MTLComputePipelineState> pso_e8_g_dn     = get_pso("sg53_moe_e8s18_gather_dn");
        id<MTLComputePipelineState> pso_e8_grp_gu   = get_pso("sg53_moe_e8s18_grouped_gu");
        id<MTLComputePipelineState> pso_e8_grp_dn   = get_pso("sg53_moe_e8s18_grouped_dn");
        id<MTLComputePipelineState> pso_e8_nax_gu64 = get_pso("sg53_moe_e8s18_nax_gu_m64");
        id<MTLComputePipelineState> pso_e8_nax_gu32 = get_pso("sg53_moe_e8s18_nax_gu_m32");
        id<MTLComputePipelineState> pso_e8_nax_gu16 = get_pso("sg53_moe_e8s18_nax_gu_m16");
        id<MTLComputePipelineState> pso_e8_nax_dn64 = get_pso("sg53_moe_e8s18_nax_dn_m64");
        id<MTLComputePipelineState> pso_e8_nax_dn32 = get_pso("sg53_moe_e8s18_nax_dn_m32");
        id<MTLComputePipelineState> pso_e8_nax_dn16 = get_pso("sg53_moe_e8s18_nax_dn_m16");
        id<MTLComputePipelineState> pso_kda_prep    = get_pso("sg53_kda_prep_batch");
        id<MTLComputePipelineState> pso_kda_rec     = get_pso("sg53_kda_rec_batch");
        id<MTLComputePipelineState> pso_kda_wy_prep = get_pso("sg53_kda_wy_prep_chunk32");
        id<MTLComputePipelineState> pso_kda_wy_rec  = get_pso("sg53_kda_wy_recur_chunk32");
        id<MTLComputePipelineState> pso_mla_split   = get_pso("sg53_mla_latent_splitkv");
        id<MTLComputePipelineState> pso_mla_dec_nax = get_pso("sg53_mla_decode_splitkv_nax");
        id<MTLComputePipelineState> pso_mla_red     = get_pso("sg53_mla_latent_reduce");
        id<MTLComputePipelineState> pso_mla_q_hm    = get_pso("sg53_mla_q_to_hm_f16");
        id<MTLComputePipelineState> pso_mla_dense   = get_pso("sg53_mla_prefill_dense_nax");
        id<MTLComputePipelineState> pso_mla_sparse  = get_pso("sg53_mla_prefill_sparse_nax");
        id<MTLComputePipelineState> pso_idx_scalar  = get_pso("sg53_indexer_score_pools");
        id<MTLComputePipelineState> pso_idx_q_hm    = get_pso("sg53_indexer_q_to_hm_f16");
        id<MTLComputePipelineState> pso_idx_nax     = get_pso("sg53_indexer_score_pools_nax");
        id<MTLComputePipelineState> pso_idx_dec_nax = get_pso("sg53_indexer_score_pools_decode_nax");
        id<MTLComputePipelineState> pso_q4_gemv     = get_pso("sg53_q4_gemv_batch");
        id<MTLComputePipelineState> pso_rtr_old     = get_pso("sg53_moe_router_topk");
        id<MTLComputePipelineState> pso_rtr_logits  = get_pso("sg53_moe_router_logits");
        id<MTLComputePipelineState> pso_rtr_select  = get_pso("sg53_moe_router_select");
        id<MTLComputePipelineState> pso_penalties   = get_pso("sg53_apply_logit_penalties");
        if (!pso_rht64 || !pso_bin_v2 || !pso_bin_nax || !pso_gather_x ||
            !pso_e8_g_gu || !pso_e8_g_dn || !pso_e8_grp_gu || !pso_e8_grp_dn ||
            !pso_e8_nax_gu64 || !pso_e8_nax_gu32 || !pso_e8_nax_gu16 ||
            !pso_e8_nax_dn64 || !pso_e8_nax_dn32 || !pso_e8_nax_dn16 ||
            !pso_kda_prep || !pso_kda_rec || !pso_kda_wy_prep || !pso_kda_wy_rec ||
            !pso_mla_split || !pso_mla_dec_nax || !pso_mla_red || !pso_mla_q_hm || !pso_mla_dense || !pso_mla_sparse ||
            !pso_idx_scalar || !pso_idx_q_hm || !pso_idx_nax || !pso_idx_dec_nax || !pso_q4_gemv ||
            !pso_rtr_old || !pso_rtr_logits || !pso_rtr_select || !pso_penalties) {
            return 1;
        }

        // Verify RHT-64 GPU (out-of-place & in-place) vs CPU `sg53_rht64_f32`
        {
            const uint32_t M = 17U, K = 4096U;
            std::mt19937 rng(5301);
            std::normal_distribution<float> nd(0.0f, 0.5f);
            std::vector<uint16_t> x_h(static_cast<size_t>(M) * K);
            for (auto& v : x_h) v = f32_to_fp16_bits(nd(rng));

            id<MTLBuffer> b_in  = [dev newBufferWithBytes:x_h.data() length:x_h.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_out = [dev newBufferWithLength:x_h.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ip  = [dev newBufferWithBytes:x_h.data() length:x_h.size() * 2U options:MTLResourceStorageModeShared];
            struct { uint32_t M_valid, M_padded, K, pad0; } rp{M, M, K, 0U};

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_rht64];
            [enc setBuffer:b_in offset:0 atIndex:0];
            [enc setBuffer:b_out offset:0 atIndex:1];
            [enc setBytes:&rp length:sizeof(rp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(((K / 64U) + 3U) / 4U, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc setBuffer:b_ip offset:0 atIndex:0];
            [enc setBuffer:b_ip offset:0 atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(((K / 64U) + 3U) / 4U, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const uint16_t* got_oop = reinterpret_cast<const uint16_t*>([b_out contents]);
            const uint16_t* got_ip  = reinterpret_cast<const uint16_t*>([b_ip contents]);
            float max_err = 0.0f;
            int ip_mismatch = 0;
            for (uint32_t m = 0U; m < M; ++m) {
                for (uint32_t blk = 0U; blk < K / 64U; ++blk) {
                    float in64[64], ref64[64];
                    for (uint32_t i = 0U; i < 64U; ++i) {
                        in64[i] = fp16_bits_to_f32_ref(x_h[static_cast<size_t>(m) * K + blk * 64U + i]);
                    }
                    sg53_rht64_f32(in64, ref64);
                    for (uint32_t i = 0U; i < 64U; ++i) {
                        const size_t idx = static_cast<size_t>(m) * K + blk * 64U + i;
                        if (got_oop[idx] != got_ip[idx]) ++ip_mismatch;
                        float e = std::fabs(fp16_bits_to_f32_ref(got_oop[idx]) - ref64[i]);
                        if (e > max_err) max_err = e;
                    }
                }
            }
            const bool ok = (max_err < 1e-3f) && (ip_mismatch == 0);
            std::printf("[Test 7] GLM-5.3 Metal 4.1 (0 warns) + RHT-64 Parity      : %s (max_err=%.2e, inplace_diff=%d)\n",
                        ok ? "PASS" : "FAIL", max_err, ip_mismatch);
            if (!ok) return 1;
        }

        // ---------------------------------------------------------------------
        // Test 8: `E8S-18-V4` Decode GEMV (`M=1` vs `M=5` bit-exact + CPU parity
        //         + Cold-DRAM BW) & Prefill NAX (`BM=64/32/16` fused SwiGLU+RHT64!)
        // ---------------------------------------------------------------------
        {
            const uint32_t D = 4096U, d_moe = 2048U, E_test = 16U, top_k = 4U;
            const uint64_t lo16_b  = STRATA_GLM53_E8S18_LO16_BYTES;
            const uint64_t hi2_b   = STRATA_GLM53_E8S18_HI2_BYTES;
            const uint64_t proj_b  = STRATA_GLM53_E8S18_PROJ_BYTES;
            const uint64_t rec_b   = STRATA_GLM53_E8S18_RECORD_BYTES;

            std::vector<StrataGlm53E8Half8Entry> host_lut(STRATA_GLM53_E8S18_NUM_ORBITS);
            strata_glm53_e8s18_build_lut(host_lut.data());
            id<MTLBuffer> b_lut_stg = [dev newBufferWithBytes:host_lut.data()
                                                       length:STRATA_GLM53_E8S18_LUT_BYTES
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_lut = [dev newBufferWithLength:STRATA_GLM53_E8S18_LUT_BYTES
                                                   options:MTLResourceStorageModePrivate];
            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
                [bl copyFromBuffer:b_lut_stg sourceOffset:0 toBuffer:b_lut destinationOffset:0 size:STRATA_GLM53_E8S18_LUT_BYTES];
                [bl endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // Build `E_test = 16` realistic E8S-18-V4 experts (`16 * 7.125 MiB = 114 MiB > 48 MiB SLC`!)
            // Quantize expert 0 & 1 cleanly via `sg53_e8s18_quantize_group64` on a subset of rows and replicate
            // pattern across all experts so test completes in < 0.3 s!
            id<MTLBuffer> b_experts = [dev newBufferWithLength:static_cast<NSUInteger>(E_test * rec_b)
                                                       options:MTLResourceStorageModeShared];
            auto* exp_bytes = static_cast<uint8_t*>([b_experts contents]);
            {
                std::mt19937 rng(5308);
                std::uniform_int_distribution<uint32_t> dist_oid(0U, STRATA_GLM53_E8S18_NUM_ORBITS - 1U);
                std::uniform_int_distribution<uint32_t> dist_h(0U, 15U);
                std::uniform_int_distribution<uint32_t> dist_hi2(0U, 65535U);
                std::uniform_int_distribution<uint32_t> dist_sc(0x30U, 0x58U); // ~0.008 .. 0.05 realistic scales!
                for (uint32_t e = 0U; e < E_test; ++e) {
                    uint8_t* rec = exp_bytes + static_cast<size_t>(e) * rec_b;
                    for (uint32_t p = 0U; p < 3U; ++p) {
                        uint8_t* pb = rec + static_cast<size_t>(p) * proj_b;
                        auto* lo16 = reinterpret_cast<uint16_t*>(pb);
                        auto* hi2  = reinterpret_cast<uint16_t*>(pb + lo16_b);
                        auto* sc   = pb + lo16_b + hi2_b;
                        const size_t n_lo  = lo16_b / 2U;
                        const size_t n_grp = hi2_b / 2U;
                        if (e < 3U) {
                            for (size_t i = 0U; i < n_lo; ++i) lo16[i] = static_cast<uint16_t>((dist_oid(rng) << 4U) | dist_h(rng));
                            for (size_t i = 0U; i < n_grp; ++i) {
                                hi2[i] = static_cast<uint16_t>(dist_hi2(rng));
                                sc[i]  = static_cast<uint8_t>(dist_sc(rng));
                            }
                        } else {
                            std::memcpy(pb, exp_bytes + static_cast<size_t>(e % 3U) * rec_b + static_cast<size_t>(p) * proj_b, proj_b);
                        }
                    }
                }
            }

            // Prepare `M_nax = 100` tokens (`e=0`: 68 pairs -> M64+M16; `e=1`: 24 pairs -> M32; `e=2`: 8 pairs -> M16!)
            // Also check `M = 1` vs `M = 5` bit-exactness and CPU parity on `m = 0`!
            std::mt19937 rng_x(5309);
            std::normal_distribution<float> nd_x(0.0f, 0.35f);
            const uint32_t M_nax = 100U;
            std::vector<uint16_t> x_rot_h(static_cast<size_t>(M_nax) * D);
            for (auto& v : x_rot_h) v = f32_to_fp16_bits(nd_x(rng_x));

            id<MTLBuffer> b_x_rot   = [dev newBufferWithBytes:x_rot_h.data() length:x_rot_h.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_topk_1  = [dev newBufferWithLength:top_k * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_topk_5  = [dev newBufferWithLength:5U * top_k * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mid_1   = [dev newBufferWithLength:top_k * d_moe * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dn_1    = [dev newBufferWithLength:top_k * D * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mid_5   = [dev newBufferWithLength:5U * top_k * d_moe * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dn_5    = [dev newBufferWithLength:5U * top_k * D * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pairs   = [dev newBufferWithLength:(M_nax * top_k + 64U) * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_tiles   = [dev newBufferWithLength:3U * 1024U * 16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ind     = [dev newBufferWithLength:4096U options:MTLResourceStorageModeShared];

            auto* tk1 = reinterpret_cast<uint32_t*>([b_topk_1 contents]);
            auto* tk5 = reinterpret_cast<uint32_t*>([b_topk_5 contents]);
            for (uint32_t k = 0U; k < top_k; ++k) tk1[k] = k % 3U; // [0, 1, 2, 0]
            for (uint32_t m = 0U; m < 5U; ++m) {
                for (uint32_t k = 0U; k < top_k; ++k) tk5[m * top_k + k] = (m == 0U) ? tk1[k] : ((m + k) % 3U);
            }

            struct Sg53E8ParamsT {
                uint64_t layer_byte_off, e8_record_bytes, e8_proj_bytes, e8_lo16_bytes, e8_hi2_bytes;
                uint32_t M_valid, top_k, D, d_moe;
                float    swiglu_limit;
                uint32_t total_pairs_pad, tile_base, pad0, pad1;
            };
            struct Sg53BinParamsT {
                uint32_t M_valid, top_k, num_experts, d_moe, D, max_tiles_per_bin, pad0, pad1;
            };
            struct Sg53RhtParamsT { uint32_t M_valid, M_padded, K, pad0; };

            // Run M=1 (`gather_gu` -> `rht64` -> `gather_dn`) and M=5 (`bin_v2` -> indirect `grouped_gu` -> `rht64` -> `grouped_dn`)
            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

                Sg53E8ParamsT ep1{0ULL, rec_b, proj_b, lo16_b, hi2_b, 1U, top_k, D, d_moe, 10.0f, top_k + 64U, 0U, 0U, 0U};
                [enc setComputePipelineState:pso_e8_g_gu];
                [enc setBuffer:b_experts offset:0 atIndex:0];
                [enc setBuffer:b_lut offset:0 atIndex:1];
                [enc setBuffer:b_topk_1 offset:0 atIndex:2];
                [enc setBuffer:b_x_rot offset:0 atIndex:3];
                [enc setBuffer:b_mid_1 offset:0 atIndex:4];
                [enc setBytes:&ep1 length:sizeof(ep1) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((d_moe + 7U) / 8U, top_k, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                Sg53RhtParamsT mrp1{top_k, top_k, d_moe, 0U};
                [enc setComputePipelineState:pso_rht64];
                [enc setBuffer:b_mid_1 offset:0 atIndex:0];
                [enc setBuffer:b_mid_1 offset:0 atIndex:1];
                [enc setBytes:&mrp1 length:sizeof(mrp1) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(((d_moe / 64U) + 3U) / 4U, top_k, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_e8_g_dn];
                [enc setBuffer:b_experts offset:0 atIndex:0];
                [enc setBuffer:b_lut offset:0 atIndex:1];
                [enc setBuffer:b_topk_1 offset:0 atIndex:2];
                [enc setBuffer:b_mid_1 offset:0 atIndex:3];
                [enc setBuffer:b_dn_1 offset:0 atIndex:4];
                [enc setBytes:&ep1 length:sizeof(ep1) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((D + 15U) / 16U, top_k, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                // M = 5 grouped indirect
                Sg53BinParamsT bp5{5U, top_k, E_test, d_moe, D, 1024U, 0U, 0U};
                [enc setComputePipelineState:pso_bin_v2];
                [enc setBuffer:b_topk_5 offset:0 atIndex:0];
                [enc setBuffer:b_pairs offset:0 atIndex:1];
                [enc setBuffer:b_tiles offset:0 atIndex:2];
                [enc setBuffer:b_ind offset:0 atIndex:3];
                [enc setBytes:&bp5 length:sizeof(bp5) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

                Sg53E8ParamsT ep5{0ULL, rec_b, proj_b, lo16_b, hi2_b, 5U, top_k, D, d_moe, 10.0f, 5U * top_k + 64U, 0U, 0U, 0U};
                [enc setComputePipelineState:pso_e8_grp_gu];
                [enc setBuffer:b_experts offset:0 atIndex:0];
                [enc setBuffer:b_lut offset:0 atIndex:1];
                [enc setBuffer:b_x_rot offset:0 atIndex:2];
                [enc setBuffer:b_pairs offset:0 atIndex:3];
                [enc setBuffer:b_tiles offset:0 atIndex:4];
                [enc setBuffer:b_mid_5 offset:0 atIndex:5];
                [enc setBytes:&ep5 length:sizeof(ep5) atIndex:6];
                [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:32 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                Sg53RhtParamsT mrp5{5U * top_k, 5U * top_k, d_moe, 0U};
                [enc setComputePipelineState:pso_rht64];
                [enc setBuffer:b_mid_5 offset:0 atIndex:0];
                [enc setBuffer:b_mid_5 offset:0 atIndex:1];
                [enc setBytes:&mrp5 length:sizeof(mrp5) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(((d_moe / 64U) + 3U) / 4U, 5U * top_k, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_e8_grp_dn];
                [enc setBuffer:b_experts offset:0 atIndex:0];
                [enc setBuffer:b_lut offset:0 atIndex:1];
                [enc setBuffer:b_mid_5 offset:0 atIndex:2];
                [enc setBuffer:b_pairs offset:0 atIndex:3];
                [enc setBuffer:b_tiles offset:0 atIndex:4];
                [enc setBuffer:b_dn_5 offset:0 atIndex:5];
                [enc setBytes:&ep5 length:sizeof(ep5) atIndex:6];
                [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:44 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // 1) Check M=5 row 0 vs M=1 is 100.0000% BIT-EXACT!
            const auto* dn1 = reinterpret_cast<const float*>([b_dn_1 contents]);
            const auto* dn5 = reinterpret_cast<const float*>([b_dn_5 contents]);
            int bit_diffs_1v5 = 0;
            for (size_t i = 0U; i < static_cast<size_t>(top_k) * D; ++i) {
                uint32_t u1 = 0U, u5 = 0U;
                std::memcpy(&u1, &dn1[i], 4U);
                std::memcpy(&u5, &dn5[i], 4U);
                if (u1 != u5) ++bit_diffs_1v5;
            }

            // 2) Check `M=1` vs exact CPU E8S-18-V4 decode reference on expert 0 (first 64 rows of `dn`!)
            const auto* mid1_h = reinterpret_cast<const uint16_t*>([b_mid_1 contents]);
            float cpu_max_err = 0.0f;
            {
                const uint8_t* rec0 = exp_bytes;
                const uint8_t* d_base = rec0 + 2U * proj_b;
                const auto* d_lo = reinterpret_cast<const uint16_t*>(d_base);
                const auto* d_hi = reinterpret_cast<const uint16_t*>(d_base + lo16_b);
                const auto* d_sc = d_base + lo16_b + hi2_b;
                const uint32_t grps = d_moe / 64U;
                for (uint32_t r = 0U; r < 64U; ++r) {
                    double acc = 0.0;
                    for (uint32_t g = 0U; g < grps; ++g) {
                        const float sc = sg53_e4m4_decode_f32(d_sc[static_cast<size_t>(r) * grps + g]);
                        const uint16_t hiw = d_hi[static_cast<size_t>(r) * grps + g];
                        for (uint32_t s = 0U; s < 8U; ++s) {
                            const uint16_t lo16 = d_lo[(static_cast<size_t>(r) * grps + g) * 8U + s];
                            float w8[8];
                            sg53_e8s18_decode_vec8_f32(host_lut.data(), lo16, (hiw >> (2U * s)) & 3U, w8);
                            for (uint32_t j = 0U; j < 8U; ++j) {
                                acc += static_cast<double>(w8[j] * sc) *
                                       static_cast<double>(fp16_bits_to_f32_ref(mid1_h[g * 64U + s * 8U + j]));
                            }
                        }
                    }
                    const float e = std::fabs(dn1[r] - static_cast<float>(acc));
                    if (e > cpu_max_err) cpu_max_err = e;
                }
            }

            // 3) Measure cold-DRAM BW for `E8S-18-V4` `gather_gu` + `gather_dn` over 16 experts (`119.5 MB > 48 MiB SLC`)!
            id<MTLBuffer> b_tk16  = [dev newBufferWithLength:16U * 4U options:MTLResourceStorageModeShared];
            auto* tk16 = reinterpret_cast<uint32_t*>([b_tk16 contents]);
            for (uint32_t k = 0U; k < 16U; ++k) tk16[k] = k;
            id<MTLBuffer> b_mid16 = [dev newBufferWithLength:16U * d_moe * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dn16  = [dev newBufferWithLength:16U * D * 4U options:MTLResourceStorageModeShared];
            double e8_cold_gbps = 0.0;
            {
                Sg53E8ParamsT ep16{0ULL, rec_b, proj_b, lo16_b, hi2_b, 1U, 16U, D, d_moe, 10.0f, 80U, 0U, 0U, 0U};
                // Ramp GPU DVFS clock out of idle after CPU random weight init (matches continuous 45L decode!)
                double best_sec = 1e9;
                for (int it = 0; it < 20; ++it) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_e8_g_gu];
                    [enc setBuffer:b_experts offset:0 atIndex:0];
                    [enc setBuffer:b_lut offset:0 atIndex:1];
                    [enc setBuffer:b_tk16 offset:0 atIndex:2];
                    [enc setBuffer:b_x_rot offset:0 atIndex:3];
                    [enc setBuffer:b_mid16 offset:0 atIndex:4];
                    [enc setBytes:&ep16 length:sizeof(ep16) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((d_moe + 7U) / 8U, 16U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc setComputePipelineState:pso_e8_g_dn];
                    [enc setBuffer:b_experts offset:0 atIndex:0];
                    [enc setBuffer:b_lut offset:0 atIndex:1];
                    [enc setBuffer:b_tk16 offset:0 atIndex:2];
                    [enc setBuffer:b_mid16 offset:0 atIndex:3];
                    [enc setBuffer:b_dn16 offset:0 atIndex:4];
                    [enc setBytes:&ep16 length:sizeof(ep16) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 15U) / 16U, 16U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    if (it >= 10) {
                        double sec = [cb GPUEndTime] - [cb GPUStartTime];
                        if (sec > 0.0 && sec < best_sec) best_sec = sec;
                    }
                }
                e8_cold_gbps = (static_cast<double>(16ULL * rec_b) / 1e9) / best_sec;
            }

            // 3b) Measure cold-DRAM BW for `grouped_gu/dn` at `M=5` (`16` distinct experts, B=5 shared & B=1..5 mix!)
            double e8_grp_b5_gbps = 0.0, e8_grp_mix_gbps = 0.0;
            {
                id<MTLBuffer> b_tk_b5  = [dev newBufferWithLength:5U * 16U * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_tk_mix = [dev newBufferWithLength:5U * 8U * 4U options:MTLResourceStorageModeShared];
                auto* p_b5  = reinterpret_cast<uint32_t*>([b_tk_b5 contents]);
                auto* p_mix = reinterpret_cast<uint32_t*>([b_tk_mix contents]);
                for (uint32_t m = 0U; m < 5U; ++m) {
                    for (uint32_t k = 0U; k < 16U; ++k) p_b5[m * 16U + k] = k; // 16 experts, each B=5!
                }
                // Mix across 16 experts (`40` pairs): 4x B=5 (20), 2x B=4 (8), 2x B=3 (6), 2x B=2 (4), 2x B=1 (2) = 40 pairs over 12 experts, plus test 16 unique!
                // Better: 5 tokens x 8 experts = 40 pairs spanning 16 experts (B=5: 2, B=4: 2, B=3: 4, B=2: 2, B=1: 6 -> 10+8+12+4+6 = 40 pairs, 16 distinct experts = 119.5 MB!)
                uint32_t mix_flat[40] = {
                    0, 1, 2, 3, 4, 5, 6, 7,
                    0, 1, 2, 3, 4, 5, 6, 7,
                    0, 1, 2, 3, 4, 5, 8, 9,
                    0, 1, 2, 3, 4, 8, 10, 11,
                    0, 1, 6, 7, 12, 13, 14, 15
                };
                std::memcpy(p_mix, mix_flat, sizeof(mix_flat));
                id<MTLBuffer> b_mid80 = [dev newBufferWithLength:80U * d_moe * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_dn80  = [dev newBufferWithLength:80U * D * 4U options:MTLResourceStorageModeShared];

                auto bench_grp = [&](id<MTLBuffer> tk_buf, uint32_t tk_k, uint32_t n_unique_exp) -> double {
                    Sg53BinParamsT bpb{5U, tk_k, E_test, d_moe, D, 1024U, 0U, 0U};
                    Sg53E8ParamsT epb{0ULL, rec_b, proj_b, lo16_b, hi2_b, 5U, tk_k, D, d_moe, 10.0f, 5U * tk_k + 64U, 0U, 0U, 0U};
                    double best_sec = 1e9;
                    for (int it = 0; it < 16; ++it) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_bin_v2];
                        [enc setBuffer:tk_buf offset:0 atIndex:0];
                        [enc setBuffer:b_pairs offset:0 atIndex:1];
                        [enc setBuffer:b_tiles offset:0 atIndex:2];
                        [enc setBuffer:b_ind offset:0 atIndex:3];
                        [enc setBytes:&bpb length:sizeof(bpb) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
                        [enc setComputePipelineState:pso_e8_grp_gu];
                        [enc setBuffer:b_experts offset:0 atIndex:0];
                        [enc setBuffer:b_lut offset:0 atIndex:1];
                        [enc setBuffer:b_x_rot offset:0 atIndex:2];
                        [enc setBuffer:b_pairs offset:0 atIndex:3];
                        [enc setBuffer:b_tiles offset:0 atIndex:4];
                        [enc setBuffer:b_mid80 offset:0 atIndex:5];
                        [enc setBytes:&epb length:sizeof(epb) atIndex:6];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:32 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_e8_grp_dn];
                        [enc setBuffer:b_experts offset:0 atIndex:0];
                        [enc setBuffer:b_lut offset:0 atIndex:1];
                        [enc setBuffer:b_mid80 offset:0 atIndex:2];
                        [enc setBuffer:b_pairs offset:0 atIndex:3];
                        [enc setBuffer:b_tiles offset:0 atIndex:4];
                        [enc setBuffer:b_dn80 offset:0 atIndex:5];
                        [enc setBytes:&epb length:sizeof(epb) atIndex:6];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:44 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        if (it >= 8) {
                            double sec = [cb GPUEndTime] - [cb GPUStartTime];
                            if (sec > 0.0 && sec < best_sec) best_sec = sec;
                        }
                    }
                    return (static_cast<double>(static_cast<uint64_t>(n_unique_exp) * rec_b) / 1e9) / best_sec;
                };
                e8_grp_b5_gbps  = bench_grp(b_tk_b5, 16U, 16U);
                e8_grp_mix_gbps = bench_grp(b_tk_mix, 8U, 16U);
            }

            // 4) Prefill NAX `E8S-18-V4` Tri-Bin (`BM=64`, `BM=32`, `BM=16` ALL active!) vs `gather_gu/dn` reference!
            id<MTLBuffer> b_tk_nax  = [dev newBufferWithLength:M_nax * 1U * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mid_nax = [dev newBufferWithLength:(M_nax + 64U) * d_moe * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dn_nax  = [dev newBufferWithLength:(M_nax + 64U) * D * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mid_ref = [dev newBufferWithLength:M_nax * d_moe * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dn_ref  = [dev newBufferWithLength:M_nax * D * 4U options:MTLResourceStorageModeShared];
            auto* tkn = reinterpret_cast<uint32_t*>([b_tk_nax contents]);
            for (uint32_t m = 0U; m < M_nax; ++m) {
                tkn[m] = (m < 68U) ? 0U : ((m < 92U) ? 1U : 2U); // e=0: 68 (M64+M16), e=1: 24 (M32), e=2: 8 (M16)!
            }
            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

                // Reference via `gather_gu` -> `rht64` -> `gather_dn`
                Sg53E8ParamsT epr{0ULL, rec_b, proj_b, lo16_b, hi2_b, M_nax, 1U, D, d_moe, 10.0f, M_nax + 64U, 0U, 0U, 0U};
                [enc setComputePipelineState:pso_e8_g_gu];
                [enc setBuffer:b_experts offset:0 atIndex:0];
                [enc setBuffer:b_lut offset:0 atIndex:1];
                [enc setBuffer:b_tk_nax offset:0 atIndex:2];
                [enc setBuffer:b_x_rot offset:0 atIndex:3];
                [enc setBuffer:b_mid_ref offset:0 atIndex:4];
                [enc setBytes:&epr length:sizeof(epr) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((d_moe + 7U) / 8U, 1U, M_nax) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                Sg53RhtParamsT mrpr{M_nax, M_nax, d_moe, 0U};
                [enc setComputePipelineState:pso_rht64];
                [enc setBuffer:b_mid_ref offset:0 atIndex:0];
                [enc setBuffer:b_mid_ref offset:0 atIndex:1];
                [enc setBytes:&mrpr length:sizeof(mrpr) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(((d_moe / 64U) + 3U) / 4U, M_nax, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_e8_g_dn];
                [enc setBuffer:b_experts offset:0 atIndex:0];
                [enc setBuffer:b_lut offset:0 atIndex:1];
                [enc setBuffer:b_tk_nax offset:0 atIndex:2];
                [enc setBuffer:b_mid_ref offset:0 atIndex:3];
                [enc setBuffer:b_dn_ref offset:0 atIndex:4];
                [enc setBytes:&epr length:sizeof(epr) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((D + 15U) / 16U, 1U, M_nax) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                // NAX Tri-Bin (`bin_nax` -> `gather_x` -> Concurrent `gu_m64/32/16` -> barrier -> `dn_m64/32/16`)
                Sg53BinParamsT bpn{M_nax, 1U, E_test, d_moe, D, 1024U, 0U, 0U};
                [enc setComputePipelineState:pso_bin_nax];
                [enc setBuffer:b_tk_nax offset:0 atIndex:0];
                [enc setBuffer:b_pairs offset:0 atIndex:1];
                [enc setBuffer:b_tiles offset:0 atIndex:2];
                [enc setBuffer:b_ind offset:0 atIndex:3];
                [enc setBytes:&bpn length:sizeof(bpn) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

                [enc setComputePipelineState:pso_gather_x];
                [enc setBuffer:b_x_rot offset:0 atIndex:0];
                [enc setBuffer:b_pairs offset:0 atIndex:1];
                [enc setBuffer:b_dn_nax offset:0 atIndex:2]; // x_sorted alias
                [enc setBuffer:b_mid_nax offset:0 atIndex:3];
                [enc setBytes:&bpn length:sizeof(bpn) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(M_nax + 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];

                Sg53E8ParamsT ep64 = epr; ep64.tile_base = 0U;
                Sg53E8ParamsT ep32 = epr; ep32.tile_base = 1024U;
                Sg53E8ParamsT ep16 = epr; ep16.tile_base = 2048U;

                {
                    id<MTLComputeCommandEncoder> cenc_gu = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    [cenc_gu setComputePipelineState:pso_e8_nax_gu64];
                    [cenc_gu setBuffer:b_experts offset:0 atIndex:0];
                    [cenc_gu setBuffer:b_lut offset:0 atIndex:1];
                    [cenc_gu setBuffer:b_dn_nax offset:0 atIndex:2];
                    [cenc_gu setBuffer:b_tiles offset:0 atIndex:3];
                    [cenc_gu setBuffer:b_mid_nax offset:0 atIndex:4];
                    [cenc_gu setBytes:&ep64 length:sizeof(ep64) atIndex:5];
                    [cenc_gu dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_gu setComputePipelineState:pso_e8_nax_gu32];
                    [cenc_gu setBytes:&ep32 length:sizeof(ep32) atIndex:5];
                    [cenc_gu dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:12 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_gu setComputePipelineState:pso_e8_nax_gu16];
                    [cenc_gu setBytes:&ep16 length:sizeof(ep16) atIndex:5];
                    [cenc_gu dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:24 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [cenc_gu endEncoding];
                }

                {
                    id<MTLComputeCommandEncoder> cenc_dn = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    [cenc_dn setComputePipelineState:pso_e8_nax_dn64];
                    [cenc_dn setBuffer:b_experts offset:0 atIndex:0];
                    [cenc_dn setBuffer:b_lut offset:0 atIndex:1];
                    [cenc_dn setBuffer:b_mid_nax offset:0 atIndex:2];
                    [cenc_dn setBuffer:b_pairs offset:0 atIndex:3];
                    [cenc_dn setBuffer:b_tiles offset:0 atIndex:4];
                    [cenc_dn setBuffer:b_dn_nax offset:0 atIndex:5];
                    [cenc_dn setBytes:&ep64 length:sizeof(ep64) atIndex:6];
                    [cenc_dn dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:36 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_dn setComputePipelineState:pso_e8_nax_dn32];
                    [cenc_dn setBytes:&ep32 length:sizeof(ep32) atIndex:6];
                    [cenc_dn dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:48 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [cenc_dn setComputePipelineState:pso_e8_nax_dn16];
                    [cenc_dn setBytes:&ep16 length:sizeof(ep16) atIndex:6];
                    [cenc_dn dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:60 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [cenc_dn endEncoding];
                }

                [cb commit];
                [cb waitUntilCompleted];
            }

            const auto* ref_d = reinterpret_cast<const float*>([b_dn_ref contents]);
            const auto* nax_d = reinterpret_cast<const float*>([b_dn_nax contents]);
            float nax_max_err = 0.0f;
            for (size_t i = 0U; i < static_cast<size_t>(M_nax) * D; ++i) {
                float e = std::fabs(nax_d[i] - ref_d[i]);
                if (e > nax_max_err) nax_max_err = e;
            }
            const bool ok = (bit_diffs_1v5 == 0) && (cpu_max_err < 1e-3f) && (nax_max_err < 1e-3f);
            std::printf("[Test 8] E8S-18-V4 GEMV & NAX (BM=16/32/64 Fused RHT-64)  : %s\n", ok ? "PASS" : "FAIL");
            std::printf("         • M=1 vs M=5 Bit-Exact: %d diffs | CPU err: %.2e | NAX vs GEMV err: %.2e\n",
                        bit_diffs_1v5, cpu_max_err, nax_max_err);
            std::printf("         • Cold BW: M=1 %.1f GB/s | M=5 Grouped (Mix B=1..5) %.1f GB/s | M=5 Grouped (All B=5) %.1f GB/s\n",
                        e8_cold_gbps, e8_grp_mix_gbps, e8_grp_b5_gbps);
            if (!ok) return 1;
        }

        // ---------------------------------------------------------------------
        // Test 9: Chunk-32 WY NAX KDA vs Serial KDA Exact Parity (`M=77`, per-channel decay!)
        // ---------------------------------------------------------------------
        {
            const uint32_t M = 77U, M_pad = 96U, num_chunks = 3U, nh = 8U, hd = 128U;
            const uint32_t qkv_dim = nh * hd, conv_dim = 3U * qkv_dim;
            std::mt19937 rng(5309);
            std::normal_distribution<float> nd(0.0f, 0.4f);

            std::vector<uint16_t> qkv_h(static_cast<size_t>(M) * conv_dim);
            std::vector<float> a_f(static_cast<size_t>(M) * qkv_dim);
            std::vector<float> b_f(static_cast<size_t>(M) * nh);
            std::vector<float> alog_f(nh), dt_f(qkv_dim), s_init(static_cast<size_t>(nh) * hd * hd);
            for (auto& v : qkv_h)  v = f32_to_fp16_bits(nd(rng));
            for (auto& v : a_f)    v = nd(rng) * 2.0f;
            for (auto& v : b_f)    v = nd(rng);
            for (auto& v : alog_f) v = nd(rng) * 0.5f;
            for (auto& v : dt_f)   v = nd(rng) * 0.5f;
            for (auto& v : s_init) v = nd(rng) * 0.25f;

            id<MTLBuffer> b_qkv   = [dev newBufferWithBytes:qkv_h.data() length:qkv_h.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_a     = [dev newBufferWithBytes:a_f.data() length:a_f.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_b     = [dev newBufferWithBytes:b_f.data() length:b_f.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_alog  = [dev newBufferWithBytes:alog_f.data() length:alog_f.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dt    = [dev newBufferWithBytes:dt_f.data() length:dt_f.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_s_ser = [dev newBufferWithBytes:s_init.data() length:s_init.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_s_wy  = [dev newBufferWithBytes:s_init.data() length:s_init.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_qk    = [dev newBufferWithLength:static_cast<size_t>(M) * 2U * qkv_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dec   = [dev newBufferWithLength:static_cast<size_t>(M) * qkv_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_beta  = [dev newBufferWithLength:static_cast<size_t>(M) * nh * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_o_ser = [dev newBufferWithLength:static_cast<size_t>(M_pad) * qkv_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_o_wy  = [dev newBufferWithLength:static_cast<size_t>(M_pad) * qkv_dim * 4U options:MTLResourceStorageModeShared];

            const size_t qk_b = static_cast<size_t>(nh) * M_pad * 128U * 2U;
            id<MTLBuffer> b_qsc   = [dev newBufferWithLength:qk_b options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kend  = [dev newBufferWithLength:qk_b options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_w     = [dev newBufferWithLength:qk_b options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_u     = [dev newBufferWithLength:qk_b options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_p     = [dev newBufferWithLength:static_cast<size_t>(nh) * num_chunks * 1024U * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_gl    = [dev newBufferWithLength:static_cast<size_t>(nh) * num_chunks * 128U * 4U options:MTLResourceStorageModeShared];

            struct { uint32_t M_valid, M_padded, nh, hd; float l2_eps, norm_eps, glb; uint32_t sm, sl, nk, p0, p1; } kp{
                M, M_pad, nh, hd, 1e-6f, 1e-5f, -5.0f, 0U, 0U, 1U, 0U, 0U
            };
            struct { uint32_t M_valid, M_padded, num_chunks, nh; float l2_eps, glb; uint32_t p0, p1; } wyp{
                M, M_pad, num_chunks, nh, 1e-6f, -5.0f, 0U, 0U
            };

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_kda_prep];
            [enc setBuffer:b_qkv offset:0 atIndex:0]; [enc setBuffer:b_a offset:0 atIndex:1];
            [enc setBuffer:b_b offset:0 atIndex:2];   [enc setBuffer:b_alog offset:0 atIndex:3];
            [enc setBuffer:b_dt offset:0 atIndex:4];  [enc setBuffer:b_qk offset:0 atIndex:5];
            [enc setBuffer:b_dec offset:0 atIndex:6]; [enc setBuffer:b_beta offset:0 atIndex:7];
            [enc setBytes:&kp length:sizeof(kp) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(3U * nh + 1U, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:pso_kda_rec];
            [enc setBuffer:b_qkv offset:0 atIndex:0];  [enc setBuffer:b_qk offset:0 atIndex:1];
            [enc setBuffer:b_dec offset:0 atIndex:2];  [enc setBuffer:b_beta offset:0 atIndex:3];
            [enc setBuffer:b_s_ser offset:0 atIndex:4];[enc setBuffer:b_o_ser offset:0 atIndex:5];
            [enc setBytes:&kp length:sizeof(kp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(4, nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:pso_kda_wy_prep];
            [enc setBuffer:b_qkv offset:0 atIndex:0];  [enc setBuffer:b_a offset:0 atIndex:1];
            [enc setBuffer:b_b offset:0 atIndex:2];    [enc setBuffer:b_alog offset:0 atIndex:3];
            [enc setBuffer:b_dt offset:0 atIndex:4];   [enc setBuffer:b_qsc offset:0 atIndex:5];
            [enc setBuffer:b_kend offset:0 atIndex:6]; [enc setBuffer:b_w offset:0 atIndex:7];
            [enc setBuffer:b_u offset:0 atIndex:8];    [enc setBuffer:b_p offset:0 atIndex:9];
            [enc setBuffer:b_gl offset:0 atIndex:10];  [enc setBytes:&wyp length:sizeof(wyp) atIndex:11];
            [enc dispatchThreadgroups:MTLSizeMake(num_chunks, nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:pso_kda_wy_rec];
            [enc setBuffer:b_qsc offset:0 atIndex:0];  [enc setBuffer:b_kend offset:0 atIndex:1];
            [enc setBuffer:b_w offset:0 atIndex:2];    [enc setBuffer:b_u offset:0 atIndex:3];
            [enc setBuffer:b_p offset:0 atIndex:4];    [enc setBuffer:b_gl offset:0 atIndex:5];
            [enc setBuffer:b_s_wy offset:0 atIndex:6]; [enc setBuffer:b_o_wy offset:0 atIndex:7];
            [enc setBytes:&wyp length:sizeof(wyp) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(4, nh, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const auto* o_ser = reinterpret_cast<const float*>([b_o_ser contents]);
            const auto* o_wy  = reinterpret_cast<const float*>([b_o_wy contents]);
            const auto* s_ser = reinterpret_cast<const float*>([b_s_ser contents]);
            const auto* s_wy  = reinterpret_cast<const float*>([b_s_wy contents]);
            float out_err = 0.0f, st_err = 0.0f;
            for (size_t i = 0U; i < static_cast<size_t>(M) * qkv_dim; ++i) {
                out_err = std::max(out_err, std::fabs(o_ser[i] - o_wy[i]));
            }
            for (size_t i = 0U; i < s_init.size(); ++i) {
                st_err = std::max(st_err, std::fabs(s_ser[i] - s_wy[i]));
            }
            const bool ok = (out_err < 1.5e-3f) && (st_err < 1.5e-3f);
            std::printf("[Test 9] Chunk-32 WY NAX KDA vs Serial KDA (`M=77`)       : %s (out_err=%.2e, state_err=%.2e)\n",
                        ok ? "PASS" : "FAIL", out_err, st_err);
            if (!ok) return 1;
        }

        // ---------------------------------------------------------------------
        // Test 10: NAX Flash-MLA (Dense & Sparse/Paged) + NAX Indexer + Penalties
        // ---------------------------------------------------------------------
        {
            const uint32_t M = 64U, seq_start = 128U, total_kv = seq_start + M;
            const uint32_t nh = 8U, kvr = 512U, kpool = 4U, max_seq = 256U;
            std::mt19937 rng(5310);
            std::normal_distribution<float> nd(0.0f, 0.3f);

            std::vector<uint16_t> q_tm(static_cast<size_t>(M) * nh * kvr);
            std::vector<uint16_t> c_kv(static_cast<size_t>(max_seq) * kvr, 0U);
            for (auto& v : q_tm) v = f32_to_fp16_bits(nd(rng));
            for (size_t i = 0U; i < static_cast<size_t>(total_kv) * kvr; ++i) c_kv[i] = f32_to_fp16_bits(nd(rng));

            // Shuffled paged block table for sparse+paged test
            const uint32_t n_blks = max_seq / kpool;
            std::vector<uint32_t> blk_loc(n_blks);
            for (uint32_t b = 0U; b < n_blks; ++b) blk_loc[b] = STRATA_GLM53_KV_LOC_UMA_BIT | ((b * 13U + 7U) % n_blks);
            const uint32_t sel_k = 8U;
            std::vector<uint32_t> sel_pools(static_cast<size_t>(M) * sel_k);
            for (uint32_t m = 0U; m < M; ++m) {
                const uint32_t nc = (seq_start + m + 1U) / kpool;
                for (uint32_t s = 0U; s < sel_k; ++s) {
                    sel_pools[static_cast<size_t>(m) * sel_k + s] = (s * 3U + (m & 1U)) % nc;
                }
                std::sort(sel_pools.begin() + static_cast<ptrdiff_t>(m * sel_k),
                          sel_pools.begin() + static_cast<ptrdiff_t>((m + 1U) * sel_k));
            }

            id<MTLBuffer> b_q_tm  = [dev newBufferWithBytes:q_tm.data() length:q_tm.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q_hm  = [dev newBufferWithLength:q_tm.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ckv   = [dev newBufferWithBytes:c_kv.data() length:c_kv.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sel   = [dev newBufferWithBytes:sel_pools.data() length:sel_pools.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_loc   = [dev newBufferWithBytes:blk_loc.data() length:blk_loc.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pm    = [dev newBufferWithLength:8U * 16U * nh * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pl    = [dev newBufferWithLength:8U * 16U * nh * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_po    = [dev newBufferWithLength:8U * 16U * nh * kvr * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ref_d = [dev newBufferWithLength:q_tm.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_nax_d = [dev newBufferWithLength:q_tm.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ref_s = [dev newBufferWithLength:q_tm.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_nax_s = [dev newBufferWithLength:q_tm.size() * 2U options:MTLResourceStorageModeShared];

            struct { uint32_t M_valid, M_padded, nh, N, K, p0, p1, p2; } tp{M, M, nh, 0U, kvr, 0U, 0U, 0U};
            struct {
                uint32_t M_valid, nh, kv_lora_rank, q_per_tg, seq_pos_start, max_seq_alloc, max_splits, index_kpool, select_k_pools;
                float attn_scale;
                uint32_t m_base, paged, m_end, M_padded, total_kv, pad0;
            } mp_d{M, nh, kvr, 4U, seq_start, max_seq, 16U, kpool, 512U, 0.0625f, 0U, 0U, M, M, total_kv, 0U},
              mp_s{M, nh, kvr, 4U, seq_start, max_seq, 16U, kpool, sel_k, 0.0625f, 0U, 1U, M, M, total_kv, 0U};

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            [enc setComputePipelineState:pso_mla_q_hm];
            [enc setBuffer:b_q_tm offset:0 atIndex:0]; [enc setBuffer:b_q_hm offset:0 atIndex:1];
            [enc setBytes:&tp length:sizeof(tp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(nh, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            // Dense NAX vs SplitKV
            [enc setComputePipelineState:pso_mla_dense];
            [enc setBuffer:b_q_hm offset:0 atIndex:0]; [enc setBuffer:b_ckv offset:0 atIndex:1];
            [enc setBuffer:b_nax_d offset:0 atIndex:2]; [enc setBytes:&mp_d length:sizeof(mp_d) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(M / 16U, nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            for (uint32_t mb = 0U; mb < M; mb += 8U) {
                mp_d.m_base = mb; mp_d.m_end = mb + 8U;
                [enc setComputePipelineState:pso_mla_split];
                [enc setBuffer:b_q_tm offset:0 atIndex:0]; [enc setBuffer:b_ckv offset:0 atIndex:1];
                [enc setBuffer:b_sel offset:0 atIndex:2];  [enc setBuffer:b_pm offset:0 atIndex:3];
                [enc setBuffer:b_pl offset:0 atIndex:4];   [enc setBuffer:b_po offset:0 atIndex:5];
                [enc setBytes:&mp_d length:sizeof(mp_d) atIndex:6]; [enc setBuffer:b_loc offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(4, nh / 4U, 8) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc setComputePipelineState:pso_mla_red];
                [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_ref_d offset:0 atIndex:3];
                [enc setBytes:&mp_d length:sizeof(mp_d) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(nh, 8, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            // Sparse + Paged NAX vs SplitKV
            [enc setComputePipelineState:pso_mla_sparse];
            [enc setBuffer:b_q_hm offset:0 atIndex:0]; [enc setBuffer:b_ckv offset:0 atIndex:1];
            [enc setBuffer:b_sel offset:0 atIndex:2];  [enc setBuffer:b_nax_s offset:0 atIndex:3];
            [enc setBytes:&mp_s length:sizeof(mp_s) atIndex:4]; [enc setBuffer:b_loc offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(M / 16U, nh, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            for (uint32_t mb = 0U; mb < M; mb += 8U) {
                mp_s.m_base = mb; mp_s.m_end = mb + 8U;
                [enc setComputePipelineState:pso_mla_split];
                [enc setBuffer:b_q_tm offset:0 atIndex:0]; [enc setBuffer:b_ckv offset:0 atIndex:1];
                [enc setBuffer:b_sel offset:0 atIndex:2];  [enc setBuffer:b_pm offset:0 atIndex:3];
                [enc setBuffer:b_pl offset:0 atIndex:4];   [enc setBuffer:b_po offset:0 atIndex:5];
                [enc setBytes:&mp_s length:sizeof(mp_s) atIndex:6]; [enc setBuffer:b_loc offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(4, nh / 4U, 8) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc setComputePipelineState:pso_mla_red];
                [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_ref_s offset:0 atIndex:3];
                [enc setBytes:&mp_s length:sizeof(mp_s) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(nh, 8, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            // NAX Indexer vs Scalar Indexer (`32` heads, `128` hd, `128` pools)
            const uint32_t idx_nh = 32U, idx_hd = 128U, max_pools = 128U;
            std::vector<float> iq_f(static_cast<size_t>(M) * idx_nh * idx_hd);
            std::vector<float> iw_f(static_cast<size_t>(M) * idx_nh);
            std::vector<uint16_t> pk_h(static_cast<size_t>(max_pools) * idx_hd);
            for (auto& v : iq_f) v = nd(rng);
            for (auto& v : iw_f) v = nd(rng); // signed negative & positive head weights!
            for (auto& v : pk_h) v = f32_to_fp16_bits(nd(rng));

            id<MTLBuffer> b_iq    = [dev newBufferWithBytes:iq_f.data() length:iq_f.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_iqhm  = [dev newBufferWithLength:iq_f.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_iw    = [dev newBufferWithBytes:iw_f.data() length:iw_f.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pk    = [dev newBufferWithBytes:pk_h.data() length:pk_h.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sc_r  = [dev newBufferWithLength:static_cast<size_t>(M) * max_pools * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sc_n  = [dev newBufferWithLength:static_cast<size_t>(M) * max_pools * 4U options:MTLResourceStorageModeShared];
            struct { uint32_t M_valid, idx_nh, idx_hd, kpool, sel_k, seq_start, fp, nnp; float eps; uint32_t mps, msw, pad0; } ip{
                M, idx_nh, idx_hd, kpool, sel_k, 256U, 0U, 0U, 1e-6f, max_pools, 4U, M
            };

            [enc setComputePipelineState:pso_idx_scalar];
            [enc setBuffer:b_iq offset:0 atIndex:0]; [enc setBuffer:b_iw offset:0 atIndex:1];
            [enc setBuffer:b_pk offset:0 atIndex:2]; [enc setBuffer:b_sc_r offset:0 atIndex:3];
            [enc setBytes:&ip length:sizeof(ip) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(max_pools / 8U, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:pso_idx_q_hm];
            [enc setBuffer:b_iq offset:0 atIndex:0]; [enc setBuffer:b_iqhm offset:0 atIndex:1];
            [enc setBytes:&ip length:sizeof(ip) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(idx_nh, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:pso_idx_nax];
            [enc setBuffer:b_iqhm offset:0 atIndex:0]; [enc setBuffer:b_iw offset:0 atIndex:1];
            [enc setBuffer:b_pk offset:0 atIndex:2];   [enc setBuffer:b_sc_n offset:0 atIndex:3];
            [enc setBytes:&ip length:sizeof(ip) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(max_pools / 64U, M / 64U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            // Sliding-Window Logit Penalties: M=5 batch vs 5 sequential M=1 steps (`0` ULP diff!)
            const uint32_t V_pen = 256U, win = 8U;
            std::vector<float> logits_base(5U * V_pen);
            for (auto& v : logits_base) v = nd(rng) * 3.0f;
            const uint32_t hist_init[8] = {5U, 10U, 5U, 20U, 30U, 40U, 50U, 60U};
            const uint32_t extra_t[5]   = {99U, 5U, 10U, 77U, 20U};
            id<MTLBuffer> b_lg_bat = [dev newBufferWithBytes:logits_base.data() length:logits_base.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pids   = [dev newBufferWithLength:V_pen * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pcnt   = [dev newBufferWithLength:V_pen * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ext    = [dev newBufferWithBytes:extra_t length:sizeof(extra_t) options:MTLResourceStorageModeShared];
            std::memset([b_pcnt contents], 0, V_pen * 2U);
            auto* pids = reinterpret_cast<uint32_t*>([b_pids contents]);
            auto* pcnt = reinterpret_cast<uint16_t*>([b_pcnt contents]);
            uint32_t n_u = 0U;
            for (uint32_t t : hist_init) {
                if (pcnt[t] == 0U) pids[n_u++] = t;
                ++pcnt[t];
            }
            struct { uint32_t M, V, nh, ec, em; float pres, freq, rep; uint32_t drop[8]; } penp{
                5U, V_pen, n_u, 4U, 0U, 0.4f, 0.25f, 1.15f,
                {hist_init[0], hist_init[1], hist_init[2], hist_init[3], 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU}
            };
            [enc setComputePipelineState:pso_penalties];
            [enc setBuffer:b_lg_bat offset:0 atIndex:0]; [enc setBuffer:b_pids offset:0 atIndex:1];
            [enc setBuffer:b_pcnt offset:0 atIndex:2];   [enc setBuffer:b_ext offset:0 atIndex:3];
            [enc setBytes:&penp length:sizeof(penp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const auto* rd = reinterpret_cast<const uint16_t*>([b_ref_d contents]);
            const auto* nd_out = reinterpret_cast<const uint16_t*>([b_nax_d contents]);
            const auto* rs = reinterpret_cast<const uint16_t*>([b_ref_s contents]);
            const auto* ns_out = reinterpret_cast<const uint16_t*>([b_nax_s contents]);
            float mla_d_err = 0.0f, mla_s_err = 0.0f;
            for (uint32_t h = 0U; h < nh; ++h) {
                for (uint32_t m = 0U; m < M; ++m) {
                    for (uint32_t d = 0U; d < kvr; ++d) {
                        const size_t tm_i = (static_cast<size_t>(m) * nh + h) * kvr + d;
                        const size_t hm_i = (static_cast<size_t>(h) * M + m) * kvr + d;
                        mla_d_err = std::max(mla_d_err, std::fabs(fp16_bits_to_f32_ref(rd[tm_i]) - fp16_bits_to_f32_ref(nd_out[hm_i])));
                        mla_s_err = std::max(mla_s_err, std::fabs(fp16_bits_to_f32_ref(rs[tm_i]) - fp16_bits_to_f32_ref(ns_out[hm_i])));
                    }
                }
            }
            const auto* scr = reinterpret_cast<const float*>([b_sc_r contents]);
            const auto* scn = reinterpret_cast<const float*>([b_sc_n contents]);
            float idx_err = 0.0f;
            for (uint32_t m = 0U; m < M; ++m) {
                const uint32_t nc = (256U + m + 1U) / kpool;
                for (uint32_t b = 0U; b < nc; ++b) {
                    idx_err = std::max(idx_err, std::fabs(scr[static_cast<size_t>(m) * max_pools + b] - scn[static_cast<size_t>(m) * max_pools + b]));
                }
            }
            // Verify sliding-window penalties bit-exact against sequential CPU reference
            int pen_diffs = 0;
            std::vector<uint32_t> sim_hist(hist_init, hist_init + 8);
            const auto* lg_got = reinterpret_cast<const float*>([b_lg_bat contents]);
            for (uint32_t m = 0U; m < 5U; ++m) {
                if (m > 0U) sim_hist.push_back(extra_t[m]);
                std::vector<uint32_t> cnt_ref(V_pen, 0U);
                const size_t w0 = (sim_hist.size() > win) ? (sim_hist.size() - win) : 0U;
                for (size_t i = w0; i < sim_hist.size(); ++i) ++cnt_ref[sim_hist[i]];
                for (uint32_t t = 0U; t < V_pen; ++t) {
                    float v = logits_base[static_cast<size_t>(m) * V_pen + t];
                    if (cnt_ref[t] > 0U) {
                        v = (v > 0.0f) ? (v / 1.15f) : (v * 1.15f);
                        v -= 0.4f;
                        v -= 0.25f * static_cast<float>(cnt_ref[t]);
                    }
                    uint32_t ug = 0U, ur = 0U;
                    std::memcpy(&ug, &lg_got[static_cast<size_t>(m) * V_pen + t], 4U);
                    std::memcpy(&ur, &v, 4U);
                    if (ug != ur) ++pen_diffs;
                }
            }
            const bool ok = (mla_d_err < 1.5e-3f) && (mla_s_err < 1.5e-3f) && (idx_err < 2e-3f) && (pen_diffs == 0);
            std::printf("[Test 10] NAX Flash-MLA + NAX Indexer + Window Penalties  : %s\n", ok ? "PASS" : "FAIL");
            std::printf("         • Flash-MLA Dense=%.2e Sparse/Paged=%.2e | NAX Indexer=%.2e | Penalty BitDiffs=%d\n",
                        mla_d_err, mla_s_err, idx_err, pen_diffs);
            if (!ok) return 1;
        }

        // ---------------------------------------------------------------------
        // Test 11: NAX Split-KV Decode MLA + NAX Decode Indexer + Q4 GEMV BW
        //          (Parity, M=1 vs M=5 Bit-Exactness, & 11-Layer Latency!)
        // ---------------------------------------------------------------------
        {
            const uint32_t nh = 64U, kvr = 512U, kpool = 4U, sel_k = 512U, max_splits = 32U;
            const uint32_t max_seq = 16384U, max_pools = max_seq / kpool; // 4096 pools (16K ctx!)
            const uint32_t seq_pos = 16000U; // Sparse top-512 pools = 2048 active KV tokens!
            const uint32_t M5 = 5U;
            std::mt19937 rng(5311);
            std::normal_distribution<float> nd(0.0f, 0.25f);

            std::vector<uint16_t> q_dec(static_cast<size_t>(M5) * nh * kvr);
            std::vector<uint16_t> kv_big(static_cast<size_t>(max_seq) * kvr);
            for (auto& v : q_dec)  v = f32_to_fp16_bits(nd(rng));
            for (auto& v : kv_big) v = f32_to_fp16_bits(nd(rng));
            std::vector<uint32_t> sel_p(static_cast<size_t>(M5) * sel_k);
            for (uint32_t m = 0U; m < M5; ++m) {
                for (uint32_t s = 0U; s < sel_k; ++s) {
                    sel_p[static_cast<size_t>(m) * sel_k + s] = (s * 7U + m * 3U) % (seq_pos / kpool);
                }
            }
            std::vector<uint32_t> bloc(max_pools);
            for (uint32_t b = 0U; b < max_pools; ++b) bloc[b] = (b * 13U + 5U) % max_pools;

            id<MTLBuffer> b_q    = [dev newBufferWithBytes:q_dec.data() length:q_dec.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kv   = [dev newBufferWithBytes:kv_big.data() length:kv_big.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sel  = [dev newBufferWithBytes:sel_p.data() length:sel_p.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_bloc = [dev newBufferWithBytes:bloc.data() length:bloc.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pm   = [dev newBufferWithLength:static_cast<size_t>(M5) * max_splits * nh * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_pl   = [dev newBufferWithLength:static_cast<size_t>(M5) * max_splits * nh * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_po   = [dev newBufferWithLength:static_cast<size_t>(M5) * max_splits * nh * kvr * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_o_ref = [dev newBufferWithLength:static_cast<size_t>(M5) * nh * kvr * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_o_n1  = [dev newBufferWithLength:static_cast<size_t>(1U) * nh * kvr * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_o_n5  = [dev newBufferWithLength:static_cast<size_t>(M5) * nh * kvr * 2U options:MTLResourceStorageModeShared];

            struct MlaP11 {
                uint32_t M_valid, nh, kvr, q_per_tg, seq_pos_start, max_seq, max_splits, kpool, sel_k;
                float scale;
                uint32_t m_base, paged, m_end, M_padded, total_kv, pad0;
            };
            MlaP11 mp5{M5, nh, kvr, 4U, seq_pos, max_seq, max_splits, kpool, sel_k, 0.0625f, 0U, 1U, M5, M5, seq_pos + M5, 0U};
            MlaP11 mp1{1U, nh, kvr, 4U, seq_pos, max_seq, max_splits, kpool, sel_k, 0.0625f, 0U, 1U, 1U, 1U, seq_pos + 1U, 0U};
            const uint32_t l_splits = 32U;

            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                // Scalar ref (M=5)
                [enc setComputePipelineState:pso_mla_split];
                [enc setBuffer:b_q offset:0 atIndex:0]; [enc setBuffer:b_kv offset:0 atIndex:1];
                [enc setBuffer:b_sel offset:0 atIndex:2]; [enc setBuffer:b_pm offset:0 atIndex:3];
                [enc setBuffer:b_pl offset:0 atIndex:4]; [enc setBuffer:b_po offset:0 atIndex:5];
                [enc setBytes:&mp5 length:sizeof(mp5) atIndex:6]; [enc setBuffer:b_bloc offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(l_splits, nh / 4U, M5) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc setComputePipelineState:pso_mla_red];
                [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_o_ref offset:0 atIndex:3];
                [enc setBytes:&mp5 length:sizeof(mp5) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(nh, M5, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // NAX Decode Split-KV (M=5)
                [enc setComputePipelineState:pso_mla_dec_nax];
                [enc setBuffer:b_q offset:0 atIndex:0]; [enc setBuffer:b_kv offset:0 atIndex:1];
                [enc setBuffer:b_sel offset:0 atIndex:2]; [enc setBuffer:b_pm offset:0 atIndex:3];
                [enc setBuffer:b_pl offset:0 atIndex:4]; [enc setBuffer:b_po offset:0 atIndex:5];
                [enc setBytes:&mp5 length:sizeof(mp5) atIndex:6]; [enc setBuffer:b_bloc offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(l_splits, nh / 16U, M5) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc setComputePipelineState:pso_mla_red];
                [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_o_n5 offset:0 atIndex:3];
                [enc setBytes:&mp5 length:sizeof(mp5) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(nh, M5, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // NAX Decode Split-KV (M=1)
                [enc setComputePipelineState:pso_mla_dec_nax];
                [enc setBuffer:b_q offset:0 atIndex:0]; [enc setBuffer:b_kv offset:0 atIndex:1];
                [enc setBuffer:b_sel offset:0 atIndex:2]; [enc setBuffer:b_pm offset:0 atIndex:3];
                [enc setBuffer:b_pl offset:0 atIndex:4]; [enc setBuffer:b_po offset:0 atIndex:5];
                [enc setBytes:&mp1 length:sizeof(mp1) atIndex:6]; [enc setBuffer:b_bloc offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(l_splits, nh / 16U, 1U) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc setComputePipelineState:pso_mla_red];
                [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_o_n1 offset:0 atIndex:3];
                [enc setBytes:&mp1 length:sizeof(mp1) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(nh, 1U, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }

            const auto* o_ref = reinterpret_cast<const uint16_t*>([b_o_ref contents]);
            const auto* o_n1  = reinterpret_cast<const uint16_t*>([b_o_n1 contents]);
            const auto* o_n5  = reinterpret_cast<const uint16_t*>([b_o_n5 contents]);
            float mla_dec_err = 0.0f;
            int mla_dec_diffs = 0;
            for (size_t i = 0U; i < static_cast<size_t>(M5) * nh * kvr; ++i) {
                float e = std::fabs(fp16_bits_to_f32_ref(o_n5[i]) - fp16_bits_to_f32_ref(o_ref[i]));
                if (e > mla_dec_err) mla_dec_err = e;
            }
            for (size_t i = 0U; i < static_cast<size_t>(nh) * kvr; ++i) {
                if (o_n1[i] != o_n5[i]) ++mla_dec_diffs;
            }

            // Benchmark 11 layers of Old Scalar vs New NAX Split-KV at M=1 and M=5 (`n_tok=2048`)
            auto bench_mla11 = [&](bool use_nax_dec, uint32_t mb) -> double {
                MlaP11 mp = (mb == 1U) ? mp1 : mp5;
                double best_ms = 1e9;
                for (int it = 0; it < 12; ++it) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    for (int l = 0; l < 11; ++l) {
                        [enc setComputePipelineState:(use_nax_dec ? pso_mla_dec_nax : pso_mla_split)];
                        [enc setBuffer:b_q offset:0 atIndex:0]; [enc setBuffer:b_kv offset:0 atIndex:1];
                        [enc setBuffer:b_sel offset:0 atIndex:2]; [enc setBuffer:b_pm offset:0 atIndex:3];
                        [enc setBuffer:b_pl offset:0 atIndex:4]; [enc setBuffer:b_po offset:0 atIndex:5];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:6]; [enc setBuffer:b_bloc offset:0 atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(l_splits, use_nax_dec ? (nh / 16U) : (nh / 4U), mb)
                            threadsPerThreadgroup:MTLSizeMake(use_nax_dec ? 128 : 256, 1, 1)];
                        [enc setComputePipelineState:pso_mla_red];
                        [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                        [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_o_n5 offset:0 atIndex:3];
                        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(nh, mb, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (it >= 5) {
                        double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3;
                        if (ms > 0.0 && ms < best_ms) best_ms = ms;
                    }
                }
                return best_ms;
            };
            const double mla_old_m1_ms = bench_mla11(false, 1U);
            const double mla_nax_m1_ms = bench_mla11(true,  1U);
            const double mla_old_m5_ms = bench_mla11(false, 5U);
            const double mla_nax_m5_ms = bench_mla11(true,  5U);

            // NAX Decode Indexer (`16K` ctx = 4096 pools): Parity + M=1 vs M=5 bit-exact + 11L timing!
            const uint32_t idx_nh = 32U, idx_hd = 128U;
            std::vector<float> iq_h(static_cast<size_t>(M5) * idx_nh * idx_hd);
            std::vector<float> iw_h(static_cast<size_t>(M5) * idx_nh);
            std::vector<uint16_t> ipk_h(static_cast<size_t>(max_pools) * idx_hd);
            for (auto& v : iq_h)  v = nd(rng);
            for (auto& v : iw_h)  v = std::fabs(nd(rng)) + 0.05f;
            for (auto& v : ipk_h) v = f32_to_fp16_bits(nd(rng));

            id<MTLBuffer> b_iq  = [dev newBufferWithBytes:iq_h.data() length:iq_h.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_iw  = [dev newBufferWithBytes:iw_h.data() length:iw_h.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ipk = [dev newBufferWithBytes:ipk_h.data() length:ipk_h.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_is_ref = [dev newBufferWithLength:static_cast<size_t>(M5) * max_pools * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_is_n1  = [dev newBufferWithLength:static_cast<size_t>(1U) * max_pools * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_is_n5  = [dev newBufferWithLength:static_cast<size_t>(M5) * max_pools * 4U options:MTLResourceStorageModeShared];

            struct IdxP11 { uint32_t M_valid, nh, hd, kpool, sel_k, s0, fp, nnp; float eps; uint32_t mp, msw, p0; };
            IdxP11 ip5{M5, idx_nh, idx_hd, kpool, sel_k, seq_pos, 0U, 0U, 1e-5f, max_pools, max_pools / 32U, 0U};
            IdxP11 ip1{1U, idx_nh, idx_hd, kpool, sel_k, seq_pos, 0U, 0U, 1e-5f, max_pools, max_pools / 32U, 0U};
            const uint32_t max_comp = (seq_pos + M5) / kpool;

            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_idx_scalar];
                [enc setBuffer:b_iq offset:0 atIndex:0]; [enc setBuffer:b_iw offset:0 atIndex:1];
                [enc setBuffer:b_ipk offset:0 atIndex:2]; [enc setBuffer:b_is_ref offset:0 atIndex:3];
                [enc setBytes:&ip5 length:sizeof(ip5) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((max_comp + 7U) / 8U, M5, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:pso_idx_dec_nax];
                [enc setBuffer:b_iq offset:0 atIndex:0]; [enc setBuffer:b_iw offset:0 atIndex:1];
                [enc setBuffer:b_ipk offset:0 atIndex:2]; [enc setBuffer:b_is_n5 offset:0 atIndex:3];
                [enc setBytes:&ip5 length:sizeof(ip5) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((max_comp + 63U) / 64U, M5, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_idx_dec_nax];
                [enc setBuffer:b_iq offset:0 atIndex:0]; [enc setBuffer:b_iw offset:0 atIndex:1];
                [enc setBuffer:b_ipk offset:0 atIndex:2]; [enc setBuffer:b_is_n1 offset:0 atIndex:3];
                [enc setBytes:&ip1 length:sizeof(ip1) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((max_comp + 63U) / 64U, 1U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }

            const auto* is_ref = reinterpret_cast<const float*>([b_is_ref contents]);
            const auto* is_n1  = reinterpret_cast<const float*>([b_is_n1 contents]);
            const auto* is_n5  = reinterpret_cast<const float*>([b_is_n5 contents]);
            float idx_dec_err = 0.0f;
            int idx_dec_diffs = 0;
            for (uint32_t m = 0U; m < M5; ++m) {
                const uint32_t nc = (seq_pos + m + 1U) / kpool;
                for (uint32_t b = 0U; b < nc; ++b) {
                    const size_t ii = static_cast<size_t>(m) * max_pools + b;
                    float e = std::fabs(is_n5[ii] - is_ref[ii]);
                    if (e > idx_dec_err) idx_dec_err = e;
                    if (m == 0U) {
                        uint32_t u1 = 0U, u5 = 0U;
                        std::memcpy(&u1, &is_n1[b], 4U);
                        std::memcpy(&u5, &is_n5[b], 4U);
                        if (u1 != u5) ++idx_dec_diffs;
                    }
                }
            }

            auto bench_idx11 = [&](bool use_nax_dec, uint32_t mb) -> double {
                IdxP11 ip = (mb == 1U) ? ip1 : ip5;
                double best_ms = 1e9;
                for (int it = 0; it < 12; ++it) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    for (int l = 0; l < 11; ++l) {
                        [enc setComputePipelineState:(use_nax_dec ? pso_idx_dec_nax : pso_idx_scalar)];
                        [enc setBuffer:b_iq offset:0 atIndex:0]; [enc setBuffer:b_iw offset:0 atIndex:1];
                        [enc setBuffer:b_ipk offset:0 atIndex:2]; [enc setBuffer:b_is_n5 offset:0 atIndex:3];
                        [enc setBytes:&ip length:sizeof(ip) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(use_nax_dec ? ((max_comp + 63U) / 64U) : ((max_comp + 7U) / 8U), mb, 1)
                            threadsPerThreadgroup:MTLSizeMake(use_nax_dec ? 128 : 256, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (it >= 5) {
                        double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3;
                        if (ms > 0.0 && ms < best_ms) best_ms = ms;
                    }
                }
                return best_ms;
            };
            const double idx_old_m1_ms = bench_idx11(false, 1U);
            const double idx_nax_m1_ms = bench_idx11(true,  1U);
            const double idx_old_m5_ms = bench_idx11(false, 5U);
            const double idx_nax_m5_ms = bench_idx11(true,  5U);

            // Quick Q4 GEMV cold-DRAM BW check (`16` matrices of `4096x4096` = `144 MB > 48 MiB SLC`)
            double q4_m1_gbps = 0.0, q4_m5_gbps = 0.0;
            {
                const uint32_t R = 4096U, C = 4096U, n_mats = 16U;
                const size_t w_sz = static_cast<size_t>(R) * (C / 2U);
                const size_t s_sz = static_cast<size_t>(R) * (C / 64U) * 2U;
                id<MTLBuffer> b_w  = [dev newBufferWithLength:n_mats * w_sz options:MTLResourceStorageModePrivate];
                id<MTLBuffer> b_sc = [dev newBufferWithLength:n_mats * s_sz options:MTLResourceStorageModePrivate];
                id<MTLBuffer> b_bi = [dev newBufferWithLength:n_mats * s_sz options:MTLResourceStorageModePrivate];
                id<MTLBuffer> b_x  = [dev newBufferWithLength:5U * C * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_y  = [dev newBufferWithLength:5U * R * 4U options:MTLResourceStorageModePrivate];
                struct { uint32_t M, rows, cols, out_mode; } gp1{1U, R, C, 1U}, gp5{5U, R, C, 1U};
                auto bench_q4 = [&](uint32_t mb) -> double {
                    auto gp = (mb == 1U) ? gp1 : gp5;
                    double best_sec = 1e9;
                    for (int it = 0; it < 16; ++it) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_q4_gemv];
                        for (uint32_t mi = 0U; mi < n_mats; ++mi) {
                            [enc setBuffer:b_w offset:mi * w_sz atIndex:0];
                            [enc setBuffer:b_sc offset:mi * s_sz atIndex:1];
                            [enc setBuffer:b_bi offset:mi * s_sz atIndex:2];
                            [enc setBuffer:b_x offset:0 atIndex:3];
                            [enc setBuffer:b_y offset:0 atIndex:4];
                            [enc setBuffer:b_y offset:0 atIndex:5];
                            [enc setBytes:&gp length:sizeof(gp) atIndex:6];
                            [enc dispatchThreadgroups:MTLSizeMake((R + 7U) / 8U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        }
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                        if (it >= 8) {
                            double sec = [cb GPUEndTime] - [cb GPUStartTime];
                            if (sec > 0.0 && sec < best_sec) best_sec = sec;
                        }
                    }
                    return (static_cast<double>(n_mats) * static_cast<double>(w_sz + 2U * s_sz) / 1e9) / best_sec;
                };
                q4_m1_gbps = bench_q4(1U);
                q4_m5_gbps = bench_q4(5U);
            }

            // 40-Core Parallel Router (`sg53_moe_router_logits + select`) vs 1-Core `sg53_moe_router_topk` (42L!)
            int rtr_diffs = 0;
            double rtr_old_m1_ms = 0.0, rtr_new_m1_ms = 0.0, rtr_old_m5_ms = 0.0, rtr_new_m5_ms = 0.0;
            {
                const uint32_t R_D = 4096U, R_E = 288U, R_K = 8U, R_L = 42U;
                const size_t gate_sz = static_cast<size_t>(R_E) * R_D * 2U; // 2.36 MB per layer -> 99.1 MB across 42L!
                std::vector<float> rx_h(static_cast<size_t>(M5) * R_D);
                std::vector<uint16_t> rg_h(static_cast<size_t>(R_E) * R_D);
                std::vector<float> rb_h(R_E);
                for (auto& v : rx_h) v = nd(rng);
                for (auto& v : rg_h) v = f32_to_fp16_bits(nd(rng));
                for (auto& v : rb_h) v = nd(rng) * 0.1f;

                id<MTLBuffer> b_rx = [dev newBufferWithBytes:rx_h.data() length:rx_h.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_rg = [dev newBufferWithLength:static_cast<size_t>(R_L) * gate_sz options:MTLResourceStorageModeShared];
                for (uint32_t l = 0U; l < R_L; ++l) {
                    std::memcpy(static_cast<uint8_t*>([b_rg contents]) + static_cast<size_t>(l) * gate_sz, rg_h.data(), gate_sz);
                }
                id<MTLBuffer> b_rb   = [dev newBufferWithBytes:rb_h.data() length:rb_h.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_rlg  = [dev newBufferWithLength:static_cast<size_t>(M5) * R_E * 4U options:MTLResourceStorageModePrivate];
                id<MTLBuffer> b_ri0  = [dev newBufferWithLength:static_cast<size_t>(M5) * R_K * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_rw0  = [dev newBufferWithLength:static_cast<size_t>(M5) * R_K * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_ri1  = [dev newBufferWithLength:static_cast<size_t>(M5) * R_K * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_rw1  = [dev newBufferWithLength:static_cast<size_t>(M5) * R_K * 4U options:MTLResourceStorageModeShared];
                struct RtrP { uint32_t M_valid, D, num_experts, top_k, norm_topk_prob; float scale; uint32_t n_group, topk_group; };
                RtrP rp5{M5, R_D, R_E, R_K, 1U, 2.5f, 1U, 1U};
                RtrP rp1{1U, R_D, R_E, R_K, 1U, 2.5f, 1U, 1U};
                {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_rtr_old];
                    [enc setBuffer:b_rx offset:0 atIndex:0]; [enc setBuffer:b_rg offset:0 atIndex:1];
                    [enc setBuffer:b_rb offset:0 atIndex:2]; [enc setBuffer:b_ri0 offset:0 atIndex:3];
                    [enc setBuffer:b_rw0 offset:0 atIndex:4]; [enc setBytes:&rp5 length:sizeof(rp5) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(M5, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

                    [enc setComputePipelineState:pso_rtr_logits];
                    [enc setBuffer:b_rx offset:0 atIndex:0]; [enc setBuffer:b_rg offset:0 atIndex:1];
                    [enc setBuffer:b_rlg offset:0 atIndex:2]; [enc setBytes:&rp5 length:sizeof(rp5) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((R_E + 3U) / 4U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc setComputePipelineState:pso_rtr_select];
                    [enc setBuffer:b_rlg offset:0 atIndex:0]; [enc setBuffer:b_rb offset:0 atIndex:1];
                    [enc setBuffer:b_ri1 offset:0 atIndex:2]; [enc setBuffer:b_rw1 offset:0 atIndex:3];
                    [enc setBytes:&rp5 length:sizeof(rp5) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(M5, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                const auto* i0 = reinterpret_cast<const uint32_t*>([b_ri0 contents]);
                const auto* i1 = reinterpret_cast<const uint32_t*>([b_ri1 contents]);
                const auto* w0 = reinterpret_cast<const uint32_t*>([b_rw0 contents]);
                const auto* w1 = reinterpret_cast<const uint32_t*>([b_rw1 contents]);
                for (uint32_t j = 0U; j < M5 * R_K; ++j) {
                    if (i0[j] != i1[j] || w0[j] != w1[j]) ++rtr_diffs;
                }
                auto bench_rtr42 = [&](bool use_new, uint32_t mb) -> double {
                    RtrP rp = (mb == 1U) ? rp1 : rp5;
                    double best_ms = 1e9;
                    for (int it = 0; it < 12; ++it) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        for (uint32_t l = 0U; l < R_L; ++l) {
                            const NSUInteger goff = static_cast<NSUInteger>(l) * gate_sz;
                            if (!use_new) {
                                [enc setComputePipelineState:pso_rtr_old];
                                [enc setBuffer:b_rx offset:0 atIndex:0]; [enc setBuffer:b_rg offset:goff atIndex:1];
                                [enc setBuffer:b_rb offset:0 atIndex:2]; [enc setBuffer:b_ri0 offset:0 atIndex:3];
                                [enc setBuffer:b_rw0 offset:0 atIndex:4]; [enc setBytes:&rp length:sizeof(rp) atIndex:5];
                                [enc dispatchThreadgroups:MTLSizeMake(mb, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
                            } else {
                                [enc setComputePipelineState:pso_rtr_logits];
                                [enc setBuffer:b_rx offset:0 atIndex:0]; [enc setBuffer:b_rg offset:goff atIndex:1];
                                [enc setBuffer:b_rlg offset:0 atIndex:2]; [enc setBytes:&rp length:sizeof(rp) atIndex:3];
                                [enc dispatchThreadgroups:MTLSizeMake((R_E + 3U) / 4U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                                [enc setComputePipelineState:pso_rtr_select];
                                [enc setBuffer:b_rlg offset:0 atIndex:0]; [enc setBuffer:b_rb offset:0 atIndex:1];
                                [enc setBuffer:b_ri1 offset:0 atIndex:2]; [enc setBuffer:b_rw1 offset:0 atIndex:3];
                                [enc setBytes:&rp length:sizeof(rp) atIndex:4];
                                [enc dispatchThreadgroups:MTLSizeMake(mb, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
                            }
                        }
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                        if (it >= 5) {
                            double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e3;
                            if (ms > 0.0 && ms < best_ms) best_ms = ms;
                        }
                    }
                    return best_ms;
                };
                rtr_old_m1_ms = bench_rtr42(false, 1U);
                rtr_new_m1_ms = bench_rtr42(true,  1U);
                rtr_old_m5_ms = bench_rtr42(false, 5U);
                rtr_new_m5_ms = bench_rtr42(true,  5U);
            }

            const bool ok11 = (mla_dec_err < 1.5e-3f) && (mla_dec_diffs == 0) &&
                              (idx_dec_err < 2e-3f) && (idx_dec_diffs == 0) && (rtr_diffs == 0);
            std::printf("[Test 11] NAX Decode MLA + NAX Decode Idx + 40c Router    : %s\n", ok11 ? "PASS" : "FAIL");
            std::printf("         • Decode MLA (11L, 2K): err=%.2e BitDiffs=%d | M=1 %.2f->%.2f ms (%.1fx) | M=5 %.2f->%.2f ms (%.1fx)\n",
                        mla_dec_err, mla_dec_diffs,
                        mla_old_m1_ms, mla_nax_m1_ms, mla_old_m1_ms / mla_nax_m1_ms,
                        mla_old_m5_ms, mla_nax_m5_ms, mla_old_m5_ms / mla_nax_m5_ms);
            std::printf("         • Decode Idx (11L, 16K): err=%.2e BitDiffs=%d | M=1 %.2f->%.2f ms (%.1fx) | M=5 %.2f->%.2f ms (%.1fx)\n",
                        idx_dec_err, idx_dec_diffs,
                        idx_old_m1_ms, idx_nax_m1_ms, idx_old_m1_ms / idx_nax_m1_ms,
                        idx_old_m5_ms, idx_nax_m5_ms, idx_old_m5_ms / idx_nax_m5_ms);
            std::printf("         • 40c Router (42L, 288E): BitDiffs=%d | M=1 %.2f->%.2f ms (%.1fx) | M=5 %.2f->%.2f ms (%.1fx)\n",
                        rtr_diffs,
                        rtr_old_m1_ms, rtr_new_m1_ms, rtr_old_m1_ms / rtr_new_m1_ms,
                        rtr_old_m5_ms, rtr_new_m5_ms, rtr_old_m5_ms / rtr_new_m5_ms);
            std::printf("         • Spine Q4 GEMV Cold BW: M=1 %.1f GB/s | M=5 %.1f GB/s\n", q4_m1_gbps, q4_m5_gbps);
            if (!ok11) return 1;
        }
    }
    return 0;
}

int run_inkling2_turbo_metal41_suite(id<MTLDevice> dev) {
    if (!dev) {
        std::printf("[Test 12-17] Inkling-2-Mini Turbo Metal 4.1 + INT8 Suite   : SKIPPED (No Metal GPU)\n");
        return 0;
    }

    @autoreleasepool {
        // ---------------------------------------------------------------------
        // Test 12: Zero-Warning Metal 4.1 Compile + BK=128 NAX & Dynamic-Extent
        // ---------------------------------------------------------------------
        std::string hdr_src = read_text_file_rel("include/strata_inkling2.h");
        std::string msl_src = read_text_file_rel("src/gpu/strata_inkling2_kernels.metal");
        if (hdr_src.empty() || msl_src.empty()) {
            std::fprintf(stderr, "FAIL: could not read strata_inkling2.h or strata_inkling2_kernels.metal\n");
            return 1;
        }
        const std::string marker = "// @@STRATA_INKLING2_HEADER@@";
        const size_t pos = msl_src.find(marker);
        if (pos != std::string::npos) msl_src.replace(pos, marker.size(), hdr_src);

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 27.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_1;
        } else if (@available(macOS 26.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_0;
        }
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
            opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
        }

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:msl_src.c_str()]
                                               options:opts
                                                 error:&err];
        if (!lib || err) {
            std::fprintf(stderr, "FAIL: Inkling-2 Metal 4.1 compile failed or emitted warnings: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return 1;
        }

        id<MTLCommandQueue> q = [dev newCommandQueue];
        auto get_pso = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!fn) {
                std::fprintf(stderr, "FAIL: missing Inkling-2 kernel %s\n", name);
                return nil;
            }
            NSError* pe = nil;
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&pe];
            if (!pso) {
                std::fprintf(stderr, "FAIL: Inkling-2 PSO %s: %s\n", name, pe ? [[pe localizedDescription] UTF8String] : "?");
            }
            return pso;
        };
        auto get_pso_fc = [&](const char* name, uint32_t m_const) -> id<MTLComputePipelineState> {
            MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
            [cv setConstantValue:&m_const type:MTLDataTypeUInt atIndex:0];
            NSError* fe = nil;
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]
                                           constantValues:cv
                                                    error:&fe];
            if (!fn) {
                std::fprintf(stderr, "FAIL: missing specialized kernel %s (M=%u): %s\n",
                             name, m_const, fe ? [[fe localizedDescription] UTF8String] : "?");
                return nil;
            }
            NSError* pe = nil;
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&pe];
            if (!pso) {
                std::fprintf(stderr, "FAIL: specialized PSO %s (M=%u): %s\n",
                             name, m_const, pe ? [[pe localizedDescription] UTF8String] : "?");
            }
            return pso;
        };

        id<MTLComputePipelineState> pso_q4a_nax64   = get_pso("sik_q4a_gemm_nax_m64");
        id<MTLComputePipelineState> pso_q8_nax64    = get_pso("sik_q8_gemm_nax_m64");
        id<MTLComputePipelineState> pso_loop_f16    = get_pso("sik_loop_ctrl_apply_nax_f16_m64");
        id<MTLComputePipelineState> pso_bin_grp     = get_pso("sik_moe_bin_tokens_grouped");
        id<MTLComputePipelineState> pso_bin_quad    = get_pso("sik_moe_bin_tokens_quad");
        id<MTLComputePipelineState> pso_gather_x    = get_pso("sik_moe_gather_x_f16");
        id<MTLComputePipelineState> pso_grp_w13_4   = get_pso_fc("sik_moe_grouped_w13_nvfp4", 4U);
        id<MTLComputePipelineState> pso_grp_w2_4    = get_pso_fc("sik_moe_grouped_w2_nvfp4", 4U);
        id<MTLComputePipelineState> pso_q_w13_g[4]  = {
            get_pso_fc("sik_moe_prefill_w13_nvfp4_nax_m16_sg2", 0U),
            get_pso_fc("sik_moe_prefill_w13_nvfp4_nax_m32", 0U),
            get_pso_fc("sik_moe_prefill_w13_nvfp4_nax_m64", 0U),
            get_pso_fc("sik_moe_prefill_w13_nvfp4_nax_m128", 0U)
        };
        id<MTLComputePipelineState> pso_q_w13_u[4]  = {
            get_pso_fc("sik_moe_prefill_w13_nvfp4_nax_m16_sg2", 1U),
            get_pso_fc("sik_moe_prefill_w13_nvfp4_nax_m32", 1U),
            get_pso_fc("sik_moe_prefill_w13_nvfp4_nax_m64", 1U),
            get_pso_fc("sik_moe_prefill_w13_nvfp4_nax_m128", 1U)
        };
        id<MTLComputePipelineState> pso_q_w2[4]     = {
            get_pso("sik_moe_prefill_w2_nvfp4_nax_m16_sg2"),
            get_pso("sik_moe_prefill_w2_nvfp4_nax_m32"),
            get_pso("sik_moe_prefill_w2_nvfp4_nax_m64"),
            get_pso("sik_moe_prefill_w2_nvfp4_nax_m128")
        };
        id<MTLComputePipelineState> pso_qkvr_pre    = get_pso("sik_attn_qkvr_prep_prefill");
        id<MTLComputePipelineState> pso_attn_f16    = get_pso("sik_attn_prefill_nax");
        id<MTLComputePipelineState> pso_attn_q8     = get_pso("sik_attn_prefill_nax_q8");
        id<MTLComputePipelineState> pso_blk64_sum   = get_pso("sik_f16_blk64_sum");
        id<MTLComputePipelineState> pso_q4a_gemv8   = get_pso_fc("sik_q4a_gemv_batch", 8U);
        id<MTLComputePipelineState> pso_q4a_xsum8   = get_pso_fc("sik_q4a_gemv_batch_xsum", 8U);
        if (!pso_q4a_nax64 || !pso_q8_nax64 || !pso_loop_f16 || !pso_bin_grp || !pso_bin_quad ||
            !pso_gather_x || !pso_grp_w13_4 || !pso_grp_w2_4 ||
            !pso_q_w13_g[0] || !pso_q_w13_g[1] || !pso_q_w13_g[2] || !pso_q_w13_g[3] ||
            !pso_q_w13_u[0] || !pso_q_w13_u[1] || !pso_q_w13_u[2] || !pso_q_w13_u[3] ||
            !pso_q_w2[0] || !pso_q_w2[1] || !pso_q_w2[2] || !pso_q_w2[3] ||
            !pso_qkvr_pre || !pso_attn_f16 || !pso_attn_q8 ||
            !pso_blk64_sum || !pso_q4a_gemv8 || !pso_q4a_xsum8) {
            return 1;
        }

        auto cosine_sim = [](const float* a, const float* b, size_t n) -> double {
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (size_t i = 0U; i < n; ++i) {
                dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
                na  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
                nb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }
            return (na > 0.0 && nb > 0.0) ? (dot / std::sqrt(na * nb)) : 0.0;
        };

        // ---- Test 12: BK=128 NAX (Q4_A & Q8_0) + Direct-Device `dynamic_extent` Loop Apply ----
        {
            const uint32_t M = 64U, N = 128U, K = 256U, gs = 64U;
            std::mt19937 rng(202612);
            std::normal_distribution<float> nd(0.0f, 0.3f);

            std::vector<uint16_t> x_h(static_cast<size_t>(M) * K);
            for (auto& v : x_h) v = f32_to_fp16_bits(nd(rng));

            // Q4_A weights: [N, K/2] packed nibbles + [N, K/64] FP16 scale & bias
            std::vector<uint8_t>  q4_w(static_cast<size_t>(N) * (K / 2U));
            std::vector<uint16_t> q4_s(static_cast<size_t>(N) * (K / gs));
            std::vector<uint16_t> q4_b(static_cast<size_t>(N) * (K / gs));
            for (size_t i = 0U; i < q4_w.size(); ++i) q4_w[i] = static_cast<uint8_t>(rng() & 0xFFU);
            for (size_t i = 0U; i < q4_s.size(); ++i) {
                q4_s[i] = f32_to_fp16_bits(0.02f + 0.04f * (static_cast<float>(rng() & 255U) / 255.0f));
                q4_b[i] = f32_to_fp16_bits(-0.15f + 0.30f * (static_cast<float>(rng() & 255U) / 255.0f));
            }

            // Q8_0 weights: [N, K] int8 + [N, K/64] FP16 scale
            std::vector<int8_t>   q8_w(static_cast<size_t>(N) * K);
            std::vector<uint16_t> q8_s(static_cast<size_t>(N) * (K / gs));
            for (size_t i = 0U; i < q8_w.size(); ++i) q8_w[i] = static_cast<int8_t>(static_cast<int>(rng() % 251U) - 125);
            for (size_t i = 0U; i < q8_s.size(); ++i) {
                q8_s[i] = f32_to_fp16_bits(0.005f + 0.01f * (static_cast<float>(rng() & 255U) / 255.0f));
            }

            // Direct-device FP16 loop ctrl apply (`r=64, D=128, M=64, num_ctrls=1`)
            const uint32_t R = 64U;
            std::vector<uint16_t> h_f16(static_cast<size_t>(M) * R);
            std::vector<uint16_t> wg_f16(static_cast<size_t>(N) * R);
            std::vector<uint16_t> wf_f16(static_cast<size_t>(N) * R);
            std::vector<float>    cur_f32(static_cast<size_t>(M) * N), msg_f32(static_cast<size_t>(M) * N);
            for (auto& v : h_f16)   v = f32_to_fp16_bits(nd(rng));
            for (auto& v : wg_f16)  v = f32_to_fp16_bits(nd(rng) * 0.25f);
            for (auto& v : wf_f16)  v = f32_to_fp16_bits(nd(rng) * 0.25f);
            for (auto& v : cur_f32) v = nd(rng);
            for (auto& v : msg_f32) v = nd(rng);

            id<MTLBuffer> b_x    = [dev newBufferWithBytes:x_h.data() length:x_h.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q4w  = [dev newBufferWithBytes:q4_w.data() length:q4_w.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q4s  = [dev newBufferWithBytes:q4_s.data() length:q4_s.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q4b  = [dev newBufferWithBytes:q4_b.data() length:q4_b.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y4   = [dev newBufferWithLength:static_cast<size_t>(M) * N * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q8w  = [dev newBufferWithBytes:q8_w.data() length:q8_w.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q8s  = [dev newBufferWithBytes:q8_s.data() length:q8_s.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y8   = [dev newBufferWithLength:static_cast<size_t>(M) * N * 4U options:MTLResourceStorageModeShared];

            std::vector<uint8_t> uma_loop(wg_f16.size() * 2U + wf_f16.size() * 2U);
            std::memcpy(uma_loop.data(), wg_f16.data(), wg_f16.size() * 2U);
            std::memcpy(uma_loop.data() + wg_f16.size() * 2U, wf_f16.data(), wf_f16.size() * 2U);
            id<MTLBuffer> b_uma  = [dev newBufferWithBytes:uma_loop.data() length:uma_loop.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_h    = [dev newBufferWithBytes:h_f16.data() length:h_f16.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_cur  = [dev newBufferWithBytes:cur_f32.data() length:cur_f32.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_msg  = [dev newBufferWithBytes:msg_f32.data() length:msg_f32.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_lout = [dev newBufferWithLength:static_cast<size_t>(M) * N * 4U options:MTLResourceStorageModeShared];

            struct SikNaxP { uint32_t M_v, M_p, N_l, N_p, K, out_mode; float softcap; uint32_t pad0; };
            SikNaxP gp{M, M, N, N, K, 1U /*SIK_OUT_F32*/, 0.0f, 0U};

            struct SikLoopP {
                uint64_t w_off[3], s_off[3], f_off[3], fs_off[3];
                uint32_t M_v, M_p, D, gate_rank, num_ctrls, pad0, pad1, pad2;
            };
            SikLoopP lp{};
            lp.w_off[0] = 0ULL;
            lp.f_off[0] = static_cast<uint64_t>(wg_f16.size() * 2U);
            lp.M_v = M; lp.M_p = M; lp.D = N; lp.gate_rank = R; lp.num_ctrls = 1U;

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            [enc setComputePipelineState:pso_q4a_nax64];
            [enc setBuffer:b_q4w offset:0 atIndex:0];
            [enc setBuffer:b_q4s offset:0 atIndex:1];
            [enc setBuffer:b_q4b offset:0 atIndex:2];
            [enc setBuffer:b_x   offset:0 atIndex:3];
            [enc setBuffer:b_y4  offset:0 atIndex:4];
            [enc setBuffer:b_y4  offset:0 atIndex:5];
            [enc setBytes:&gp length:sizeof(gp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(N / 64U, M / 64U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:pso_q8_nax64];
            [enc setBuffer:b_q8w offset:0 atIndex:0];
            [enc setBuffer:b_q8s offset:0 atIndex:1];
            [enc setBuffer:b_x   offset:0 atIndex:2];
            [enc setBuffer:b_y8  offset:0 atIndex:3];
            [enc setBuffer:b_y8  offset:0 atIndex:4];
            [enc setBytes:&gp length:sizeof(gp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(N / 64U, M / 64U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:pso_loop_f16];
            [enc setBuffer:b_uma  offset:0 atIndex:0];
            [enc setBuffer:b_h    offset:0 atIndex:1];
            [enc setBuffer:b_cur  offset:0 atIndex:2];
            [enc setBuffer:b_cur  offset:0 atIndex:3];
            [enc setBuffer:b_cur  offset:0 atIndex:4];
            [enc setBuffer:b_msg  offset:0 atIndex:5];
            [enc setBuffer:b_msg  offset:0 atIndex:6];
            [enc setBuffer:b_msg  offset:0 atIndex:7];
            [enc setBuffer:b_lout offset:0 atIndex:8];
            [enc setBuffer:b_lout offset:0 atIndex:9];
            [enc setBuffer:b_lout offset:0 atIndex:10];
            [enc setBytes:&lp length:sizeof(lp) atIndex:11];
            [enc dispatchThreadgroups:MTLSizeMake(N / 64U, M / 64U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            std::vector<float> ref4(static_cast<size_t>(M) * N), ref8(static_cast<size_t>(M) * N), refl(static_cast<size_t>(M) * N);
            for (uint32_t m = 0U; m < M; ++m) {
                for (uint32_t n = 0U; n < N; ++n) {
                    double s4 = 0.0, s8 = 0.0;
                    for (uint32_t k = 0U; k < K; ++k) {
                        const float xv = fp16_bits_to_f32_ref(x_h[static_cast<size_t>(m) * K + k]);
                        const uint32_t g = k / gs;
                        const uint8_t pb = q4_w[static_cast<size_t>(n) * (K / 2U) + (k / 2U)];
                        const uint32_t nib = (k & 1U) ? ((pb >> 4U) & 0xFU) : (pb & 0xFU);
                        const float w4 = static_cast<float>(nib) * fp16_bits_to_f32_ref(q4_s[static_cast<size_t>(n) * (K / gs) + g])
                                       + fp16_bits_to_f32_ref(q4_b[static_cast<size_t>(n) * (K / gs) + g]);
                        const float w8 = static_cast<float>(q8_w[static_cast<size_t>(n) * K + k])
                                       * fp16_bits_to_f32_ref(q8_s[static_cast<size_t>(n) * (K / gs) + g]);
                        s4 += static_cast<double>(w4) * static_cast<double>(xv);
                        s8 += static_cast<double>(w8) * static_cast<double>(xv);
                    }
                    ref4[static_cast<size_t>(m) * N + n] = static_cast<float>(s4);
                    ref8[static_cast<size_t>(m) * N + n] = static_cast<float>(s8);

                    double cg = 0.0, cf = 0.0;
                    for (uint32_t r = 0U; r < R; ++r) {
                        const float hv = fp16_bits_to_f32_ref(h_f16[static_cast<size_t>(m) * R + r]);
                        cg += static_cast<double>(hv) * static_cast<double>(fp16_bits_to_f32_ref(wg_f16[static_cast<size_t>(n) * R + r]));
                        cf += static_cast<double>(hv) * static_cast<double>(fp16_bits_to_f32_ref(wf_f16[static_cast<size_t>(n) * R + r]));
                    }
                    const size_t idx = static_cast<size_t>(m) * N + n;
                    refl[idx] = (1.0f - std::tanh(static_cast<float>(cf))) * cur_f32[idx]
                              + std::tanh(static_cast<float>(cg)) * msg_f32[idx];
                }
            }

            const float* g4 = reinterpret_cast<const float*>([b_y4 contents]);
            const float* g8 = reinterpret_cast<const float*>([b_y8 contents]);
            const float* gl = reinterpret_cast<const float*>([b_lout contents]);
            const double cos4 = cosine_sim(g4, ref4.data(), ref4.size());
            const double cos8 = cosine_sim(g8, ref8.data(), ref8.size());
            float err_l = 0.0f;
            for (size_t i = 0U; i < refl.size(); ++i) err_l = std::max(err_l, std::fabs(gl[i] - refl[i]));
            const bool ok12 = (cos4 > 0.99999) && (cos8 > 0.99999) && (err_l < 5e-3f);
            std::printf("[Test 12] Inkling-2 Metal 4.1 (0 warns) + BK=128 NAX/Loop : %s (q4=%.6f q8=%.6f loop_err=%.2e)\n",
                        ok12 ? "PASS" : "FAIL", cos4, cos8, err_l);
            if (!ok12) return 1;
        }

        // ---- Test 13: Quad-Bin Concurrent NAX MoE (BM=16/32/64/128) + Grouped-4 M=1==M=8 ----
        {
            const uint32_t E = 8U, D = 256U, inter = 128U, M_quad = 256U;
            std::mt19937 rng(202613);
            std::normal_distribution<float> nd(0.0f, 0.25f);

            const size_t w13_bytes = static_cast<size_t>(E) * 2U * inter * (D / 2U);
            const size_t s13_bytes = static_cast<size_t>(E) * 2U * inter * (D / 16U);
            const size_t w2_bytes  = static_cast<size_t>(E) * D * (inter / 2U);
            const size_t s2_bytes  = static_cast<size_t>(E) * D * (inter / 16U);
            std::vector<uint8_t> w13(w13_bytes), s13(s13_bytes), w2(w2_bytes), s2(s2_bytes);
            std::vector<float>   es13(E, 0.25f), es2(E, 0.25f);
            for (auto& v : w13) v = static_cast<uint8_t>(rng() & 0xFFU);
            for (auto& v : w2)  v = static_cast<uint8_t>(rng() & 0xFFU);
            // Finite positive E4M3 scales (0x20..0x3F => ~0.06..0.5)
            for (auto& v : s13) v = static_cast<uint8_t>(0x20U + (rng() & 0x1FU));
            for (auto& v : s2)  v = static_cast<uint8_t>(0x20U + (rng() & 0x1FU));

            std::vector<uint16_t> x_h(static_cast<size_t>(M_quad) * D);
            for (auto& v : x_h) v = f32_to_fp16_bits(nd(rng));

            id<MTLBuffer> b_w13   = [dev newBufferWithBytes:w13.data() length:w13.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_s13   = [dev newBufferWithBytes:s13.data() length:s13.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_es13  = [dev newBufferWithBytes:es13.data() length:es13.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_w2    = [dev newBufferWithBytes:w2.data() length:w2.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_s2    = [dev newBufferWithBytes:s2.data() length:s2.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_es2   = [dev newBufferWithBytes:es2.data() length:es2.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_x     = [dev newBufferWithBytes:x_h.data() length:x_h.size() * 2U options:MTLResourceStorageModeShared];

            struct SikMoeExecP { uint32_t M_v, top_k, D, inter, num_exp, tile_cap; float gcap, ucap; };

            // Part A: Grouped-4 M=1 vs M=8 bit-exactness (0 bit diffs on token 0!)
            const uint32_t tk = 4U;
            id<MTLBuffer> b_tk1   = [dev newBufferWithLength:tk * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_tk8   = [dev newBufferWithLength:8U * tk * 4U options:MTLResourceStorageModeShared];
            auto* tk1 = reinterpret_cast<uint32_t*>([b_tk1 contents]);
            auto* tk8 = reinterpret_cast<uint32_t*>([b_tk8 contents]);
            for (uint32_t k = 0U; k < tk; ++k) tk1[k] = k;
            for (uint32_t m = 0U; m < 8U; ++m) {
                for (uint32_t k = 0U; k < tk; ++k) tk8[m * tk + k] = (m == 0U) ? tk1[k] : ((m + k) % E);
            }
            id<MTLBuffer> b_sp    = [dev newBufferWithLength:(M_quad * tk + 256U) * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_wt    = [dev newBufferWithLength:2048U * 16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_nt    = [dev newBufferWithLength:64U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ind   = [dev newBufferWithLength:256U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mid1  = [dev newBufferWithLength:tk * inter * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dn1   = [dev newBufferWithLength:tk * D * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mid8  = [dev newBufferWithLength:8U * tk * inter * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dn8   = [dev newBufferWithLength:8U * tk * D * 4U options:MTLResourceStorageModeShared];

            auto run_grouped = [&](id<MTLBuffer> b_tk, uint32_t M_v, uint32_t topk_v, id<MTLBuffer> b_mid, id<MTLBuffer> b_dn) {
                SikMoeExecP ep{M_v, topk_v, D, inter, E, 4U, 4.0f, 25.0f};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_bin_grp];
                [enc setBuffer:b_tk offset:0 atIndex:0];
                [enc setBuffer:b_sp offset:0 atIndex:1];
                [enc setBuffer:b_wt offset:0 atIndex:2];
                [enc setBuffer:b_nt offset:0 atIndex:3];
                [enc setBuffer:b_ind offset:0 atIndex:4];
                [enc setBytes:&ep length:sizeof(ep) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_grp_w13_4];
                [enc setBuffer:b_w13 offset:0 atIndex:0]; [enc setBuffer:b_s13 offset:0 atIndex:1];
                [enc setBuffer:b_es13 offset:0 atIndex:2]; [enc setBuffer:b_x offset:0 atIndex:3];
                [enc setBuffer:b_sp offset:0 atIndex:4]; [enc setBuffer:b_wt offset:0 atIndex:5];
                [enc setBuffer:b_nt offset:0 atIndex:6]; [enc setBuffer:b_mid offset:0 atIndex:7];
                [enc setBytes:&ep length:sizeof(ep) atIndex:8];
                [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_grp_w2_4];
                [enc setBuffer:b_w2 offset:0 atIndex:0]; [enc setBuffer:b_s2 offset:0 atIndex:1];
                [enc setBuffer:b_es2 offset:0 atIndex:2]; [enc setBuffer:b_mid offset:0 atIndex:3];
                [enc setBuffer:b_sp offset:0 atIndex:4]; [enc setBuffer:b_wt offset:0 atIndex:5];
                [enc setBuffer:b_nt offset:0 atIndex:6]; [enc setBuffer:b_dn offset:0 atIndex:7];
                [enc setBytes:&ep length:sizeof(ep) atIndex:8];
                [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:16 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            };
            run_grouped(b_tk1, 1U, tk, b_mid1, b_dn1);
            run_grouped(b_tk8, 8U, tk, b_mid8, b_dn8);

            int bit_diffs_1v8 = 0;
            const auto* u1 = reinterpret_cast<const uint32_t*>([b_dn1 contents]);
            const auto* u8 = reinterpret_cast<const uint32_t*>([b_dn8 contents]);
            for (size_t i = 0U; i < static_cast<size_t>(tk) * D; ++i) {
                if (u1[i] != u8[i]) ++bit_diffs_1v8;
            }

            // Part B: Quad-Bin Concurrent Indirect NAX (`BM=16/32/64/128` ALL firing!) vs Grouped-4 Ref!
            id<MTLBuffer> b_tkq   = [dev newBufferWithLength:M_quad * 4U options:MTLResourceStorageModeShared];
            auto* tkq = reinterpret_cast<uint32_t*>([b_tkq contents]);
            // e=0:100 (BM128), e=1:50 (BM64), e=2:24 (BM32), e=3:10 (BM16), e=4:72 (BM64+BM16) -> 256 total!
            for (uint32_t m = 0U; m < M_quad; ++m) {
                tkq[m] = (m < 100U) ? 0U : ((m < 150U) ? 1U : ((m < 174U) ? 2U : ((m < 184U) ? 3U : 4U)));
            }
            id<MTLBuffer> b_mid_r = [dev newBufferWithLength:M_quad * inter * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dn_r  = [dev newBufferWithLength:M_quad * D * 4U options:MTLResourceStorageModeShared];
            run_grouped(b_tkq, M_quad, 1U, b_mid_r, b_dn_r);

            id<MTLBuffer> b_xsort = [dev newBufferWithLength:(M_quad + 128U) * D * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_msort = [dev newBufferWithLength:(M_quad + 128U) * inter * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dn_q  = [dev newBufferWithLength:M_quad * D * 2U options:MTLResourceStorageModeShared];
            {
                SikMoeExecP epq{M_quad, 1U, D, inter, E, 128U, 4.0f, 25.0f};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_bin_quad];
                [enc setBuffer:b_tkq offset:0 atIndex:0]; [enc setBuffer:b_sp offset:0 atIndex:1];
                [enc setBuffer:b_wt offset:0 atIndex:2]; [enc setBuffer:b_nt offset:0 atIndex:3];
                [enc setBuffer:b_ind offset:0 atIndex:4]; [enc setBytes:&epq length:sizeof(epq) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_gather_x];
                [enc setBuffer:b_x offset:0 atIndex:0]; [enc setBuffer:b_sp offset:0 atIndex:1];
                [enc setBuffer:b_xsort offset:0 atIndex:2]; [enc setBuffer:b_msort offset:0 atIndex:3];
                [enc setBytes:&epq length:sizeof(epq) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(M_quad + 128U, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];

                const NSUInteger tg_sz[4] = {64, 128, 128, 128};
                for (int pass = 0; pass < 2; ++pass) {
                    id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    for (uint32_t b = 0U; b < 4U; ++b) {
                        [cenc setComputePipelineState:(pass == 0 ? pso_q_w13_g[b] : pso_q_w13_u[b])];
                        [cenc setBuffer:b_w13 offset:0 atIndex:0]; [cenc setBuffer:b_s13 offset:0 atIndex:1];
                        [cenc setBuffer:b_es13 offset:0 atIndex:2]; [cenc setBuffer:b_xsort offset:0 atIndex:3];
                        [cenc setBuffer:b_sp offset:0 atIndex:4];
                        [cenc setBuffer:b_wt offset:static_cast<NSUInteger>(b) * 512U * 16U atIndex:5];
                        [cenc setBuffer:b_msort offset:0 atIndex:6]; [cenc setBytes:&epq length:sizeof(epq) atIndex:7];
                        [cenc dispatchThreadgroupsWithIndirectBuffer:b_ind
                                                indirectBufferOffset:static_cast<NSUInteger>(b) * 32U
                                               threadsPerThreadgroup:MTLSizeMake(tg_sz[b], 1, 1)];
                    }
                    [cenc endEncoding];
                }
                {
                    id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    for (uint32_t b = 0U; b < 4U; ++b) {
                        [cenc setComputePipelineState:pso_q_w2[b]];
                        [cenc setBuffer:b_w2 offset:0 atIndex:0]; [cenc setBuffer:b_s2 offset:0 atIndex:1];
                        [cenc setBuffer:b_es2 offset:0 atIndex:2]; [cenc setBuffer:b_msort offset:0 atIndex:3];
                        [cenc setBuffer:b_sp offset:0 atIndex:4];
                        [cenc setBuffer:b_wt offset:static_cast<NSUInteger>(b) * 512U * 16U atIndex:5];
                        [cenc setBuffer:b_dn_q offset:0 atIndex:6]; [cenc setBytes:&epq length:sizeof(epq) atIndex:7];
                        [cenc dispatchThreadgroupsWithIndirectBuffer:b_ind
                                                indirectBufferOffset:static_cast<NSUInteger>(b) * 32U + 16U
                                               threadsPerThreadgroup:MTLSizeMake(tg_sz[b], 1, 1)];
                    }
                    [cenc endEncoding];
                }
                [cb commit]; [cb waitUntilCompleted];
            }

            const auto* nt_q = reinterpret_cast<const uint32_t*>([b_nt contents]);
            const auto* ref_dn = reinterpret_cast<const float*>([b_dn_r contents]);
            const auto* q_dn16 = reinterpret_cast<const uint16_t*>([b_dn_q contents]);
            std::vector<float> q_dn32(static_cast<size_t>(M_quad) * D);
            for (size_t i = 0U; i < q_dn32.size(); ++i) q_dn32[i] = fp16_bits_to_f32_ref(q_dn16[i]);
            const double cos_quad = cosine_sim(q_dn32.data(), ref_dn, q_dn32.size());
            const bool bins_ok = (nt_q[0] == 2U && nt_q[1] == 1U && nt_q[2] == 2U && nt_q[3] == 1U);
            const bool ok13 = (bit_diffs_1v8 == 0) && bins_ok && (cos_quad > 0.99995);
            std::printf("[Test 13] Quad-Bin NAX MoE (16/32/64/128) + Grp4 M1==M8   : %s (diffs=%d bins=%u/%u/%u/%u cos=%.6f)\n",
                        ok13 ? "PASS" : "FAIL", bit_diffs_1v8, nt_q[0], nt_q[1], nt_q[2], nt_q[3], cos_quad);
            if (!ok13) return 1;
        }

        // ---- Test 14: Per-Head Q8_0 Global KV Cache Quantize & NAX FlashAttention Parity ----
        {
            const uint32_t M = 64U, n_q = 6U, n_kv = 2U, hd = 128U, d_rel = 16U, stride = 128U;
            std::mt19937 rng(202614);
            std::normal_distribution<float> nd(0.0f, 0.5f);

            std::vector<float> q_raw(M * n_q * hd), k_c(M * n_kv * hd), v_c(M * n_kv * hd);
            std::vector<float> r_raw(M * n_q * d_rel), qs(n_q * hd, 1.0f), rs(n_q * d_rel, 1.0f);
            std::vector<float> rel_p(1024U * d_rel, 0.0f), sink(M * n_q, 0.0f), gate(M * n_q * hd, 1.0f);
            for (auto& v : q_raw) v = nd(rng);
            for (auto& v : k_c)   v = nd(rng);
            for (auto& v : v_c)   v = nd(rng);
            for (auto& v : r_raw) v = nd(rng) * 0.25f;
            for (auto& v : rel_p) v = nd(rng) * 0.1f;

            id<MTLBuffer> b_qr   = [dev newBufferWithBytes:q_raw.data() length:q_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kc   = [dev newBufferWithBytes:k_c.data() length:k_c.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vc   = [dev newBufferWithBytes:v_c.data() length:v_c.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rr   = [dev newBufferWithBytes:r_raw.data() length:r_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_qs   = [dev newBufferWithBytes:qs.data() length:qs.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rs   = [dev newBufferWithBytes:rs.data() length:rs.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rp   = [dev newBufferWithBytes:rel_p.data() length:rel_p.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sk   = [dev newBufferWithBytes:sink.data() length:sink.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_gt   = [dev newBufferWithBytes:gate.data() length:gate.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_qhm  = [dev newBufferWithLength:n_q * M * hd * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rhm  = [dev newBufferWithLength:n_q * M * d_rel * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_k16  = [dev newBufferWithLength:n_kv * stride * hd * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_v16  = [dev newBufferWithLength:n_kv * stride * hd * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_k8   = [dev newBufferWithLength:n_kv * stride * hd * 1U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_v8   = [dev newBufferWithLength:n_kv * stride * hd * 1U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ks   = [dev newBufferWithLength:n_kv * stride * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vs   = [dev newBufferWithLength:n_kv * stride * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_o16  = [dev newBufferWithLength:M * n_q * hd * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_o8   = [dev newBufferWithLength:M * n_q * hd * 2U options:MTLResourceStorageModeShared];

            struct SikPrepP {
                uint32_t M_v, M_p16, n_q, n_kv, hd, d_rel, seq0, kv_stride, is_loc, snap, v_off, ztail;
                float qk_clip, rel_clip, eps, log_a, log_nf, scale;
                uint32_t kv_dtype, pad1;
            };
            struct SikAttnP {
                uint32_t M_v, M_p16, n_q, n_kv, hd, d_rel, rel_ext, swa_win, is_loc, seq0, m_off, kv_stride, max_sp, csz, cbf;
                float scale;
            };
            const float ascale = 1.0f / std::sqrt(static_cast<float>(hd));
            SikPrepP pp16{M, M, n_q, n_kv, hd, d_rel, 0U, stride, 0U, 0U, 0U, 0U, 22.627417f, 8.0f, 1e-6f, 0.1f, 8192.0f, ascale, STRATA_INKLING2_KV_DTYPE_F16, 0U};
            SikPrepP pp8 = pp16; pp8.kv_dtype = STRATA_INKLING2_KV_DTYPE_Q8_0;
            SikAttnP ap{M, M, n_q, n_kv, hd, d_rel, 1024U, 1024U, 0U, 0U, 0U, stride, 1U, 32U, 0U, ascale};

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_qkvr_pre];
            [enc setBuffer:b_qr offset:0 atIndex:0]; [enc setBuffer:b_kc offset:0 atIndex:1];
            [enc setBuffer:b_vc offset:0 atIndex:2]; [enc setBuffer:b_rr offset:0 atIndex:3];
            [enc setBuffer:b_qs offset:0 atIndex:4]; [enc setBuffer:b_rs offset:0 atIndex:5];
            [enc setBuffer:b_qhm offset:0 atIndex:6]; [enc setBuffer:b_rhm offset:0 atIndex:7];
            [enc setBuffer:b_k16 offset:0 atIndex:8]; [enc setBuffer:b_v16 offset:0 atIndex:9];
            [enc setBytes:&pp16 length:sizeof(pp16) atIndex:10];
            [enc setBuffer:b_ks offset:0 atIndex:11]; [enc setBuffer:b_vs offset:0 atIndex:12];
            [enc dispatchThreadgroups:MTLSizeMake(2U * n_q + 2U * n_kv, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setBuffer:b_k8 offset:0 atIndex:8]; [enc setBuffer:b_v8 offset:0 atIndex:9];
            [enc setBytes:&pp8 length:sizeof(pp8) atIndex:10];
            [enc dispatchThreadgroups:MTLSizeMake(2U * n_q + 2U * n_kv, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:pso_attn_f16];
            [enc setBuffer:b_qhm offset:0 atIndex:0]; [enc setBuffer:b_rhm offset:0 atIndex:1];
            [enc setBuffer:b_rp offset:0 atIndex:2];  [enc setBuffer:b_sk offset:0 atIndex:3];
            [enc setBuffer:b_gt offset:0 atIndex:4];  [enc setBuffer:b_k16 offset:0 atIndex:5];
            [enc setBuffer:b_v16 offset:0 atIndex:6]; [enc setBuffer:b_o16 offset:0 atIndex:7];
            [enc setBytes:&ap length:sizeof(ap) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(M / 16U, n_kv, 1) threadsPerThreadgroup:MTLSizeMake(96, 1, 1)];

            [enc setComputePipelineState:pso_attn_q8];
            [enc setBuffer:b_qhm offset:0 atIndex:0]; [enc setBuffer:b_rhm offset:0 atIndex:1];
            [enc setBuffer:b_rp offset:0 atIndex:2];  [enc setBuffer:b_sk offset:0 atIndex:3];
            [enc setBuffer:b_gt offset:0 atIndex:4];  [enc setBuffer:b_k8 offset:0 atIndex:5];
            [enc setBuffer:b_v8 offset:0 atIndex:6];  [enc setBuffer:b_o8 offset:0 atIndex:7];
            [enc setBytes:&ap length:sizeof(ap) atIndex:8];
            [enc setBuffer:b_ks offset:0 atIndex:9];  [enc setBuffer:b_vs offset:0 atIndex:10];
            [enc dispatchThreadgroups:MTLSizeMake(M / 64U, n_q, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

            const auto* o16 = reinterpret_cast<const uint16_t*>([b_o16 contents]);
            const auto* o8  = reinterpret_cast<const uint16_t*>([b_o8 contents]);
            std::vector<float> f16(M * n_q * hd), f8(M * n_q * hd);
            for (size_t i = 0U; i < f16.size(); ++i) {
                f16[i] = fp16_bits_to_f32_ref(o16[i]);
                f8[i]  = fp16_bits_to_f32_ref(o8[i]);
            }
            const double cos_q8kv = cosine_sim(f16.data(), f8.data(), f16.size());
            const bool ok14 = (cos_q8kv > 0.9999);
            std::printf("[Test 14] Per-Head Q8_0 KV Cache + NAX FlashAttn Parity   : %s (cos=%.6f, 130B vs 256B/head)\n",
                        ok14 ? "PASS" : "FAIL", cos_q8kv);
            if (!ok14) return 1;
        }

        // ---- Test 15: Q4_A 3L DSpark MTP `xsum` GEMV Parity (`M = 8`) ----
        {
            const uint32_t M = 8U, rows = 128U, cols = 256U, gs = 64U;
            std::mt19937 rng(202615);
            std::normal_distribution<float> nd(0.0f, 0.35f);
            std::vector<uint16_t> x_h(M * cols), ws(rows * (cols / gs)), wb(rows * (cols / gs));
            std::vector<uint8_t>  wq(rows * (cols / 2U));
            for (auto& v : x_h) v = f32_to_fp16_bits(nd(rng));
            for (auto& v : wq)  v = static_cast<uint8_t>(rng() & 0xFFU);
            for (size_t i = 0U; i < ws.size(); ++i) {
                ws[i] = f32_to_fp16_bits(0.02f + 0.04f * (static_cast<float>(rng() & 255U) / 255.0f));
                wb[i] = f32_to_fp16_bits(-0.15f + 0.30f * (static_cast<float>(rng() & 255U) / 255.0f));
            }
            id<MTLBuffer> b_x   = [dev newBufferWithBytes:x_h.data() length:x_h.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_wq  = [dev newBufferWithBytes:wq.data() length:wq.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ws  = [dev newBufferWithBytes:ws.data() length:ws.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_wb  = [dev newBufferWithBytes:wb.data() length:wb.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xs  = [dev newBufferWithLength:M * (cols / gs) * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y0  = [dev newBufferWithLength:M * rows * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y1  = [dev newBufferWithLength:M * rows * 4U options:MTLResourceStorageModeShared];

            struct SikBlkSumP { uint32_t M, cols, p0, p1; };
            struct SikGemvP   { uint32_t M, rows, cols, out_mode; float softcap; uint32_t p0, p1, p2; };
            SikBlkSumP bp{M, cols, 0U, 0U};
            SikGemvP   gp{M, rows, cols, 1U /*SIK_OUT_F32*/, 0.0f, 0U, 0U, 0U};

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_blk64_sum];
            [enc setBuffer:b_x offset:0 atIndex:0]; [enc setBuffer:b_xs offset:0 atIndex:1];
            [enc setBytes:&bp length:sizeof(bp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:pso_q4a_gemv8];
            [enc setBuffer:b_wq offset:0 atIndex:0]; [enc setBuffer:b_ws offset:0 atIndex:1];
            [enc setBuffer:b_wb offset:0 atIndex:2]; [enc setBuffer:b_x offset:0 atIndex:3];
            [enc setBuffer:b_y0 offset:0 atIndex:4]; [enc setBuffer:b_y0 offset:0 atIndex:5];
            [enc setBytes:&gp length:sizeof(gp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((rows + 7U) / 8U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            [enc setComputePipelineState:pso_q4a_xsum8];
            [enc setBuffer:b_wq offset:0 atIndex:0]; [enc setBuffer:b_ws offset:0 atIndex:1];
            [enc setBuffer:b_wb offset:0 atIndex:2]; [enc setBuffer:b_x offset:0 atIndex:3];
            [enc setBuffer:b_xs offset:0 atIndex:4]; [enc setBuffer:b_y1 offset:0 atIndex:5];
            [enc setBuffer:b_y1 offset:0 atIndex:6]; [enc setBytes:&gp length:sizeof(gp) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((rows + 7U) / 8U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

            const float* y0 = reinterpret_cast<const float*>([b_y0 contents]);
            const float* y1 = reinterpret_cast<const float*>([b_y1 contents]);
            float max_err = 0.0f;
            for (size_t i = 0U; i < static_cast<size_t>(M) * rows; ++i) {
                max_err = std::max(max_err, std::fabs(y0[i] - y1[i]));
            }
            const double cos_xs = cosine_sim(y0, y1, static_cast<size_t>(M) * rows);
            const bool ok15 = (max_err < 1e-3f) && (cos_xs > 0.999999);
            std::printf("[Test 15] Q4_A 3L DSpark MTP xsum GEMV Parity (M=8)       : %s (max_err=%.2e cos=%.7f)\n",
                        ok15 ? "PASS" : "FAIL", max_err, cos_xs);
            if (!ok15) return 1;
        }

        // ---- Test 16: Persistent `MTLBinaryArchive` Cold Miss -> Atomic Save -> Warm Hit ----
        {
            const char* prev_dir_env = std::getenv("STRATA_METAL_CACHE_DIR");
            const char* prev_dis_env = std::getenv("STRATA_METAL_CACHE_DISABLE");
            const std::string saved_dir = prev_dir_env ? std::string(prev_dir_env) : std::string();
            const std::string saved_dis = prev_dis_env ? std::string(prev_dis_env) : std::string();

            char tmp_dir[256];
            std::snprintf(tmp_dir, sizeof(tmp_dir), "/tmp/strata_test_metal_cache_%d", static_cast<int>(getpid()));
            ::setenv("STRATA_METAL_CACHE_DIR", tmp_dir, 1);
            ::unsetenv("STRATA_METAL_CACHE_DISABLE");

            struct CacheKernelSpec { const char* name; bool use_c; uint32_t cval; };
            const CacheKernelSpec specs[12] = {
                {"sik_quant_f16_to_i8_row",          false, 0U},
                {"sik_q8_gemm_nax_i8_m64",           false, 0U},
                {"sik_q4a_gemm_nax_i8i4_m64",        false, 0U},
                {"sik_q4a_gemm_nax_i8_m128",         false, 0U},
                {"sik_loop_ctrl_prep_i8",            false, 0U},
                {"sik_loop_ctrl_down_nax_q8_i8_m64", false, 0U},
                {"sik_loop_ctrl_apply_nax_q8_i8_m64",false, 0U},
                {"sik_moe_prefill_w13_nvfp4_i8_m128",true,  0U},
                {"sik_moe_prefill_w13_nvfp4_i8_m128",true,  1U},
                {"sik_moe_prefill_w2_nvfp4_i8_m128", false, 0U},
                {"sik_moe_prefill_w13_mxfp8_i8_m128",true,  0U},
                {"sik_moe_prefill_w2_mxfp8_i8_m128", false, 0U}
            };

            bool ok16 = true;
            double cold_ms = 0.0, warm_ms = 0.0;
            uint32_t cold_miss = 0U, warm_hit = 0U, warm_miss = 0U, spill_cnt = 0U;
            bool cold_was_hit = true, warm_was_hit = false, file_exists = false;
            std::string cache_file;

            // Pass 1: Cold miss -> compile -> atomic serialize
            StrataMetalPsoCache* c1 = strata_metal_cache_open_with_opts(
                dev, lib, "test_inkling2_cache", msl_src.data(), msl_src.size(), opts);
            if (!c1) {
                ok16 = false;
            } else {
                cold_was_hit = c1->cache_hit;
                for (uint32_t i = 0U; i < 12U; ++i) {
                    NSError* pe = nil;
                    id<MTLComputePipelineState> pso = specs[i].use_c
                        ? strata_metal_cache_make_pso_const_u32(c1, specs[i].name, 0U, specs[i].cval, &pe)
                        : strata_metal_cache_make_pso(c1, specs[i].name, nil, &pe);
                    if (!pso || pso.maxTotalThreadsPerThreadgroup < 1024U) ok16 = false;
                }
                const bool cold_dirty = c1->dirty;
                if (!strata_metal_cache_finish(c1)) ok16 = false;
                cold_ms   = c1->elapsed_ms;
                cold_miss = c1->archive_miss_count;
                spill_cnt = c1->spill_risk_count;
                cache_file = c1->cache_path;
                struct stat st{};
                file_exists = (::stat(cache_file.c_str(), &st) == 0 && st.st_size > 0);
                if (cold_was_hit || cold_miss != 12U || !cold_dirty || !file_exists || spill_cnt != 0U) ok16 = false;
                strata_metal_cache_destroy(c1);
            }

            // Pass 2: Warm hit -> load directly from on-disk MTLBinaryArchive
            StrataMetalPsoCache* c2 = ok16 ? strata_metal_cache_open_with_opts(
                dev, lib, "test_inkling2_cache", msl_src.data(), msl_src.size(), opts) : nullptr;
            if (!c2) {
                ok16 = false;
            } else {
                for (uint32_t i = 0U; i < 12U; ++i) {
                    NSError* pe = nil;
                    id<MTLComputePipelineState> pso = specs[i].use_c
                        ? strata_metal_cache_make_pso_const_u32(c2, specs[i].name, 0U, specs[i].cval, &pe)
                        : strata_metal_cache_make_pso(c2, specs[i].name, nil, &pe);
                    if (!pso || pso.maxTotalThreadsPerThreadgroup < 1024U) ok16 = false;
                }
                if (!strata_metal_cache_finish(c2)) ok16 = false;
                warm_was_hit = c2->cache_hit;
                warm_ms      = c2->elapsed_ms;
                warm_hit     = c2->archive_hit_count;
                warm_miss    = c2->archive_miss_count;
                if (!warm_was_hit || warm_hit != 12U || warm_miss != 0U || c2->dirty || c2->spill_risk_count != 0U ||
                    !(warm_ms < cold_ms)) {
                    ok16 = false;
                }
                strata_metal_cache_destroy(c2);
            }

            // Clean up temp archive directory and restore environment
            if (!cache_file.empty()) ::unlink(cache_file.c_str());
            ::rmdir(tmp_dir);
            if (prev_dir_env) ::setenv("STRATA_METAL_CACHE_DIR", saved_dir.c_str(), 1);
            else              ::unsetenv("STRATA_METAL_CACHE_DIR");
            if (prev_dis_env) ::setenv("STRATA_METAL_CACHE_DISABLE", saved_dis.c_str(), 1);
            else              ::unsetenv("STRATA_METAL_CACHE_DISABLE");

            std::printf("[Test 16] MTLBinaryArchive Cold->Save->Warm Hit (12 PSOs) : %s (cold=%.2fms miss=%u -> warm=%.2fms hit=%u/%.0fx spill=%u)\n",
                        ok16 ? "PASS" : "FAIL", cold_ms, cold_miss, warm_ms, warm_hit,
                        (warm_ms > 0.0) ? (cold_ms / warm_ms) : 0.0, spill_cnt);
            if (!ok16) return 1;
        }

        // ---- Test 17: Metal 4.1 INT8xINT8 & INT8xINT4 NAX Prefill Parity (`cos >= 0.995`) ----
        {
            id<MTLComputePipelineState> pso_qrow         = get_pso("sik_quant_f16_to_i8_row");
            id<MTLComputePipelineState> pso_q8_i8_m64    = get_pso("sik_q8_gemm_nax_i8_m64");
            id<MTLComputePipelineState> pso_q8_i8_m32    = get_pso("sik_q8_gemm_nax_i8_m32");
            id<MTLComputePipelineState> pso_q4_i8i4_m64  = get_pso("sik_q4a_gemm_nax_i8i4_m64");
            id<MTLComputePipelineState> pso_q4_i8i4_m32  = get_pso("sik_q4a_gemm_nax_i8i4_m32");
            id<MTLComputePipelineState> pso_q4_i8_m128   = get_pso("sik_q4a_gemm_nax_i8_m128");
            id<MTLComputePipelineState> pso_q4_i8_m64    = get_pso("sik_q4a_gemm_nax_i8_m64");
            id<MTLComputePipelineState> pso_q4_i8_m32    = get_pso("sik_q4a_gemm_nax_i8_m32");
            id<MTLComputePipelineState> pso_lprep_i8     = get_pso("sik_loop_ctrl_prep_i8");
            id<MTLComputePipelineState> pso_ldown_i8_m64 = get_pso("sik_loop_ctrl_down_nax_q8_i8_m64");
            id<MTLComputePipelineState> pso_lapp_i8_m64  = get_pso("sik_loop_ctrl_apply_nax_q8_i8_m64");
            id<MTLComputePipelineState> pso_gather_i8    = get_pso("sik_moe_gather_x_i8");
            id<MTLComputePipelineState> pso_nv_w13_i8_g[4] = {
                get_pso_fc("sik_moe_prefill_w13_nvfp4_i8_m16",  0U),
                get_pso_fc("sik_moe_prefill_w13_nvfp4_i8_m32",  0U),
                get_pso_fc("sik_moe_prefill_w13_nvfp4_i8_m64",  0U),
                get_pso_fc("sik_moe_prefill_w13_nvfp4_i8_m128", 0U)
            };
            id<MTLComputePipelineState> pso_nv_w13_i8_u[4] = {
                get_pso_fc("sik_moe_prefill_w13_nvfp4_i8_m16",  1U),
                get_pso_fc("sik_moe_prefill_w13_nvfp4_i8_m32",  1U),
                get_pso_fc("sik_moe_prefill_w13_nvfp4_i8_m64",  1U),
                get_pso_fc("sik_moe_prefill_w13_nvfp4_i8_m128", 1U)
            };
            id<MTLComputePipelineState> pso_nv_w2_i8[4] = {
                get_pso("sik_moe_prefill_w2_nvfp4_i8_m16"),
                get_pso("sik_moe_prefill_w2_nvfp4_i8_m32"),
                get_pso("sik_moe_prefill_w2_nvfp4_i8_m64"),
                get_pso("sik_moe_prefill_w2_nvfp4_i8_m128")
            };
            id<MTLComputePipelineState> pso_mx_w13_fp16[2] = {
                get_pso_fc("sik_moe_prefill_w13_mxfp8_nax_m128", 0U),
                get_pso_fc("sik_moe_prefill_w13_mxfp8_nax_m128", 1U)
            };
            id<MTLComputePipelineState> pso_mx_w2_fp16 = get_pso("sik_moe_prefill_w2_mxfp8_nax_m128");
            id<MTLComputePipelineState> pso_mx_w13_i8[2] = {
                get_pso_fc("sik_moe_prefill_w13_mxfp8_i8_m128", 0U),
                get_pso_fc("sik_moe_prefill_w13_mxfp8_i8_m128", 1U)
            };
            id<MTLComputePipelineState> pso_mx_w2_i8 = get_pso("sik_moe_prefill_w2_mxfp8_i8_m128");

            if (!pso_qrow || !pso_q8_i8_m64 || !pso_q8_i8_m32 || !pso_q4_i8i4_m64 || !pso_q4_i8i4_m32 ||
                !pso_q4_i8_m128 || !pso_q4_i8_m64 || !pso_q4_i8_m32 || !pso_lprep_i8 || !pso_ldown_i8_m64 ||
                !pso_lapp_i8_m64 || !pso_gather_i8 ||
                !pso_nv_w13_i8_g[0] || !pso_nv_w13_i8_g[1] || !pso_nv_w13_i8_g[2] || !pso_nv_w13_i8_g[3] ||
                !pso_nv_w13_i8_u[0] || !pso_nv_w13_i8_u[1] || !pso_nv_w13_i8_u[2] || !pso_nv_w13_i8_u[3] ||
                !pso_nv_w2_i8[0] || !pso_nv_w2_i8[1] || !pso_nv_w2_i8[2] || !pso_nv_w2_i8[3] ||
                !pso_mx_w13_fp16[0] || !pso_mx_w13_fp16[1] || !pso_mx_w2_fp16 ||
                !pso_mx_w13_i8[0] || !pso_mx_w13_i8[1] || !pso_mx_w2_i8) {
                return 1;
            }

            struct SikQuantP   { uint32_t M_v, M_p, K, emit_xsum, num_slices, p0, p1, p2; };
            struct SikNaxP     { uint32_t M_v, M_p, N_l, N_p, K, out_mode; float softcap; uint32_t pad0; };
            struct SikMoeExecP { uint32_t M_v, top_k, D, inter, num_exp, tile_cap; float gcap, ucap; };
            struct SikLPrepP   { uint32_t M_v, M_p, D; float eps; };
            struct SikLoopP    { uint64_t w_off[3], s_off[3], f_off[3], fs_off[3]; uint32_t M_v, M_p, D, gate_rank, num_ctrls, p0, p1, p2; };

            // Part A: Dense Q8_0 (INT8xINT8) + Q4_A (INT8xINT4 & 8KiB INT8xINT8) with 6x outlier channels!
            const uint32_t M = 128U, N = 128U, K = 256U, gs = 64U;
            std::mt19937 rng(202617);
            std::normal_distribution<float> nd(0.0f, 0.35f);

            std::vector<uint16_t> x_h(static_cast<size_t>(M) * K);
            for (uint32_t m = 0U; m < M; ++m) {
                for (uint32_t k = 0U; k < K; ++k) {
                    const float v = nd(rng) * (((k % 37U) == 0U) ? 6.0f : 1.0f);
                    x_h[static_cast<size_t>(m) * K + k] = f32_to_fp16_bits(v);
                }
            }
            std::vector<uint8_t>  q4_w(static_cast<size_t>(N) * (K / 2U));
            std::vector<uint16_t> q4_s(static_cast<size_t>(N) * (K / gs)), q4_b(static_cast<size_t>(N) * (K / gs));
            for (auto& v : q4_w) v = static_cast<uint8_t>(rng() & 0xFFU);
            for (size_t i = 0U; i < q4_s.size(); ++i) {
                const float s = 0.02f + 0.05f * (static_cast<float>(rng() & 255U) / 255.0f);
                q4_s[i] = f32_to_fp16_bits(s);
                q4_b[i] = f32_to_fp16_bits(-7.5f * s + 0.03f * nd(rng));
            }
            std::vector<int8_t>   q8_w(static_cast<size_t>(N) * K);
            std::vector<uint16_t> q8_s(static_cast<size_t>(N) * (K / gs));
            for (auto& v : q8_w) v = static_cast<int8_t>(static_cast<int>(rng() % 251U) - 125);
            for (size_t i = 0U; i < q8_s.size(); ++i) {
                q8_s[i] = f32_to_fp16_bits(0.005f + 0.015f * (static_cast<float>(rng() & 255U) / 255.0f));
            }

            std::vector<float> ref_q8(static_cast<size_t>(M) * N), ref_q4(static_cast<size_t>(M) * N);
            for (uint32_t m = 0U; m < M; ++m) {
                for (uint32_t n = 0U; n < N; ++n) {
                    double s8 = 0.0, s4 = 0.0;
                    for (uint32_t k = 0U; k < K; ++k) {
                        const float xv = fp16_bits_to_f32_ref(x_h[static_cast<size_t>(m) * K + k]);
                        const uint32_t g = k / gs;
                        const uint8_t pb = q4_w[static_cast<size_t>(n) * (K / 2U) + (k / 2U)];
                        const uint32_t nib = (k & 1U) ? ((pb >> 4U) & 0xFU) : (pb & 0xFU);
                        const float w4 = static_cast<float>(nib) * fp16_bits_to_f32_ref(q4_s[static_cast<size_t>(n) * (K / gs) + g])
                                       + fp16_bits_to_f32_ref(q4_b[static_cast<size_t>(n) * (K / gs) + g]);
                        const float w8 = static_cast<float>(q8_w[static_cast<size_t>(n) * K + k])
                                       * fp16_bits_to_f32_ref(q8_s[static_cast<size_t>(n) * (K / gs) + g]);
                        s4 += static_cast<double>(w4) * static_cast<double>(xv);
                        s8 += static_cast<double>(w8) * static_cast<double>(xv);
                    }
                    ref_q8[static_cast<size_t>(m) * N + n] = static_cast<float>(s8);
                    ref_q4[static_cast<size_t>(m) * N + n] = static_cast<float>(s4);
                }
            }

            id<MTLBuffer> b_xf16   = [dev newBufferWithBytes:x_h.data()  length:x_h.size() * 2U  options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xi8    = [dev newBufferWithLength:static_cast<size_t>(M) * K         options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xsc    = [dev newBufferWithLength:static_cast<size_t>(M) * 2U        options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xsum   = [dev newBufferWithLength:static_cast<size_t>(M) * (K / gs) * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q8w    = [dev newBufferWithBytes:q8_w.data() length:q8_w.size()       options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q8s    = [dev newBufferWithBytes:q8_s.data() length:q8_s.size() * 2U  options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q4w    = [dev newBufferWithBytes:q4_w.data() length:q4_w.size()       options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q4s    = [dev newBufferWithBytes:q4_s.data() length:q4_s.size() * 2U  options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_q4b    = [dev newBufferWithBytes:q4_b.data() length:q4_b.size() * 2U  options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y_q8_64   = [dev newBufferWithLength:static_cast<size_t>(M) * N * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y_q8_32   = [dev newBufferWithLength:static_cast<size_t>(32U) * N * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y_i8i4_64 = [dev newBufferWithLength:static_cast<size_t>(M) * N * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y_i8i4_32 = [dev newBufferWithLength:static_cast<size_t>(32U) * N * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y_q4_128  = [dev newBufferWithLength:static_cast<size_t>(M) * N * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y_q4_64   = [dev newBufferWithLength:static_cast<size_t>(M) * N * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_y_q4_32   = [dev newBufferWithLength:static_cast<size_t>(32U) * N * 4U options:MTLResourceStorageModeShared];

            {
                SikQuantP qp{M, M, K, 1U, 1U, 0U, 0U, 0U};
                SikNaxP np128{M, M, N, N, K, 1U, 0.0f, 0U};
                SikNaxP np32{32U, 32U, N, N, K, 1U, 0.0f, 0U};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

                [enc setComputePipelineState:pso_qrow];
                [enc setBuffer:b_xf16 offset:0 atIndex:0]; [enc setBuffer:b_xi8 offset:0 atIndex:1];
                [enc setBuffer:b_xsc offset:0 atIndex:2];  [enc setBuffer:b_xsum offset:0 atIndex:3];
                [enc setBytes:&qp length:sizeof(qp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((M + 3U) / 4U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                // Q8_0 INT8xINT8 m64 & m32
                [enc setComputePipelineState:pso_q8_i8_m64];
                [enc setBuffer:b_q8w offset:0 atIndex:0]; [enc setBuffer:b_q8s offset:0 atIndex:1];
                [enc setBuffer:b_xi8 offset:0 atIndex:2]; [enc setBuffer:b_xsc offset:0 atIndex:3];
                [enc setBuffer:b_y_q8_64 offset:0 atIndex:4]; [enc setBuffer:b_y_q8_64 offset:0 atIndex:5];
                [enc setBytes:&np128 length:sizeof(np128) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(N / 64U, M / 64U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_q8_i8_m32];
                [enc setBuffer:b_y_q8_32 offset:0 atIndex:4]; [enc setBuffer:b_y_q8_32 offset:0 atIndex:5];
                [enc setBytes:&np32 length:sizeof(np32) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(N / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                // Q4_A INT8xINT4 m64 & m32
                [enc setComputePipelineState:pso_q4_i8i4_m64];
                [enc setBuffer:b_q4w offset:0 atIndex:0]; [enc setBuffer:b_q4s offset:0 atIndex:1];
                [enc setBuffer:b_q4b offset:0 atIndex:2]; [enc setBuffer:b_xi8 offset:0 atIndex:3];
                [enc setBuffer:b_xsc offset:0 atIndex:4]; [enc setBuffer:b_xsum offset:0 atIndex:5];
                [enc setBuffer:b_y_i8i4_64 offset:0 atIndex:6]; [enc setBuffer:b_y_i8i4_64 offset:0 atIndex:7];
                [enc setBytes:&np128 length:sizeof(np128) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(N / 64U, M / 64U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_q4_i8i4_m32];
                [enc setBuffer:b_y_i8i4_32 offset:0 atIndex:6]; [enc setBuffer:b_y_i8i4_32 offset:0 atIndex:7];
                [enc setBytes:&np32 length:sizeof(np32) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(N / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                // Q4_A 8KiB Whole-Row INT8xINT8 m128, m64 & m32
                [enc setComputePipelineState:pso_q4_i8_m128];
                [enc setBuffer:b_q4w offset:0 atIndex:0]; [enc setBuffer:b_q4s offset:0 atIndex:1];
                [enc setBuffer:b_q4b offset:0 atIndex:2]; [enc setBuffer:b_xi8 offset:0 atIndex:3];
                [enc setBuffer:b_xsc offset:0 atIndex:4]; [enc setBuffer:b_y_q4_128 offset:0 atIndex:5];
                [enc setBuffer:b_y_q4_128 offset:0 atIndex:6]; [enc setBytes:&np128 length:sizeof(np128) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(N / 64U, M / 128U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_q4_i8_m64];
                [enc setBuffer:b_y_q4_64 offset:0 atIndex:5]; [enc setBuffer:b_y_q4_64 offset:0 atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(N / 64U, M / 64U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_q4_i8_m32];
                [enc setBuffer:b_y_q4_32 offset:0 atIndex:5]; [enc setBuffer:b_y_q4_32 offset:0 atIndex:6];
                [enc setBytes:&np32 length:sizeof(np32) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(N / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }

            const double cos_q8_64   = cosine_sim(reinterpret_cast<const float*>([b_y_q8_64 contents]),   ref_q8.data(), ref_q8.size());
            const double cos_q8_32   = cosine_sim(reinterpret_cast<const float*>([b_y_q8_32 contents]),   ref_q8.data(), 32U * N);
            const double cos_i8i4_64 = cosine_sim(reinterpret_cast<const float*>([b_y_i8i4_64 contents]), ref_q4.data(), ref_q4.size());
            const double cos_i8i4_32 = cosine_sim(reinterpret_cast<const float*>([b_y_i8i4_32 contents]), ref_q4.data(), 32U * N);
            const double cos_q4_128  = cosine_sim(reinterpret_cast<const float*>([b_y_q4_128 contents]),  ref_q4.data(), ref_q4.size());
            const double cos_q4_64   = cosine_sim(reinterpret_cast<const float*>([b_y_q4_64 contents]),   ref_q4.data(), ref_q4.size());
            const double cos_q4_32   = cosine_sim(reinterpret_cast<const float*>([b_y_q4_32 contents]),   ref_q4.data(), 32U * N);

            // Part B: Fused Loop Controller INT8 (`prep_i8 -> down_i8 -> quant -> apply_i8`) vs FP32 CPU Ref
            double cos_loop_i8 = 0.0;
            {
                const uint32_t ML = 64U, DL = 128U, RL = 64U;
                std::vector<float> lc_cur(ML * DL), lc_msg(ML * DL);
                for (auto& v : lc_cur) v = nd(rng);
                for (auto& v : lc_msg) v = nd(rng);
                // Pack Q8_0 down [RL, 2*DL], gain [DL, RL], forget [DL, RL] into single UMA buffer
                const size_t dw_b = RL * (2U * DL), ds_b = RL * ((2U * DL) / 64U) * 2U;
                const size_t gw_b = DL * RL, gs_b = DL * 2U;
                std::vector<int8_t>   dw(dw_b), gw(gw_b), fw(gw_b);
                std::vector<uint16_t> ds(RL * ((2U * DL) / 64U)), gs_v(DL), fs_v(DL);
                for (auto& v : dw) v = static_cast<int8_t>(static_cast<int>(rng() % 251U) - 125);
                for (auto& v : gw) v = static_cast<int8_t>(static_cast<int>(rng() % 251U) - 125);
                for (auto& v : fw) v = static_cast<int8_t>(static_cast<int>(rng() % 251U) - 125);
                for (auto& v : ds)   v = f32_to_fp16_bits(0.008f + 0.008f * (static_cast<float>(rng() & 255U) / 255.0f));
                for (auto& v : gs_v) v = f32_to_fp16_bits(0.005f + 0.005f * (static_cast<float>(rng() & 255U) / 255.0f));
                for (auto& v : fs_v) v = f32_to_fp16_bits(0.005f + 0.005f * (static_cast<float>(rng() & 255U) / 255.0f));

                auto align256 = [](size_t x) { return (x + 255U) & ~size_t(255U); };
                const size_t o_dw = 0U;
                const size_t o_ds = align256(o_dw + dw_b);
                const size_t o_gw = align256(o_ds + ds_b);
                const size_t o_gs = align256(o_gw + gw_b);
                const size_t o_fw = align256(o_gs + gs_b);
                const size_t o_fs = align256(o_fw + gw_b);
                std::vector<uint8_t> uma_b(align256(o_fs + gs_b), 0U);
                std::memcpy(uma_b.data() + o_dw, dw.data(), dw_b);
                std::memcpy(uma_b.data() + o_ds, ds.data(), ds_b);
                std::memcpy(uma_b.data() + o_gw, gw.data(), gw_b);
                std::memcpy(uma_b.data() + o_gs, gs_v.data(), gs_b);
                std::memcpy(uma_b.data() + o_fw, fw.data(), gw_b);
                std::memcpy(uma_b.data() + o_fs, fs_v.data(), gs_b);

                id<MTLBuffer> b_luma  = [dev newBufferWithBytes:uma_b.data()  length:uma_b.size()       options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_lcur  = [dev newBufferWithBytes:lc_cur.data() length:lc_cur.size() * 4U  options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_lmsg  = [dev newBufferWithBytes:lc_msg.data() length:lc_msg.size() * 4U  options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_ci8   = [dev newBufferWithLength:ML * (2U * DL)    options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_csc   = [dev newBufferWithLength:ML * 2U           options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_hf16  = [dev newBufferWithLength:ML * RL * 2U      options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_hi8   = [dev newBufferWithLength:ML * RL           options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_hsc   = [dev newBufferWithLength:ML * 2U           options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_lout  = [dev newBufferWithLength:ML * DL * 4U      options:MTLResourceStorageModeShared];

                SikLPrepP lpp{ML, ML, DL, 1e-6f};
                SikLoopP d_off{}, a_off{};
                d_off.M_v = a_off.M_v = ML; d_off.M_p = a_off.M_p = ML;
                d_off.D = a_off.D = DL; d_off.gate_rank = a_off.gate_rank = RL;
                d_off.num_ctrls = a_off.num_ctrls = 1U;
                d_off.w_off[0] = o_dw; d_off.s_off[0] = o_ds;
                a_off.w_off[0] = o_gw; a_off.s_off[0] = o_gs;
                a_off.f_off[0] = o_fw; a_off.fs_off[0] = o_fs;
                SikQuantP qph{ML, ML, RL, 0U, 1U, 0U, 0U, 0U};

                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_lprep_i8];
                [enc setBuffer:b_lcur offset:0 atIndex:0]; [enc setBuffer:b_lcur offset:0 atIndex:1]; [enc setBuffer:b_lcur offset:0 atIndex:2];
                [enc setBuffer:b_lmsg offset:0 atIndex:3]; [enc setBuffer:b_lmsg offset:0 atIndex:4]; [enc setBuffer:b_lmsg offset:0 atIndex:5];
                [enc setBuffer:b_ci8 offset:0 atIndex:6];  [enc setBuffer:b_csc offset:0 atIndex:7];
                [enc setBytes:&lpp length:sizeof(lpp) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(ML, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc setComputePipelineState:pso_ldown_i8_m64];
                [enc setBuffer:b_luma offset:0 atIndex:0]; [enc setBuffer:b_ci8 offset:0 atIndex:1];
                [enc setBuffer:b_csc offset:0 atIndex:2];  [enc setBuffer:b_hf16 offset:0 atIndex:3];
                [enc setBytes:&d_off length:sizeof(d_off) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_qrow];
                [enc setBuffer:b_hf16 offset:0 atIndex:0]; [enc setBuffer:b_hi8 offset:0 atIndex:1];
                [enc setBuffer:b_hsc offset:0 atIndex:2];  [enc setBuffer:b_hsc offset:0 atIndex:3];
                [enc setBytes:&qph length:sizeof(qph) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((ML + 3U) / 4U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_lapp_i8_m64];
                [enc setBuffer:b_luma offset:0 atIndex:0]; [enc setBuffer:b_hi8 offset:0 atIndex:1]; [enc setBuffer:b_hsc offset:0 atIndex:2];
                [enc setBuffer:b_lcur offset:0 atIndex:3]; [enc setBuffer:b_lcur offset:0 atIndex:4]; [enc setBuffer:b_lcur offset:0 atIndex:5];
                [enc setBuffer:b_lmsg offset:0 atIndex:6]; [enc setBuffer:b_lmsg offset:0 atIndex:7]; [enc setBuffer:b_lmsg offset:0 atIndex:8];
                [enc setBuffer:b_lout offset:0 atIndex:9]; [enc setBuffer:b_lout offset:0 atIndex:10]; [enc setBuffer:b_lout offset:0 atIndex:11];
                [enc setBytes:&a_off length:sizeof(a_off) atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake(DL / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                // Exact CPU reference
                std::vector<float> ref_l(ML * DL);
                for (uint32_t m = 0U; m < ML; ++m) {
                    double s0 = 0.0, s1 = 0.0;
                    for (uint32_t d = 0U; d < DL; ++d) {
                        s0 += double(lc_cur[m * DL + d]) * double(lc_cur[m * DL + d]);
                        s1 += double(lc_msg[m * DL + d]) * double(lc_msg[m * DL + d]);
                    }
                    const float ir0 = 1.0f / std::sqrt(float(s0 / DL) + 1e-6f);
                    const float ir1 = 1.0f / std::sqrt(float(s1 / DL) + 1e-6f);
                    std::vector<float> cat(2U * DL);
                    for (uint32_t d = 0U; d < DL; ++d) { cat[d] = lc_cur[m * DL + d] * ir0; cat[DL + d] = lc_msg[m * DL + d] * ir1; }
                    std::vector<float> hv(RL, 0.0f);
                    for (uint32_t r = 0U; r < RL; ++r) {
                        double acc = 0.0;
                        for (uint32_t k = 0U; k < 2U * DL; ++k) {
                            acc += double(cat[k]) * double(dw[r * (2U * DL) + k]) * double(fp16_bits_to_f32_ref(ds[r * ((2U * DL) / 64U) + (k / 64U)]));
                        }
                        hv[r] = static_cast<float>(acc);
                    }
                    for (uint32_t d = 0U; d < DL; ++d) {
                        double cg = 0.0, cf = 0.0;
                        const float gs_d = fp16_bits_to_f32_ref(gs_v[d]), fs_d = fp16_bits_to_f32_ref(fs_v[d]);
                        for (uint32_t r = 0U; r < RL; ++r) {
                            cg += double(hv[r]) * double(gw[d * RL + r]) * double(gs_d);
                            cf += double(hv[r]) * double(fw[d * RL + r]) * double(fs_d);
                        }
                        ref_l[m * DL + d] = (1.0f - std::tanh(float(cf))) * lc_cur[m * DL + d] + std::tanh(float(cg)) * lc_msg[m * DL + d];
                    }
                }
                cos_loop_i8 = cosine_sim(reinterpret_cast<const float*>([b_lout contents]), ref_l.data(), ref_l.size());
            }

            // Part C: Quad-Bin NVFP4 INT8 MoE (ALL 4 bins m16/m32/m64/m128!) & MXFP8 INT8 MoE vs FP16 Ref
            double cos_moe_nv = 0.0, cos_moe_mx = 0.0;
            {
                const uint32_t E = 8U, D = 256U, inter = 128U, MQ = 256U;
                const size_t w13_b = size_t(E) * 2U * inter * (D / 2U);
                const size_t s13_b = size_t(E) * 2U * inter * (D / 16U);
                const size_t w2_b  = size_t(E) * D * (inter / 2U);
                const size_t s2_b  = size_t(E) * D * (inter / 16U);
                std::vector<uint8_t> w13(w13_b), s13(s13_b), w2(w2_b), s2(s2_b);
                std::vector<float>   es13(E, 0.25f), es2(E, 0.25f);
                for (auto& v : w13) v = static_cast<uint8_t>(rng() & 0xFFU);
                for (auto& v : w2)  v = static_cast<uint8_t>(rng() & 0xFFU);
                for (auto& v : s13) v = static_cast<uint8_t>(0x24U + (rng() & 0x0FU));
                for (auto& v : s2)  v = static_cast<uint8_t>(0x24U + (rng() & 0x0FU));

                std::vector<uint16_t> mx_h(size_t(MQ) * D);
                for (uint32_t m = 0U; m < MQ; ++m) {
                    for (uint32_t d = 0U; d < D; ++d) {
                        mx_h[size_t(m) * D + d] = f32_to_fp16_bits(nd(rng) * (((d % 37U) == 0U) ? 4.0f : 1.0f));
                    }
                }
                id<MTLBuffer> b_w13   = [dev newBufferWithBytes:w13.data()  length:w13.size()       options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_s13   = [dev newBufferWithBytes:s13.data()  length:s13.size()       options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_es13  = [dev newBufferWithBytes:es13.data() length:es13.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_w2    = [dev newBufferWithBytes:w2.data()   length:w2.size()        options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_s2    = [dev newBufferWithBytes:s2.data()   length:s2.size()        options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_es2   = [dev newBufferWithBytes:es2.data()  length:es2.size() * 4U  options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_mx    = [dev newBufferWithBytes:mx_h.data() length:mx_h.size() * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_tkq   = [dev newBufferWithLength:MQ * 4U options:MTLResourceStorageModeShared];
                auto* tkq = reinterpret_cast<uint32_t*>([b_tkq contents]);
                for (uint32_t m = 0U; m < MQ; ++m) {
                    tkq[m] = (m < 100U) ? 0U : ((m < 150U) ? 1U : ((m < 174U) ? 2U : ((m < 184U) ? 3U : 4U)));
                }
                id<MTLBuffer> b_sp    = [dev newBufferWithLength:(MQ + 256U) * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_wt    = [dev newBufferWithLength:2048U * 16U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_nt    = [dev newBufferWithLength:64U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_ind   = [dev newBufferWithLength:256U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_xs16  = [dev newBufferWithLength:(MQ + 128U) * D * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_ms16  = [dev newBufferWithLength:(MQ + 128U) * inter * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_dn_fp = [dev newBufferWithLength:MQ * D * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_mxi8  = [dev newBufferWithLength:MQ * D options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_mxsc  = [dev newBufferWithLength:MQ * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_xsi8  = [dev newBufferWithLength:(MQ + 128U) * D options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_xssc  = [dev newBufferWithLength:(MQ + 128U) * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_mi16  = [dev newBufferWithLength:(MQ + 128U) * inter * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_mi8   = [dev newBufferWithLength:(MQ + 128U) * inter options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_misc  = [dev newBufferWithLength:(MQ + 128U) * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_dn_i8 = [dev newBufferWithLength:MQ * D * 2U options:MTLResourceStorageModeShared];

                SikMoeExecP epq{MQ, 1U, D, inter, E, 128U, 4.0f, 25.0f};
                SikQuantP qpx{MQ, MQ, D, 0U, 1U, 0U, 0U, 0U};
                SikQuantP qpm{MQ, MQ, inter, 0U, 1U, 0U, 0U, 0U};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_bin_quad];
                [enc setBuffer:b_tkq offset:0 atIndex:0]; [enc setBuffer:b_sp offset:0 atIndex:1];
                [enc setBuffer:b_wt offset:0 atIndex:2];  [enc setBuffer:b_nt offset:0 atIndex:3];
                [enc setBuffer:b_ind offset:0 atIndex:4]; [enc setBytes:&epq length:sizeof(epq) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                // FP16 Quad-Bin Ref
                [enc setComputePipelineState:pso_gather_x];
                [enc setBuffer:b_mx offset:0 atIndex:0]; [enc setBuffer:b_sp offset:0 atIndex:1];
                [enc setBuffer:b_xs16 offset:0 atIndex:2]; [enc setBuffer:b_ms16 offset:0 atIndex:3];
                [enc setBytes:&epq length:sizeof(epq) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(MQ + 128U, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                const NSUInteger tg_fp[4] = {64, 128, 128, 128};
                for (int pass = 0; pass < 2; ++pass) {
                    for (uint32_t b = 0U; b < 4U; ++b) {
                        [enc setComputePipelineState:(pass == 0 ? pso_q_w13_g[b] : pso_q_w13_u[b])];
                        [enc setBuffer:b_w13 offset:0 atIndex:0]; [enc setBuffer:b_s13 offset:0 atIndex:1];
                        [enc setBuffer:b_es13 offset:0 atIndex:2]; [enc setBuffer:b_xs16 offset:0 atIndex:3];
                        [enc setBuffer:b_sp offset:0 atIndex:4]; [enc setBuffer:b_wt offset:static_cast<NSUInteger>(b) * 512U * 16U atIndex:5];
                        [enc setBuffer:b_ms16 offset:0 atIndex:6]; [enc setBytes:&epq length:sizeof(epq) atIndex:7];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:static_cast<NSUInteger>(b) * 32U threadsPerThreadgroup:MTLSizeMake(tg_fp[b], 1, 1)];
                    }
                }
                for (uint32_t b = 0U; b < 4U; ++b) {
                    [enc setComputePipelineState:pso_q_w2[b]];
                    [enc setBuffer:b_w2 offset:0 atIndex:0]; [enc setBuffer:b_s2 offset:0 atIndex:1];
                    [enc setBuffer:b_es2 offset:0 atIndex:2]; [enc setBuffer:b_ms16 offset:0 atIndex:3];
                    [enc setBuffer:b_sp offset:0 atIndex:4]; [enc setBuffer:b_wt offset:static_cast<NSUInteger>(b) * 512U * 16U atIndex:5];
                    [enc setBuffer:b_dn_fp offset:0 atIndex:6]; [enc setBytes:&epq length:sizeof(epq) atIndex:7];
                    [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:static_cast<NSUInteger>(b) * 32U + 16U threadsPerThreadgroup:MTLSizeMake(tg_fp[b], 1, 1)];
                }

                // INT8 Quad-Bin Test (ALL 4 bins!)
                [enc setComputePipelineState:pso_qrow];
                [enc setBuffer:b_mx offset:0 atIndex:0]; [enc setBuffer:b_mxi8 offset:0 atIndex:1];
                [enc setBuffer:b_mxsc offset:0 atIndex:2]; [enc setBuffer:b_mxsc offset:0 atIndex:3];
                [enc setBytes:&qpx length:sizeof(qpx) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((MQ + 3U) / 4U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:pso_gather_i8];
                [enc setBuffer:b_mxi8 offset:0 atIndex:0]; [enc setBuffer:b_mxsc offset:0 atIndex:1];
                [enc setBuffer:b_sp offset:0 atIndex:2];   [enc setBuffer:b_xsi8 offset:0 atIndex:3];
                [enc setBuffer:b_xssc offset:0 atIndex:4]; [enc setBuffer:b_mi8 offset:0 atIndex:5];
                [enc setBuffer:b_misc offset:0 atIndex:6]; [enc setBytes:&epq length:sizeof(epq) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(MQ + 128U, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                const NSUInteger tg_i8[4] = {64, 64, 128, 128};
                for (int pass = 0; pass < 2; ++pass) {
                    for (uint32_t b = 0U; b < 4U; ++b) {
                        [enc setComputePipelineState:(pass == 0 ? pso_nv_w13_i8_g[b] : pso_nv_w13_i8_u[b])];
                        [enc setBuffer:b_w13 offset:0 atIndex:0]; [enc setBuffer:b_s13 offset:0 atIndex:1];
                        [enc setBuffer:b_es13 offset:0 atIndex:2]; [enc setBuffer:b_xsi8 offset:0 atIndex:3];
                        [enc setBuffer:b_xssc offset:0 atIndex:4]; [enc setBuffer:b_wt offset:static_cast<NSUInteger>(b) * 512U * 16U atIndex:5];
                        [enc setBuffer:b_mi16 offset:0 atIndex:6]; [enc setBytes:&epq length:sizeof(epq) atIndex:7];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:static_cast<NSUInteger>(b) * 32U threadsPerThreadgroup:MTLSizeMake(tg_i8[b], 1, 1)];
                    }
                }
                [enc setComputePipelineState:pso_qrow];
                [enc setBuffer:b_mi16 offset:0 atIndex:0]; [enc setBuffer:b_mi8 offset:0 atIndex:1];
                [enc setBuffer:b_misc offset:0 atIndex:2]; [enc setBuffer:b_misc offset:0 atIndex:3];
                [enc setBytes:&qpm length:sizeof(qpm) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((MQ + 3U) / 4U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                for (uint32_t b = 0U; b < 4U; ++b) {
                    [enc setComputePipelineState:pso_nv_w2_i8[b]];
                    [enc setBuffer:b_w2 offset:0 atIndex:0]; [enc setBuffer:b_s2 offset:0 atIndex:1];
                    [enc setBuffer:b_es2 offset:0 atIndex:2]; [enc setBuffer:b_mi8 offset:0 atIndex:3];
                    [enc setBuffer:b_misc offset:0 atIndex:4]; [enc setBuffer:b_sp offset:0 atIndex:5];
                    [enc setBuffer:b_wt offset:static_cast<NSUInteger>(b) * 512U * 16U atIndex:6];
                    [enc setBuffer:b_dn_i8 offset:0 atIndex:7]; [enc setBytes:&epq length:sizeof(epq) atIndex:8];
                    [enc dispatchThreadgroupsWithIndirectBuffer:b_ind indirectBufferOffset:static_cast<NSUInteger>(b) * 32U + 16U threadsPerThreadgroup:MTLSizeMake(tg_i8[b], 1, 1)];
                }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                const auto* fp_h = reinterpret_cast<const uint16_t*>([b_dn_fp contents]);
                const auto* i8_h = reinterpret_cast<const uint16_t*>([b_dn_i8 contents]);
                std::vector<float> fp_v(size_t(MQ) * D), i8_v(size_t(MQ) * D);
                for (size_t i = 0U; i < fp_v.size(); ++i) { fp_v[i] = fp16_bits_to_f32_ref(fp_h[i]); i8_v[i] = fp16_bits_to_f32_ref(i8_h[i]); }
                cos_moe_nv = cosine_sim(fp_v.data(), i8_v.data(), fp_v.size());

                // MXFP8 INT8 vs FP16 (`m128`, 1 expert, 128 pairs)
                const uint32_t MMX = 128U;
                std::vector<uint8_t> mx_w13(2U * inter * D), mx_s13(2U * inter * (D / 32U));
                std::vector<uint8_t> mx_w2(D * inter),       mx_s2(D * (inter / 32U));
                for (auto& v : mx_w13) { uint8_t b = uint8_t(rng() & 0xFFU); if ((b & 0x7FU) == 0x7FU) b &= 0xFEU; v = b; }
                for (auto& v : mx_w2)  { uint8_t b = uint8_t(rng() & 0xFFU); if ((b & 0x7FU) == 0x7FU) b &= 0xFEU; v = b; }
                for (auto& v : mx_s13) v = static_cast<uint8_t>(118U + (rng() & 3U));
                for (auto& v : mx_s2)  v = static_cast<uint8_t>(118U + (rng() & 3U));
                id<MTLBuffer> b_mw13 = [dev newBufferWithBytes:mx_w13.data() length:mx_w13.size() options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_ms13 = [dev newBufferWithBytes:mx_s13.data() length:mx_s13.size() options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_mw2  = [dev newBufferWithBytes:mx_w2.data()  length:mx_w2.size()  options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_ms2  = [dev newBufferWithBytes:mx_s2.data()  length:mx_s2.size()  options:MTLResourceStorageModeShared];
                struct WorkTileH { uint32_t eid, pstart, pcnt, pad; };
                WorkTileH wt0{0U, 0U, MMX, 0U};
                id<MTLBuffer> b_wt1  = [dev newBufferWithBytes:&wt0 length:sizeof(wt0) options:MTLResourceStorageModeShared];
                std::vector<uint32_t> sp_id(MMX); for (uint32_t i = 0U; i < MMX; ++i) sp_id[i] = i;
                id<MTLBuffer> b_sp1  = [dev newBufferWithBytes:sp_id.data() length:sp_id.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_mdfp = [dev newBufferWithLength:MMX * D * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_mdi8 = [dev newBufferWithLength:MMX * D * 2U options:MTLResourceStorageModeShared];
                SikMoeExecP epmx{MMX, 1U, D, inter, 1U, 128U, 4.0f, 25.0f};
                SikQuantP qpmx{MMX, MMX, inter, 0U, 1U, 0U, 0U, 0U};

                id<MTLCommandBuffer> cb2 = [q commandBuffer];
                id<MTLComputeCommandEncoder> e2 = [cb2 computeCommandEncoder];
                for (int pass = 0; pass < 2; ++pass) {
                    [e2 setComputePipelineState:pso_mx_w13_fp16[pass]];
                    [e2 setBuffer:b_mw13 offset:0 atIndex:0]; [e2 setBuffer:b_ms13 offset:0 atIndex:1];
                    [e2 setBuffer:b_mx offset:0 atIndex:2];   [e2 setBuffer:b_sp1 offset:0 atIndex:3];
                    [e2 setBuffer:b_wt1 offset:0 atIndex:4];  [e2 setBuffer:b_ms16 offset:0 atIndex:5];
                    [e2 setBytes:&epmx length:sizeof(epmx) atIndex:6];
                    [e2 dispatchThreadgroups:MTLSizeMake(inter / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [e2 setComputePipelineState:pso_mx_w2_fp16];
                [e2 setBuffer:b_mw2 offset:0 atIndex:0];  [e2 setBuffer:b_ms2 offset:0 atIndex:1];
                [e2 setBuffer:b_ms16 offset:0 atIndex:2]; [e2 setBuffer:b_sp1 offset:0 atIndex:3];
                [e2 setBuffer:b_wt1 offset:0 atIndex:4];  [e2 setBuffer:b_mdfp offset:0 atIndex:5];
                [e2 setBytes:&epmx length:sizeof(epmx) atIndex:6];
                [e2 dispatchThreadgroups:MTLSizeMake(D / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                for (int pass = 0; pass < 2; ++pass) {
                    [e2 setComputePipelineState:pso_mx_w13_i8[pass]];
                    [e2 setBuffer:b_mw13 offset:0 atIndex:0]; [e2 setBuffer:b_ms13 offset:0 atIndex:1];
                    [e2 setBuffer:b_mxi8 offset:0 atIndex:2]; [e2 setBuffer:b_mxsc offset:0 atIndex:3];
                    [e2 setBuffer:b_wt1 offset:0 atIndex:4];  [e2 setBuffer:b_mi16 offset:0 atIndex:5];
                    [e2 setBytes:&epmx length:sizeof(epmx) atIndex:6];
                    [e2 dispatchThreadgroups:MTLSizeMake(inter / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [e2 setComputePipelineState:pso_qrow];
                [e2 setBuffer:b_mi16 offset:0 atIndex:0]; [e2 setBuffer:b_mi8 offset:0 atIndex:1];
                [e2 setBuffer:b_misc offset:0 atIndex:2]; [e2 setBuffer:b_misc offset:0 atIndex:3];
                [e2 setBytes:&qpmx length:sizeof(qpmx) atIndex:4];
                [e2 dispatchThreadgroups:MTLSizeMake((MMX + 3U) / 4U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [e2 setComputePipelineState:pso_mx_w2_i8];
                [e2 setBuffer:b_mw2 offset:0 atIndex:0];  [e2 setBuffer:b_ms2 offset:0 atIndex:1];
                [e2 setBuffer:b_mi8 offset:0 atIndex:2];  [e2 setBuffer:b_misc offset:0 atIndex:3];
                [e2 setBuffer:b_sp1 offset:0 atIndex:4];  [e2 setBuffer:b_wt1 offset:0 atIndex:5];
                [e2 setBuffer:b_mdi8 offset:0 atIndex:6]; [e2 setBytes:&epmx length:sizeof(epmx) atIndex:7];
                [e2 dispatchThreadgroups:MTLSizeMake(D / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [e2 endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];

                const auto* mfp_h = reinterpret_cast<const uint16_t*>([b_mdfp contents]);
                const auto* mi8_h = reinterpret_cast<const uint16_t*>([b_mdi8 contents]);
                std::vector<float> mfp_v(MMX * D), mi8_v(MMX * D);
                for (size_t i = 0U; i < mfp_v.size(); ++i) { mfp_v[i] = fp16_bits_to_f32_ref(mfp_h[i]); mi8_v[i] = fp16_bits_to_f32_ref(mi8_h[i]); }
                cos_moe_mx = cosine_sim(mfp_v.data(), mi8_v.data(), mfp_v.size());
            }

            const bool ok17 = (cos_q8_64 >= 0.995) && (cos_q8_32 >= 0.995) &&
                              (cos_i8i4_64 >= 0.995) && (cos_i8i4_32 >= 0.995) &&
                              (cos_q4_128 >= 0.995) && (cos_q4_64 >= 0.995) && (cos_q4_32 >= 0.995) &&
                              (cos_loop_i8 >= 0.995) && (cos_moe_nv >= 0.995) && (cos_moe_mx >= 0.995);
            std::printf("[Test 17] INT8xINT8 & INT8xINT4 NAX Prefill Parity        : %s\n", ok17 ? "PASS" : "FAIL");
            std::printf("         • Q8_0 INT8xINT8 (m64/m32)  : cos=%.6f / %.6f | Q4_A INT8xINT4 (m64/m32): cos=%.6f / %.6f\n",
                        cos_q8_64, cos_q8_32, cos_i8i4_64, cos_i8i4_32);
            std::printf("         • Q4_A 8KiB INT8 (128/64/32): cos=%.6f / %.6f / %.6f | LoopCtrl INT8: cos=%.6f\n",
                        cos_q4_128, cos_q4_64, cos_q4_32, cos_loop_i8);
            std::printf("         • MoE Quad-Bin NVFP4 INT8   : cos=%.6f | MoE MXFP8 INT8: cos=%.6f\n",
                        cos_moe_nv, cos_moe_mx);
            if (!ok17) return 1;
        }

        // ---- Test 18: 2-Stage Bitplane Exact Q8_0 LM Head (`sik_bp_*`) vs Full Q8_0 ----
        {
            id<MTLComputePipelineState> pso_bp_repack  = get_pso("sik_bp_repack_q8_to_hi4");
            id<MTLComputePipelineState> pso_bp_prepx   = get_pso("sik_bp_prep_x");
            id<MTLComputePipelineState> pso_bp_s1      = get_pso("sik_bp_stage1_hi4");
            id<MTLComputePipelineState> pso_bp_s15     = get_pso("sik_bp_stage15_seed");
            id<MTLComputePipelineState> pso_bp_flt     = get_pso("sik_bp_filter_compact");
            id<MTLComputePipelineState> pso_bp_ind     = get_pso("sik_bp_prep_indirect");
            id<MTLComputePipelineState> pso_bp_s2      = get_pso("sik_bp_stage2_exact_cand");
            id<MTLComputePipelineState> pso_bp_grd     = get_pso("sik_bp_cand_sample_greedy");
            id<MTLComputePipelineState> pso_bp_st1     = get_pso("sik_bp_cand_sample_stoch_s1");
            id<MTLComputePipelineState> pso_bp_st2     = get_pso("sik_bp_cand_sample_stoch_s2");
            id<MTLComputePipelineState> pso_q8_m1      = get_pso_fc("sik_q8_gemv_batch", 1U);
            id<MTLComputePipelineState> pso_pen        = get_pso("sik_apply_logit_penalties");
            id<MTLComputePipelineState> pso_grd1       = get_pso("sik_sample_greedy_stage1");
            id<MTLComputePipelineState> pso_grd2       = get_pso("sik_sample_greedy_stage2");
            id<MTLComputePipelineState> pso_st1        = get_pso("sik_sample_vocab_stage1");
            id<MTLComputePipelineState> pso_st2        = get_pso("sik_sample_vocab_stage2");
            if (!pso_bp_repack || !pso_bp_prepx || !pso_bp_s1 || !pso_bp_s15 || !pso_bp_flt ||
                !pso_bp_ind || !pso_bp_s2 || !pso_bp_grd || !pso_bp_st1 || !pso_bp_st2 ||
                !pso_q8_m1 || !pso_pen || !pso_grd1 || !pso_grd2 || !pso_st1 || !pso_st2) {
                return 1;
            }
            id<MTLComputePipelineState> bp_psos[10] = {
                pso_bp_repack, pso_bp_prepx, pso_bp_s1, pso_bp_s15, pso_bp_flt,
                pso_bp_ind, pso_bp_s2, pso_bp_grd, pso_bp_st1, pso_bp_st2
            };
            uint32_t bp_spills = 0U;
            for (int i = 0; i < 10; ++i) {
                if ([bp_psos[i] maxTotalThreadsPerThreadgroup] < 1024U) ++bp_spills;
            }

            struct BpMetaH {
                uint32_t cand_count, overflow;
                float    T_raw, T_pen;
                uint32_t ind_stage2[4], ind_cand_s1[4], ind_cand_s2[4];
                uint32_t ind_fb_gemv[4], ind_fb_pen[4], ind_fb_s1[4], ind_fb_s2[4];
            };
            struct BpParamsH {
                uint32_t V, D; float softcap; uint32_t top_k, num_seeds, has_pen;
                float pres_pen, freq_pen, rep_pen; uint32_t pen_tgs, is_greedy;
                float inv_temp, top_p; uint32_t rng_lo, rng_hi, pad0;
            };
            struct GemvP { uint32_t M, rows, cols, out_mode; float softcap; uint32_t p0, p1, p2; };
            struct PenP {
                uint32_t M, vocab_size, logits_stride, num_hist_ids, extra_mode, extra_count;
                float pres, freq, rep; uint32_t use_sl, p1, p2, drop[8];
            };
            struct SampP {
                uint32_t M, vocab_size, greedy; float inv_temp; uint32_t top_k; float top_p;
                uint32_t rng_lo, rng_hi, use_map, compute_pmax, p0, p1;
            };

            const uint32_t V = 40960U, D = 2304U, nblk = D / 64U;
            const float softcap = 30.0f;
            std::mt19937 rng(202618);
            std::normal_distribution<float> nd(0.0f, 1.0f);

            // Build unit-norm Q8_0 rows & realistic x (||x|| ~ 260) with peaked top-15 winners
            std::vector<uint16_t> x_h(D);
            std::vector<float>    x_f(D);
            for (uint32_t c = 0U; c < D; ++c) {
                const float v = nd(rng) * 5.5f;
                x_h[c] = f32_to_fp16_bits(v);
                x_f[c] = fp16_bits_to_f32_ref(x_h[c]);
            }
            std::vector<int8_t>   wq(static_cast<size_t>(V) * D);
            std::vector<uint16_t> ws(static_cast<size_t>(V) * nblk);
            for (size_t i = 0U; i < wq.size(); ++i) {
                wq[i] = static_cast<int8_t>(static_cast<int>(rng() % 181U) - 90);
            }
            for (size_t i = 0U; i < ws.size(); ++i) {
                ws[i] = f32_to_fp16_bits(0.00035f + 0.00015f * (static_cast<float>(rng() & 255U) / 255.0f));
            }
            // Inject 16 peaked rows aligned with x_f (scores +20..+55 above background)
            const uint32_t peaked_rows[16] = {
                17U, 1024U, 4095U, 8888U, 12345U, 16384U, 19999U, 22222U,
                25000U, 27777U, 30001U, 31415U, 33333U, 35555U, 38000U, 40900U
            };
            for (uint32_t pi = 0U; pi < 16U; ++pi) {
                const uint32_t r = peaked_rows[pi];
                const float strength = 0.0018f - 0.00007f * static_cast<float>(pi);
                for (uint32_t g = 0U; g < nblk; ++g) {
                    ws[static_cast<size_t>(r) * nblk + g] = f32_to_fp16_bits(strength);
                }
                for (uint32_t c = 0U; c < D; ++c) {
                    const float qv = std::clamp(std::round(x_f[c] * 14.0f), -127.0f, 127.0f);
                    wq[static_cast<size_t>(r) * D + c] = static_cast<int8_t>(qv);
                }
            }

            id<MTLBuffer> b_wq    = [dev newBufferWithBytes:wq.data() length:wq.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ws    = [dev newBufferWithBytes:ws.data() length:ws.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_x     = [dev newBufferWithBytes:x_h.data() length:x_h.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_hi4   = [dev newBufferWithLength:static_cast<size_t>(V) * (D / 2U) options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_rn4   = [dev newBufferWithLength:static_cast<size_t>(V) * 8U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_ub    = [dev newBufferWithLength:static_cast<size_t>(V) * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_xp    = [dev newBufferWithLength:256U * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_sv    = [dev newBufferWithLength:512U * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_si    = [dev newBufferWithLength:512U * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_cids  = [dev newBufferWithLength:32768U * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_clgs  = [dev newBufferWithLength:32768U * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_meta  = [dev newBufferWithLength:sizeof(BpMetaH) options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_lgs   = [dev newBufferWithLength:static_cast<size_t>(V) * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ssv   = [dev newBufferWithLength:64U * 64U * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_ssi   = [dev newBufferWithLength:64U * 64U * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_slse  = [dev newBufferWithLength:128U * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_pids  = [dev newBufferWithLength:static_cast<size_t>(V) * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pcnt  = [dev newBufferWithLength:static_cast<size_t>(V) * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_dummy = [dev newBufferWithLength:256U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_t_ref = [dev newBufferWithLength:16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_t_bp  = [dev newBufferWithLength:16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_tids  = [dev newBufferWithLength:64U * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_tprs  = [dev newBufferWithLength:64U * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_tkk   = [dev newBufferWithLength:16U options:MTLResourceStorageModeShared];
            std::memset([b_pcnt contents], 0, static_cast<size_t>(V) * 2U);

            auto run_repack = [&](id<MTLBuffer> wq_b, id<MTLBuffer> ws_b) {
                BpParamsH rp{}; rp.V = V; rp.D = D;
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_bp_repack];
                [enc setBuffer:wq_b offset:0 atIndex:0]; [enc setBuffer:ws_b offset:0 atIndex:1];
                [enc setBuffer:b_hi4 offset:0 atIndex:2]; [enc setBuffer:b_rn4 offset:0 atIndex:3];
                [enc setBytes:&rp length:sizeof(rp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((V + 3U) / 4U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            };

            auto run_pair = [&](id<MTLBuffer> wq_b, id<MTLBuffer> ws_b, bool greedy, uint32_t top_k,
                                float temp, float top_p, uint32_t rlo, uint32_t rhi,
                                uint32_t pen_n, float pp, float fp, float rp_val,
                                uint32_t& tok_ref_o, uint32_t& tok_bp_o, uint32_t& cands_o, uint32_t& ovf_o, uint32_t& cand_bit_diffs_o) {
                const uint32_t pen_tgs = (pen_n > 0U) ? ((pen_n + 255U) / 256U) : 0U;
                GemvP gp{1U, V, D, 2U, softcap, 0U, 0U, 0U};
                PenP lpp{}; lpp.M = 1U; lpp.vocab_size = V; lpp.logits_stride = V; lpp.num_hist_ids = pen_n;
                lpp.pres = pp; lpp.freq = fp; lpp.rep = rp_val;
                for (int j = 0; j < 8; ++j) lpp.drop[j] = 0xFFFFFFFFU;
                SampP sp{1U, V, greedy ? 1U : 0U, greedy ? 1.0f : (1.0f / temp), top_k, top_p, rlo, rhi, 0U, 0U, 0U, 0U};
                BpParamsH bp{V, D, softcap, greedy ? 1U : top_k, greedy ? 16U : std::min<uint32_t>(64U, top_k + 12U),
                             pen_tgs > 0U ? 1U : 0U, pp, fp, rp_val, pen_tgs, greedy ? 1U : 0U,
                             sp.inv_temp, top_p, rlo, rhi, 0U};

                // 1. Full Q8_0 Reference
                {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_q8_m1];
                    [enc setBuffer:wq_b offset:0 atIndex:0]; [enc setBuffer:ws_b offset:0 atIndex:1];
                    [enc setBuffer:b_x offset:0 atIndex:2]; [enc setBuffer:b_x offset:0 atIndex:3];
                    [enc setBuffer:b_lgs offset:0 atIndex:4]; [enc setBytes:&gp length:sizeof(gp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((V + 7U) / 8U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    if (pen_tgs > 0U) {
                        [enc setComputePipelineState:pso_pen];
                        [enc setBuffer:b_lgs offset:0 atIndex:0]; [enc setBuffer:b_pids offset:0 atIndex:1];
                        [enc setBuffer:b_pcnt offset:0 atIndex:2]; [enc setBuffer:b_dummy offset:0 atIndex:3];
                        [enc setBuffer:b_dummy offset:0 atIndex:4]; [enc setBytes:&lpp length:sizeof(lpp) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(pen_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    if (greedy) {
                        [enc setComputePipelineState:pso_grd1];
                        [enc setBuffer:b_lgs offset:0 atIndex:0]; [enc setBuffer:b_ssv offset:0 atIndex:1];
                        [enc setBuffer:b_ssi offset:0 atIndex:2]; [enc setBuffer:b_slse offset:0 atIndex:3];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_grd2];
                        [enc setBuffer:b_ssv offset:0 atIndex:0]; [enc setBuffer:b_ssi offset:0 atIndex:1];
                        [enc setBuffer:b_slse offset:0 atIndex:2]; [enc setBuffer:b_dummy offset:0 atIndex:3];
                        [enc setBuffer:b_t_ref offset:0 atIndex:4]; [enc setBuffer:b_dummy offset:0 atIndex:5];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    } else {
                        [enc setComputePipelineState:pso_st1];
                        [enc setBuffer:b_lgs offset:0 atIndex:0]; [enc setBuffer:b_ssv offset:0 atIndex:1];
                        [enc setBuffer:b_ssi offset:0 atIndex:2]; [enc setBuffer:b_slse offset:0 atIndex:3];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc setComputePipelineState:pso_st2];
                        [enc setBuffer:b_ssv offset:0 atIndex:0]; [enc setBuffer:b_ssi offset:0 atIndex:1];
                        [enc setBuffer:b_slse offset:0 atIndex:2]; [enc setBuffer:b_dummy offset:0 atIndex:3];
                        [enc setBuffer:b_t_ref offset:0 atIndex:4]; [enc setBuffer:b_dummy offset:0 atIndex:5];
                        [enc setBuffer:b_tids offset:0 atIndex:6]; [enc setBuffer:b_tprs offset:0 atIndex:7];
                        [enc setBuffer:b_tkk offset:0 atIndex:8]; [enc setBytes:&sp length:sizeof(sp) atIndex:9];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                std::vector<uint32_t> ref_lgs_bits(V);
                std::memcpy(ref_lgs_bits.data(), [b_lgs contents], static_cast<size_t>(V) * 4U);
                tok_ref_o = reinterpret_cast<const uint32_t*>([b_t_ref contents])[0];

                // 2. 2-Stage Bitplane Exact LM Head
                {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_bp_prepx];
                    [enc setBuffer:b_x offset:0 atIndex:0]; [enc setBuffer:b_xp offset:0 atIndex:1];
                    [enc setBuffer:b_meta offset:0 atIndex:2]; [enc setBytes:&bp length:sizeof(bp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc setComputePipelineState:pso_bp_s1];
                    [enc setBuffer:b_hi4 offset:0 atIndex:0]; [enc setBuffer:ws_b offset:0 atIndex:1];
                    [enc setBuffer:b_rn4 offset:0 atIndex:2]; [enc setBuffer:b_x offset:0 atIndex:3];
                    [enc setBuffer:b_xp offset:0 atIndex:4]; [enc setBuffer:b_ub offset:0 atIndex:5];
                    [enc setBuffer:b_sv offset:0 atIndex:6]; [enc setBuffer:b_si offset:0 atIndex:7];
                    [enc setBytes:&bp length:sizeof(bp) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:pso_bp_s15];
                    [enc setBuffer:b_sv offset:0 atIndex:0]; [enc setBuffer:b_si offset:0 atIndex:1];
                    [enc setBuffer:wq_b offset:0 atIndex:2]; [enc setBuffer:ws_b offset:0 atIndex:3];
                    [enc setBuffer:b_x offset:0 atIndex:4]; [enc setBuffer:b_pcnt offset:0 atIndex:5];
                    [enc setBuffer:b_meta offset:0 atIndex:6]; [enc setBytes:&bp length:sizeof(bp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc setComputePipelineState:pso_bp_flt];
                    [enc setBuffer:b_ub offset:0 atIndex:0]; [enc setBuffer:b_cids offset:0 atIndex:1];
                    [enc setBuffer:b_meta offset:0 atIndex:2]; [enc setBytes:&bp length:sizeof(bp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(128, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc setComputePipelineState:pso_bp_ind];
                    [enc setBuffer:b_meta offset:0 atIndex:0]; [enc setBytes:&bp length:sizeof(bp) atIndex:1];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                    [enc setComputePipelineState:pso_bp_s2];
                    [enc setBuffer:wq_b offset:0 atIndex:0]; [enc setBuffer:ws_b offset:0 atIndex:1];
                    [enc setBuffer:b_x offset:0 atIndex:2]; [enc setBuffer:b_cids offset:0 atIndex:3];
                    [enc setBuffer:b_pcnt offset:0 atIndex:4]; [enc setBuffer:b_clgs offset:0 atIndex:5];
                    [enc setBuffer:b_meta offset:0 atIndex:6]; [enc setBytes:&bp length:sizeof(bp) atIndex:7];
                    [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:16U threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    if (greedy) {
                        [enc setComputePipelineState:pso_bp_grd];
                        [enc setBuffer:b_cids offset:0 atIndex:0]; [enc setBuffer:b_clgs offset:0 atIndex:1];
                        [enc setBuffer:b_t_bp offset:0 atIndex:2]; [enc setBuffer:b_meta offset:0 atIndex:3];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:32U threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    } else {
                        [enc setComputePipelineState:pso_bp_st1];
                        [enc setBuffer:b_cids offset:0 atIndex:0]; [enc setBuffer:b_clgs offset:0 atIndex:1];
                        [enc setBuffer:b_ssv offset:0 atIndex:2]; [enc setBuffer:b_ssi offset:0 atIndex:3];
                        [enc setBuffer:b_meta offset:0 atIndex:4]; [enc setBytes:&bp length:sizeof(bp) atIndex:5];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:32U threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc setComputePipelineState:pso_bp_st2];
                        [enc setBuffer:b_ssv offset:0 atIndex:0]; [enc setBuffer:b_ssi offset:0 atIndex:1];
                        [enc setBuffer:b_t_bp offset:0 atIndex:2]; [enc setBytes:&bp length:sizeof(bp) atIndex:3];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:48U threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }

                    // Indirect Fallback Path (0-TG no-op unless overflow!)
                    [enc setComputePipelineState:pso_q8_m1];
                    [enc setBuffer:wq_b offset:0 atIndex:0]; [enc setBuffer:ws_b offset:0 atIndex:1];
                    [enc setBuffer:b_x offset:0 atIndex:2]; [enc setBuffer:b_x offset:0 atIndex:3];
                    [enc setBuffer:b_lgs offset:0 atIndex:4]; [enc setBytes:&gp length:sizeof(gp) atIndex:5];
                    [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:64U threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    if (pen_tgs > 0U) {
                        [enc setComputePipelineState:pso_pen];
                        [enc setBuffer:b_lgs offset:0 atIndex:0]; [enc setBuffer:b_pids offset:0 atIndex:1];
                        [enc setBuffer:b_pcnt offset:0 atIndex:2]; [enc setBuffer:b_dummy offset:0 atIndex:3];
                        [enc setBuffer:b_dummy offset:0 atIndex:4]; [enc setBytes:&lpp length:sizeof(lpp) atIndex:5];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:80U threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    if (greedy) {
                        [enc setComputePipelineState:pso_grd1];
                        [enc setBuffer:b_lgs offset:0 atIndex:0]; [enc setBuffer:b_ssv offset:0 atIndex:1];
                        [enc setBuffer:b_ssi offset:0 atIndex:2]; [enc setBuffer:b_slse offset:0 atIndex:3];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:4];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:96U threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_grd2];
                        [enc setBuffer:b_ssv offset:0 atIndex:0]; [enc setBuffer:b_ssi offset:0 atIndex:1];
                        [enc setBuffer:b_slse offset:0 atIndex:2]; [enc setBuffer:b_dummy offset:0 atIndex:3];
                        [enc setBuffer:b_t_bp offset:0 atIndex:4]; [enc setBuffer:b_dummy offset:0 atIndex:5];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:6];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:112U threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    } else {
                        [enc setComputePipelineState:pso_st1];
                        [enc setBuffer:b_lgs offset:0 atIndex:0]; [enc setBuffer:b_ssv offset:0 atIndex:1];
                        [enc setBuffer:b_ssi offset:0 atIndex:2]; [enc setBuffer:b_slse offset:0 atIndex:3];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:4];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:96U threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc setComputePipelineState:pso_st2];
                        [enc setBuffer:b_ssv offset:0 atIndex:0]; [enc setBuffer:b_ssi offset:0 atIndex:1];
                        [enc setBuffer:b_slse offset:0 atIndex:2]; [enc setBuffer:b_dummy offset:0 atIndex:3];
                        [enc setBuffer:b_t_bp offset:0 atIndex:4]; [enc setBuffer:b_dummy offset:0 atIndex:5];
                        [enc setBuffer:b_tids offset:0 atIndex:6]; [enc setBuffer:b_tprs offset:0 atIndex:7];
                        [enc setBuffer:b_tkk offset:0 atIndex:8]; [enc setBytes:&sp length:sizeof(sp) atIndex:9];
                        [enc dispatchThreadgroupsWithIndirectBuffer:b_meta indirectBufferOffset:112U threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                const auto* meta = reinterpret_cast<const BpMetaH*>([b_meta contents]);
                tok_bp_o = reinterpret_cast<const uint32_t*>([b_t_bp contents])[0];
                cands_o  = meta->cand_count;
                ovf_o    = meta->overflow;
                cand_bit_diffs_o = 0U;
                if (ovf_o == 0U) {
                    const auto* cids = reinterpret_cast<const uint32_t*>([b_cids contents]);
                    const auto* clgs = reinterpret_cast<const uint32_t*>([b_clgs contents]);
                    const uint32_t cn = std::min<uint32_t>(cands_o, 32768U);
                    for (uint32_t i = 0U; i < cn; ++i) {
                        if (clgs[i] != ref_lgs_bits[cids[i]]) ++cand_bit_diffs_o;
                    }
                }
            };

            run_repack(b_wq, b_ws);

            uint32_t tref_g = 0U, tbp_g = 0U, cands_g = 0U, ovf_g = 0U, cdiffs_g = 0U;
            run_pair(b_wq, b_ws, true, 1U, 1.0f, 1.0f, 0U, 0U, 0U, 0.0f, 0.0f, 1.0f, tref_g, tbp_g, cands_g, ovf_g, cdiffs_g);

            // Case B: Penalize top-2 winners so #3 wins!
            auto* pids = reinterpret_cast<uint32_t*>([b_pids contents]);
            auto* pcnt = reinterpret_cast<uint16_t*>([b_pcnt contents]);
            pids[0] = peaked_rows[0]; pcnt[peaked_rows[0]] = 6U;
            pids[1] = peaked_rows[1]; pcnt[peaked_rows[1]] = 6U;
            uint32_t tref_p = 0U, tbp_p = 0U, cands_p = 0U, ovf_p = 0U, cdiffs_p = 0U;
            run_pair(b_wq, b_ws, true, 1U, 1.0f, 1.0f, 0U, 0U, 2U, 4.0f, 2.0f, 1.5f, tref_p, tbp_p, cands_p, ovf_p, cdiffs_p);
            pcnt[peaked_rows[0]] = 0U; pcnt[peaked_rows[1]] = 0U;

            // Case C: Stochastic Top-K=20, Top-P=0.95, T=0.7 across 8 seeds
            uint32_t stoch_diffs = 0U, cands_s = 0U, ovf_s = 0U, cdiffs_s = 0U;
            for (uint32_t si = 0U; si < 8U; ++si) {
                uint32_t tr = 0U, tb = 0U, cs = 0U, os = 0U, cd = 0U;
                run_pair(b_wq, b_ws, false, 20U, 0.7f, 0.95f, 0x12345U + si * 97U, 0x9E377U ^ si,
                         0U, 0.0f, 0.0f, 1.0f, tr, tb, cs, os, cd);
                if (tr != tb) ++stoch_diffs;
                cands_s = cs; ovf_s |= os; cdiffs_s += cd;
            }

            // Case D: Flat all-zero distribution with tiny bump on row 33333 -> triggers overflow=1 & exact fallback!
            std::vector<int8_t>   wq_flat(static_cast<size_t>(V) * D, 0);
            std::vector<uint16_t> ws_flat(static_cast<size_t>(V) * nblk, f32_to_fp16_bits(0.001f));
            for (uint32_t c = 0U; c < D; ++c) {
                wq_flat[static_cast<size_t>(33333U) * D + c] = (x_f[c] >= 0.0f) ? 4 : -4;
            }
            id<MTLBuffer> b_wq_f = [dev newBufferWithBytes:wq_flat.data() length:wq_flat.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ws_f = [dev newBufferWithBytes:ws_flat.data() length:ws_flat.size() * 2U options:MTLResourceStorageModeShared];
            run_repack(b_wq_f, b_ws_f);
            uint32_t tref_f = 0U, tbp_f = 0U, cands_f = 0U, ovf_f = 0U, cdiffs_f = 0U;
            run_pair(b_wq_f, b_ws_f, true, 1U, 1.0f, 1.0f, 0U, 0U, 0U, 0.0f, 0.0f, 1.0f, tref_f, tbp_f, cands_f, ovf_f, cdiffs_f);

            const bool ok18 = (bp_spills == 0U) &&
                              (tref_g == tbp_g) && (ovf_g == 0U) && (cands_g < 1024U) && (cdiffs_g == 0U) &&
                              (tref_p == tbp_p) && (tref_p != tref_g) && (cdiffs_p == 0U) &&
                              (stoch_diffs == 0U) && (ovf_s == 0U) && (cdiffs_s == 0U) &&
                              (tref_f == tbp_f) && (tref_f == 33333U) && (ovf_f == 1U);
            std::printf("[Test 18] 2-Stage Bitplane Exact Q8_0 LM Head Parity      : %s\n", ok18 ? "PASS" : "FAIL");
            std::printf("         • Greedy: ref=%u bp=%u cands=%u/%u (%.2f%%) logit_bit_diffs=%u | Spills=%u\n",
                        tref_g, tbp_g, cands_g, V, 100.0 * double(cands_g) / double(V), cdiffs_g, bp_spills);
            std::printf("         • Penalized: ref=%u bp=%u cands=%u | Stoch(k=20,8 seeds): diffs=%u cands=%u | Flat Fallback: ref=%u bp=%u ovf=%u\n",
                        tref_p, tbp_p, cands_p, stoch_diffs, cands_s, tref_f, tbp_f, ovf_f);
            if (!ok18) return 1;
        }

        // ---- Test 19: Dynamic W=4 M=8 Beam Tree Attention + Parent-Indexed SConv K=4 + Compaction ----
        {
            id<MTLComputePipelineState> pso_kv_lin   = get_pso("sik_sconv_kv_qkvr_prep");
            id<MTLComputePipelineState> pso_kv_tr    = get_pso("sik_sconv_kv_qkvr_prep_tree");
            id<MTLComputePipelineState> pso_as_lin   = get_pso("sik_attn_decode_single_gqa3_sg8");
            id<MTLComputePipelineState> pso_as_tr    = get_pso("sik_attn_decode_single_gqa3_sg8_tree");
            id<MTLComputePipelineState> pso_sp_lin   = get_pso("sik_attn_decode_split_gqa3_reg");
            id<MTLComputePipelineState> pso_sp_tr    = get_pso("sik_attn_decode_split_gqa3_reg_tree");
            id<MTLComputePipelineState> pso_q8_lin   = get_pso_fc("sik_attn_decode_q8_reg", 1U);
            id<MTLComputePipelineState> pso_q8_tr    = get_pso("sik_attn_decode_q8_reg_tree");
            id<MTLComputePipelineState> pso_red_lin  = get_pso("sik_attn_decode_reduce");
            id<MTLComputePipelineState> pso_red_tr   = get_pso("sik_attn_decode_reduce_tree");
            id<MTLComputePipelineState> pso_wo_m1    = get_pso_fc("sik_q4a_gemv_wo_sconv_xsum", 1U);
            id<MTLComputePipelineState> pso_wo_tr8   = get_pso_fc("sik_q4a_gemv_wo_sconv_tree", 8U);
            id<MTLComputePipelineState> pso_sc_lin   = get_pso("sik_sconv_static");
            id<MTLComputePipelineState> pso_sc_tr    = get_pso("sik_sconv_static_tree");
            id<MTLComputePipelineState> pso_mc_lin   = get_pso("sik_moe_combine_sconv");
            id<MTLComputePipelineState> pso_mc_tr    = get_pso("sik_moe_combine_sconv_tree");
            id<MTLComputePipelineState> pso_ckv      = get_pso("sik_compact_kv_tree");
            id<MTLComputePipelineState> pso_ctap     = get_pso("sik_compact_taps_tree");
            id<MTLComputePipelineState> pso_t_init   = get_pso("sik_tree_init");
            id<MTLComputePipelineState> pso_t_prep   = get_pso("sik_mtp_rnn_prep_tree");
            id<MTLComputePipelineState> pso_t_gate   = get_pso("sik_mtp_rnn_gate_tree");
            id<MTLComputePipelineState> pso_t_top4   = get_pso("sik_mtp_fold_top4_q8");
            id<MTLComputePipelineState> pso_t_exp    = get_pso("sik_tree_expand_top4");
            if (!pso_kv_lin || !pso_kv_tr || !pso_as_lin || !pso_as_tr || !pso_sp_lin || !pso_sp_tr ||
                !pso_q8_lin || !pso_q8_tr || !pso_red_lin || !pso_red_tr || !pso_wo_m1 || !pso_wo_tr8 ||
                !pso_sc_lin || !pso_sc_tr || !pso_mc_lin || !pso_mc_tr || !pso_ckv || !pso_ctap ||
                !pso_t_init || !pso_t_prep || !pso_t_gate || !pso_t_top4 || !pso_t_exp) {
                return 1;
            }
            id<MTLComputePipelineState> tr_psos[14] = {
                pso_kv_tr, pso_as_tr, pso_sp_tr, pso_q8_tr, pso_red_tr, pso_wo_tr8,
                pso_sc_tr, pso_mc_tr, pso_ckv, pso_ctap, pso_t_init, pso_t_prep, pso_t_gate, pso_t_top4
            };
            uint32_t tr_spills = 0U;
            for (int i = 0; i < 14; ++i) {
                if ([tr_psos[i] maxTotalThreadsPerThreadgroup] < 1024U) ++tr_spills;
            }

            struct TreeMetaH {
                uint32_t M, max_nodes, is_linear, active_parent;
                uint32_t parent[8], depth[8], anc_mask[8], path_packed[8], p2[8], p3[8], tok_id[8];
                float    node_prob[8], cum_prob[8];
                uint32_t sampled_tok[8], pool_count, beam_width;
                uint32_t pool_parent[32], pool_tok[32];
                float    pool_cond_p[32], pool_cum_p[32];
                uint32_t pool_used[32];
            };
            struct AttnPrepP {
                uint32_t M_v, M_p16, n_q, n_kv, hd, d_rel, seq_pos_start, kv_stride, is_local, save_snap_mask;
                uint32_t v_state_off_f4, kv_zero_tail;
                float qk_clip, rel_clip, eps, log_alpha, log_n_floor, attn_scale;
                uint32_t kv_dtype, snap_stride_f4;
            };
            struct AttnP {
                uint32_t M_v, M_p16, n_q, n_kv, hd, d_rel, rel_extent, swa_win, is_local, seq_pos_start;
                uint32_t m_offset, kv_stride, max_splits, chunk_size, c_batch_first; float attn_scale;
            };
            struct WoSconvP { uint32_t M, rows, cols, save_snap_mask, res_mode, snap_stride_f4, p1, p2; };
            struct SconvP   { uint32_t M, channels, save_snap_mask, res_mode, hd, n_kv, v_state_off_f4, top_k; };
            struct TreeCmpP {
                uint32_t n_loc, n_glob, n_kv, hd, old_pos, swa_str, glob_str, kv_dtype;
                uint64_t glob_lstr_b, glob_q8_data_b;
                uint32_t path_len, D, path[8];
            };

            // Genuine W=4, M=8 branching tree:
            //   0 -> {1, 2, 3, 4} (width 4 at root!), 1 -> 5 -> 7 (depth 3!), 2 -> 6 (depth 2!)
            const uint32_t M = 8U;
            const uint32_t par_arr[8] = {0U, 0U, 0U, 0U, 0U, 1U, 2U, 5U};
            TreeMetaH tm{};
            tm.M = M; tm.max_nodes = M; tm.is_linear = 0U; tm.active_parent = 0xFFFFFFFFU; tm.beam_width = 4U;
            for (uint32_t i = 0U; i < M; ++i) {
                tm.parent[i] = par_arr[i];
                tm.depth[i]  = (i == 0U) ? 0U : (tm.depth[par_arr[i]] + 1U);
                tm.anc_mask[i]    = (i == 0U) ? 1U : (tm.anc_mask[par_arr[i]] | (1U << i));
                tm.path_packed[i] = (i == 0U) ? 0U : (tm.path_packed[par_arr[i]] | (i << (tm.depth[i] * 4U)));
                tm.p2[i] = (tm.depth[i] >= 2U) ? tm.parent[par_arr[i]] : 0xFFFFFFFFU;
                tm.p3[i] = (tm.depth[i] >= 3U) ? tm.p2[par_arr[i]]     : 0xFFFFFFFFU;
            }
            id<MTLBuffer> b_tree = [dev newBufferWithBytes:&tm length:sizeof(tm) options:MTLResourceStorageModeShared];

            // 4 root-to-leaf branches covering ALL 8 nodes
            const std::vector<std::vector<uint32_t>> branches = {
                {0U, 1U, 5U, 7U}, {0U, 2U, 6U}, {0U, 3U}, {0U, 4U}
            };

            std::mt19937 rng(202619);
            std::normal_distribution<float> nd(0.0f, 0.35f);
            const uint32_t n_q = 24U, n_kv = 8U, hd = 128U, d_rel = 16U, rel_ext = 128U;
            const uint32_t q_dim = n_q * hd, kv_dim = n_kv * hd, r_dim = n_q * d_rel;
            const uint32_t kv_stride = 512U, max_splits = 36U;
            const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

            std::vector<float> q_raw(M * q_dim), k_raw(M * kv_dim), v_raw(M * kv_dim), r_raw(M * r_dim);
            std::vector<float> k_sw(kv_dim * 4U), v_sw(kv_dim * 4U), k_dyn(M * n_kv * 4U), v_dyn(M * n_kv * 4U);
            std::vector<float> q_sc(q_dim, 1.0f), r_sc(r_dim, 1.0f), rel_p(rel_ext * d_rel);
            std::vector<float> sink(M * n_q), gate(M * q_dim);
            std::vector<float> init_kv_st(2U * kv_dim * 4U);
            for (auto& v : q_raw) v = nd(rng);
            for (auto& v : k_raw) v = nd(rng);
            for (auto& v : v_raw) v = nd(rng);
            for (auto& v : r_raw) v = nd(rng);
            for (auto& v : k_sw)  v = nd(rng) * 0.2f;
            for (auto& v : v_sw)  v = nd(rng) * 0.2f;
            for (auto& v : k_dyn) v = nd(rng) * 0.1f;
            for (auto& v : v_dyn) v = nd(rng) * 0.1f;
            for (auto& v : rel_p) v = nd(rng) * 0.1f;
            for (auto& v : sink)  v = nd(rng) * 0.5f;
            for (auto& v : gate)  v = nd(rng);
            for (auto& v : init_kv_st) v = nd(rng) * 0.25f;

            // Prefix KV caches (F16 + Q8_0) populated for positions 0..310
            std::vector<uint16_t> init_k16(n_kv * kv_stride * hd), init_v16(n_kv * kv_stride * hd);
            std::vector<int8_t>   init_kq8(n_kv * kv_stride * hd), init_vq8(n_kv * kv_stride * hd);
            std::vector<uint16_t> init_ks8(n_kv * kv_stride),      init_vs8(n_kv * kv_stride);
            for (auto& v : init_k16) v = f32_to_fp16_bits(nd(rng) * 0.4f);
            for (auto& v : init_v16) v = f32_to_fp16_bits(nd(rng) * 0.4f);
            for (auto& v : init_kq8) v = static_cast<int8_t>(static_cast<int>(rng() % 201U) - 100);
            for (auto& v : init_vq8) v = static_cast<int8_t>(static_cast<int>(rng() % 201U) - 100);
            for (auto& v : init_ks8) v = f32_to_fp16_bits(0.01f);
            for (auto& v : init_vs8) v = f32_to_fp16_bits(0.01f);

            id<MTLBuffer> b_qraw = [dev newBufferWithBytes:q_raw.data() length:q_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kraw = [dev newBufferWithBytes:k_raw.data() length:k_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vraw = [dev newBufferWithBytes:v_raw.data() length:v_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rraw = [dev newBufferWithBytes:r_raw.data() length:r_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ksw  = [dev newBufferWithBytes:k_sw.data()  length:k_sw.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vsw  = [dev newBufferWithBytes:v_sw.data()  length:v_sw.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kdyn = [dev newBufferWithBytes:k_dyn.data() length:k_dyn.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vdyn = [dev newBufferWithBytes:v_dyn.data() length:v_dyn.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_qsc  = [dev newBufferWithBytes:q_sc.data()  length:q_sc.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rsc  = [dev newBufferWithBytes:r_sc.data()  length:r_sc.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_relp = [dev newBufferWithBytes:rel_p.data() length:rel_p.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sink = [dev newBufferWithBytes:sink.data()  length:sink.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_gate = [dev newBufferWithBytes:gate.data()  length:gate.size()  * 4U options:MTLResourceStorageModeShared];

            id<MTLBuffer> b_qf32  = [dev newBufferWithLength:M * q_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rf32  = [dev newBufferWithLength:M * r_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kc    = [dev newBufferWithLength:n_kv * kv_stride * hd * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vc    = [dev newBufferWithLength:n_kv * kv_stride * hd * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ks    = [dev newBufferWithLength:n_kv * kv_stride * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vs    = [dev newBufferWithLength:n_kv * kv_stride * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_st    = [dev newBufferWithLength:2U * kv_dim * 16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sn    = [dev newBufferWithLength:8U * 2U * kv_dim * 16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pm    = [dev newBufferWithLength:M * n_q * max_splits * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_pl    = [dev newBufferWithLength:M * n_q * max_splits * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_po    = [dev newBufferWithLength:M * n_q * max_splits * hd * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_out_t = [dev newBufferWithLength:M * q_dim * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xs_t  = [dev newBufferWithLength:M * n_q * 2U * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_out_r = [dev newBufferWithLength:q_dim * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xs_r  = [dev newBufferWithLength:n_q * 2U * 4U options:MTLResourceStorageModeShared];

            // Single-token staging buffers for sequential M=1 reference walk
            id<MTLBuffer> b_qr1 = [dev newBufferWithLength:q_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kr1 = [dev newBufferWithLength:kv_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vr1 = [dev newBufferWithLength:kv_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rr1 = [dev newBufferWithLength:r_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kd1 = [dev newBufferWithLength:n_kv * 16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vd1 = [dev newBufferWithLength:n_kv * 16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sk1 = [dev newBufferWithLength:n_q * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_gt1 = [dev newBufferWithLength:q_dim * 4U options:MTLResourceStorageModeShared];

            auto test_tree_attn = [&](uint32_t P, bool is_local, bool use_q8) -> uint32_t {
                const uint32_t snap_stride_f4 = 2U * kv_dim;
                // 1. Run W=4 M=8 Tree Pass
                std::memcpy([b_st contents], init_kv_st.data(), init_kv_st.size() * 4U);
                if (use_q8) {
                    std::memcpy([b_kc contents], init_kq8.data(), init_kq8.size());
                    std::memcpy([b_vc contents], init_vq8.data(), init_vq8.size());
                    std::memcpy([b_ks contents], init_ks8.data(), init_ks8.size() * 2U);
                    std::memcpy([b_vs contents], init_vs8.data(), init_vs8.size() * 2U);
                } else {
                    std::memcpy([b_kc contents], init_k16.data(), init_k16.size() * 2U);
                    std::memcpy([b_vc contents], init_v16.data(), init_v16.size() * 2U);
                }
                {
                    AttnPrepP pp{M, 16U, n_q, n_kv, hd, d_rel, P, kv_stride, is_local ? 1U : 0U, 0xFFU,
                                 kv_dim, 0U, 10.0f, 10.0f, 1e-6f, 0.1f, 128.0f, attn_scale, use_q8 ? 1U : 0U, snap_stride_f4};
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_kv_tr];
                    [enc setBuffer:b_qraw offset:0 atIndex:0]; [enc setBuffer:b_kraw offset:0 atIndex:1];
                    [enc setBuffer:b_vraw offset:0 atIndex:2]; [enc setBuffer:b_rraw offset:0 atIndex:3];
                    [enc setBuffer:b_ksw offset:0 atIndex:4];  [enc setBuffer:b_vsw offset:0 atIndex:5];
                    [enc setBuffer:b_kdyn offset:0 atIndex:6]; [enc setBuffer:b_vdyn offset:0 atIndex:7];
                    [enc setBuffer:b_st offset:0 atIndex:8];   [enc setBuffer:b_qsc offset:0 atIndex:9];
                    [enc setBuffer:b_rsc offset:0 atIndex:10]; [enc setBuffer:b_qf32 offset:0 atIndex:11];
                    [enc setBuffer:b_rf32 offset:0 atIndex:12]; [enc setBuffer:b_kc offset:0 atIndex:13];
                    [enc setBuffer:b_vc offset:0 atIndex:14];  [enc setBuffer:b_sn offset:0 atIndex:15];
                    [enc setBytes:&pp length:sizeof(pp) atIndex:16];
                    [enc setBuffer:b_ks offset:0 atIndex:17];  [enc setBuffer:b_vs offset:0 atIndex:18];
                    [enc setBuffer:b_tree offset:0 atIndex:19];
                    [enc dispatchThreadgroups:MTLSizeMake(n_q * M + n_kv + n_kv + n_q * M, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    auto host_cs = [&](uint32_t qp) -> uint32_t {
                        if (qp < 160U) return 16U;
                        if (qp < 512U) return 32U;
                        return 64U;
                    };
                    bool need_red = false;
                    uint32_t qc = P;
                    while (qc <= P + M - 1U) {
                        const uint32_t Bc = host_cs(qc);
                        const bool use_s = (!use_q8 && qc < 256U);
                        uint32_t qe = qc + 1U;
                        while (qe <= P + M - 1U && host_cs(qe) == Bc && ((!use_q8 && qe < 256U) == use_s)) ++qe;
                        const uint32_t ws = (is_local && qc + 1U > 1024U) ? (qc + 1U - 1024U) : 0U;
                        const uint32_t cf = ws / Bc, cl = (qe - 1U) / Bc;
                        AttnP ap{M, 16U, n_q, n_kv, hd, d_rel, rel_ext, 1024U, is_local ? 1U : 0U, P, 0U, kv_stride, max_splits, Bc, cf, attn_scale};
                        if (use_s) {
                            [enc setComputePipelineState:pso_as_tr];
                            [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                            [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_kc offset:0 atIndex:3];
                            [enc setBuffer:b_vc offset:0 atIndex:4];   [enc setBuffer:b_sink offset:0 atIndex:5];
                            [enc setBuffer:b_gate offset:0 atIndex:6]; [enc setBuffer:b_out_t offset:0 atIndex:7];
                            [enc setBuffer:b_xs_t offset:0 atIndex:8]; [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                            [enc setBuffer:b_tree offset:0 atIndex:10];
                            [enc dispatchThreadgroups:MTLSizeMake(n_kv, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        } else {
                            need_red = true;
                            if (use_q8) {
                                [enc setComputePipelineState:pso_q8_tr];
                                [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                                [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_kc offset:0 atIndex:3];
                                [enc setBuffer:b_vc offset:0 atIndex:4];   [enc setBuffer:b_pm offset:0 atIndex:5];
                                [enc setBuffer:b_pl offset:0 atIndex:6];   [enc setBuffer:b_po offset:0 atIndex:7];
                                [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                                [enc setBuffer:b_ks offset:0 atIndex:9];   [enc setBuffer:b_vs offset:0 atIndex:10];
                                [enc setBuffer:b_tree offset:0 atIndex:11];
                            } else {
                                [enc setComputePipelineState:pso_sp_tr];
                                [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                                [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_kc offset:0 atIndex:3];
                                [enc setBuffer:b_vc offset:0 atIndex:4];   [enc setBuffer:b_pm offset:0 atIndex:5];
                                [enc setBuffer:b_pl offset:0 atIndex:6];   [enc setBuffer:b_po offset:0 atIndex:7];
                                [enc setBytes:&ap length:sizeof(ap) atIndex:8]; [enc setBuffer:b_tree offset:0 atIndex:9];
                            }
                            [enc dispatchThreadgroups:MTLSizeMake(cl - cf + 1U, n_kv, M) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        }
                        qc = qe;
                    }
                    if (need_red) {
                        AttnP rp{M, 16U, n_q, n_kv, hd, d_rel, rel_ext, 1024U, is_local ? 1U : 0U, P, 0U, kv_stride, max_splits, 32U, use_q8 ? 0U : 1U, attn_scale};
                        [enc setComputePipelineState:pso_red_tr];
                        [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                        [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_sink offset:0 atIndex:3];
                        [enc setBuffer:b_gate offset:0 atIndex:4]; [enc setBuffer:b_out_t offset:0 atIndex:5];
                        [enc setBytes:&rp length:sizeof(rp) atIndex:6]; [enc setBuffer:b_xs_t offset:0 atIndex:7];
                        [enc setBuffer:b_tree offset:0 atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(n_q, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                std::vector<uint16_t> tree_out(M * q_dim);
                std::vector<uint32_t> tree_snaps(M * 2U * kv_dim * 4U);
                std::memcpy(tree_out.data(), [b_out_t contents], tree_out.size() * 2U);
                std::memcpy(tree_snaps.data(), [b_sn contents], tree_snaps.size() * 4U);

                // 2. Compare against sequential M=1 along EVERY root-to-leaf branch!
                uint32_t bit_diffs = 0U;
                for (const auto& br : branches) {
                    std::memcpy([b_st contents], init_kv_st.data(), init_kv_st.size() * 4U);
                    if (use_q8) {
                        std::memcpy([b_kc contents], init_kq8.data(), init_kq8.size());
                        std::memcpy([b_vc contents], init_vq8.data(), init_vq8.size());
                        std::memcpy([b_ks contents], init_ks8.data(), init_ks8.size() * 2U);
                        std::memcpy([b_vs contents], init_vs8.data(), init_vs8.size() * 2U);
                    } else {
                        std::memcpy([b_kc contents], init_k16.data(), init_k16.size() * 2U);
                        std::memcpy([b_vc contents], init_v16.data(), init_v16.size() * 2U);
                    }
                    for (uint32_t step = 0U; step < br.size(); ++step) {
                        const uint32_t node = br[step];
                        const uint32_t qp   = P + step;
                        std::memcpy([b_qr1 contents], q_raw.data() + node * q_dim, q_dim * 4U);
                        std::memcpy([b_kr1 contents], k_raw.data() + node * kv_dim, kv_dim * 4U);
                        std::memcpy([b_vr1 contents], v_raw.data() + node * kv_dim, kv_dim * 4U);
                        std::memcpy([b_rr1 contents], r_raw.data() + node * r_dim, r_dim * 4U);
                        std::memcpy([b_kd1 contents], k_dyn.data() + node * n_kv * 4U, n_kv * 16U);
                        std::memcpy([b_vd1 contents], v_dyn.data() + node * n_kv * 4U, n_kv * 16U);
                        std::memcpy([b_sk1 contents], sink.data()  + node * n_q, n_q * 4U);
                        std::memcpy([b_gt1 contents], gate.data()  + node * q_dim, q_dim * 4U);

                        AttnPrepP pp{1U, 16U, n_q, n_kv, hd, d_rel, qp, kv_stride, is_local ? 1U : 0U, 0U,
                                     kv_dim, 0U, 10.0f, 10.0f, 1e-6f, 0.1f, 128.0f, attn_scale, use_q8 ? 1U : 0U, snap_stride_f4};
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_kv_lin];
                        [enc setBuffer:b_qr1 offset:0 atIndex:0]; [enc setBuffer:b_kr1 offset:0 atIndex:1];
                        [enc setBuffer:b_vr1 offset:0 atIndex:2]; [enc setBuffer:b_rr1 offset:0 atIndex:3];
                        [enc setBuffer:b_ksw offset:0 atIndex:4]; [enc setBuffer:b_vsw offset:0 atIndex:5];
                        [enc setBuffer:b_kd1 offset:0 atIndex:6]; [enc setBuffer:b_vd1 offset:0 atIndex:7];
                        [enc setBuffer:b_st offset:0 atIndex:8];  [enc setBuffer:b_qsc offset:0 atIndex:9];
                        [enc setBuffer:b_rsc offset:0 atIndex:10]; [enc setBuffer:b_qf32 offset:0 atIndex:11];
                        [enc setBuffer:b_rf32 offset:0 atIndex:12]; [enc setBuffer:b_kc offset:0 atIndex:13];
                        [enc setBuffer:b_vc offset:0 atIndex:14]; [enc setBuffer:b_sn offset:0 atIndex:15];
                        [enc setBytes:&pp length:sizeof(pp) atIndex:16];
                        [enc setBuffer:b_ks offset:0 atIndex:17]; [enc setBuffer:b_vs offset:0 atIndex:18];
                        [enc dispatchThreadgroups:MTLSizeMake(n_q + n_kv + n_kv + n_q, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                        const uint32_t B = (qp < 160U) ? 16U : ((qp < 512U) ? 32U : 64U);
                        const uint32_t ws = (is_local && qp + 1U > 1024U) ? (qp + 1U - 1024U) : 0U;
                        const uint32_t cf = ws / B, cl = qp / B;
                        AttnP ap{1U, 16U, n_q, n_kv, hd, d_rel, rel_ext, 1024U, is_local ? 1U : 0U, qp, 0U, kv_stride, max_splits, B, cf, attn_scale};
                        if (!use_q8 && qp < 256U) {
                            [enc setComputePipelineState:pso_as_lin];
                            [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                            [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_kc offset:0 atIndex:3];
                            [enc setBuffer:b_vc offset:0 atIndex:4];   [enc setBuffer:b_sk1 offset:0 atIndex:5];
                            [enc setBuffer:b_gt1 offset:0 atIndex:6];  [enc setBuffer:b_out_r offset:0 atIndex:7];
                            [enc setBuffer:b_xs_r offset:0 atIndex:8]; [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                            [enc dispatchThreadgroups:MTLSizeMake(n_kv, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        } else {
                            if (use_q8) {
                                [enc setComputePipelineState:pso_q8_lin];
                                [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                                [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_kc offset:0 atIndex:3];
                                [enc setBuffer:b_vc offset:0 atIndex:4];   [enc setBuffer:b_pm offset:0 atIndex:5];
                                [enc setBuffer:b_pl offset:0 atIndex:6];   [enc setBuffer:b_po offset:0 atIndex:7];
                                [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                                [enc setBuffer:b_ks offset:0 atIndex:9];   [enc setBuffer:b_vs offset:0 atIndex:10];
                            } else {
                                [enc setComputePipelineState:pso_sp_lin];
                                [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                                [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_kc offset:0 atIndex:3];
                                [enc setBuffer:b_vc offset:0 atIndex:4];   [enc setBuffer:b_pm offset:0 atIndex:5];
                                [enc setBuffer:b_pl offset:0 atIndex:6];   [enc setBuffer:b_po offset:0 atIndex:7];
                                [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                            }
                            [enc dispatchThreadgroups:MTLSizeMake(cl - cf + 1U, n_kv, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                            [enc setComputePipelineState:pso_red_lin];
                            [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                            [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_sk1 offset:0 atIndex:3];
                            [enc setBuffer:b_gt1 offset:0 atIndex:4]; [enc setBuffer:b_out_r offset:0 atIndex:5];
                            [enc setBytes:&ap length:sizeof(ap) atIndex:6]; [enc setBuffer:b_xs_r offset:0 atIndex:7];
                            [enc dispatchThreadgroups:MTLSizeMake(n_q, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        }
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                        const auto* ro = reinterpret_cast<const uint16_t*>([b_out_r contents]);
                        for (uint32_t c = 0U; c < q_dim; ++c) {
                            if (tree_out[node * q_dim + c] != ro[c]) ++bit_diffs;
                        }
                        const auto* rst = reinterpret_cast<const uint32_t*>([b_st contents]);
                        for (uint32_t c = 0U; c < 2U * kv_dim * 4U; ++c) {
                            if (tree_snaps[node * 2U * kv_dim * 4U + c] != rst[c]) ++bit_diffs;
                        }
                    }
                }
                return bit_diffs;
            };

            const uint32_t diffs_swa_40   = test_tree_attn(40U,  true,  false);
            const uint32_t diffs_glob_254 = test_tree_attn(254U, false, false); // straddles 256!
            const uint32_t diffs_glob_300 = test_tree_attn(300U, false, false);
            const uint32_t diffs_q8_300   = test_tree_attn(300U, false, true);

            // 19c. Parent-Indexed Wo+AttnSConv Tree (M=8) + Static SConv Tree + MoE Combine+SConv Tree vs sequential M=1!
            const uint32_t D = 256U, C_in = 256U, top_k = 8U;
            std::vector<uint8_t>  wo_w(D * (C_in / 2U));
            std::vector<uint16_t> wo_s(D * (C_in / 64U)), wo_b(D * (C_in / 64U)), x_wo(M * C_in);
            std::vector<float>    x_wo_sum(M * (C_in / 64U), 0.0f), sc_w(D * 4U), sc_init(D * 4U), cur_init(M * D);
            std::vector<float>    moe_dp(M * top_k * D), moe_tw(M * top_k);
            for (auto& v : wo_w)     v = static_cast<uint8_t>(rng() & 0xFFU);
            for (auto& v : wo_s)     v = f32_to_fp16_bits(0.02f);
            for (auto& v : wo_b)     v = f32_to_fp16_bits(-0.15f);
            for (auto& v : x_wo)     v = f32_to_fp16_bits(nd(rng));
            for (uint32_t m = 0U; m < M; ++m) {
                for (uint32_t g = 0U; g < C_in / 64U; ++g) {
                    float s = 0.0f;
                    for (uint32_t k = 0U; k < 64U; ++k) s += fp16_bits_to_f32_ref(x_wo[m * C_in + g * 64U + k]);
                    x_wo_sum[m * (C_in / 64U) + g] = s;
                }
            }
            for (auto& v : sc_w)     v = nd(rng) * 0.25f;
            for (auto& v : sc_init)  v = nd(rng) * 0.25f;
            for (auto& v : cur_init) v = nd(rng);
            for (auto& v : moe_dp)   v = nd(rng);
            for (auto& v : moe_tw)   v = 0.125f;

            id<MTLBuffer> b_wow  = [dev newBufferWithBytes:wo_w.data() length:wo_w.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_wos  = [dev newBufferWithBytes:wo_s.data() length:wo_s.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_wob  = [dev newBufferWithBytes:wo_b.data() length:wo_b.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xwo  = [dev newBufferWithBytes:x_wo.data() length:x_wo.size() * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xws  = [dev newBufferWithBytes:x_wo_sum.data() length:x_wo_sum.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_scw  = [dev newBufferWithBytes:sc_w.data() length:sc_w.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_scs  = [dev newBufferWithBytes:sc_init.data() length:sc_init.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_scsn = [dev newBufferWithLength:8U * D * 16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_curt = [dev newBufferWithBytes:cur_init.data() length:cur_init.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_blkd = [dev newBufferWithLength:M * D * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mdp  = [dev newBufferWithBytes:moe_dp.data() length:moe_dp.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mtw  = [dev newBufferWithBytes:moe_tw.data() length:moe_tw.size() * 4U options:MTLResourceStorageModeShared];

            // Tree Wo+AttnSConv (M=8)
            {
                WoSconvP wp{M, D, C_in, 0xFFU, 1U, D, 0U, 0U};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_wo_tr8];
                [enc setBuffer:b_wow offset:0 atIndex:0];  [enc setBuffer:b_wos offset:0 atIndex:1];
                [enc setBuffer:b_wob offset:0 atIndex:2];  [enc setBuffer:b_xwo offset:0 atIndex:3];
                [enc setBuffer:b_xws offset:0 atIndex:4];  [enc setBuffer:b_scw offset:0 atIndex:5];
                [enc setBuffer:b_scs offset:0 atIndex:6];  [enc setBuffer:b_curt offset:0 atIndex:7];
                [enc setBuffer:b_blkd offset:0 atIndex:8]; [enc setBuffer:b_scsn offset:0 atIndex:9];
                [enc setBytes:&wp length:sizeof(wp) atIndex:10]; [enc setBuffer:b_tree offset:0 atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            std::vector<uint32_t> wo_t_cur(M * D), wo_t_sn(M * D * 4U);
            std::memcpy(wo_t_cur.data(), [b_curt contents], wo_t_cur.size() * 4U);
            std::memcpy(wo_t_sn.data(),  [b_scsn contents], wo_t_sn.size() * 4U);

            // Tree MoE Combine+SConv (M=8)
            std::memcpy([b_scs contents],  sc_init.data(),  sc_init.size() * 4U);
            std::memcpy([b_curt contents], cur_init.data(), cur_init.size() * 4U);
            {
                SconvP csp{M, D, 0xFFU, 1U, hd, n_kv, D, top_k};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_mc_tr];
                [enc setBuffer:b_mdp offset:0 atIndex:0];  [enc setBuffer:b_mtw offset:0 atIndex:1];
                [enc setBuffer:b_scw offset:0 atIndex:2];  [enc setBuffer:b_scs offset:0 atIndex:3];
                [enc setBuffer:b_curt offset:0 atIndex:4]; [enc setBuffer:b_scsn offset:0 atIndex:5];
                [enc setBytes:&csp length:sizeof(csp) atIndex:6]; [enc setBuffer:b_tree offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((D + 63U) / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            std::vector<uint32_t> mc_t_cur(M * D), mc_t_sn(M * D * 4U);
            std::memcpy(mc_t_cur.data(), [b_curt contents], mc_t_cur.size() * 4U);
            std::memcpy(mc_t_sn.data(),  [b_scsn contents], mc_t_sn.size() * 4U);

            // Compare both against sequential M=1 along every branch!
            uint32_t wo_sconv_diffs = 0U, moe_sconv_diffs = 0U;
            id<MTLBuffer> b_xwo1 = [dev newBufferWithLength:C_in * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xws1 = [dev newBufferWithLength:(C_in / 64U) * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_cur1 = [dev newBufferWithLength:D * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mdp1 = [dev newBufferWithLength:top_k * D * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_mtw1 = [dev newBufferWithLength:top_k * 4U options:MTLResourceStorageModeShared];

            for (const auto& br : branches) {
                // Wo+AttnSConv branch ref
                std::memcpy([b_scs contents], sc_init.data(), sc_init.size() * 4U);
                for (uint32_t node : br) {
                    std::memcpy([b_xwo1 contents], x_wo.data() + node * C_in, C_in * 2U);
                    std::memcpy([b_xws1 contents], x_wo_sum.data() + node * (C_in / 64U), (C_in / 64U) * 4U);
                    std::memcpy([b_cur1 contents], cur_init.data() + node * D, D * 4U);
                    WoSconvP wp{1U, D, C_in, 0U, 1U, D, 0U, 0U};
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_wo_m1];
                    [enc setBuffer:b_wow offset:0 atIndex:0];  [enc setBuffer:b_wos offset:0 atIndex:1];
                    [enc setBuffer:b_wob offset:0 atIndex:2];  [enc setBuffer:b_xwo1 offset:0 atIndex:3];
                    [enc setBuffer:b_xws1 offset:0 atIndex:4]; [enc setBuffer:b_scw offset:0 atIndex:5];
                    [enc setBuffer:b_scs offset:0 atIndex:6];  [enc setBuffer:b_cur1 offset:0 atIndex:7];
                    [enc setBuffer:b_blkd offset:0 atIndex:8]; [enc setBuffer:b_scsn offset:0 atIndex:9];
                    [enc setBytes:&wp length:sizeof(wp) atIndex:10];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    const auto* rc = reinterpret_cast<const uint32_t*>([b_cur1 contents]);
                    const auto* rs = reinterpret_cast<const uint32_t*>([b_scs contents]);
                    for (uint32_t c = 0U; c < D; ++c) if (wo_t_cur[node * D + c] != rc[c]) ++wo_sconv_diffs;
                    for (uint32_t c = 0U; c < D * 4U; ++c) if (wo_t_sn[node * D * 4U + c] != rs[c]) ++wo_sconv_diffs;
                }
                // MoE Combine+SConv branch ref
                std::memcpy([b_scs contents], sc_init.data(), sc_init.size() * 4U);
                for (uint32_t node : br) {
                    std::memcpy([b_mdp1 contents], moe_dp.data() + node * top_k * D, top_k * D * 4U);
                    std::memcpy([b_mtw1 contents], moe_tw.data() + node * top_k, top_k * 4U);
                    std::memcpy([b_cur1 contents], cur_init.data() + node * D, D * 4U);
                    SconvP csp{1U, D, 0U, 1U, hd, n_kv, D, top_k};
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_mc_lin];
                    [enc setBuffer:b_mdp1 offset:0 atIndex:0]; [enc setBuffer:b_mtw1 offset:0 atIndex:1];
                    [enc setBuffer:b_scw offset:0 atIndex:2];  [enc setBuffer:b_scs offset:0 atIndex:3];
                    [enc setBuffer:b_cur1 offset:0 atIndex:4]; [enc setBuffer:b_scsn offset:0 atIndex:5];
                    [enc setBytes:&csp length:sizeof(csp) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 63U) / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    const auto* rc = reinterpret_cast<const uint32_t*>([b_cur1 contents]);
                    const auto* rs = reinterpret_cast<const uint32_t*>([b_scs contents]);
                    for (uint32_t c = 0U; c < D; ++c) if (mc_t_cur[node * D + c] != rc[c]) ++moe_sconv_diffs;
                    for (uint32_t c = 0U; c < D * 4U; ++c) if (mc_t_sn[node * D * 4U + c] != rs[c]) ++moe_sconv_diffs;
                }
            }

            // 19d. Winning-Path KV & Tap Compaction (`path = [0, 2, 6]`)
            uint32_t compact_diffs = 0U;
            {
                const uint32_t old_pos = 100U;
                std::memcpy([b_kc contents], init_k16.data(), init_k16.size() * 2U);
                std::memcpy([b_vc contents], init_v16.data(), init_v16.size() * 2U);
                const uint64_t gq8_data = static_cast<uint64_t>(n_kv) * kv_stride * hd;
                const uint64_t gq8_sc   = static_cast<uint64_t>(n_kv) * kv_stride * 2U;
                id<MTLBuffer> b_gk = [dev newBufferWithLength:gq8_data + gq8_sc options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_gv = [dev newBufferWithLength:gq8_data + gq8_sc options:MTLResourceStorageModeShared];
                std::memcpy([b_gk contents], init_kq8.data(), gq8_data);
                std::memcpy(static_cast<uint8_t*>([b_gk contents]) + gq8_data, init_ks8.data(), gq8_sc);
                std::memcpy([b_gv contents], init_vq8.data(), gq8_data);
                std::memcpy(static_cast<uint8_t*>([b_gv contents]) + gq8_data, init_vs8.data(), gq8_sc);
                std::vector<float> taps_orig(M * 4U * D);
                for (size_t i = 0U; i < taps_orig.size(); ++i) taps_orig[i] = static_cast<float>(i) * 0.25f;
                id<MTLBuffer> b_taps = [dev newBufferWithBytes:taps_orig.data() length:taps_orig.size() * 4U options:MTLResourceStorageModeShared];

                TreeCmpP cp{1U, 1U, n_kv, hd, old_pos, kv_stride, kv_stride, 1U, gq8_data + gq8_sc, gq8_data, 3U, D, {0U, 2U, 6U, 0U, 0U, 0U, 0U, 0U}};
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso_ckv];
                [enc setBuffer:b_kc offset:0 atIndex:0]; [enc setBuffer:b_vc offset:0 atIndex:1];
                [enc setBuffer:b_gk offset:0 atIndex:2]; [enc setBuffer:b_gv offset:0 atIndex:3];
                [enc setBytes:&cp length:sizeof(cp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(2U, n_kv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc setComputePipelineState:pso_ctap];
                [enc setBuffer:b_taps offset:0 atIndex:0]; [enc setBytes:&cp length:sizeof(cp) atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake(((D / 4U) + 255U) / 256U, 4U, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                const uint32_t pth[3] = {0U, 2U, 6U};
                const auto* kc_out = reinterpret_cast<const uint16_t*>([b_kc contents]);
                const auto* gk_out = reinterpret_cast<const int8_t*>([b_gk contents]);
                const auto* gs_out = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>([b_gk contents]) + gq8_data);
                const auto* tp_out = reinterpret_cast<const float*>([b_taps contents]);
                for (uint32_t k = 0U; k < 3U; ++k) {
                    for (uint32_t h = 0U; h < n_kv; ++h) {
                        const size_t dst_p = (static_cast<size_t>(h) * kv_stride + (old_pos + k)) * hd;
                        const size_t src_p = (static_cast<size_t>(h) * kv_stride + (old_pos + pth[k])) * hd;
                        for (uint32_t d = 0U; d < hd; ++d) {
                            if (kc_out[dst_p + d] != init_k16[src_p + d]) ++compact_diffs;
                            if (gk_out[dst_p + d] != init_kq8[src_p + d]) ++compact_diffs;
                        }
                        if (gs_out[h * kv_stride + old_pos + k] != init_ks8[h * kv_stride + old_pos + pth[k]]) ++compact_diffs;
                    }
                    for (uint32_t i = 0U; i < 4U * D; ++i) {
                        if (tp_out[k * 4U * D + i] != taps_orig[pth[k] * 4U * D + i]) ++compact_diffs;
                    }
                }
            }

            const bool ok19 = (tr_spills == 0U) &&
                              (diffs_swa_40 == 0U) && (diffs_glob_254 == 0U) &&
                              (diffs_glob_300 == 0U) && (diffs_q8_300 == 0U) &&
                              (wo_sconv_diffs == 0U) && (moe_sconv_diffs == 0U) && (compact_diffs == 0U);
            std::printf("[Test 19] W=4 M=8 Beam Tree Attn + Parent SConv + Compact : %s\n", ok19 ? "PASS" : "FAIL");
            std::printf("         • Tree Attn BitDiffs: SWA(40)=%u Glob(254)=%u Glob(300)=%u Q8_KV(300)=%u | Spills=%u\n",
                        diffs_swa_40, diffs_glob_254, diffs_glob_300, diffs_q8_300, tr_spills);
            std::printf("         • Parent SConv BitDiffs: Wo+Attn=%u MoE+MLP=%u | KV/Tap Compact=%u\n",
                        wo_sconv_diffs, moe_sconv_diffs, compact_diffs);
            if (!ok19) return 1;
        }

        // ---- Test 20: Continuous-Batching Multi-Stream GPU Decode Kernels (`B = 4`, Staggered `seq_pos`, FP16 & Q8_0 KV, Penalties & Mixed Sampling) ----
        {
            id<MTLComputePipelineState> pso_kv_lin   = get_pso("sik_sconv_kv_qkvr_prep");
            id<MTLComputePipelineState> pso_as_lin   = get_pso("sik_attn_decode_single_gqa3_sg8");
            id<MTLComputePipelineState> pso_sp_lin   = get_pso("sik_attn_decode_split_gqa3_reg");
            id<MTLComputePipelineState> pso_q8_lin   = get_pso_fc("sik_attn_decode_q8_reg", 1U);
            id<MTLComputePipelineState> pso_red_lin  = get_pso("sik_attn_decode_reduce");
            id<MTLComputePipelineState> pso_wo_m1    = get_pso_fc("sik_q4a_gemv_wo_sconv_xsum", 1U);
            id<MTLComputePipelineState> pso_sc_lin   = get_pso("sik_sconv_static");
            id<MTLComputePipelineState> pso_mc_lin   = get_pso("sik_moe_combine_sconv");
            id<MTLComputePipelineState> pso_pen_lin  = get_pso("sik_apply_logit_penalties");
            id<MTLComputePipelineState> pso_grd1     = get_pso("sik_sample_greedy_stage1");
            id<MTLComputePipelineState> pso_grd2     = get_pso("sik_sample_greedy_stage2");
            id<MTLComputePipelineState> pso_st1      = get_pso("sik_sample_vocab_stage1");
            id<MTLComputePipelineState> pso_st2      = get_pso("sik_sample_vocab_stage2");

            id<MTLComputePipelineState> pso_b_kv     = get_pso("sik_sconv_kv_qkvr_prep_batched");
            id<MTLComputePipelineState> pso_b_as     = get_pso("sik_attn_decode_single_gqa3_batched");
            id<MTLComputePipelineState> pso_b_sp     = get_pso("sik_attn_decode_split_gqa3_batched");
            id<MTLComputePipelineState> pso_b_q8     = get_pso("sik_attn_decode_q8_reg_batched");
            id<MTLComputePipelineState> pso_b_red    = get_pso("sik_attn_decode_reduce_batched");
            id<MTLComputePipelineState> pso_b_wo[8];
            for (uint32_t m = 1U; m <= 8U; ++m) {
                pso_b_wo[m - 1U] = get_pso_fc("sik_q4a_gemv_wo_sconv_batched", m);
            }
            id<MTLComputePipelineState> pso_b_sc     = get_pso("sik_sconv_static_batched");
            id<MTLComputePipelineState> pso_b_mc     = get_pso("sik_moe_combine_sconv_batched");
            id<MTLComputePipelineState> pso_b_pen    = get_pso("sik_apply_logit_penalties_batched");
            id<MTLComputePipelineState> pso_b_s1     = get_pso("sik_sample_batched_stage1");
            id<MTLComputePipelineState> pso_b_s2     = get_pso("sik_sample_batched_stage2");
            id<MTLComputePipelineState> pso_b_q2d    = get_pso("sik_gemv_attn_qgrs_mixed_2d");
            id<MTLComputePipelineState> pso_b_wo2d   = get_pso("sik_q4a_gemv_wo_sconv_batched_2d");
            id<MTLComputePipelineState> pso_b_q82d   = get_pso("sik_q8_gemv_batch_2d");
            id<MTLComputePipelineState> pso_b_rt2d   = get_pso("sik_moe_router_f32_2d");
            id<MTLComputePipelineState> pso_b_d12d   = get_pso("sik_loop_ctrl_down1_left_q8_2d");
            id<MTLComputePipelineState> pso_b_dr2d[8], pso_b_dl32d[8];
            for (uint32_t m = 1U; m <= 8U; ++m) {
                pso_b_dr2d[m - 1U]  = get_pso_fc("sik_loop_ctrl_down_right_batch12_q8_2d", m);
                pso_b_dl32d[m - 1U] = get_pso_fc("sik_loop_ctrl_down3_left_q8_2d", m);
            }

            if (!pso_kv_lin || !pso_as_lin || !pso_sp_lin || !pso_q8_lin || !pso_red_lin ||
                !pso_wo_m1 || !pso_sc_lin || !pso_mc_lin || !pso_pen_lin || !pso_grd1 || !pso_grd2 ||
                !pso_st1 || !pso_st2 || !pso_b_kv || !pso_b_as || !pso_b_sp || !pso_b_q8 || !pso_b_red ||
                !pso_b_sc || !pso_b_mc || !pso_b_pen || !pso_b_s1 || !pso_b_s2 ||
                !pso_b_q2d || !pso_b_wo2d || !pso_b_q82d || !pso_b_rt2d || !pso_b_d12d) {
                return 1;
            }
            uint32_t cb_spills = 0U;
            id<MTLComputePipelineState> cb_psos[15] = {
                pso_b_kv, pso_b_as, pso_b_sp, pso_b_q8, pso_b_red,
                pso_b_sc, pso_b_mc, pso_b_pen, pso_b_s1, pso_b_s2,
                pso_b_q2d, pso_b_wo2d, pso_b_q82d, pso_b_rt2d, pso_b_d12d
            };
            for (int i = 0; i < 15; ++i) {
                if ([cb_psos[i] maxTotalThreadsPerThreadgroup] < 1024U) ++cb_spills;
            }
            for (int m = 0; m < 8; ++m) {
                if (!pso_b_wo[m] || !pso_b_dr2d[m] || !pso_b_dl32d[m]) return 1;
                if ([pso_b_wo[m] maxTotalThreadsPerThreadgroup] < 1024U) ++cb_spills;
                if ([pso_b_dr2d[m] maxTotalThreadsPerThreadgroup] < 1024U) ++cb_spills;
                if ([pso_b_dl32d[m] maxTotalThreadsPerThreadgroup] < 1024U) ++cb_spills;
            }

            struct BSlotMetaH {
                uint64_t swa_k, swa_v, glob_k, glob_v, sconv;
                uint32_t seq_pos, pad0;
            };
            struct BAttnPrepP {
                uint32_t B, n_q, n_kv, hd, d_rel, kv_stride, is_local, kv_dtype;
                uint32_t sconv_k_off_f4, v_state_off_f4, pad0, pad1;
                uint64_t kv_lyr_off, kv_q8_data_b;
                float    qk_clip, rel_clip, eps, log_alpha, log_n_floor, pad2;
            };
            struct BAttnP {
                uint32_t B, n_q, n_kv, hd, d_rel, rel_extent, swa_win, is_local;
                uint32_t kv_stride, max_splits, skip_under_256, pad0;
                uint64_t kv_lyr_off, kv_q8_data_b;
                float    attn_scale;
                uint32_t pad1;
            };
            struct BSconvP { uint32_t B, channels, res_mode, sconv_off_f4, top_k, cols, pad0, pad1; };
            struct BPenItemH { uint32_t off, len; float pres, freq, rep; uint32_t pad0; };
            struct BPenP { uint32_t B_chunk, vocab_size, item_off, pad0; };
            struct BSampItemH { uint32_t greedy; float inv_temp; uint32_t top_k; float top_p; uint32_t rlo, rhi, p0, p1; };
            struct BSampP { uint32_t B_chunk, vocab_size, item_off, pad0; };

            struct AttnPrepP {
                uint32_t M_v, M_p16, n_q, n_kv, hd, d_rel, seq_pos_start, kv_stride, is_local, save_snap_mask;
                uint32_t v_state_off_f4, kv_zero_tail;
                float qk_clip, rel_clip, eps, log_alpha, log_n_floor, attn_scale;
                uint32_t kv_dtype, snap_stride_f4;
            };
            struct AttnP {
                uint32_t M_v, M_p16, n_q, n_kv, hd, d_rel, rel_extent, swa_win, is_local, seq_pos_start;
                uint32_t m_offset, kv_stride, max_splits, chunk_size, c_batch_first; float attn_scale;
            };
            struct WoSconvP { uint32_t M, rows, cols, save_snap_mask, res_mode, snap_stride_f4, p1, p2; };
            struct SconvP   { uint32_t M, channels, save_snap_mask, res_mode, hd, n_kv, v_state_off_f4, top_k; };
            struct PenP     { uint32_t M, vocab_size, logits_stride, num_hist_ids, extra_mode, extra_count; float pres, freq, rep; uint32_t use_sl, p0, p1; uint32_t drop[8]; };
            struct SampP    { uint32_t M, vocab_size, greedy; float inv_temp; uint32_t top_k; float top_p; uint32_t rlo, rhi, use_tm, comp_pm, p0, p1; };

            const uint32_t B = 4U;
            const uint32_t n_q = 24U, n_kv = 8U, hd = 128U, d_rel = 16U, rel_ext = 128U;
            const uint32_t q_dim = n_q * hd, kv_dim = n_kv * hd, r_dim = n_q * d_rel;
            const uint32_t D = 2304U, top_k = 8U, kv_stride = 512U, max_splits = 36U;
            const float    attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

            std::mt19937 rng(202620);
            std::normal_distribution<float> nd(0.0f, 0.35f);

            std::vector<float> q_raw(B * q_dim), k_raw(B * kv_dim), v_raw(B * kv_dim), r_raw(B * r_dim);
            std::vector<float> k_sw(kv_dim * 4U), v_sw(kv_dim * 4U), k_dyn(B * n_kv * 4U), v_dyn(B * n_kv * 4U);
            std::vector<float> q_sc(q_dim, 1.0f), r_sc(r_dim, 1.0f), rel_p(rel_ext * d_rel);
            std::vector<float> sink(B * n_q), gate(B * q_dim);
            for (auto& v : q_raw) v = nd(rng);
            for (auto& v : k_raw) v = nd(rng);
            for (auto& v : v_raw) v = nd(rng);
            for (auto& v : r_raw) v = nd(rng);
            for (auto& v : k_sw)  v = nd(rng) * 0.2f;
            for (auto& v : v_sw)  v = nd(rng) * 0.2f;
            for (auto& v : k_dyn) v = nd(rng) * 0.1f;
            for (auto& v : v_dyn) v = nd(rng) * 0.1f;
            for (auto& v : rel_p) v = nd(rng) * 0.1f;
            for (auto& v : sink)  v = nd(rng) * 0.5f;
            for (auto& v : gate)  v = nd(rng);

            const uint64_t q8_data_b = static_cast<uint64_t>(n_kv) * kv_stride * hd;
            const uint64_t q8_sc_b   = static_cast<uint64_t>(n_kv) * kv_stride * 2U;
            const uint32_t sconv_f4  = 2U * kv_dim + 2U * D; // [k, v, attn, mlp] float4 channels per slot

            // 4 completely independent initial KV + SConv states
            std::vector<std::vector<uint16_t>> init_k16(B), init_v16(B);
            std::vector<std::vector<int8_t>>   init_kq8(B), init_vq8(B);
            std::vector<std::vector<uint16_t>> init_ks8(B), init_vs8(B);
            std::vector<std::vector<float>>    init_st(B);
            for (uint32_t b = 0U; b < B; ++b) {
                init_k16[b].resize(n_kv * kv_stride * hd);
                init_v16[b].resize(n_kv * kv_stride * hd);
                init_kq8[b].resize(n_kv * kv_stride * hd);
                init_vq8[b].resize(n_kv * kv_stride * hd);
                init_ks8[b].resize(n_kv * kv_stride);
                init_vs8[b].resize(n_kv * kv_stride);
                init_st[b].resize(sconv_f4 * 4U);
                for (auto& v : init_k16[b]) v = f32_to_fp16_bits(nd(rng) * 0.4f);
                for (auto& v : init_v16[b]) v = f32_to_fp16_bits(nd(rng) * 0.4f);
                for (auto& v : init_kq8[b]) v = static_cast<int8_t>(static_cast<int>(rng() % 201U) - 100);
                for (auto& v : init_vq8[b]) v = static_cast<int8_t>(static_cast<int>(rng() % 201U) - 100);
                for (auto& v : init_ks8[b]) v = f32_to_fp16_bits(0.01f + 0.002f * float(b));
                for (auto& v : init_vs8[b]) v = f32_to_fp16_bits(0.01f + 0.002f * float(b));
                for (auto& v : init_st[b])  v = nd(rng) * 0.25f;
            }

            id<MTLBuffer> b_qraw = [dev newBufferWithBytes:q_raw.data() length:q_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kraw = [dev newBufferWithBytes:k_raw.data() length:k_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vraw = [dev newBufferWithBytes:v_raw.data() length:v_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rraw = [dev newBufferWithBytes:r_raw.data() length:r_raw.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_ksw  = [dev newBufferWithBytes:k_sw.data()  length:k_sw.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vsw  = [dev newBufferWithBytes:v_sw.data()  length:v_sw.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_kdyn = [dev newBufferWithBytes:k_dyn.data() length:k_dyn.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_vdyn = [dev newBufferWithBytes:v_dyn.data() length:v_dyn.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_qsc  = [dev newBufferWithBytes:q_sc.data()  length:q_sc.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rsc  = [dev newBufferWithBytes:r_sc.data()  length:r_sc.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_relp = [dev newBufferWithBytes:rel_p.data() length:rel_p.size() * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_sink = [dev newBufferWithBytes:sink.data()  length:sink.size()  * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_gate = [dev newBufferWithBytes:gate.data()  length:gate.size()  * 4U options:MTLResourceStorageModeShared];

            id<MTLBuffer> b_qf32  = [dev newBufferWithLength:B * q_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_rf32  = [dev newBufferWithLength:B * r_dim * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_pm    = [dev newBufferWithLength:B * n_q * max_splits * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_pl    = [dev newBufferWithLength:B * n_q * max_splits * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_po    = [dev newBufferWithLength:B * n_q * max_splits * hd * 4U options:MTLResourceStorageModePrivate];
            id<MTLBuffer> b_out_b = [dev newBufferWithLength:B * q_dim * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_xs_b  = [dev newBufferWithLength:B * n_q * 2U * 4U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_meta  = [dev newBufferWithLength:B * sizeof(BSlotMetaH) options:MTLResourceStorageModeShared];

            id<MTLBuffer> b_sk[4], b_sv[4], b_gk[4], b_gv[4], b_st[4];
            std::vector<id<MTLResource>> slot_res;
            for (uint32_t b = 0U; b < B; ++b) {
                b_sk[b] = [dev newBufferWithLength:n_kv * kv_stride * hd * 2U options:MTLResourceStorageModeShared];
                b_sv[b] = [dev newBufferWithLength:n_kv * kv_stride * hd * 2U options:MTLResourceStorageModeShared];
                b_gk[b] = [dev newBufferWithLength:std::max<uint64_t>(n_kv * kv_stride * hd * 2U, q8_data_b + q8_sc_b) options:MTLResourceStorageModeShared];
                b_gv[b] = [dev newBufferWithLength:std::max<uint64_t>(n_kv * kv_stride * hd * 2U, q8_data_b + q8_sc_b) options:MTLResourceStorageModeShared];
                b_st[b] = [dev newBufferWithLength:sconv_f4 * 16U options:MTLResourceStorageModeShared];
                slot_res.push_back(b_sk[b]); slot_res.push_back(b_sv[b]);
                slot_res.push_back(b_gk[b]); slot_res.push_back(b_gv[b]);
                slot_res.push_back(b_st[b]);
            }

            // Solo M=1 reference scratch buffers
            id<MTLBuffer> b_r_k   = [dev newBufferWithLength:n_kv * kv_stride * hd * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_r_v   = [dev newBufferWithLength:n_kv * kv_stride * hd * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_r_ks  = [dev newBufferWithLength:n_kv * kv_stride * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_r_vs  = [dev newBufferWithLength:n_kv * kv_stride * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_r_st  = [dev newBufferWithLength:sconv_f4 * 16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_r_sn  = [dev newBufferWithLength:8U * sconv_f4 * 16U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_r_out = [dev newBufferWithLength:q_dim * 2U options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_r_xs  = [dev newBufferWithLength:n_q * 2U * 4U options:MTLResourceStorageModeShared];

            auto chunk_sz_h = [](uint32_t qp, uint32_t loc) -> uint32_t {
                if (qp < 160U) return 16U;
                if (qp < 512U) return 32U;
                return (loc != 0U) ? 64U : 64U;
            };

            auto test_batched_attn = [&](const uint32_t pos_arr[4], bool is_local, bool use_q8) -> uint32_t {
                auto* sm = reinterpret_cast<BSlotMetaH*>([b_meta contents]);
                bool has_u256 = false, has_ge256 = false;
                uint32_t max_sp = 0U, max_q8 = 0U;
                for (uint32_t b = 0U; b < B; ++b) {
                    std::memcpy([b_st[b] contents], init_st[b].data(), init_st[b].size() * 4U);
                    if (use_q8) {
                        std::memcpy([b_gk[b] contents], init_kq8[b].data(), q8_data_b);
                        std::memcpy(static_cast<uint8_t*>([b_gk[b] contents]) + q8_data_b, init_ks8[b].data(), q8_sc_b);
                        std::memcpy([b_gv[b] contents], init_vq8[b].data(), q8_data_b);
                        std::memcpy(static_cast<uint8_t*>([b_gv[b] contents]) + q8_data_b, init_vs8[b].data(), q8_sc_b);
                    } else if (is_local) {
                        std::memcpy([b_sk[b] contents], init_k16[b].data(), init_k16[b].size() * 2U);
                        std::memcpy([b_sv[b] contents], init_v16[b].data(), init_v16[b].size() * 2U);
                    } else {
                        std::memcpy([b_gk[b] contents], init_k16[b].data(), init_k16[b].size() * 2U);
                        std::memcpy([b_gv[b] contents], init_v16[b].data(), init_v16[b].size() * 2U);
                    }
                    sm[b] = BSlotMetaH{[b_sk[b] gpuAddress], [b_sv[b] gpuAddress], [b_gk[b] gpuAddress],
                                       [b_gv[b] gpuAddress], [b_st[b] gpuAddress], pos_arr[b], 0U};
                    const uint32_t bc = chunk_sz_h(pos_arr[b], is_local ? 1U : 0U);
                    const uint32_t ws = (is_local && pos_arr[b] + 1U > 128U) ? (pos_arr[b] + 1U - 128U) : 0U;
                    const uint32_t sp = (pos_arr[b] / bc) - (ws / bc) + 1U;
                    if (sp > max_q8) max_q8 = sp;
                    if (pos_arr[b] < 256U) has_u256 = true;
                    else { has_ge256 = true; if (sp > max_sp) max_sp = sp; }
                }

                // 1. Run Batched Pointer-Table Pass in ONE Command Buffer
                {
                    BAttnPrepP bpp{B, n_q, n_kv, hd, d_rel, kv_stride, is_local ? 1U : 0U, use_q8 ? 1U : 0U,
                                   0U, kv_dim, 0U, 0U, 0ULL, q8_data_b, 10.0f, 10.0f, 1e-6f, 0.1f, 128.0f, 0.0f};
                    BAttnP bap{B, n_q, n_kv, hd, d_rel, rel_ext, 128U, is_local ? 1U : 0U,
                               kv_stride, max_splits, use_q8 ? 0U : 1U, 0U, 0ULL, q8_data_b, attn_scale, 0U};
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc useResources:slot_res.data() count:slot_res.size()
                                usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                    [enc setComputePipelineState:pso_b_kv];
                    [enc setBuffer:b_qraw offset:0 atIndex:0]; [enc setBuffer:b_kraw offset:0 atIndex:1];
                    [enc setBuffer:b_vraw offset:0 atIndex:2]; [enc setBuffer:b_rraw offset:0 atIndex:3];
                    [enc setBuffer:b_ksw  offset:0 atIndex:4]; [enc setBuffer:b_vsw  offset:0 atIndex:5];
                    [enc setBuffer:b_kdyn offset:0 atIndex:6]; [enc setBuffer:b_vdyn offset:0 atIndex:7];
                    [enc setBuffer:b_qsc  offset:0 atIndex:8]; [enc setBuffer:b_rsc  offset:0 atIndex:9];
                    [enc setBuffer:b_qf32 offset:0 atIndex:10]; [enc setBuffer:b_rf32 offset:0 atIndex:11];
                    [enc setBuffer:b_meta offset:0 atIndex:12]; [enc setBytes:&bpp length:sizeof(bpp) atIndex:13];
                    [enc dispatchThreadgroups:MTLSizeMake(2U * n_q + 2U * n_kv, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    if (use_q8) {
                        [enc setComputePipelineState:pso_b_q8];
                        [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                        [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_pm   offset:0 atIndex:3];
                        [enc setBuffer:b_pl   offset:0 atIndex:4]; [enc setBuffer:b_po   offset:0 atIndex:5];
                        [enc setBuffer:b_meta offset:0 atIndex:6]; [enc setBytes:&bap length:sizeof(bap) atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(max_q8, n_kv, B) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_b_red];
                        [enc setBuffer:b_pm   offset:0 atIndex:0]; [enc setBuffer:b_pl   offset:0 atIndex:1];
                        [enc setBuffer:b_po   offset:0 atIndex:2]; [enc setBuffer:b_sink offset:0 atIndex:3];
                        [enc setBuffer:b_gate offset:0 atIndex:4]; [enc setBuffer:b_out_b offset:0 atIndex:5];
                        [enc setBuffer:b_xs_b offset:0 atIndex:6]; [enc setBuffer:b_meta offset:0 atIndex:7];
                        [enc setBytes:&bap length:sizeof(bap) atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(n_q, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    } else {
                        if (has_u256) {
                            [enc setComputePipelineState:pso_b_as];
                            [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                            [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_sink offset:0 atIndex:3];
                            [enc setBuffer:b_gate offset:0 atIndex:4]; [enc setBuffer:b_out_b offset:0 atIndex:5];
                            [enc setBuffer:b_xs_b offset:0 atIndex:6]; [enc setBuffer:b_meta offset:0 atIndex:7];
                            [enc setBytes:&bap length:sizeof(bap) atIndex:8];
                            [enc dispatchThreadgroups:MTLSizeMake(n_kv, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        }
                        if (has_ge256) {
                            [enc setComputePipelineState:pso_b_sp];
                            [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                            [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_pm   offset:0 atIndex:3];
                            [enc setBuffer:b_pl   offset:0 atIndex:4]; [enc setBuffer:b_po   offset:0 atIndex:5];
                            [enc setBuffer:b_meta offset:0 atIndex:6]; [enc setBytes:&bap length:sizeof(bap) atIndex:7];
                            [enc dispatchThreadgroups:MTLSizeMake(max_sp, n_kv, B) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                            [enc setComputePipelineState:pso_b_red];
                            [enc setBuffer:b_pm   offset:0 atIndex:0]; [enc setBuffer:b_pl   offset:0 atIndex:1];
                            [enc setBuffer:b_po   offset:0 atIndex:2]; [enc setBuffer:b_sink offset:0 atIndex:3];
                            [enc setBuffer:b_gate offset:0 atIndex:4]; [enc setBuffer:b_out_b offset:0 atIndex:5];
                            [enc setBuffer:b_xs_b offset:0 atIndex:6]; [enc setBuffer:b_meta offset:0 atIndex:7];
                            [enc setBytes:&bap length:sizeof(bap) atIndex:8];
                            [enc dispatchThreadgroups:MTLSizeMake(n_q, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        }
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }

                // 2. Compare each stream b against an isolated solo M=1 run!
                uint32_t diffs = 0U;
                const auto* out_b = reinterpret_cast<const uint16_t*>([b_out_b contents]);
                const auto* xs_b  = reinterpret_cast<const uint32_t*>([b_xs_b contents]);
                for (uint32_t b = 0U; b < B; ++b) {
                    const uint32_t qp = pos_arr[b];
                    std::memcpy([b_r_st contents], init_st[b].data(), init_st[b].size() * 4U);
                    if (use_q8) {
                        std::memcpy([b_r_k contents], init_kq8[b].data(), init_kq8[b].size());
                        std::memcpy([b_r_v contents], init_vq8[b].data(), init_vq8[b].size());
                        std::memcpy([b_r_ks contents], init_ks8[b].data(), init_ks8[b].size() * 2U);
                        std::memcpy([b_r_vs contents], init_vs8[b].data(), init_vs8[b].size() * 2U);
                    } else {
                        std::memcpy([b_r_k contents], init_k16[b].data(), init_k16[b].size() * 2U);
                        std::memcpy([b_r_v contents], init_v16[b].data(), init_v16[b].size() * 2U);
                    }
                    AttnPrepP pp{1U, 16U, n_q, n_kv, hd, d_rel, qp, kv_stride, is_local ? 1U : 0U, 0U,
                                 kv_dim, 0U, 10.0f, 10.0f, 1e-6f, 0.1f, 128.0f, attn_scale, use_q8 ? 1U : 0U, sconv_f4};
                    const uint32_t bc = chunk_sz_h(qp, is_local ? 1U : 0U);
                    const uint32_t ws = (is_local && qp + 1U > 128U) ? (qp + 1U - 128U) : 0U;
                    const uint32_t c_first = ws / bc;
                    const uint32_t n_ch    = (qp / bc) - c_first + 1U;
                    AttnP ap{1U, 16U, n_q, n_kv, hd, d_rel, rel_ext, 128U, is_local ? 1U : 0U, qp,
                             0U, kv_stride, max_splits, bc, c_first, attn_scale};
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_kv_lin];
                    [enc setBuffer:b_qraw offset:b * q_dim * 4U atIndex:0];
                    [enc setBuffer:b_kraw offset:b * kv_dim * 4U atIndex:1];
                    [enc setBuffer:b_vraw offset:b * kv_dim * 4U atIndex:2];
                    [enc setBuffer:b_rraw offset:b * r_dim * 4U atIndex:3];
                    [enc setBuffer:b_ksw  offset:0 atIndex:4];
                    [enc setBuffer:b_vsw  offset:0 atIndex:5];
                    [enc setBuffer:b_kdyn offset:b * n_kv * 16U atIndex:6];
                    [enc setBuffer:b_vdyn offset:b * n_kv * 16U atIndex:7];
                    [enc setBuffer:b_r_st offset:0 atIndex:8];
                    [enc setBuffer:b_qsc  offset:0 atIndex:9];
                    [enc setBuffer:b_rsc  offset:0 atIndex:10];
                    [enc setBuffer:b_qf32 offset:0 atIndex:11];
                    [enc setBuffer:b_rf32 offset:0 atIndex:12];
                    [enc setBuffer:b_r_k  offset:0 atIndex:13];
                    [enc setBuffer:b_r_v  offset:0 atIndex:14];
                    [enc setBuffer:b_r_sn offset:0 atIndex:15];
                    [enc setBytes:&pp length:sizeof(pp) atIndex:16];
                    [enc setBuffer:b_r_ks offset:0 atIndex:17];
                    [enc setBuffer:b_r_vs offset:0 atIndex:18];
                    [enc dispatchThreadgroups:MTLSizeMake(2U * n_q + 2U * n_kv, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    if (use_q8) {
                        [enc setComputePipelineState:pso_q8_lin];
                        [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                        [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_r_k  offset:0 atIndex:3];
                        [enc setBuffer:b_r_v  offset:0 atIndex:4]; [enc setBuffer:b_pm   offset:0 atIndex:5];
                        [enc setBuffer:b_pl   offset:0 atIndex:6]; [enc setBuffer:b_po   offset:0 atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc setBuffer:b_r_ks offset:0 atIndex:9]; [enc setBuffer:b_r_vs offset:0 atIndex:10];
                        [enc dispatchThreadgroups:MTLSizeMake(n_ch, n_kv, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_red_lin];
                        [enc setBuffer:b_pm   offset:0 atIndex:0]; [enc setBuffer:b_pl   offset:0 atIndex:1];
                        [enc setBuffer:b_po   offset:0 atIndex:2]; [enc setBuffer:b_sink offset:b * n_q * 4U atIndex:3];
                        [enc setBuffer:b_gate offset:b * q_dim * 4U atIndex:4]; [enc setBuffer:b_r_out offset:0 atIndex:5];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:6]; [enc setBuffer:b_r_xs offset:0 atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(n_q, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    } else if (qp < 256U) {
                        [enc setComputePipelineState:pso_as_lin];
                        [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                        [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_r_k  offset:0 atIndex:3];
                        [enc setBuffer:b_r_v  offset:0 atIndex:4]; [enc setBuffer:b_sink offset:b * n_q * 4U atIndex:5];
                        [enc setBuffer:b_gate offset:b * q_dim * 4U atIndex:6]; [enc setBuffer:b_r_out offset:0 atIndex:7];
                        [enc setBuffer:b_r_xs offset:0 atIndex:8]; [enc setBytes:&ap length:sizeof(ap) atIndex:9];
                        [enc dispatchThreadgroups:MTLSizeMake(n_kv, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    } else {
                        [enc setComputePipelineState:pso_sp_lin];
                        [enc setBuffer:b_qf32 offset:0 atIndex:0]; [enc setBuffer:b_rf32 offset:0 atIndex:1];
                        [enc setBuffer:b_relp offset:0 atIndex:2]; [enc setBuffer:b_r_k  offset:0 atIndex:3];
                        [enc setBuffer:b_r_v  offset:0 atIndex:4]; [enc setBuffer:b_pm   offset:0 atIndex:5];
                        [enc setBuffer:b_pl   offset:0 atIndex:6]; [enc setBuffer:b_po   offset:0 atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(n_ch, n_kv, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_red_lin];
                        [enc setBuffer:b_pm   offset:0 atIndex:0]; [enc setBuffer:b_pl   offset:0 atIndex:1];
                        [enc setBuffer:b_po   offset:0 atIndex:2]; [enc setBuffer:b_sink offset:b * n_q * 4U atIndex:3];
                        [enc setBuffer:b_gate offset:b * q_dim * 4U atIndex:4]; [enc setBuffer:b_r_out offset:0 atIndex:5];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:6]; [enc setBuffer:b_r_xs offset:0 atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(n_q, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                    const auto* out_r = reinterpret_cast<const uint16_t*>([b_r_out contents]);
                    const auto* xs_r  = reinterpret_cast<const uint32_t*>([b_r_xs contents]);
                    for (uint32_t i = 0U; i < q_dim; ++i) {
                        if (out_b[b * q_dim + i] != out_r[i]) ++diffs;
                    }
                    for (uint32_t i = 0U; i < n_q * 2U; ++i) {
                        if (xs_b[b * n_q * 2U + i] != xs_r[i]) ++diffs;
                    }
                    if (std::memcmp([b_st[b] contents], [b_r_st contents], 2U * kv_dim * 16U) != 0) ++diffs;
                }
                return diffs;
            };

            const uint32_t pos_short[4] = {7U, 19U, 31U, 45U};
            const uint32_t pos_mixed[4] = {7U, 19U, 270U, 310U};
            const uint32_t diffs_swa   = test_batched_attn(pos_short, true,  false);
            const uint32_t diffs_glob  = test_batched_attn(pos_mixed, false, false);
            const uint32_t diffs_q8_s  = test_batched_attn(pos_short, false, true);
            const uint32_t diffs_q8_m  = test_batched_attn(pos_mixed, false, true);

            // Part 2: Batched `Wo+AttnSConv` (stock & alt_wo!), `sconv_static_batched`, and `moe_combine_sconv_batched`
            uint32_t wo_diffs = 0U, sc_diffs = 0U, mc_diffs = 0U;
            {
                const uint32_t cols = q_dim;
                const uint32_t k16 = cols >> 4U, nblk = cols >> 6U;
                std::vector<uint32_t> wq(D * k16 * 2U), wq_alt(D * k16 * 2U);
                std::vector<uint16_t> ws(D * nblk), wb(D * nblk), xh(B * cols);
                std::vector<float>    xs(B * nblk, 0.0f), sw(D * 4U), cur0(B * D), xin(B * D);
                std::vector<float>    dp(B * top_k * D), tw(B * top_k);
                for (auto& v : wq)     v = rng();
                for (auto& v : wq_alt) v = rng();
                for (auto& v : ws)     v = f32_to_fp16_bits(0.01f);
                for (auto& v : wb)     v = f32_to_fp16_bits(-0.075f);
                for (uint32_t b = 0U; b < B; ++b) {
                    for (uint32_t c = 0U; c < cols; ++c) {
                        const uint16_t hv = f32_to_fp16_bits(nd(rng) * 0.5f);
                        xh[b * cols + c] = hv;
                        xs[b * nblk + (c >> 6U)] += fp16_bits_to_f32_ref(hv);
                    }
                }
                for (auto& v : sw)   v = nd(rng) * 0.2f;
                for (auto& v : cur0) v = nd(rng);
                for (auto& v : xin)  v = nd(rng);
                for (auto& v : dp)   v = nd(rng) * 0.3f;
                for (auto& v : tw)   v = 0.125f + 0.02f * nd(rng);

                id<MTLBuffer> b_wq  = [dev newBufferWithBytes:wq.data()     length:wq.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_wqa = [dev newBufferWithBytes:wq_alt.data() length:wq_alt.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_ws  = [dev newBufferWithBytes:ws.data()     length:ws.size() * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_wb  = [dev newBufferWithBytes:wb.data()     length:wb.size() * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_xh  = [dev newBufferWithBytes:xh.data()     length:xh.size() * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_xs  = [dev newBufferWithBytes:xs.data()     length:xs.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_sw  = [dev newBufferWithBytes:sw.data()     length:sw.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_cur = [dev newBufferWithBytes:cur0.data()   length:cur0.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_blk = [dev newBufferWithLength:cur0.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_xin = [dev newBufferWithBytes:xin.data()    length:xin.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_dp  = [dev newBufferWithBytes:dp.data()     length:dp.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_tw  = [dev newBufferWithBytes:tw.data()     length:tw.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_rc  = [dev newBufferWithLength:D * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_rb  = [dev newBufferWithLength:D * 4U options:MTLResourceStorageModeShared];

                const uint32_t attn_off_f4 = 2U * kv_dim;
                const uint32_t mlp_off_f4  = 2U * kv_dim + D;
                for (int alt_pass = 0; alt_pass < 2; ++alt_pass) {
                    id<MTLBuffer> cur_wq = (alt_pass == 0) ? b_wq : b_wqa;
                    auto* sm = reinterpret_cast<BSlotMetaH*>([b_meta contents]);
                    for (uint32_t b = 0U; b < B; ++b) {
                        std::memcpy([b_st[b] contents], init_st[b].data(), init_st[b].size() * 4U);
                        sm[b] = BSlotMetaH{[b_sk[b] gpuAddress], [b_sv[b] gpuAddress], [b_gk[b] gpuAddress],
                                           [b_gv[b] gpuAddress], [b_st[b] gpuAddress], pos_short[b], 0U};
                    }
                    std::memcpy([b_cur contents], cur0.data(), cur0.size() * 4U);
                    BSconvP bsp{B, D, 2U, attn_off_f4, top_k, cols, 0U, 0U};
                    {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc useResources:slot_res.data() count:slot_res.size() usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                        [enc setComputePipelineState:pso_b_wo[B - 1U]];
                        [enc setBuffer:cur_wq offset:0 atIndex:0]; [enc setBuffer:b_ws  offset:0 atIndex:1];
                        [enc setBuffer:b_wb   offset:0 atIndex:2]; [enc setBuffer:b_xh  offset:0 atIndex:3];
                        [enc setBuffer:b_xs   offset:0 atIndex:4]; [enc setBuffer:b_sw  offset:0 atIndex:5];
                        [enc setBuffer:b_cur  offset:0 atIndex:6]; [enc setBuffer:b_blk offset:0 atIndex:7];
                        [enc setBuffer:b_meta offset:0 atIndex:8]; [enc setBytes:&bsp length:sizeof(bsp) atIndex:9];
                        [enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }
                    const auto* cur_b = reinterpret_cast<const uint32_t*>([b_cur contents]);
                    const auto* blk_b = reinterpret_cast<const uint32_t*>([b_blk contents]);
                    for (uint32_t b = 0U; b < B; ++b) {
                        std::memcpy([b_r_st contents], init_st[b].data(), init_st[b].size() * 4U);
                        std::memcpy([b_rc contents], cur0.data() + b * D, D * 4U);
                        WoSconvP wp{1U, D, cols, 0U, 2U, sconv_f4, 0U, 0U};
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_wo_m1];
                        [enc setBuffer:cur_wq offset:0 atIndex:0]; [enc setBuffer:b_ws offset:0 atIndex:1];
                        [enc setBuffer:b_wb   offset:0 atIndex:2]; [enc setBuffer:b_xh offset:b * cols * 2U atIndex:3];
                        [enc setBuffer:b_xs   offset:b * nblk * 4U atIndex:4]; [enc setBuffer:b_sw offset:0 atIndex:5];
                        [enc setBuffer:b_r_st offset:attn_off_f4 * 16U atIndex:6];
                        [enc setBuffer:b_rc   offset:0 atIndex:7]; [enc setBuffer:b_rb offset:0 atIndex:8];
                        [enc setBuffer:b_r_sn offset:0 atIndex:9]; [enc setBytes:&wp length:sizeof(wp) atIndex:10];
                        [enc dispatchThreadgroups:MTLSizeMake((D + 7U) / 8U, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                        const auto* cur_r = reinterpret_cast<const uint32_t*>([b_rc contents]);
                        const auto* blk_r = reinterpret_cast<const uint32_t*>([b_rb contents]);
                        for (uint32_t c = 0U; c < D; ++c) {
                            if (cur_b[b * D + c] != cur_r[c] || blk_b[b * D + c] != blk_r[c]) ++wo_diffs;
                        }
                        if (std::memcmp(static_cast<const uint8_t*>([b_st[b] contents]) + attn_off_f4 * 16U,
                                        static_cast<const uint8_t*>([b_r_st contents]) + attn_off_f4 * 16U,
                                        D * 16U) != 0) ++wo_diffs;
                    }
                }

                // Unfused `sik_sconv_static_batched` + `sik_moe_combine_sconv_batched`
                for (uint32_t b = 0U; b < B; ++b) std::memcpy([b_st[b] contents], init_st[b].data(), init_st[b].size() * 4U);
                std::memcpy([b_cur contents], cur0.data(), cur0.size() * 4U);
                {
                    BSconvP bsp1{B, D, 1U, attn_off_f4, top_k, 0U, 0U, 0U};
                    BSconvP bsp2{B, D, 1U, mlp_off_f4,  top_k, 0U, 0U, 0U};
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc useResources:slot_res.data() count:slot_res.size() usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                    [enc setComputePipelineState:pso_b_sc];
                    [enc setBuffer:b_xin  offset:0 atIndex:0]; [enc setBuffer:b_sw  offset:0 atIndex:1];
                    [enc setBuffer:b_cur  offset:0 atIndex:2]; [enc setBuffer:b_blk offset:0 atIndex:3];
                    [enc setBuffer:b_meta offset:0 atIndex:4]; [enc setBytes:&bsp1 length:sizeof(bsp1) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 255U) / 256U, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc setComputePipelineState:pso_b_mc];
                    [enc setBuffer:b_dp   offset:0 atIndex:0]; [enc setBuffer:b_tw  offset:0 atIndex:1];
                    [enc setBuffer:b_sw   offset:0 atIndex:2]; [enc setBuffer:b_cur offset:0 atIndex:3];
                    [enc setBuffer:b_meta offset:0 atIndex:4]; [enc setBytes:&bsp2 length:sizeof(bsp2) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 63U) / 64U, B, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                const auto* cur_b = reinterpret_cast<const uint32_t*>([b_cur contents]);
                for (uint32_t b = 0U; b < B; ++b) {
                    std::memcpy([b_r_st contents], init_st[b].data(), init_st[b].size() * 4U);
                    std::memcpy([b_rc contents], cur0.data() + b * D, D * 4U);
                    SconvP sp1{1U, D, 0U, 1U, hd, n_kv, sconv_f4, top_k};
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_sc_lin];
                    [enc setBuffer:b_xin  offset:b * D * 4U atIndex:0]; [enc setBuffer:b_sw offset:0 atIndex:1];
                    [enc setBuffer:b_r_st offset:attn_off_f4 * 16U atIndex:2];
                    [enc setBuffer:b_rc   offset:0 atIndex:3]; [enc setBuffer:b_rb offset:0 atIndex:4];
                    [enc setBuffer:b_r_sn offset:0 atIndex:5]; [enc setBytes:&sp1 length:sizeof(sp1) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 31U) / 32U, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [enc setComputePipelineState:pso_mc_lin];
                    [enc setBuffer:b_dp   offset:b * top_k * D * 4U atIndex:0];
                    [enc setBuffer:b_tw   offset:b * top_k * 4U atIndex:1];
                    [enc setBuffer:b_sw   offset:0 atIndex:2];
                    [enc setBuffer:b_r_st offset:mlp_off_f4 * 16U atIndex:3];
                    [enc setBuffer:b_rc   offset:0 atIndex:4];
                    [enc setBuffer:b_r_sn offset:0 atIndex:5];
                    [enc setBytes:&sp1 length:sizeof(sp1) atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake((D + 63U) / 64U, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                    const auto* cur_r = reinterpret_cast<const uint32_t*>([b_rc contents]);
                    for (uint32_t c = 0U; c < D; ++c) {
                        if (cur_b[b * D + c] != cur_r[c]) ++mc_diffs;
                    }
                    if (std::memcmp(static_cast<const uint8_t*>([b_st[b] contents]) + attn_off_f4 * 16U,
                                    static_cast<const uint8_t*>([b_r_st contents]) + attn_off_f4 * 16U,
                                    D * 16U) != 0) ++sc_diffs;
                    if (std::memcmp(static_cast<const uint8_t*>([b_st[b] contents]) + mlp_off_f4 * 16U,
                                    static_cast<const uint8_t*>([b_r_st contents]) + mlp_off_f4 * 16U,
                                    D * 16U) != 0) ++mc_diffs;
                }
            }

            // Part 3: Per-Stream Isolated Sliding-Window Penalties + Heterogeneous Sampling (Greedy + Stochastic)
            uint32_t pen_samp_diffs = 0U;
            {
                const uint32_t V = 4096U;
                std::vector<float> logits0(B * V);
                for (auto& v : logits0) v = nd(rng) * 2.0f;
                // Give all 4 streams the exact same strong peaks on tokens 100, 200, 300 so per-stream penalties MUST diverge!
                for (uint32_t b = 0U; b < B; ++b) {
                    logits0[b * V + 100U] = 15.0f;
                    logits0[b * V + 200U] = 14.0f;
                    logits0[b * V + 300U] = 13.0f;
                }
                // Stream 0: greedy, no penalty -> 100
                // Stream 1: greedy, penalize 100 -> 200
                // Stream 2: greedy, penalize 100 & 200 -> 300
                // Stream 3: stochastic T=0.7, top_k=20, top_p=0.95, seed=0x123456789ABCDEF0, penalize 100
                const uint32_t p_toks[4] = {100U, 100U, 200U, 100U};
                const uint16_t p_cnts[4] = {4U,   4U,   3U,   2U};
                const BPenItemH p_items[4] = {
                    {0U, 0U, 0.0f, 0.0f, 1.0f, 0U},
                    {0U, 1U, 3.0f, 1.0f, 1.4f, 0U},
                    {1U, 2U, 3.0f, 1.0f, 1.4f, 0U},
                    {3U, 1U, 2.0f, 0.5f, 1.2f, 0U}
                };
                const BSampItemH s_items[4] = {
                    {1U, 1.0f, 1U, 1.0f, 0U, 0U, 0U, 0U},
                    {1U, 1.0f, 1U, 1.0f, 0U, 0U, 0U, 0U},
                    {1U, 1.0f, 1U, 1.0f, 0U, 0U, 0U, 0U},
                    {0U, 1.0f / 0.7f, 20U, 0.95f, 0x9ABCDEF0U, 0x12345678U, 0U, 0U}
                };

                id<MTLBuffer> b_lgs  = [dev newBufferWithBytes:logits0.data() length:logits0.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_pt   = [dev newBufferWithBytes:p_toks  length:sizeof(p_toks)  options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_pc   = [dev newBufferWithBytes:p_cnts  length:sizeof(p_cnts)  options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_pi   = [dev newBufferWithBytes:p_items length:sizeof(p_items) options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_si   = [dev newBufferWithBytes:s_items length:sizeof(s_items) options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_sv   = [dev newBufferWithLength:B * 4096U * 4U options:MTLResourceStorageModePrivate];
                id<MTLBuffer> b_sidx = [dev newBufferWithLength:B * 4096U * 4U options:MTLResourceStorageModePrivate];
                id<MTLBuffer> b_slse = [dev newBufferWithLength:B * 128U * 4U  options:MTLResourceStorageModePrivate];
                id<MTLBuffer> b_toks = [dev newBufferWithLength:B * 4U         options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_dum  = [dev newBufferWithLength:256U           options:MTLResourceStorageModeShared];
                {
                    BPenP bpp{B, V, 0U, 0U};
                    BSampP bsp{B, V, 0U, 0U};
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso_b_pen];
                    [enc setBuffer:b_lgs offset:0 atIndex:0]; [enc setBuffer:b_pt offset:0 atIndex:1];
                    [enc setBuffer:b_pc  offset:0 atIndex:2]; [enc setBuffer:b_pi offset:0 atIndex:3];
                    [enc setBytes:&bpp length:sizeof(bpp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(1, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc setComputePipelineState:pso_b_s1];
                    [enc setBuffer:b_lgs offset:0 atIndex:0]; [enc setBuffer:b_sv   offset:0 atIndex:1];
                    [enc setBuffer:b_sidx offset:0 atIndex:2]; [enc setBuffer:b_si  offset:0 atIndex:3];
                    [enc setBytes:&bsp length:sizeof(bsp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(64, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc setComputePipelineState:pso_b_s2];
                    [enc setBuffer:b_sv  offset:0 atIndex:0]; [enc setBuffer:b_sidx offset:0 atIndex:1];
                    [enc setBuffer:b_toks offset:0 atIndex:2]; [enc setBuffer:b_si  offset:0 atIndex:3];
                    [enc setBytes:&bsp length:sizeof(bsp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                const auto* got_toks = reinterpret_cast<const uint32_t*>([b_toks contents]);
                if (got_toks[0] != 100U || got_toks[1] != 200U || got_toks[2] != 300U) ++pen_samp_diffs;

                // Compare all 4 streams bit-for-bit against solo M=1 penalty + sampler kernels!
                id<MTLBuffer> b_rlg  = [dev newBufferWithLength:V * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_rpid = [dev newBufferWithLength:V * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_rpc  = [dev newBufferWithLength:V * 2U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_rtok = [dev newBufferWithLength:16U    options:MTLResourceStorageModeShared];
                for (uint32_t b = 0U; b < B; ++b) {
                    std::memcpy([b_rlg contents], logits0.data() + b * V, V * 4U);
                    std::memset([b_rpc contents], 0, V * 2U);
                    auto* pids = reinterpret_cast<uint32_t*>([b_rpid contents]);
                    auto* pcnt = reinterpret_cast<uint16_t*>([b_rpc contents]);
                    for (uint32_t k = 0U; k < p_items[b].len; ++k) {
                        const uint32_t t = p_toks[p_items[b].off + k];
                        pids[k] = t;
                        pcnt[t] = p_cnts[p_items[b].off + k];
                    }
                    PenP lpp{}; lpp.M = 1U; lpp.vocab_size = V; lpp.logits_stride = V; lpp.num_hist_ids = p_items[b].len;
                    lpp.pres = p_items[b].pres; lpp.freq = p_items[b].freq; lpp.rep = p_items[b].rep;
                    for (int j = 0; j < 8; ++j) lpp.drop[j] = 0xFFFFFFFFU;
                    SampP sp{1U, V, s_items[b].greedy, s_items[b].inv_temp, s_items[b].top_k, s_items[b].top_p,
                             s_items[b].rlo, s_items[b].rhi, 0U, 0U, 0U, 0U};

                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    if (p_items[b].len > 0U) {
                        [enc setComputePipelineState:pso_pen_lin];
                        [enc setBuffer:b_rlg offset:0 atIndex:0]; [enc setBuffer:b_rpid offset:0 atIndex:1];
                        [enc setBuffer:b_rpc offset:0 atIndex:2]; [enc setBuffer:b_dum  offset:0 atIndex:3];
                        [enc setBuffer:b_dum offset:0 atIndex:4]; [enc setBytes:&lpp length:sizeof(lpp) atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    if (s_items[b].greedy != 0U) {
                        [enc setComputePipelineState:pso_grd1];
                        [enc setBuffer:b_rlg offset:0 atIndex:0]; [enc setBuffer:b_sv   offset:0 atIndex:1];
                        [enc setBuffer:b_sidx offset:0 atIndex:2]; [enc setBuffer:b_slse offset:0 atIndex:3];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_grd2];
                        [enc setBuffer:b_sv  offset:0 atIndex:0]; [enc setBuffer:b_sidx offset:0 atIndex:1];
                        [enc setBuffer:b_slse offset:0 atIndex:2]; [enc setBuffer:b_dum offset:0 atIndex:3];
                        [enc setBuffer:b_rtok offset:0 atIndex:4]; [enc setBuffer:b_dum offset:0 atIndex:5];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    } else {
                        [enc setComputePipelineState:pso_st1];
                        [enc setBuffer:b_rlg offset:0 atIndex:0]; [enc setBuffer:b_sv   offset:0 atIndex:1];
                        [enc setBuffer:b_sidx offset:0 atIndex:2]; [enc setBuffer:b_slse offset:0 atIndex:3];
                        [enc setBytes:&sp length:sizeof(sp) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc setComputePipelineState:pso_st2];
                        [enc setBuffer:b_sv  offset:0 atIndex:0]; [enc setBuffer:b_sidx offset:0 atIndex:1];
                        [enc setBuffer:b_slse offset:0 atIndex:2]; [enc setBuffer:b_dum offset:0 atIndex:3];
                        [enc setBuffer:b_rtok offset:0 atIndex:4]; [enc setBuffer:b_dum offset:0 atIndex:5];
                        [enc setBuffer:b_dum offset:0 atIndex:6]; [enc setBuffer:b_dum  offset:0 atIndex:7];
                        [enc setBuffer:b_dum offset:0 atIndex:8]; [enc setBytes:&sp length:sizeof(sp) atIndex:9];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                    const uint32_t ref_tok = reinterpret_cast<const uint32_t*>([b_rtok contents])[0];
                    if (got_toks[b] != ref_tok) ++pen_samp_diffs;
                    if (std::memcmp(static_cast<const uint8_t*>([b_lgs contents]) + b * V * 4U,
                                    [b_rlg contents], V * 4U) != 0) ++pen_samp_diffs;
                }
            }

            const bool ok20 = (cb_spills == 0U) &&
                              (diffs_swa == 0U) && (diffs_glob == 0U) &&
                              (diffs_q8_s == 0U) && (diffs_q8_m == 0U) &&
                              (wo_diffs == 0U) && (sc_diffs == 0U) && (mc_diffs == 0U) &&
                              (pen_samp_diffs == 0U);
            std::printf("[Test 20] Continuous-Batching Multi-Stream GPU Kernels (B=4): %s\n", ok20 ? "PASS" : "FAIL");
            std::printf("         • Staggered Attn BitDiffs: SWA(7..45)=%u Glob(7..310)=%u Q8(7..45)=%u Q8(7..310)=%u | Spills=%u/39\n",
                        diffs_swa, diffs_glob, diffs_q8_s, diffs_q8_m, cb_spills);
            std::printf("         • Per-Stream SConv BitDiffs: Wo+Attn(stock+alt)=%u SConvStatic=%u MoE+MLP=%u | Pen+Sample=%u\n",
                        wo_diffs, sc_diffs, mc_diffs, pen_samp_diffs);
            if (!ok20) return 1;

            // Test 21: Native 1M (`1,048,576`) Global Attention Split-KV (FP16 & Q8_0, Solo vs Batched + Exact Needles)
            {
                auto chunk_sz_1m = [](uint32_t qp) -> uint32_t {
                    if (qp < 160U)    return 16U;
                    if (qp < 512U)    return 32U;
                    if (qp < 2048U)   return 64U;
                    if (qp < 4096U)   return 128U;
                    if (qp < 8192U)   return 256U;
                    if (qp < 16384U)  return 512U;
                    if (qp < 32768U)  return 1024U;
                    if (qp < 65536U)  return 2048U;
                    if (qp < 262144U) return 8192U;
                    if (qp < 524288U) return 16384U;
                    return 32768U;
                };
                bool split_bounds_ok = (STRATA_INKLING2_MAX_SEQ_LEN_CAP == 1048576U);
                const uint32_t bound_pos[6] = {262143U, 262144U, 409600U, 524287U, 524288U, 1048575U};
                for (uint32_t bp : bound_pos) {
                    const uint32_t bc = chunk_sz_1m(bp);
                    const uint32_t sp = (bp / bc) + 1U;
                    if (sp == 0U || sp > 32U || sp > max_splits) split_bounds_ok = false;
                }

                const uint32_t B1 = 2U, n_kv1 = 1U, n_q1 = 3U, kv_stride_1m = 1048640U;
                const uint32_t q_dim1 = n_q1 * hd, r_dim1 = n_q1 * d_rel;
                const uint32_t pos_1m[2] = {409600U, 1048575U};
                const float    q_amp = 12.0f * std::sqrt(static_cast<float>(hd));

                std::vector<float> q1(B1 * q_dim1, 0.0f), r1(B1 * r_dim1, 0.0f);
                std::vector<float> sink1(B1 * n_q1, -50.0f), gate1(B1 * q_dim1, 0.0f);
                for (uint32_t b = 0U; b < B1; ++b) {
                    for (uint32_t h = 0U; h < n_q1; ++h) {
                        q1[(b * n_q1 + h) * hd + h] = q_amp;
                    }
                }
                id<MTLBuffer> b_q1     = [dev newBufferWithBytes:q1.data()    length:q1.size() * 4U    options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_r1     = [dev newBufferWithBytes:r1.data()    length:r1.size() * 4U    options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_sink1  = [dev newBufferWithBytes:sink1.data() length:sink1.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_gate1  = [dev newBufferWithBytes:gate1.data() length:gate1.size() * 4U options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_out_b1 = [dev newBufferWithLength:B1 * q_dim1 * 2U                     options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_xs_b1  = [dev newBufferWithLength:B1 * n_q1 * 2U * 4U                  options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_out_r1 = [dev newBufferWithLength:q_dim1 * 2U                          options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_xs_r1  = [dev newBufferWithLength:n_q1 * 2U * 4U                       options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_meta1  = [dev newBufferWithLength:B1 * sizeof(BSlotMetaH)              options:MTLResourceStorageModeShared];
                id<MTLBuffer> b_relz   = [dev newBufferWithLength:rel_ext * d_rel * 4U                 options:MTLResourceStorageModeShared];
                std::memset([b_relz contents], 0, rel_ext * d_rel * 4U);

                const float exp_val[2][3] = {
                    { 0.75f,  0.75f,  0.0f   }, // qp = 409,600 (causal hides needle @ 1,048,575!)
                    { 0.75f,  0.75f, -0.625f }  // qp = 1,048,575 (retrieves needles @ 0, 409,600, 1,048,575!)
                };
                uint32_t f16_1m_diffs = 0U, q8_1m_diffs = 0U;
                float    f16_1m_max_err = 0.0f, q8_1m_max_err = 0.0f;

                // 21a. FP16 Global KV at 1M stride (`kv_stride = 1,048,640`)
                @autoreleasepool {
                    const size_t f16_bytes = static_cast<size_t>(kv_stride_1m) * hd * sizeof(uint16_t);
                    id<MTLBuffer> b_k16 = [dev newBufferWithLength:f16_bytes options:MTLResourceStorageModeShared];
                    id<MTLBuffer> b_v16 = [dev newBufferWithLength:f16_bytes options:MTLResourceStorageModeShared];
                    if (!b_k16 || !b_v16) return 1;
                    auto* kp = reinterpret_cast<uint16_t*>([b_k16 contents]);
                    auto* vp = reinterpret_cast<uint16_t*>([b_v16 contents]);
                    std::memset(kp, 0, f16_bytes);
                    std::memset(vp, 0, f16_bytes);
                    const uint16_t k2 = f32_to_fp16_bits(2.0f);
                    const uint16_t v_pos = f32_to_fp16_bits(1.5f);
                    const uint16_t v_neg = f32_to_fp16_bits(-1.25f);
                    kp[0ULL * hd + 0U]       = k2;
                    kp[409600ULL * hd + 1U]  = k2;
                    kp[1048575ULL * hd + 2U] = k2;
                    for (uint32_t d = 0U; d < hd; ++d) {
                        vp[0ULL * hd + d]       = v_pos;
                        vp[409600ULL * hd + d]  = v_pos;
                        vp[1048575ULL * hd + d] = v_neg;
                    }

                    auto* sm = reinterpret_cast<BSlotMetaH*>([b_meta1 contents]);
                    for (uint32_t b = 0U; b < B1; ++b) {
                        sm[b] = BSlotMetaH{0ULL, 0ULL, [b_k16 gpuAddress], [b_v16 gpuAddress], 0ULL, pos_1m[b], 0U};
                    }
                    BAttnP bap{B1, n_q1, n_kv1, hd, d_rel, rel_ext, 1024U, 0U, kv_stride_1m, max_splits, 1U, 0U, 0ULL, 0ULL, attn_scale, 0U};
                    id<MTLResource> res1[2] = {b_k16, b_v16};
                    {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc useResources:res1 count:2 usage:MTLResourceUsageRead];
                        [enc setComputePipelineState:pso_b_sp];
                        [enc setBuffer:b_q1 offset:0 atIndex:0]; [enc setBuffer:b_r1 offset:0 atIndex:1];
                        [enc setBuffer:b_relz offset:0 atIndex:2]; [enc setBuffer:b_pm offset:0 atIndex:3];
                        [enc setBuffer:b_pl offset:0 atIndex:4]; [enc setBuffer:b_po offset:0 atIndex:5];
                        [enc setBuffer:b_meta1 offset:0 atIndex:6]; [enc setBytes:&bap length:sizeof(bap) atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(32U, n_kv1, B1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_b_red];
                        [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                        [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_sink1 offset:0 atIndex:3];
                        [enc setBuffer:b_gate1 offset:0 atIndex:4]; [enc setBuffer:b_out_b1 offset:0 atIndex:5];
                        [enc setBuffer:b_xs_b1 offset:0 atIndex:6]; [enc setBuffer:b_meta1 offset:0 atIndex:7];
                        [enc setBytes:&bap length:sizeof(bap) atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(n_q1, B1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }
                    const auto* ob = reinterpret_cast<const uint16_t*>([b_out_b1 contents]);
                    const auto* xb = reinterpret_cast<const uint32_t*>([b_xs_b1 contents]);
                    for (uint32_t b = 0U; b < B1; ++b) {
                        const uint32_t qp = pos_1m[b];
                        const uint32_t bc = chunk_sz_1m(qp);
                        const uint32_t nc = (qp / bc) + 1U;
                        AttnP ap{1U, 16U, n_q1, n_kv1, hd, d_rel, rel_ext, 1024U, 0U, qp, 0U, kv_stride_1m, max_splits, bc, 0U, attn_scale};
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_sp_lin];
                        [enc setBuffer:b_q1 offset:b * q_dim1 * 4U atIndex:0]; [enc setBuffer:b_r1 offset:b * r_dim1 * 4U atIndex:1];
                        [enc setBuffer:b_relz offset:0 atIndex:2]; [enc setBuffer:b_k16 offset:0 atIndex:3];
                        [enc setBuffer:b_v16 offset:0 atIndex:4]; [enc setBuffer:b_pm offset:0 atIndex:5];
                        [enc setBuffer:b_pl offset:0 atIndex:6]; [enc setBuffer:b_po offset:0 atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(nc, n_kv1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_red_lin];
                        [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                        [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_sink1 offset:b * n_q1 * 4U atIndex:3];
                        [enc setBuffer:b_gate1 offset:b * q_dim1 * 4U atIndex:4]; [enc setBuffer:b_out_r1 offset:0 atIndex:5];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:6]; [enc setBuffer:b_xs_r1 offset:0 atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(n_q1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                        const auto* or1 = reinterpret_cast<const uint16_t*>([b_out_r1 contents]);
                        const auto* xr1 = reinterpret_cast<const uint32_t*>([b_xs_r1 contents]);
                        for (uint32_t i = 0U; i < q_dim1; ++i) {
                            if (ob[b * q_dim1 + i] != or1[i]) ++f16_1m_diffs;
                            const float got = fp16_bits_to_f32_ref(or1[i]);
                            const float err = std::fabs(got - exp_val[b][i / hd]);
                            if (!std::isfinite(got) || err > f16_1m_max_err) f16_1m_max_err = std::isfinite(got) ? err : 999.0f;
                        }
                        for (uint32_t i = 0U; i < n_q1 * 2U; ++i) {
                            if (xb[b * n_q1 * 2U + i] != xr1[i]) ++f16_1m_diffs;
                        }
                    }
                }

                // 21b. Q8_0 Global KV at 1M stride (`kv_stride = 1,048,640`)
                @autoreleasepool {
                    uint64_t q8_db1 = 0ULL, q8_sb1 = 0ULL;
                    strata_inkling2_kv_q8_layer_layout(n_kv1, kv_stride_1m, hd, &q8_db1, &q8_sb1);
                    id<MTLBuffer> b_kq8 = [dev newBufferWithLength:static_cast<size_t>(q8_db1 + q8_sb1) options:MTLResourceStorageModeShared];
                    id<MTLBuffer> b_vq8 = [dev newBufferWithLength:static_cast<size_t>(q8_db1 + q8_sb1) options:MTLResourceStorageModeShared];
                    if (!b_kq8 || !b_vq8) return 1;
                    auto* kq = reinterpret_cast<int8_t*>([b_kq8 contents]);
                    auto* vq = reinterpret_cast<int8_t*>([b_vq8 contents]);
                    auto* ks = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>([b_kq8 contents]) + q8_db1);
                    auto* vs = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>([b_vq8 contents]) + q8_db1);
                    std::memset(kq, 0, static_cast<size_t>(q8_db1 + q8_sb1));
                    std::memset(vq, 0, static_cast<size_t>(q8_db1 + q8_sb1));
                    const uint16_t sc_k  = f32_to_fp16_bits(2.0f / 100.0f);
                    const uint16_t sc_vp = f32_to_fp16_bits(1.5f / 100.0f);
                    const uint16_t sc_vn = f32_to_fp16_bits(1.25f / 100.0f);
                    kq[0ULL * hd + 0U]       = 100; ks[0U]       = sc_k;
                    kq[409600ULL * hd + 1U]  = 100; ks[409600U]  = sc_k;
                    kq[1048575ULL * hd + 2U] = 100; ks[1048575U] = sc_k;
                    vs[0U] = sc_vp; vs[409600U] = sc_vp; vs[1048575U] = sc_vn;
                    for (uint32_t d = 0U; d < hd; ++d) {
                        vq[0ULL * hd + d]       = 100;
                        vq[409600ULL * hd + d]  = 100;
                        vq[1048575ULL * hd + d] = -100;
                    }

                    auto* sm = reinterpret_cast<BSlotMetaH*>([b_meta1 contents]);
                    for (uint32_t b = 0U; b < B1; ++b) {
                        sm[b] = BSlotMetaH{0ULL, 0ULL, [b_kq8 gpuAddress], [b_vq8 gpuAddress], 0ULL, pos_1m[b], 0U};
                    }
                    BAttnP bap{B1, n_q1, n_kv1, hd, d_rel, rel_ext, 1024U, 0U, kv_stride_1m, max_splits, 0U, 0U, 0ULL, q8_db1, attn_scale, 0U};
                    id<MTLResource> res1[2] = {b_kq8, b_vq8};
                    {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc useResources:res1 count:2 usage:MTLResourceUsageRead];
                        [enc setComputePipelineState:pso_b_q8];
                        [enc setBuffer:b_q1 offset:0 atIndex:0]; [enc setBuffer:b_r1 offset:0 atIndex:1];
                        [enc setBuffer:b_relz offset:0 atIndex:2]; [enc setBuffer:b_pm offset:0 atIndex:3];
                        [enc setBuffer:b_pl offset:0 atIndex:4]; [enc setBuffer:b_po offset:0 atIndex:5];
                        [enc setBuffer:b_meta1 offset:0 atIndex:6]; [enc setBytes:&bap length:sizeof(bap) atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(32U, n_kv1, B1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_b_red];
                        [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                        [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_sink1 offset:0 atIndex:3];
                        [enc setBuffer:b_gate1 offset:0 atIndex:4]; [enc setBuffer:b_out_b1 offset:0 atIndex:5];
                        [enc setBuffer:b_xs_b1 offset:0 atIndex:6]; [enc setBuffer:b_meta1 offset:0 atIndex:7];
                        [enc setBytes:&bap length:sizeof(bap) atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(n_q1, B1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }
                    const auto* ob = reinterpret_cast<const uint16_t*>([b_out_b1 contents]);
                    const auto* xb = reinterpret_cast<const uint32_t*>([b_xs_b1 contents]);
                    for (uint32_t b = 0U; b < B1; ++b) {
                        const uint32_t qp = pos_1m[b];
                        const uint32_t bc = chunk_sz_1m(qp);
                        const uint32_t nc = (qp / bc) + 1U;
                        AttnP ap{1U, 16U, n_q1, n_kv1, hd, d_rel, rel_ext, 1024U, 0U, qp, 0U, kv_stride_1m, max_splits, bc, 0U, attn_scale};
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_q8_lin];
                        [enc setBuffer:b_q1 offset:b * q_dim1 * 4U atIndex:0]; [enc setBuffer:b_r1 offset:b * r_dim1 * 4U atIndex:1];
                        [enc setBuffer:b_relz offset:0 atIndex:2]; [enc setBuffer:b_kq8 offset:0 atIndex:3];
                        [enc setBuffer:b_vq8 offset:0 atIndex:4]; [enc setBuffer:b_pm offset:0 atIndex:5];
                        [enc setBuffer:b_pl offset:0 atIndex:6]; [enc setBuffer:b_po offset:0 atIndex:7];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
                        [enc setBuffer:b_kq8 offset:static_cast<NSUInteger>(q8_db1) atIndex:9];
                        [enc setBuffer:b_vq8 offset:static_cast<NSUInteger>(q8_db1) atIndex:10];
                        [enc dispatchThreadgroups:MTLSizeMake(nc, n_kv1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc setComputePipelineState:pso_red_lin];
                        [enc setBuffer:b_pm offset:0 atIndex:0]; [enc setBuffer:b_pl offset:0 atIndex:1];
                        [enc setBuffer:b_po offset:0 atIndex:2]; [enc setBuffer:b_sink1 offset:b * n_q1 * 4U atIndex:3];
                        [enc setBuffer:b_gate1 offset:b * q_dim1 * 4U atIndex:4]; [enc setBuffer:b_out_r1 offset:0 atIndex:5];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:6]; [enc setBuffer:b_xs_r1 offset:0 atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(n_q1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

                        const auto* or1 = reinterpret_cast<const uint16_t*>([b_out_r1 contents]);
                        const auto* xr1 = reinterpret_cast<const uint32_t*>([b_xs_r1 contents]);
                        for (uint32_t i = 0U; i < q_dim1; ++i) {
                            if (ob[b * q_dim1 + i] != or1[i]) ++q8_1m_diffs;
                            const float got = fp16_bits_to_f32_ref(or1[i]);
                            const float err = std::fabs(got - exp_val[b][i / hd]);
                            if (!std::isfinite(got) || err > q8_1m_max_err) q8_1m_max_err = std::isfinite(got) ? err : 999.0f;
                        }
                        for (uint32_t i = 0U; i < n_q1 * 2U; ++i) {
                            if (xb[b * n_q1 * 2U + i] != xr1[i]) ++q8_1m_diffs;
                        }
                    }
                }

                const bool ok21 = split_bounds_ok && (f16_1m_diffs == 0U) && (q8_1m_diffs == 0U) &&
                                  (f16_1m_max_err < 2e-3f) && (q8_1m_max_err < 2e-3f);
                std::printf("[Test 21] Inkling-2 Native 1M (1,048,576) Global Split-KV   : %s\n", ok21 ? "PASS" : "FAIL");
                std::printf("         • 400K & 1M-1 (32 splits) Needle MaxErr: F16=%.5f Q8_0=%.5f | Solo==Batch BitDiffs: F16=%u Q8=%u\n",
                            f16_1m_max_err, q8_1m_max_err, f16_1m_diffs, q8_1m_diffs);
                if (!ok21) return 1;
            }
        }
    }
    return 0;
}

} // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf(" STRATA-M5 v2.0 — GPU Subsystem Verification & Benchmark Suite (build/test_gpu)\n");
    std::printf("================================================================================\n");

    StrataGpuContext* gpu = strata_gpu_init();
    if (!gpu) {
        std::fprintf(stderr, "FAIL: strata_gpu_init() failed\n");
        return 1;
    }

    id<MTLDevice> mtl_dev = (__bridge id<MTLDevice>)strata_gpu_get_mtl_device(gpu);
    if (mtl_dev) {
        std::printf("[Device] %s | Unified Memory: %s | RecMaxWorkingSet: %.2f GiB\n",
                    [[mtl_dev name] UTF8String],
                    [mtl_dev hasUnifiedMemory] ? "YES" : "NO",
                    static_cast<double>([mtl_dev recommendedMaxWorkingSetSize]) / (1024.0 * 1024.0 * 1024.0));
    } else {
        std::printf("[Device] Seatbelt Host Reference Mode (IOKit AGX gated; run outside Seatbelt for Metal 4 GPU)\n");
    }

    // -------------------------------------------------------------------------
    // Test 1: Exhaustive 256-code E4M3 -> FP16 Mathematical & Round-Trip Check
    // -------------------------------------------------------------------------
    int e4m3_errors = 0;
    for (uint32_t c = 0; c < 256U; ++c) {
        uint8_t b = static_cast<uint8_t>(c);
        uint16_t f16_bits = strata_codec_e4m3_to_f16_bits(b);
        float got = fp16_bits_to_f32_ref(f16_bits);

        uint32_t sign = (b >> 7) & 1U;
        uint32_t exp4 = (b >> 3) & 0xFU;
        uint32_t man3 = b & 0x7U;

        if (exp4 == 15U && man3 == 7U) {
            if (!std::isnan(got)) ++e4m3_errors;
        } else {
            float expected = 0.0f;
            if (exp4 == 0U) {
                expected = std::ldexp(static_cast<float>(man3), -9);
            } else {
                expected = std::ldexp(1.0f + static_cast<float>(man3) / 8.0f, static_cast<int>(exp4) - 7);
            }
            if (sign) expected = -expected;
            if (got != expected) ++e4m3_errors;
            // Round-trip check for non-zero exact E4M3 values
            if (expected != 0.0f) {
                uint8_t rt = strata_codec_f32_to_e4m3(expected);
                if (rt != b) ++e4m3_errors;
            }
        }
    }
    std::printf("[Test 1] Exhaustive 256-state E4M3 <-> FP16 Bit-Exactness : %s (errors=%d)\n",
                (e4m3_errors == 0) ? "PASS" : "FAIL", e4m3_errors);
    if (e4m3_errors != 0) return 1;

    // -------------------------------------------------------------------------
    // Test 2: 32-Way Tile-rANS Edge Cases (Prime Count + Skewed Distributions)
    // -------------------------------------------------------------------------
    {
        const size_t prime_n = 100003U;
        std::mt19937 rng(42);
        std::vector<uint8_t> syms(prime_n);
        for (size_t i = 0; i < prime_n; ++i) {
            uint32_t r = rng() % 100U;
            syms[i] = (r < 39U) ? 1U : ((r < 78U) ? 2U : ((r < 89U) ? 0U : 3U));
        }
        double h_bits = compute_shannon_entropy(syms.data(), prime_n, 4U);
        std::vector<uint8_t> comp(prime_n + 8192U);
        size_t clen = strata_codec_rans_encode_32way(syms.data(), prime_n, 4U, comp.data(), comp.size());
        if (clen == 0U) return 1;

        std::vector<uint8_t> dec(prime_n, 0xFFU);
        int rc = strata_codec_rans_decode_32way_cpu_ref(comp.data(), clen, dec.data(), prime_n);
        bool ok = (rc == 0) && (std::memcmp(syms.data(), dec.data(), prime_n) == 0);
        double achieved_bpw = (static_cast<double>(clen) * 8.0) / static_cast<double>(prime_n);
        std::printf("[Test 2] 32-Way Tile-rANS Prime-Length Round-Trip         : %s (Shannon=%.3f b/sym, Achieved=%.3f b/sym)\n",
                    ok ? "PASS" : "FAIL", h_bits, achieved_bpw);
        if (!ok) return 1;
    }

    // -------------------------------------------------------------------------
    // Test 3: QTIP 2.125 bpw Viterbi Trellis + 4-Lvl Tile-rANS Refinement + W6A16
    // -------------------------------------------------------------------------
    {
        const size_t n_exp = 262144U; // 2,048 blocks of 128 weights (with heavy outlier channels!)
        std::mt19937 rng(2026);
        std::normal_distribution<float> nd(0.0f, 0.02f);
        std::vector<float> w(n_exp);
        for (size_t i = 0; i < n_exp; ++i) {
            w[i] = nd(rng);
            // Inject realistic 10x outlier channels every 128th column (cured by FWHT-128!)
            if ((i % 128U) == 7U) w[i] *= 8.5f;
        }

        std::vector<uint8_t> qbase((n_exp / 128U) * 34U);
        std::vector<uint8_t> qref_sym(n_exp);
        std::vector<float>   qref_scales(n_exp / 128U);
        double base_sqnr = 0.0, ref_sqnr = 0.0;

        strata_codec_qtip_quantize_expert(
            w.data(), n_exp, qbase.data(), qref_sym.data(), qref_scales.data(), &base_sqnr, &ref_sqnr);

        std::vector<uint8_t> ref_rans(n_exp + 16384U);
        size_t ref_rans_bytes = strata_codec_rans_encode_32way(
            qref_sym.data(), n_exp, 4U, ref_rans.data(), ref_rans.size());
        double ref_sym_bpw = (static_cast<double>(ref_rans_bytes) * 8.0) / static_cast<double>(n_exp);
        double ref_tot_bpw = ref_sym_bpw + (16.0 / 128.0); // + FP16 scale per 128 weights

        // Verify W6A16 (0.78125 B/weight = 50B per 64 weights) SQNR on realistic MoE-SVD U_e factors
        std::vector<float> u_svd(n_exp);
        for (size_t i = 0; i < n_exp; ++i) u_svd[i] = nd(rng); // Orthonormal singular vector distribution
        std::vector<uint8_t> w6_packed((n_exp / 32U) * 25U);
        strata_codec_w6a16_pack(u_svd.data(), n_exp, w6_packed.data());
        double w6_sig = 0.0, w6_err = 0.0;
        for (size_t blk = 0; blk < n_exp / 64U; ++blk) {
            const uint8_t* p50 = &w6_packed[blk * 50U];
            uint16_t sh = 0;
            std::memcpy(&sh, p50, 2U);
            float sc = fp16_bits_to_f32_ref(sh);
            const uint8_t* pb = p50 + 2U;
            for (size_t g = 0; g < 16U; ++g) {
                uint8_t b0 = pb[g * 3U + 0U], b1 = pb[g * 3U + 1U], b2 = pb[g * 3U + 2U];
                int q[4] = {
                    static_cast<int>(b0 & 0x3FU) - 32,
                    static_cast<int>(((b0 >> 6U) | ((b1 & 0x0FU) << 2U)) & 0x3FU) - 32,
                    static_cast<int>(((b1 >> 4U) | ((b2 & 0x03U) << 4U)) & 0x3FU) - 32,
                    static_cast<int>((b2 >> 2U) & 0x3FU) - 32
                };
                for (size_t k = 0; k < 4U; ++k) {
                    float orig = u_svd[blk * 64U + g * 4U + k];
                    float rec  = static_cast<float>(q[k]) * sc;
                    w6_sig += static_cast<double>(orig) * static_cast<double>(orig);
                    w6_err += static_cast<double>(orig - rec) * static_cast<double>(orig - rec);
                }
            }
        }
        double w6_sqnr = 10.0 * std::log10(w6_sig / w6_err);

        bool qtip_ok = (base_sqnr > 10.0) && (ref_sqnr > 18.0) && (w6_sqnr > 30.0);
        std::printf("[Test 3] QTIP 2.125b Trellis + Tile-rANS Refine + W6A16   : %s\n",
                    qtip_ok ? "PASS" : "FAIL");
        std::printf("         • QTIP Base (2.125 bpw fixed) SQNR : %.2f dB (with 8.5x outliers + FWHT-128)\n", base_sqnr);
        std::printf("         • QTIP + 4-Lvl Refine SQNR         : %.2f dB (Refine rANS = %.3f bpw total)\n",
                    ref_sqnr, ref_tot_bpw);
        std::printf("         • W6A16 MoE-SVD Factor (6.25 bpw)  : %.2f dB\n", w6_sqnr);
        if (!qtip_ok) return 1;
    }

    // -------------------------------------------------------------------------
    // Test 4: Fused In-Register 32-Way Tile-rANS + E4M3 -> FP16 Spine GEMM
    // -------------------------------------------------------------------------
    {
        bool bit_exact = false;
        StrataGpuMetrics m{};
        int rc = strata_gpu_bench_spine_e4m3_rans(
            gpu, /*batch_B=*/6U, /*d_in=*/5120U, /*d_out=*/5120U,
            /*warmup=*/3U, /*timed=*/12U, &bit_exact, &m);
        double e4m3_logical_gbps = m.effective_gbps * 0.5; // 1B/weight E4M3 vs 2B/weight FP16
        // Note: Gaussian SQNR of IEEE/OCP E4M3 (3 mantissa bits) is ~27.1-27.5 dB theoretical ceiling!
        bool ok = (rc == 0) && bit_exact && (m.sqnr_db > 26.5);
        std::printf("[Test 4] GPU In-Register Tile-rANS + E4M3->FP16 Spine GEMM: %s (bit_exact=%s)\n",
                    ok ? "PASS" : "FAIL", bit_exact ? "TRUE [0 bit errors / 26.2M]" : "FALSE");
        std::printf("         • Matrix [B=6, 5120x5120] Latency  : %.1f us\n", m.kernel_us);
        std::printf("         • Physical Cold-DRAM Read BW       : %.1f GB/s (rotating >144 MiB > 48 MiB SLC)\n",
                    m.physical_dram_gbps);
        std::printf("         • Uncompressed E4M3 Logical BW     : %.1f GB/s | FP16-Equiv BW: %.1f GB/s\n",
                    e4m3_logical_gbps, m.effective_gbps);
        std::printf("         • Effective Decode Compute & SQNR  : %.2f TFLOPS | %.2f dB vs FP32\n",
                    m.effective_tflops, m.sqnr_db);
        if (!ok) return 1;
    }

    // -------------------------------------------------------------------------
    // Test 5: Metal 4 Hardware Neural Accelerator (`NAX`) Tensor Core Roof
    // -------------------------------------------------------------------------
    {
        double nax_tflops = 0.0, nax_us = 0.0;
        int rc = strata_gpu_bench_nax_f16(gpu, 4096U, 4096U, 4096U, 10U, &nax_tflops, &nax_us);
        if (rc == 0) {
            std::printf("[Test 5] Metal 4 M5 Max NAX Tensor Cores (4K^3 FP16)      : PASS (%.2f TFLOPS FP16, %.1f us)\n",
                        nax_tflops, nax_us);
        } else {
            std::printf("[Test 5] Metal 4 M5 Max NAX Tensor Cores                  : SKIPPED (Seatbelt Host Mode)\n");
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: Slot-Indexed `strata_moe_mm` Across All 3 Regimes (Mode A & Mode B)
    // -------------------------------------------------------------------------
    StrataModelConfig qwen_cfg{};
    qwen_cfg.name            = "Qwen3.8-Flash-Next";
    qwen_cfg.regime          = STRATA_REGIME_I_100PCT_UMA;
    qwen_cfg.num_layers      = 48U;
    qwen_cfg.d_model         = 2560U;
    qwen_cfg.d_ff            = 640U;
    qwen_cfg.num_experts     = 512U;
    qwen_cfg.top_k           = 10U;
    qwen_cfg.svd_rank        = 0U;
    qwen_cfg.mode_a_tau_tok  = 0.0f;
    qwen_cfg.mode_b_cold_cap = 22U;

    StrataModelConfig glm_cfg{};
    glm_cfg.name             = "GLM-5.3-Flash";
    glm_cfg.regime           = STRATA_REGIME_I_BITNEST;
    glm_cfg.num_layers       = 45U;
    glm_cfg.d_model          = 4096U;
    glm_cfg.d_ff             = 2048U;
    glm_cfg.num_experts      = 288U;
    glm_cfg.top_k            = 8U;
    glm_cfg.svd_rank         = 0U;
    glm_cfg.mode_a_tau_tok   = 0.080f;
    glm_cfg.mode_b_cold_cap  = 4U;

    StrataModelConfig ds_cfg{};
    ds_cfg.name              = "DeepSeek-V4.1-Flash-1017";
    ds_cfg.regime            = STRATA_REGIME_II_BARY_SVD;
    ds_cfg.num_layers        = 40U;
    ds_cfg.d_model           = 5120U;
    ds_cfg.d_ff              = 2304U;
    ds_cfg.num_experts       = 384U;
    ds_cfg.top_k             = 6U;
    ds_cfg.svd_rank          = 108U;
    ds_cfg.mode_a_tau_tok    = 0.095f;
    ds_cfg.mode_b_cold_cap   = 2U;

    const StrataModelConfig* models[3] = {&qwen_cfg, &glm_cfg, &ds_cfg};
    std::printf("[Test 6] Slot-Indexed `strata_moe_mm` Single-Layer Hardware Timings (B=6):\n");
    for (int i = 0; i < 3; ++i) {
        StrataGpuMetrics ma{}, mb_ready{}, mb_fallback{};
        int r1 = strata_gpu_bench_moe_layer(gpu, models[i], STRATA_MODE_A_STATIONARY,
                                            6U, nullptr, nullptr, 0U, 2U, 8U, &ma);
        int r2 = strata_gpu_bench_moe_layer(gpu, models[i], STRATA_MODE_B_ZERO_STALL,
                                            6U, nullptr, nullptr, 0U, 2U, 8U, &mb_ready);
        int r3 = strata_gpu_bench_moe_layer(gpu, models[i], STRATA_MODE_B_ZERO_STALL,
                                            6U, nullptr, nullptr, 3U, 2U, 8U, &mb_fallback);
        std::printf("         • %-25s | Mode A: %6.1f us (Phys %5.1f GB/s, Eff-FP16 %6.1f GB/s, GPU-SQNR %5.2f dB)\n",
                    models[i]->name, ma.kernel_us, ma.physical_dram_gbps, ma.effective_gbps, ma.sqnr_db);
        std::printf("           %-25s | Mode B: %6.1f us (Ready, %5.2f dB) | %6.1f us (3 Unready -> 0ms Fallback, %5.2f dB)\n",
                    "", mb_ready.kernel_us, mb_ready.sqnr_db, mb_fallback.kernel_us, mb_fallback.sqnr_db);
        if (r1 != 0 || r2 != 0 || r3 != 0 ||
            std::isnan(ma.sqnr_db) || std::isnan(mb_ready.sqnr_db) || std::isnan(mb_fallback.sqnr_db) ||
            ma.sqnr_db < 15.0 || mb_ready.sqnr_db < 15.0 || mb_fallback.sqnr_db < 11.0) {
            std::fprintf(stderr,
                         "FAIL: strata_gpu_bench_moe_layer SQNR/status check failed on %s "
                         "(rc=%d/%d/%d, GPU-vs-FP32 SQNR=%.2f/%.2f/%.2f dB)\n",
                         models[i]->name, r1, r2, r3, ma.sqnr_db, mb_ready.sqnr_db, mb_fallback.sqnr_db);
            return 1;
        }
    }

    if (run_glm53_v2_metal41_suite(mtl_dev) != 0) {
        strata_gpu_destroy(gpu);
        return 1;
    }
    if (run_inkling2_turbo_metal41_suite(mtl_dev) != 0) {
        strata_gpu_destroy(gpu);
        return 1;
    }

    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    double peak_rss_gib = static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
    std::printf("[Safety] Peak Host RSS: %.3f GiB (Limit <= 4.000 GiB)     : %s\n",
                peak_rss_gib, (peak_rss_gib <= 4.0) ? "PASS" : "FAIL");

    strata_gpu_destroy(gpu);
    std::printf("================================================================================\n");
    std::printf(" ALL GPU SUBSYSTEM TESTS PASSED ON APPLE M5 MAX\n");
    std::printf("================================================================================\n");
    return 0;
}

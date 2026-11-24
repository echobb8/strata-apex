#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#if defined(__HAVE_TENSOR__)
#include <metal_tensor>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

// Inlined by src/model/strata_inkling2_vision.mm at runtime from include/strata_inkling2_vision.h:
// @@STRATA_INKLING2_VISION_HEADER@@
#ifndef STRATA_INKLING2_VISION_H
#include "../../include/strata_inkling2_vision.h"
#endif

using namespace metal;
#if defined(__HAVE_TENSOR__)
using namespace mpp::tensor_ops;
#endif

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 410)
template <typename T>
using sikv_strip_as = metal::remove_addrspace_t<T>;
#else
template <typename T>
using sikv_strip_as = T;
#endif

// ============================================================================
// 1. Numerically Exact Helpers (erf-GELU, FP16 clamp)
// ============================================================================

inline half sikv_clamp_f16(float v) {
    return half(clamp(v, -65504.0f, 65504.0f));
}

// Abramowitz & Stegun 7.1.26: |err| < 1.5e-7 across all x in (-inf, +inf)
inline float sikv_erf_f32(float x) {
    const float ax = fabs(x);
    const float t  = 1.0f / (1.0f + 0.3275911f * ax);
    const float p  = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                      - 0.284496736f) * t + 0.254829592f) * t;
    const float e  = 1.0f - p * exp(-ax * ax);
    return (x >= 0.0f) ? e : -e;
}

inline float sikv_gelu_erf(float x) {
    return 0.5f * x * (1.0f + sikv_erf_f32(x * 0.7071067811865475244f));
}

// ============================================================================
// 2. Metal 4.1 NAX FP16 GEMM (`Y = X @ W^T [+ bias] [-> erf-GELU]`)
//    Supports direct FP32 or FP16 output; zero TG staging, zero register spill.
// ============================================================================

#if defined(__HAVE_TENSOR__)
template <int BM, int BN, int SG>
inline void sikv_f16_gemm_nax_impl(
    device const half*       w_f16,
    device const half*       x_f16,
    device const float*      bias_f32,
    device half*             y_f16,
    device float*            y_f32,
    constant SikvGemmParams& p,
    uint2                    tgid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * uint(BN);
    if (m0 >= p.M_padded || n0 >= p.N) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(
        const_cast<device half*>(x_f16), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tW(
        const_cast<device half*>(w_f16), dextents<int, 2>(int(p.K), int(p.N)));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX = tX.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW = tW.template slice<dynamic_extent, BN>(0, int(n0));

    // Fast hardware direct store when no bias, no GELU, FP32 output, and BN-aligned
    if (p.has_bias == 0u && p.apply_gelu == 0u && p.out_f16 == 0u && (p.N % uint(BN)) == 0u) {
        tensor<device float, dextents<int, 2>, tensor_inline> tY(
            y_f32, dextents<int, 2>(int(p.N), int(p.M_padded)));
        auto sY = tY.template slice<BN, BM>(int(n0), int(m0));
        op.run(sX, sW, sY);
        return;
    }

    auto cT = op.template get_destination_cooperative_tensor<
        sikv_strip_as<decltype(sX)>, sikv_strip_as<decltype(sW)>, float>();
    op.run(sX, sW, cT);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx        = cT.get_multidimensional_index(i);
            const uint mg   = m0 + uint(idx[1]);
            const uint ng   = n0 + uint(idx[0]);
            if (mg < p.M_padded && ng < p.N) {
                const ulong out_idx = ulong(mg) * ulong(p.N) + ulong(ng);
                if (mg < p.M_valid) {
                    float val = cT[i];
                    if (p.has_bias != 0u)   val += bias_f32[ng];
                    if (p.apply_gelu != 0u) val  = sikv_gelu_erf(val);
                    if (p.out_f16 != 0u)    y_f16[out_idx] = sikv_clamp_f16(val);
                    else                    y_f32[out_idx] = val;
                } else {
                    if (p.out_f16 != 0u)    y_f16[out_idx] = 0.0h;
                    else                    y_f32[out_idx] = 0.0f;
                }
            }
        }
    }
}
#endif

kernel void sikv_f16_gemm_nax_64x128_sg4(
    device const half*       w_f16    [[buffer(0)]],
    device const half*       x_f16    [[buffer(1)]],
    device const float*      bias_f32 [[buffer(2)]],
    device half*             y_f16    [[buffer(3)]],
    device float*            y_f32    [[buffer(4)]],
    constant SikvGemmParams& p        [[buffer(5)]],
    uint2                    tgid     [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sikv_f16_gemm_nax_impl<64, 128, 4>(w_f16, x_f16, bias_f32, y_f16, y_f32, p, tgid);
#else
    (void)w_f16; (void)x_f16; (void)bias_f32; (void)y_f16; (void)y_f32; (void)p; (void)tgid;
#endif
}

kernel void sikv_f16_gemm_nax_32x64_sg2(
    device const half*       w_f16    [[buffer(0)]],
    device const half*       x_f16    [[buffer(1)]],
    device const float*      bias_f32 [[buffer(2)]],
    device half*             y_f16    [[buffer(3)]],
    device float*            y_f32    [[buffer(4)]],
    constant SikvGemmParams& p        [[buffer(5)]],
    uint2                    tgid     [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sikv_f16_gemm_nax_impl<32, 64, 2>(w_f16, x_f16, bias_f32, y_f16, y_f32, p, tgid);
#else
    (void)w_f16; (void)x_f16; (void)bias_f32; (void)y_f16; (void)y_f32; (void)p; (void)tgid;
#endif
}

// Fallback GEMM for non-NAX hardware (1 thread per output element, vectorized K loop)
kernel void sikv_f16_gemm_fallback(
    device const half*       w_f16    [[buffer(0)]],
    device const half*       x_f16    [[buffer(1)]],
    device const float*      bias_f32 [[buffer(2)]],
    device half*             y_f16    [[buffer(3)]],
    device float*            y_f32    [[buffer(4)]],
    constant SikvGemmParams& p        [[buffer(5)]],
    uint2                    gid      [[thread_position_in_grid]]
) {
    const uint ng = gid.x;
    const uint mg = gid.y;
    if (mg >= p.M_padded || ng >= p.N) return;
    const ulong out_idx = ulong(mg) * ulong(p.N) + ulong(ng);
    if (mg >= p.M_valid) {
        if (p.out_f16 != 0u) y_f16[out_idx] = 0.0h;
        else                 y_f32[out_idx] = 0.0f;
        return;
    }
    device const half* xrow = x_f16 + ulong(mg) * ulong(p.K);
    device const half* wrow = w_f16 + ulong(ng) * ulong(p.K);
    float acc = 0.0f;
    uint k = 0u;
    for (; k + 4u <= p.K; k += 4u) {
        acc += float(xrow[k + 0u]) * float(wrow[k + 0u])
             + float(xrow[k + 1u]) * float(wrow[k + 1u])
             + float(xrow[k + 2u]) * float(wrow[k + 2u])
             + float(xrow[k + 3u]) * float(wrow[k + 3u]);
    }
    for (; k < p.K; ++k) {
        acc += float(xrow[k]) * float(wrow[k]);
    }
    if (p.has_bias != 0u)   acc += bias_f32[ng];
    if (p.apply_gelu != 0u) acc  = sikv_gelu_erf(acc);
    if (p.out_f16 != 0u)    y_f16[out_idx] = sikv_clamp_f16(acc);
    else                    y_f32[out_idx] = acc;
}

// ============================================================================
// 3. Embeddings Assembly (CLS + Bilinear 32x32 PosEmb) + Pre-LayerNorm (`ln_pre`)
//    1 threadgroup (256 threads = 8 simdgroups) per token row `s in [0 .. S-1]`.
// ============================================================================

kernel void sikv_embed_assemble_ln_pre(
    device const float*             patch_emb_f32 [[buffer(0)]], // [num_patches, D]
    device const float*             class_emb     [[buffer(1)]], // [D]
    device const float*             cls_pos_emb   [[buffer(2)]], // [D]
    device const float*             pos_emb       [[buffer(3)]], // [32*32, D]
    device const float*             ln_pre_w      [[buffer(4)]], // [D]
    device const float*             ln_pre_b      [[buffer(5)]], // [D]
    device float*                   x_out_f32     [[buffer(6)]], // [S_padded, D]
    constant SikvEmbedAssembleParams& p           [[buffer(7)]],
    uint                            tgid          [[threadgroup_position_in_grid]],
    uint                            tid           [[thread_index_in_threadgroup]],
    uint                            sgid          [[simdgroup_index_in_threadgroup]],
    uint                            lane          [[thread_index_in_simdgroup]]
) {
    const uint s = tgid;
    if (s >= p.S_padded) return;

    device float* out_row = x_out_f32 + ulong(s) * ulong(p.D);
    if (s >= p.S) {
        for (uint c = tid; c < p.D; c += 256u) {
            out_row[c] = 0.0f;
        }
        return;
    }

    // Precompute bilinear indices/weights for patch rows (identical across all threads in TG)
    uint idx00 = 0u, idx01 = 0u, idx10 = 0u, idx11 = 0u;
    float w00 = 0.0f, w01 = 0.0f, w10 = 0.0f, w11 = 0.0f;
    if (s > 0u) {
        const uint patch_i = s - 1u;
        const uint in_col  = patch_i & 1u;
        const uint in_row  = (patch_i >> 1u) & 1u;
        const uint blk     = patch_i >> 2u;
        const uint blk_row = blk / p.grid_w;
        const uint blk_col = blk - blk_row * p.grid_w;
        const uint row     = blk_row * 2u + in_row;
        const uint col     = blk_col * 2u + in_col;
        const uint H_p     = p.grid_h * 2u;
        const uint W_p     = p.grid_w * 2u;
        const uint side    = p.posemb_grid;

        const float src_y = (float(row) + 0.5f) * float(side) / float(H_p) - 0.5f;
        const float src_x = (float(col) + 0.5f) * float(side) / float(W_p) - 0.5f;
        const float fy    = floor(src_y);
        const float fx    = floor(src_x);
        const int   y0    = clamp(int(fy), 0, int(side) - 1);
        const int   y1    = clamp(int(fy) + 1, 0, int(side) - 1);
        const int   x0    = clamp(int(fx), 0, int(side) - 1);
        const int   x1    = clamp(int(fx) + 1, 0, int(side) - 1);
        const float wy1   = src_y - fy;
        const float wy0   = 1.0f - wy1;
        const float wx1   = src_x - fx;
        const float wx0   = 1.0f - wx1;

        idx00 = uint(y0) * side + uint(x0);
        idx01 = uint(y0) * side + uint(x1);
        idx10 = uint(y1) * side + uint(x0);
        idx11 = uint(y1) * side + uint(x1);
        w00   = wy0 * wx0;
        w01   = wy0 * wx1;
        w10   = wy1 * wx0;
        w11   = wy1 * wx1;
    }

    // D = 1024, 256 threads => exactly 4 elements per thread (supports general D up to 1024 cleanly)
    float local_vals[4];
    float sum_v = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
        const uint c = tid + k * 256u;
        float v = 0.0f;
        if (c < p.D) {
            if (s == 0u) {
                v = class_emb[c] + cls_pos_emb[c];
            } else {
                const ulong p_off = ulong(s - 1u) * ulong(p.D) + ulong(c);
                const float pe = w00 * pos_emb[ulong(idx00) * ulong(p.D) + c]
                               + w01 * pos_emb[ulong(idx01) * ulong(p.D) + c]
                               + w10 * pos_emb[ulong(idx10) * ulong(p.D) + c]
                               + w11 * pos_emb[ulong(idx11) * ulong(p.D) + c];
                v = patch_emb_f32[p_off] + pe;
            }
            sum_v += v;
        }
        local_vals[k] = v;
    }

    threadgroup float tg_red[8];
    threadgroup float tg_mean;
    threadgroup float tg_inv_std;
    float sg_sum = simd_sum(sum_v);
    if (lane == 0u) tg_red[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float x = (lane < 8u) ? tg_red[lane] : 0.0f;
        float total = simd_sum(x);
        if (lane == 0u) tg_mean = total / float(p.D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = tg_mean;

    float sum_sq = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
        const uint c = tid + k * 256u;
        if (c < p.D) {
            const float d = local_vals[k] - mean;
            sum_sq += d * d;
        }
    }
    float sg_sq = simd_sum(sum_sq);
    if (lane == 0u) tg_red[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float x = (lane < 8u) ? tg_red[lane] : 0.0f;
        float total_sq = simd_sum(x);
        if (lane == 0u) tg_inv_std = rsqrt(total_sq / float(p.D) + p.ln_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_std = tg_inv_std;

    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
        const uint c = tid + k * 256u;
        if (c < p.D) {
            out_row[c] = (local_vals[k] - mean) * inv_std * ln_pre_w[c] + ln_pre_b[c];
        }
    }
}

// ============================================================================
// 4. Standalone LayerNorm (`F32 -> F16`, 256 threads/row)
// ============================================================================

kernel void sikv_layernorm_f32_to_f16(
    device const float*          x_f32   [[buffer(0)]],
    device const float*          ln_w    [[buffer(1)]],
    device const float*          ln_b    [[buffer(2)]],
    device half*                 out_f16 [[buffer(3)]],
    constant SikvLayerNormParams& p      [[buffer(4)]],
    uint                         tgid    [[threadgroup_position_in_grid]],
    uint                         tid     [[thread_index_in_threadgroup]],
    uint                         sgid    [[simdgroup_index_in_threadgroup]],
    uint                         lane    [[thread_index_in_simdgroup]]
) {
    const uint s = tgid;
    if (s >= p.S_padded) return;

    device half* out_row = out_f16 + ulong(s) * ulong(p.D);
    if (s >= p.S) {
        for (uint c = tid; c < p.D; c += 256u) {
            out_row[c] = 0.0h;
        }
        return;
    }

    device const float* in_row = x_f32 + ulong(s) * ulong(p.D);
    float local_vals[4];
    float sum_v = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
        const uint c = tid + k * 256u;
        float v = (c < p.D) ? in_row[c] : 0.0f;
        local_vals[k] = v;
        sum_v += v;
    }

    threadgroup float tg_red[8];
    threadgroup float tg_mean;
    threadgroup float tg_inv_std;
    float sg_sum = simd_sum(sum_v);
    if (lane == 0u) tg_red[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float x = (lane < 8u) ? tg_red[lane] : 0.0f;
        float total = simd_sum(x);
        if (lane == 0u) tg_mean = total / float(p.D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = tg_mean;

    float sum_sq = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
        const uint c = tid + k * 256u;
        if (c < p.D) {
            const float d = local_vals[k] - mean;
            sum_sq += d * d;
        }
    }
    float sg_sq = simd_sum(sum_sq);
    if (lane == 0u) tg_red[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float x = (lane < 8u) ? tg_red[lane] : 0.0f;
        float total_sq = simd_sum(x);
        if (lane == 0u) tg_inv_std = rsqrt(total_sq / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_std = tg_inv_std;

    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
        const uint c = tid + k * 256u;
        if (c < p.D) {
            const float normed = (local_vals[k] - mean) * inv_std * ln_w[c] + ln_b[c];
            out_row[c] = sikv_clamp_f16(normed);
        }
    }
}

// ============================================================================
// 5. QKV Head Split + 2D Axial Pairwise RoPE (`Q/K/V -> [H, S, 64]` F16)
//    Official Inkling-2-Mini 2D RoPE:
//      - CLS token (s == 0): (pos_h = 0, pos_w = 0) -> identity rotation
//      - Patch token (s >= 1): 1-indexed (row + 1, col + 1)
//      - Pair m in [0..15]  (dims 0..31):  pos_w * theta^(-2*m / 32)
//      - Pair m in [16..31] (dims 32..63): pos_h * theta^(-2*(m-16) / 32)
//      - Pairwise rotation: (x0*cos - x1*sin, x1*cos + x0*sin)
// ============================================================================

kernel void sikv_qkv_split_rope_f16(
    device const float*       qkv_f32 [[buffer(0)]], // [S, 3*D]
    device half*              q_f16   [[buffer(1)]], // [num_heads, S_padded, head_dim] (pre-scaled by 1/sqrt(head_dim))
    device half*              k_f16   [[buffer(2)]], // [num_heads, S_padded, head_dim]
    device half*              v_f16   [[buffer(3)]], // [num_heads, S_padded, head_dim]
    constant SikvQkvRopeParams& p     [[buffer(4)]],
    uint2                     gid     [[thread_position_in_grid]]
) {
    // gid.x in [0 .. num_heads * (head_dim/2) - 1] = [0 .. 511], gid.y = s in [0 .. S_padded - 1]
    const uint half_hd = p.head_dim >> 1u; // 32
    const uint total_pairs = p.num_heads * half_hd;
    if (gid.x >= total_pairs || gid.y >= p.S_padded) return;

    const uint s  = gid.y;
    const uint h  = gid.x / half_hd;
    const uint m  = gid.x - h * half_hd; // 0..31
    const uint d0 = m * 2u;
    const uint d1 = d0 + 1u;

    const ulong dst = (ulong(h) * ulong(p.S_padded) + ulong(s)) * ulong(p.head_dim);
    if (s >= p.S) {
        q_f16[dst + d0] = 0.0h;
        q_f16[dst + d1] = 0.0h;
        k_f16[dst + d0] = 0.0h;
        k_f16[dst + d1] = 0.0h;
        v_f16[dst + d0] = 0.0h;
        v_f16[dst + d1] = 0.0h;
        return;
    }

    uint pos_h = 0u;
    uint pos_w = 0u;
    if (s > 0u) {
        const uint patch_i = s - 1u;
        const uint in_col  = patch_i & 1u;
        const uint in_row  = (patch_i >> 1u) & 1u;
        const uint blk     = patch_i >> 2u;
        const uint blk_row = blk / p.grid_w;
        const uint blk_col = blk - blk_row * p.grid_w;
        pos_h = blk_row * 2u + in_row + 1u;
        pos_w = blk_col * 2u + in_col + 1u;
    }

    const uint spatial_half = half_hd >> 1u; // 16
    const uint freq_idx     = (m < spatial_half) ? m : (m - spatial_half);
    const float coord       = (m < spatial_half) ? float(pos_w) : float(pos_h);
    const float inv_freq    = pow(p.rope_theta, -float(2u * freq_idx) / float(p.head_dim >> 1u));
    const float angle       = coord * inv_freq;
    const float cs          = cos(angle);
    const float sn          = sin(angle);
    const float q_scale     = rsqrt(float(p.head_dim));

    const ulong row_base = ulong(s) * ulong(3u * p.D);
    const ulong q_off    = row_base + ulong(h * p.head_dim);
    const ulong k_off    = row_base + ulong(p.D + h * p.head_dim);
    const ulong v_off    = row_base + ulong(2u * p.D + h * p.head_dim);

    const float q0 = qkv_f32[q_off + d0];
    const float q1 = qkv_f32[q_off + d1];
    const float k0 = qkv_f32[k_off + d0];
    const float k1 = qkv_f32[k_off + d1];
    const float v0 = qkv_f32[v_off + d0];
    const float v1 = qkv_f32[v_off + d1];

    q_f16[dst + d0] = sikv_clamp_f16((q0 * cs - q1 * sn) * q_scale);
    q_f16[dst + d1] = sikv_clamp_f16((q1 * cs + q0 * sn) * q_scale);
    k_f16[dst + d0] = sikv_clamp_f16(k0 * cs - k1 * sn);
    k_f16[dst + d1] = sikv_clamp_f16(k1 * cs + k0 * sn);
    v_f16[dst + d0] = sikv_clamp_f16(v0);
    v_f16[dst + d1] = sikv_clamp_f16(v1);
}

// ============================================================================
// 6a. Metal 4.1 NAX Tensor-Core Bidirectional FlashAttention (`HEAD_DIM = 64`)
//     4 independent simdgroups (128 threads) per TG -> 64 query rows x 1 head, BK = 32.
//     12.75 KiB TG mem, 0 register spills (`maxTG = 1024`), O(S) memory.
// ============================================================================

kernel void sikv_flash_attn_hd64_nax(
    device const half*       q_f16   [[buffer(0)]], // [H, S_padded, 64] (pre-scaled by 1/8)
    device const half*       k_f16   [[buffer(1)]], // [H, S_padded, 64]
    device const half*       v_f16   [[buffer(2)]], // [H, S_padded, 64]
    device half*             out_f16 [[buffer(3)]], // [S_padded, H * 64]
    constant SikvAttnParams& p       [[buffer(4)]],
    uint2                    tgid    [[threadgroup_position_in_grid]],
    uint                     sgid    [[simdgroup_index_in_threadgroup]],
    uint                     lane    [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    const uint head = tgid.y;
    const uint m0   = tgid.x * 64u + sgid * 16u;
    if (head >= p.num_heads || m0 >= p.S_padded) return;

    if (m0 >= p.S) {
        const uint rows = min(16u, p.S_padded - m0);
        for (uint r = 0u; r < rows; ++r) {
            const ulong dst = ulong(m0 + r) * ulong(p.num_heads * 64u) + ulong(head * 64u);
            out_f16[dst + lane * 2u]      = 0.0h;
            out_f16[dst + lane * 2u + 1u] = 0.0h;
        }
        return;
    }

    threadgroup float S_tg[4][16][32];
    threadgroup half  P_tg[4][16][32];
    threadgroup float M_tg[4][16];
    threadgroup float L_tg[4][16];
    threadgroup float A_tg[4][16];

    if (lane < 16u) {
        M_tg[sgid][lane] = -INFINITY;
        L_tg[sgid][lane] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const ulong head_base = ulong(head) * ulong(p.S_padded) * 64ul;
    device half* qh = const_cast<device half*>(q_f16) + head_base;
    device half* kh = const_cast<device half*>(k_f16) + head_base;
    device half* vh = const_cast<device half*>(v_f16) + head_base;

    tensor<device half, dextents<int, 2>, tensor_inline> tQ(qh, dextents<int, 2>(64, int(p.S_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tK(kh, dextents<int, 2>(64, int(p.S_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tV(vh, dextents<int, 2>(64, int(p.S_padded)));
    tensor<threadgroup float, extents<int, 32, 16>, tensor_inline> tS(&S_tg[sgid][0][0], extents<int, 32, 16>());
    tensor<threadgroup half,  extents<int, 32, 16>, tensor_inline> tP(&P_tg[sgid][0][0], extents<int, 32, 16>());

    constexpr auto desc_qk = matmul2d_descriptor(
        16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;
    constexpr auto desc_pv = matmul2d_descriptor(
        16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    auto sQ  = tQ.slice<64, 16>(0, int(m0));
    auto qCT = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(sQ);

    auto sV_init = tV.slice<64, 32>(0, 0);
    auto oCT = op_pv.get_destination_cooperative_tensor<sikv_strip_as<decltype(tP)>, sikv_strip_as<decltype(sV_init)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT.get_capacity(); ++i) {
        if (oCT.is_valid_element(i)) oCT[i] = 0.0f;
    }

    for (uint kb = 0u; kb < p.S; kb += 32u) {
        auto sK  = tK.slice<64, 32>(0, int(kb));
        auto sCT = op_qk.get_destination_cooperative_tensor<sikv_strip_as<decltype(qCT)>, sikv_strip_as<decltype(sK)>, float>();
        op_qk.run(qCT, sK, sCT);
        sCT.store(tS);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const bool valid_k = (kb + lane) < p.S;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 16u; ++r) {
            const float sc   = valid_k ? S_tg[sgid][r][lane] : -INFINITY;
            const float rmax = simd_max(sc);
            const float mp   = M_tg[sgid][r];
            const float mn   = max(mp, rmax);
            const float al   = (mp == -INFINITY) ? 0.0f : exp(mp - mn);
            const float pv   = valid_k ? exp(sc - mn) : 0.0f;
            const float ps   = simd_sum(pv);
            P_tg[sgid][r][lane] = half(pv);
            if (lane == 0u) {
                M_tg[sgid][r] = mn;
                L_tg[sgid][r] = fma(L_tg[sgid][r], al, ps);
                A_tg[sgid][r] = al;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT.get_capacity(); ++i) {
            if (oCT.is_valid_element(i)) {
                auto idx = oCT.get_multidimensional_index(i);
                oCT[i] *= A_tg[sgid][uint(idx[1])];
            }
        }
        auto sV = tV.slice<64, 32>(0, int(kb));
        op_pv.run(tP, sV, oCT);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT.get_capacity(); ++i) {
        if (oCT.is_valid_element(i)) {
            auto idx = oCT.get_multidimensional_index(i);
            const uint r  = uint(idx[1]);
            const uint c  = uint(idx[0]);
            const uint mg = m0 + r;
            if (mg < p.S_padded) {
                const ulong dst = ulong(mg) * ulong(p.num_heads * 64u) + ulong(head * 64u) + ulong(c);
                if (mg < p.S) {
                    const float inv_l = (L_tg[sgid][r] > 0.0f) ? (1.0f / L_tg[sgid][r]) : 0.0f;
                    out_f16[dst] = sikv_clamp_f16(oCT[i] * inv_l);
                } else {
                    out_f16[dst] = 0.0h;
                }
            }
        }
    }
#else
    (void)q_f16; (void)k_f16; (void)v_f16; (void)out_f16; (void)p; (void)tgid; (void)sgid; (void)lane;
#endif
}

// ============================================================================
// 6b. Fallback Bidirectional Tiled FlashAttention (`HEAD_DIM = 64`, non-NAX)
//     4 simdgroups (128 threads) per TG -> 4 query rows x 1 head, BK = 32.
// ============================================================================

kernel void sikv_flash_attn_hd64(
    device const half*     q_f16   [[buffer(0)]], // [H, S_padded, 64] (pre-scaled by 1/8)
    device const half*     k_f16   [[buffer(1)]], // [H, S_padded, 64]
    device const half*     v_f16   [[buffer(2)]], // [H, S_padded, 64]
    device half*           out_f16 [[buffer(3)]], // [S_padded, H * 64]
    constant SikvAttnParams& p     [[buffer(4)]],
    uint2                  tgid    [[threadgroup_position_in_grid]],
    uint                   tid     [[thread_index_in_threadgroup]],
    uint                   sgid    [[simdgroup_index_in_threadgroup]],
    uint                   lane    [[thread_index_in_simdgroup]]
) {
    const uint head  = tgid.y;
    const uint q_idx = tgid.x * 4u + sgid;
    if (head >= p.num_heads) return;

    const uint d0 = lane * 2u;
    const uint d1 = d0 + 1u;

    const ulong head_base = ulong(head) * ulong(p.S_padded) * 64ul;
    const bool valid_q    = (q_idx < p.S);

    float q0 = 0.0f;
    float q1 = 0.0f;
    if (valid_q) {
        const ulong q_off = head_base + ulong(q_idx) * 64ul;
        q0 = float(q_f16[q_off + d0]);
        q1 = float(q_f16[q_off + d1]);
    }

    float o0  = 0.0f;
    float o1  = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;

    threadgroup half K_tile[32][64];
    threadgroup half V_tile[32][64];

    for (uint kv0 = 0u; kv0 < p.S; kv0 += 32u) {
        const uint tile_len = min(32u, p.S - kv0);

        const uint load_row = tid >> 2u;         // 0..31
        const uint load_col = (tid & 3u) * 16u;  // 0, 16, 32, 48
        if (load_row < tile_len) {
            const ulong kv_off = head_base + ulong(kv0 + load_row) * 64ul + ulong(load_col);
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 16u; ++i) {
                K_tile[load_row][load_col + i] = k_f16[kv_off + i];
                V_tile[load_row][load_col + i] = v_f16[kv_off + i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_q) {
            float s_lane = -INFINITY;
            for (uint k = 0u; k < tile_len; ++k) {
                const float prod = q0 * float(K_tile[k][d0]) + q1 * float(K_tile[k][d1]);
                const float sc   = simd_sum(prod);
                if (lane == k) s_lane = sc;
            }

            const float m_tile = simd_max(s_lane);
            const float m_new  = max(m_i, m_tile);
            const float alpha  = (m_i == -INFINITY) ? 0.0f : exp(m_i - m_new);
            const float w_lane = (lane < tile_len) ? exp(s_lane - m_new) : 0.0f;
            const float l_tile = simd_sum(w_lane);

            o0 *= alpha;
            o1 *= alpha;
            l_i = l_i * alpha + l_tile;
            m_i = m_new;

            for (uint k = 0u; k < tile_len; ++k) {
                const float wk = simd_broadcast(w_lane, ushort(k));
                o0 = fma(wk, float(V_tile[k][d0]), o0);
                o1 = fma(wk, float(V_tile[k][d1]), o1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q_idx < p.S_padded) {
        const ulong dst = ulong(q_idx) * ulong(p.num_heads * 64u) + ulong(head * 64u);
        if (valid_q) {
            const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            out_f16[dst + d0] = sikv_clamp_f16(o0 * inv_l);
            out_f16[dst + d1] = sikv_clamp_f16(o1 * inv_l);
        } else {
            out_f16[dst + d0] = 0.0h;
            out_f16[dst + d1] = 0.0h;
        }
    }
}

// ============================================================================
// 7. Fused LayerScale Residual Update + Optional Next LayerNorm (`F32 -> F16`)
//    `x_f32[s, c] += gamma[c] * y_f32[s, c]`; if `do_ln`: emit `LN(x_f32)` to `out_f16`.
// ============================================================================

kernel void sikv_res_scale_ln(
    device float*                 x_f32    [[buffer(0)]], // [S_padded, D] (in/out)
    device const float*           y_f32    [[buffer(1)]], // [S_padded, D]
    device const float*           gamma    [[buffer(2)]], // [D]
    device const float*           next_lnw [[buffer(3)]], // [D] (used if do_ln)
    device const float*           next_lnb [[buffer(4)]], // [D] (used if do_ln)
    device half*                  out_f16  [[buffer(5)]], // [S_padded, D] (used if do_ln)
    constant SikvResScaleLnParams& p       [[buffer(6)]],
    uint                          tgid     [[threadgroup_position_in_grid]],
    uint                          tid      [[thread_index_in_threadgroup]],
    uint                          sgid     [[simdgroup_index_in_threadgroup]],
    uint                          lane     [[thread_index_in_simdgroup]]
) {
    const uint s = tgid;
    if (s >= p.S_padded) return;

    if (s >= p.S) {
        if (p.do_ln != 0u) {
            device half* out_row = out_f16 + ulong(s) * ulong(p.D);
            for (uint c = tid; c < p.D; c += 256u) {
                out_row[c] = 0.0h;
            }
        }
        return;
    }

    device float*       x_row = x_f32 + ulong(s) * ulong(p.D);
    device const float* y_row = y_f32 + ulong(s) * ulong(p.D);

    float local_vals[4];
    float sum_v = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
        const uint c = tid + k * 256u;
        float v = 0.0f;
        if (c < p.D) {
            v = fma(gamma[c], y_row[c], x_row[c]);
            x_row[c] = v;
            sum_v += v;
        }
        local_vals[k] = v;
    }

    if (p.do_ln == 0u) return;

    threadgroup float tg_red[8];
    threadgroup float tg_mean;
    threadgroup float tg_inv_std;
    float sg_sum = simd_sum(sum_v);
    if (lane == 0u) tg_red[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float x = (lane < 8u) ? tg_red[lane] : 0.0f;
        float total = simd_sum(x);
        if (lane == 0u) tg_mean = total / float(p.D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = tg_mean;

    float sum_sq = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
        const uint c = tid + k * 256u;
        if (c < p.D) {
            const float d = local_vals[k] - mean;
            sum_sq += d * d;
        }
    }
    float sg_sq = simd_sum(sum_sq);
    if (lane == 0u) tg_red[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float x = (lane < 8u) ? tg_red[lane] : 0.0f;
        float total_sq = simd_sum(x);
        if (lane == 0u) tg_inv_std = rsqrt(total_sq / float(p.D) + p.ln_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_std = tg_inv_std;

    device half* out_row = out_f16 + ulong(s) * ulong(p.D);
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 4u; ++k) {
        const uint c = tid + k * 256u;
        if (c < p.D) {
            const float normed = (local_vals[k] - mean) * inv_std * next_lnw[c] + next_lnb[c];
            out_row[c] = sikv_clamp_f16(normed);
        }
    }
}

// ============================================================================
// 8. Drop CLS + 2x2 Spatial Merge Cast (`x_f32[1..4*N_vis, 1024] -> [N_vis_padded, 4096]` F16)
//    Patches are already stored in block-major `(br, bc, in_r, in_c)` order,
//    so dropping row 0 produces contiguous `[TL, TR, BL, BR]` merged tokens!
// ============================================================================

kernel void sikv_drop_cls_f32_to_f16(
    device const float* x_f32        [[buffer(0)]], // [1 + 4*N_vis, D=1024]
    device half*        merged_f16   [[buffer(1)]], // [N_vis_padded, 4*D=4096]
    constant uint4&     dims         [[buffer(2)]], // (N_vis, N_vis_padded, merged_dim=4096, D=1024)
    uint2               gid          [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint tok = gid.y;
    const uint N_vis        = dims.x;
    const uint N_vis_padded = dims.y;
    const uint merged_dim   = dims.z;
    const uint D            = dims.w;
    if (col >= merged_dim || tok >= N_vis_padded) return;

    const ulong dst = ulong(tok) * ulong(merged_dim) + ulong(col);
    if (tok >= N_vis) {
        merged_f16[dst] = 0.0h;
        return;
    }
    // Skip token 0 (CLS): offset by `D` floats in `x_f32`
    const ulong src = ulong(D) + ulong(tok) * ulong(merged_dim) + ulong(col);
    merged_f16[dst] = sikv_clamp_f16(x_f32[src]);
}

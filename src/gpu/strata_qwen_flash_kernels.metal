#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#if defined(__HAVE_TENSOR__)
#include <metal_tensor>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

// Inlined by strata_qwen_flash_gpu.mm at runtime from ../../include/strata_qwen_flash.h:
// @@STRATA_QWEN_FLASH_HEADER@@
#ifndef STRATA_QWEN_FLASH_H
#include "../../include/strata_qwen_flash.h"
#endif

using namespace metal;
#if defined(__HAVE_TENSOR__)
using namespace mpp::tensor_ops;
#endif

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 410)
template <typename T>
using sqf_strip_as = metal::remove_addrspace_t<T>;
#else
template <typename T>
using sqf_strip_as = T;
#endif

// Metal Function Constant for Exact-M / Max-B Zero-Branch PSO Specialization (eliminates unused `case 6..8` register bloat!)
constant uint SQF_CONST_M [[function_constant(0)]];

/* ============================================================================
 * 1. Numerically Safe IEEE-754 Helpers & Lossless 4-Bit Affine Unpack
 * ============================================================================ */

inline half sqf_clamp_f16(float x) {
    return half(clamp(x, -65504.0f, 65504.0f));
}

inline half4 sqf_clamp_f16x4(float4 x) {
    return half4(clamp(x, float4(-65504.0f), float4(65504.0f)));
}

inline float sqf_silu(float x) {
    return x / (1.0f + exp(-x));
}

inline float sqf_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float sqf_softplus(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

// Non-negative float -> monotonically order-preserving uint32
inline uint sqf_nonneg_f32_to_u32(float x) {
    float clean = (x > 0.0f) ? x : 0.0f;
    return as_type<uint>(clean);
}

// Lossless IEEE-754 binary16 trick: `0x6400 | nib` == `1024.0h + half(nib)` for nib in 0..15
inline void sqf_unpack_u32_nibbles_h4(uint w, thread half4& lo, thread half4& hi) {
    ushort4 n_lo(ushort(w & 0xFu),
                 ushort((w >> 4u) & 0xFu),
                 ushort((w >> 8u) & 0xFu),
                 ushort((w >> 12u) & 0xFu));
    ushort4 n_hi(ushort((w >> 16u) & 0xFu),
                 ushort((w >> 20u) & 0xFu),
                 ushort((w >> 24u) & 0xFu),
                 ushort((w >> 28u) & 0xFu));
    lo = as_type<half4>(ushort4(0x6400u) | n_lo) - half4(1024.0h);
    hi = as_type<half4>(ushort4(0x6400u) | n_hi) - half4(1024.0h);
}

// Dual-rate Apple M5 FP16 `half4` pairwise FMA + FP32 accumulation (safe up to |x|=2183, 100% bit-exact K>0 == K=0!)
inline float sqf_dot_h4x4(
    half4 n0, half4 n1, half4 n2, half4 n3,
    half4 x0, half4 x1, half4 x2, half4 x3
) {
    const half4  h0 = fma(n0, x0, n1 * x1);
    const half4  h1 = fma(n2, x2, n3 * x3);
    const float4 st = float4(h0) + float4(h1);
    return (st.x + st.y) + (st.z + st.w);
}

inline float sqf_sum_h4x4(half4 x0, half4 x1, half4 x2, half4 x3) {
    const float4 xs4 = float4((x0 + x1) + (x2 + x3));
    return (xs4.x + xs4.y) + (xs4.z + xs4.w);
}

/* ============================================================================
 * 2. Zero-Fill, Buffer Copy, F32->F16 Pad & 4-Bit Affine Embedding Lookup
 * ============================================================================ */

kernel void sqf_zero_u32(
    device uint4*  buf      [[buffer(0)]],
    constant uint& count_u4 [[buffer(1)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid < count_u4) {
        buf[gid] = uint4(0u);
    }
}

kernel void sqf_copy_u32(
    device const uint4* src      [[buffer(0)]],
    device uint4*       dst      [[buffer(1)]],
    constant uint&      count_u4 [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid < count_u4) {
        dst[gid] = src[gid];
    }
}

struct SqfPadCastParams {
    uint M_valid;
    uint M_padded;
    uint D;
    uint pad;
};

kernel void sqf_f32_to_f16_pad(
    device const float*       src_f32 [[buffer(0)]], // [M_valid, D]
    device half*              dst_f16 [[buffer(1)]], // [M_padded, D]
    constant SqfPadCastParams& p      [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]       // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_padded) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    const ulong off4 = ulong(m) * d4_cnt + idx4;
    if (m >= p.M_valid) {
        reinterpret_cast<device half4*>(dst_f16)[off4] = half4(0.0h);
        return;
    }
    float4 v = reinterpret_cast<device const float4*>(src_f32)[off4];
    reinterpret_cast<device half4*>(dst_f16)[off4] = sqf_clamp_f16x4(v);
}

struct SqfEmbedParams {
    uint M;
    uint cols;
    uint vocab_size;
    uint hc_tile_count; // 4 for main model entry (`[M, 4, cols]`), 1 for single stream (`[M, cols]`)
};

kernel void sqf_embed_lookup_q4(
    device const uint4*   weight_u4 [[buffer(0)]], // [vocab_size, cols / 32]
    device const half*    scales    [[buffer(1)]], // [vocab_size, cols / 64]
    device const half*    biases    [[buffer(2)]], // [vocab_size, cols / 64]
    device const uint*    token_ids [[buffer(3)]], // [M]
    device float*         out_f32   [[buffer(4)]], // [M, hc_tile_count * cols]
    constant SqfEmbedParams& p      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]      // 0..255
) {
    if (tgid >= p.M) return;
    const uint u4_per_row  = p.cols >> 5u;
    const uint grp_per_row = p.cols >> 6u;
    const uint tok         = min(token_ids[tgid], p.vocab_size - 1u);

    for (uint u = tid; u < u4_per_row; u += 256u) {
        const uint  grp = u >> 1u;
        const uint4 q   = weight_u4[ulong(tok) * u4_per_row + u];
        const float sc  = float(scales[ulong(tok) * grp_per_row + grp]);
        const float bi  = float(biases[ulong(tok) * grp_per_row + grp]);

        half4 n0, n1, n2, n3, n4, n5, n6, n7;
        sqf_unpack_u32_nibbles_h4(q.x, n0, n1);
        sqf_unpack_u32_nibbles_h4(q.y, n2, n3);
        sqf_unpack_u32_nibbles_h4(q.z, n4, n5);
        sqf_unpack_u32_nibbles_h4(q.w, n6, n7);

        float4 v0 = fma(float4(n0), float4(sc), float4(bi));
        float4 v1 = fma(float4(n1), float4(sc), float4(bi));
        float4 v2 = fma(float4(n2), float4(sc), float4(bi));
        float4 v3 = fma(float4(n3), float4(sc), float4(bi));
        float4 v4 = fma(float4(n4), float4(sc), float4(bi));
        float4 v5 = fma(float4(n5), float4(sc), float4(bi));
        float4 v6 = fma(float4(n6), float4(sc), float4(bi));
        float4 v7 = fma(float4(n7), float4(sc), float4(bi));

        for (uint s = 0u; s < p.hc_tile_count; ++s) {
            device float4* dst = reinterpret_cast<device float4*>(
                out_f32 + (ulong(tgid) * p.hc_tile_count + s) * p.cols + u * 32u);
            dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
            dst[4] = v4; dst[5] = v5; dst[6] = v6; dst[7] = v7;
        }
    }
}

/* ============================================================================
 * 3. Multi-M (`M = 1 .. 8`) Coalesced Q4 & FP16 GEMV Suite
 *    - `uint2` (`16` weights/lane = `512` weights/iter) -> 100% lane occupancy
 *      on `D = 2560` (`5` iters) and `cols = 6144` (`12` iters)!
 *    - Unpacks weights ONCE across `m = 0 .. M - 1` (`1x` DRAM read for MTP
 *      speculative verify `M = 2..5`) with 100% bit-identical per-token math!
 * ============================================================================ */

#define SQF_OUT_F16     0u
#define SQF_OUT_F32     1u
#define SQF_OUT_RES_F32 2u

inline void sqf_q4_gemv4_rows_batch(
    device const uint2* w_u2,   // [rows_padded, cols / 16]
    device const half*  scales, // [rows_padded, cols / 64]
    device const half*  biases, // [rows_padded, cols / 64]
    device const half*  x_in,   // [M, cols]
    uint M,
    uint row0,
    uint rows_logical,
    uint cols,
    uint lane,
    thread float4 (&out_m)[8]
) {
    const uint u2_per_row  = cols >> 4u;
    const uint grp_per_row = cols >> 6u;

    for (uint m = 0u; m < M; ++m) {
        out_m[m] = float4(0.0f);
    }

    const uint r0 = min(row0 + 0u, rows_logical - 1u);
    const uint r1 = min(row0 + 1u, rows_logical - 1u);
    const uint r2 = min(row0 + 2u, rows_logical - 1u);
    const uint r3 = min(row0 + 3u, rows_logical - 1u);

    device const uint2* r0_w = w_u2 + ulong(r0) * u2_per_row;
    device const uint2* r1_w = w_u2 + ulong(r1) * u2_per_row;
    device const uint2* r2_w = w_u2 + ulong(r2) * u2_per_row;
    device const uint2* r3_w = w_u2 + ulong(r3) * u2_per_row;

    device const half* r0_s = scales + ulong(r0) * grp_per_row;
    device const half* r1_s = scales + ulong(r1) * grp_per_row;
    device const half* r2_s = scales + ulong(r2) * grp_per_row;
    device const half* r3_s = scales + ulong(r3) * grp_per_row;

    device const half* r0_b = biases + ulong(r0) * grp_per_row;
    device const half* r1_b = biases + ulong(r1) * grp_per_row;
    device const half* r2_b = biases + ulong(r2) * grp_per_row;
    device const half* r3_b = biases + ulong(r3) * grp_per_row;

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint grp_idx = u2_idx >> 2u; // 4 * 16 = 64 = exact group_size!

        const uint2 q0 = r0_w[u2_idx];
        const uint2 q1 = r1_w[u2_idx];
        const uint2 q2 = r2_w[u2_idx];
        const uint2 q3 = r3_w[u2_idx];

        const float4 sc = float4(float(r0_s[grp_idx]), float(r1_s[grp_idx]), float(r2_s[grp_idx]), float(r3_s[grp_idx]));
        const float4 bi = float4(float(r0_b[grp_idx]), float(r1_b[grp_idx]), float(r2_b[grp_idx]), float(r3_b[grp_idx]));

        half4 n00, n01, n02, n03;
        half4 n10, n11, n12, n13;
        half4 n20, n21, n22, n23;
        half4 n30, n31, n32, n33;
        sqf_unpack_u32_nibbles_h4(q0.x, n00, n01); sqf_unpack_u32_nibbles_h4(q0.y, n02, n03);
        sqf_unpack_u32_nibbles_h4(q1.x, n10, n11); sqf_unpack_u32_nibbles_h4(q1.y, n12, n13);
        sqf_unpack_u32_nibbles_h4(q2.x, n20, n21); sqf_unpack_u32_nibbles_h4(q2.y, n22, n23);
        sqf_unpack_u32_nibbles_h4(q3.x, n30, n31); sqf_unpack_u32_nibbles_h4(q3.y, n32, n33);

        for (uint m = 0u; m < M; ++m) {
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + u2_idx * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float xs = sqf_sum_h4x4(x0, x1, x2, x3);

            out_m[m].x = fma(sc.x, sqf_dot_h4x4(n00, n01, n02, n03, x0, x1, x2, x3), fma(bi.x, xs, out_m[m].x));
            out_m[m].y = fma(sc.y, sqf_dot_h4x4(n10, n11, n12, n13, x0, x1, x2, x3), fma(bi.y, xs, out_m[m].y));
            out_m[m].z = fma(sc.z, sqf_dot_h4x4(n20, n21, n22, n23, x0, x1, x2, x3), fma(bi.z, xs, out_m[m].z));
            out_m[m].w = fma(sc.w, sqf_dot_h4x4(n30, n31, n32, n33, x0, x1, x2, x3), fma(bi.w, xs, out_m[m].w));
        }
    }

    for (uint m = 0u; m < M; ++m) {
        out_m[m] = float4(simd_sum(out_m[m].x), simd_sum(out_m[m].y), simd_sum(out_m[m].z), simd_sum(out_m[m].w));
    }
}

template <uint M_FIX>
inline void sqf_q4_gemv2_rows_fixed(
    device const uint2* w_u2,
    device const half*  scales,
    device const half*  biases,
    device const half*  x_in,
    uint row0,
    uint rows_logical,
    uint cols,
    uint lane,
    thread float2 (&out_m)[M_FIX]
) {
    const uint u2_per_row  = cols >> 4u;
    const uint grp_per_row = cols >> 6u;

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        out_m[m] = float2(0.0f);
    }

    const uint r0 = min(row0 + 0u, rows_logical - 1u);
    const uint r1 = min(row0 + 1u, rows_logical - 1u);

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 q0      = w_u2[ulong(r0) * u2_per_row + u2_idx];
        const uint2 q1      = w_u2[ulong(r1) * u2_per_row + u2_idx];

        const float2 sc = float2(float(scales[ulong(r0) * grp_per_row + grp_idx]),
                                 float(scales[ulong(r1) * grp_per_row + grp_idx]));
        const float2 bi = float2(float(biases[ulong(r0) * grp_per_row + grp_idx]),
                                 float(biases[ulong(r1) * grp_per_row + grp_idx]));

        half4 n00, n01, n02, n03;
        half4 n10, n11, n12, n13;
        sqf_unpack_u32_nibbles_h4(q0.x, n00, n01); sqf_unpack_u32_nibbles_h4(q0.y, n02, n03);
        sqf_unpack_u32_nibbles_h4(q1.x, n10, n11); sqf_unpack_u32_nibbles_h4(q1.y, n12, n13);

        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + u2_idx * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float xs = sqf_sum_h4x4(x0, x1, x2, x3);

            out_m[m].x = fma(sc.x, sqf_dot_h4x4(n00, n01, n02, n03, x0, x1, x2, x3), fma(bi.x, xs, out_m[m].x));
            out_m[m].y = fma(sc.y, sqf_dot_h4x4(n10, n11, n12, n13, x0, x1, x2, x3), fma(bi.y, xs, out_m[m].y));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        out_m[m] = float2(simd_sum(out_m[m].x), simd_sum(out_m[m].y));
    }
}

template <uint M_FIX>
inline void sqf_q4_gemv2_pair_fixed(
    device const uint2* w_u2,
    device const half*  scales,
    device const half*  biases,
    device const half*  x_in,
    uint r0,
    uint r1,
    uint cols,
    uint lane,
    thread float2 (&out_m)[M_FIX]
) {
    const uint u2_per_row  = cols >> 4u;
    const uint grp_per_row = cols >> 6u;

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        out_m[m] = float2(0.0f);
    }

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 q0      = w_u2[ulong(r0) * u2_per_row + u2_idx];
        const uint2 q1      = w_u2[ulong(r1) * u2_per_row + u2_idx];

        const float2 sc = float2(float(scales[ulong(r0) * grp_per_row + grp_idx]),
                                 float(scales[ulong(r1) * grp_per_row + grp_idx]));
        const float2 bi = float2(float(biases[ulong(r0) * grp_per_row + grp_idx]),
                                 float(biases[ulong(r1) * grp_per_row + grp_idx]));

        half4 n00, n01, n02, n03;
        half4 n10, n11, n12, n13;
        sqf_unpack_u32_nibbles_h4(q0.x, n00, n01); sqf_unpack_u32_nibbles_h4(q0.y, n02, n03);
        sqf_unpack_u32_nibbles_h4(q1.x, n10, n11); sqf_unpack_u32_nibbles_h4(q1.y, n12, n13);

        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + u2_idx * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float xs = sqf_sum_h4x4(x0, x1, x2, x3);

            out_m[m].x = fma(sc.x, sqf_dot_h4x4(n00, n01, n02, n03, x0, x1, x2, x3), fma(bi.x, xs, out_m[m].x));
            out_m[m].y = fma(sc.y, sqf_dot_h4x4(n10, n11, n12, n13, x0, x1, x2, x3), fma(bi.y, xs, out_m[m].y));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        out_m[m] = float2(simd_sum(out_m[m].x), simd_sum(out_m[m].y));
    }
}

template <uint M_FIX>
inline void sqf_store_gemv2(
    device half* y_f16,
    device float* y_f32,
    uint row0,
    uint rows,
    uint out_mode,
    thread const float2 (&res)[M_FIX]
) {
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const ulong base = ulong(m) * rows + row0;
        if (row0 + 2u <= rows) {
            if (out_mode == SQF_OUT_F16) {
                y_f16[base + 0u] = sqf_clamp_f16(res[m].x);
                y_f16[base + 1u] = sqf_clamp_f16(res[m].y);
            } else if (out_mode == SQF_OUT_F32) {
                *reinterpret_cast<device float2*>(y_f32 + base) = res[m];
            } else {
                device float2* dst = reinterpret_cast<device float2*>(y_f32 + base);
                *dst = *dst + res[m];
            }
        } else {
            if (out_mode == SQF_OUT_F16)      y_f16[base] = sqf_clamp_f16(res[m].x);
            else if (out_mode == SQF_OUT_F32) y_f32[base] = res[m].x;
            else                              y_f32[base] += res[m].x;
        }
    }
}

struct SqfGemvBatchParams {
    uint M;
    uint rows;
    uint cols;
    uint out_mode; // SQF_OUT_*
};

// Zero-spill `2` rows/SG (`8` rows/TG) Q4 GEMV — specialized per exact `M = 1..8` via `SQF_CONST_M`!
kernel void sqf_q4_gemv_batch(
    device const uint2*          w_u2   [[buffer(0)]],
    device const half*           scales [[buffer(1)]],
    device const half*           biases [[buffer(2)]],
    device const half*           x_in   [[buffer(3)]], // [M, cols]
    device half*                 y_f16  [[buffer(4)]], // [M, rows]
    device float*                y_f32  [[buffer(5)]], // [M, rows]
    constant SqfGemvBatchParams& p      [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],      // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 4u + sgid) * 2u;
    if (row0 >= p.rows) return;

    #define SQF_Q4_CASE(MF) { \
        float2 res[MF]; \
        sqf_q4_gemv2_rows_fixed<MF>(w_u2, scales, biases, x_in, row0, p.rows, p.cols, lane, res); \
        if (lane == 0u) sqf_store_gemv2<MF>(y_f16, y_f32, row0, p.rows, p.out_mode, res); \
        break; \
    }
    switch (SQF_CONST_M) {
        case 1u: SQF_Q4_CASE(1u)
        case 2u: SQF_Q4_CASE(2u)
        case 3u: SQF_Q4_CASE(3u)
        case 4u: SQF_Q4_CASE(4u)
        case 5u: SQF_Q4_CASE(5u)
        case 6u: SQF_Q4_CASE(6u)
        case 7u: SQF_Q4_CASE(7u)
        case 8u: SQF_Q4_CASE(8u)
        default: break;
    }
    #undef SQF_Q4_CASE
}

// Zero-Waste `8`-Lane/Row (`4` rows/SG = `16` rows/TG) Q4 GEMV for `cols = 640` (`shared_down_proj`, `5` full waves = 100% SIMD util!)
template <uint M_FIX>
inline void sqf_q4_8lane_fixed(
    device const uint2* w_u2,
    device const half*  scales,
    device const half*  biases,
    device const half*  x_in,
    device half*        y_f16,
    device float*       y_f32,
    uint row,
    uint rows,
    uint cols,
    uint out_mode,
    uint lane8
) {
    const uint u2_per_row  = cols >> 4u;
    const uint grp_per_row = cols >> 6u;
    device const uint2* rw = w_u2   + ulong(row) * u2_per_row;
    device const half*  rs = scales + ulong(row) * grp_per_row;
    device const half*  rb = biases + ulong(row) * grp_per_row;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint u2_idx = lane8; u2_idx < u2_per_row; u2_idx += 8u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 q0      = rw[u2_idx];
        const float sc      = float(rs[grp_idx]);
        const float bi      = float(rb[grp_idx]);

        half4 n0, n1, n2, n3;
        sqf_unpack_u32_nibbles_h4(q0.x, n0, n1);
        sqf_unpack_u32_nibbles_h4(q0.y, n2, n3);

        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + u2_idx * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float xs = sqf_sum_h4x4(x0, x1, x2, x3);
            acc[m] = fma(sc, sqf_dot_h4x4(n0, n1, n2, n3, x0, x1, x2, x3), fma(bi, xs, acc[m]));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float v = acc[m];
        v += simd_shuffle_xor(v, 4u);
        v += simd_shuffle_xor(v, 2u);
        v += simd_shuffle_xor(v, 1u);
        if (lane8 == 0u) {
            const ulong idx = ulong(m) * rows + row;
            if (out_mode == SQF_OUT_F16)      y_f16[idx] = sqf_clamp_f16(v);
            else if (out_mode == SQF_OUT_F32) y_f32[idx] = v;
            else                              y_f32[idx] += v;
        }
    }
}

kernel void sqf_q4_gemv_8lane(
    device const uint2*          w_u2   [[buffer(0)]],
    device const half*           scales [[buffer(1)]],
    device const half*           biases [[buffer(2)]],
    device const half*           x_in   [[buffer(3)]],
    device half*                 y_f16  [[buffer(4)]],
    device float*                y_f32  [[buffer(5)]],
    constant SqfGemvBatchParams& p      [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = (tgid * 4u + sgid) * 4u + (lane >> 3u);
    if (row >= p.rows) return;
    const uint lane8 = lane & 7u;
    #define SQF_Q4_8L_CASE(MF) { sqf_q4_8lane_fixed<MF>(w_u2, scales, biases, x_in, y_f16, y_f32, row, p.rows, p.cols, p.out_mode, lane8); break; }
    switch (SQF_CONST_M) {
        case 1u: SQF_Q4_8L_CASE(1u)
        case 2u: SQF_Q4_8L_CASE(2u)
        case 3u: SQF_Q4_8L_CASE(3u)
        case 4u: SQF_Q4_8L_CASE(4u)
        case 5u: SQF_Q4_8L_CASE(5u)
        case 6u: SQF_Q4_8L_CASE(6u)
        case 7u: SQF_Q4_8L_CASE(7u)
        case 8u: SQF_Q4_8L_CASE(8u)
        default: break;
    }
    #undef SQF_Q4_8L_CASE
}

struct SqfShortlistLmHeadParams {
    uint M;
    uint cols;
    uint v_short;    // e.g. 65536 (contiguous BPE high-frequency head `[0 .. v_short - 1]`)
    uint tail_start; // e.g. 248044 (special tokens `[tail_start .. v_total - 1]`)
    uint v_total;    // 248320
    uint v_out;      // v_short + (v_total - tail_start)
    uint pad0;
    uint pad1;
};

// Zero-Copy Shortlisted-Vocab MTP Draft `lm_head` (`[0..v_short-1] U [tail_start..v_total-1]`, specialized via `SQF_CONST_M`)
kernel void sqf_q4_lm_head_shortlist(
    device const uint2*                w_u2   [[buffer(0)]],
    device const half*                 scales [[buffer(1)]],
    device const half*                 biases [[buffer(2)]],
    device const half*                 x_in   [[buffer(3)]], // [M, cols]
    device float*                      y_f32  [[buffer(4)]], // [M, v_out]
    constant SqfShortlistLmHeadParams& p      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],            // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint out0 = (tgid * 4u + sgid) * 2u;
    if (out0 >= p.v_out) return;
    const uint out1 = min(out0 + 1u, p.v_out - 1u);

    const uint r0 = min((out0 < p.v_short) ? out0 : (p.tail_start + (out0 - p.v_short)), p.v_total - 1u);
    const uint r1 = min((out1 < p.v_short) ? out1 : (p.tail_start + (out1 - p.v_short)), p.v_total - 1u);

    #define SQF_SL_CASE(MF) { \
        float2 res[MF]; \
        sqf_q4_gemv2_pair_fixed<MF>(w_u2, scales, biases, x_in, r0, r1, p.cols, lane, res); \
        if (lane == 0u) sqf_store_gemv2<MF>(nullptr, y_f32, out0, p.v_out, SQF_OUT_F32, res); \
        break; \
    }
    switch (SQF_CONST_M) {
        case 1u: SQF_SL_CASE(1u)
        case 2u: SQF_SL_CASE(2u)
        default: break;
    }
    #undef SQF_SL_CASE
}

struct SqfPackFrspecParams {
    uint num_rows;    // e.g. 32768 (SLC-pinned FR-Spec draft lm_head!)
    uint cols;        // D = 2560
    uint src_v_total; // full vocab V = 248320
    uint pad0;
};

// One-time (< 0.5 ms) GPU compaction of 32K FR-Spec `lm_head` Q4_A rows into a contiguous 45.0 MiB SLC-pinned buffer!
// Grid: `(num_rows, 1, 1)`, `128` threads/TG
kernel void sqf_pack_frspec_q4(
    device const uint2*           src_w   [[buffer(0)]],
    device const half*            src_s   [[buffer(1)]],
    device const half*            src_b   [[buffer(2)]],
    device const uint*            ids     [[buffer(3)]],
    device uint2*                 dst_w   [[buffer(4)]],
    device half*                  dst_s   [[buffer(5)]],
    device half*                  dst_b   [[buffer(6)]],
    constant SqfPackFrspecParams& p       [[buffer(7)]],
    uint out_row [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]]
) {
    if (out_row >= p.num_rows) return;
    const uint src_row     = min(ids[out_row], p.src_v_total - 1u);
    const uint u2_per_row  = p.cols >> 4u;
    const uint grp_per_row = p.cols >> 6u;

    device const uint2* sw = src_w + ulong(src_row) * u2_per_row;
    device uint2*       dw = dst_w + ulong(out_row) * u2_per_row;
    for (uint i = tid; i < u2_per_row; i += 128u) {
        dw[i] = sw[i];
    }

    device const half* ss = src_s + ulong(src_row) * grp_per_row;
    device const half* sb = src_b + ulong(src_row) * grp_per_row;
    device half*       ds = dst_s + ulong(out_row) * grp_per_row;
    device half*       db = dst_b + ulong(out_row) * grp_per_row;
    for (uint g = tid; g < grp_per_row; g += 128u) {
        ds[g] = ss[g];
        db[g] = sb[g];
    }
}

// Self-healing (< 1 us/row) GPU dynamic context-slot patcher for the 32K SLC-pinned FR-Spec `lm_head`!
// Patches `count` rows (`dst_slots[idx] <- src_rows[idx]`) inline via `setBytes` (0 extra buffers, 0 async races!).
// Grid: `(count, 1, 1)`, `128` threads/TG
kernel void sqf_patch_frspec_q4(
    device const uint2*           src_w     [[buffer(0)]],
    device const half*            src_s     [[buffer(1)]],
    device const half*            src_b     [[buffer(2)]],
    constant uint*                dst_slots [[buffer(3)]],
    constant uint*                src_rows  [[buffer(4)]],
    device uint2*                 dst_w     [[buffer(5)]],
    device half*                  dst_s     [[buffer(6)]],
    device half*                  dst_b     [[buffer(7)]],
    constant SqfPackFrspecParams& p         [[buffer(8)]], // num_rows=count, cols=D, src_v_total=V, pad0=dst_v_total
    uint idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (idx >= p.num_rows) return;
    const uint out_row     = dst_slots[idx];
    const uint src_row     = src_rows[idx];
    if (out_row >= p.pad0 || src_row >= p.src_v_total) return;

    const uint u2_per_row  = p.cols >> 4u;
    const uint grp_per_row = p.cols >> 6u;

    device const uint2* sw = src_w + ulong(src_row) * u2_per_row;
    device uint2*       dw = dst_w + ulong(out_row) * u2_per_row;
    for (uint i = tid; i < u2_per_row; i += 128u) {
        dw[i] = sw[i];
    }

    device const half* ss = src_s + ulong(src_row) * grp_per_row;
    device const half* sb = src_b + ulong(src_row) * grp_per_row;
    device half*       ds = dst_s + ulong(out_row) * grp_per_row;
    device half*       db = dst_b + ulong(out_row) * grp_per_row;
    for (uint g = tid; g < grp_per_row; g += 128u) {
        ds[g] = ss[g];
        db[g] = sb[g];
    }
}

struct SqfLinQzGemvParams {
    uint M;
    uint qkv_rows;
    uint z_rows;
    uint cols;
};

// Single-Dispatch Fused `(lin_in_proj_qkv + lin_in_proj_z)` Q4 GEMV (`2` rows/SG, `8` rows/TG, `M = 1..8`)
kernel void sqf_q4_gemv_lin_qz_batch(
    device const uint2* qkv_w [[buffer(0)]],
    device const half*  qkv_s [[buffer(1)]],
    device const half*  qkv_b [[buffer(2)]],
    device const uint2* z_w   [[buffer(3)]],
    device const half*  z_s   [[buffer(4)]],
    device const half*  z_b   [[buffer(5)]],
    device const half*  x_in  [[buffer(6)]],
    device half*        qkv_o [[buffer(7)]],
    device half*        z_o   [[buffer(8)]],
    constant SqfLinQzGemvParams& p [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow = (tgid * 4u + sgid) * 2u;
    const uint b1   = p.qkv_rows;
    const uint b2   = b1 + p.z_rows;
    if (vrow >= b2) return;

    const bool is_qkv = (vrow < b1);
    device const uint2* ww = is_qkv ? qkv_w : z_w;
    device const half*  ws = is_qkv ? qkv_s : z_s;
    device const half*  wb = is_qkv ? qkv_b : z_b;
    device half*        oo = is_qkv ? qkv_o : z_o;
    const uint          rr = is_qkv ? vrow : (vrow - b1);
    const uint          nr = is_qkv ? p.qkv_rows : p.z_rows;

    #define SQF_LQZ_CASE(MF) { \
        float2 res[MF]; \
        sqf_q4_gemv2_rows_fixed<MF>(ww, ws, wb, x_in, rr, nr, p.cols, lane, res); \
        if (lane == 0u) sqf_store_gemv2<MF>(oo, nullptr, rr, nr, SQF_OUT_F16, res); \
        break; \
    }
    switch (SQF_CONST_M) {
        case 1u: SQF_LQZ_CASE(1u)
        case 2u: SQF_LQZ_CASE(2u)
        case 3u: SQF_LQZ_CASE(3u)
        case 4u: SQF_LQZ_CASE(4u)
        case 5u: SQF_LQZ_CASE(5u)
        case 6u: SQF_LQZ_CASE(6u)
        case 7u: SQF_LQZ_CASE(7u)
        case 8u: SQF_LQZ_CASE(8u)
        default: break;
    }
    #undef SQF_LQZ_CASE
}

struct SqfAttnQkvGemvParams {
    uint M;
    uint q_rows;
    uint kv_rows;
    uint cols;
};

// Single-Dispatch Fused `(attn_q_proj + attn_k_proj + attn_v_proj)` Q4 GEMV (`2` rows/SG, `8` rows/TG, `M = 1..8`)
kernel void sqf_q4_gemv_attn_qkv_batch(
    device const uint2* q_w   [[buffer(0)]],
    device const half*  q_s   [[buffer(1)]],
    device const half*  q_b   [[buffer(2)]],
    device const uint2* k_w   [[buffer(3)]],
    device const half*  k_s   [[buffer(4)]],
    device const half*  k_b   [[buffer(5)]],
    device const uint2* v_w   [[buffer(6)]],
    device const half*  v_s   [[buffer(7)]],
    device const half*  v_b   [[buffer(8)]],
    device const half*  x_in  [[buffer(9)]],
    device half*        q_out [[buffer(10)]],
    device half*        k_out [[buffer(11)]],
    device half*        v_out [[buffer(12)]],
    constant SqfAttnQkvGemvParams& p [[buffer(13)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow = (tgid * 4u + sgid) * 2u;
    const uint b1   = p.q_rows;
    const uint b2   = b1 + p.kv_rows;
    const uint b3   = b2 + p.kv_rows;
    if (vrow >= b3) return;

    device const uint2* ww = (vrow < b1) ? q_w : ((vrow < b2) ? k_w : v_w);
    device const half*  ws = (vrow < b1) ? q_s : ((vrow < b2) ? k_s : v_s);
    device const half*  wb = (vrow < b1) ? q_b : ((vrow < b2) ? k_b : v_b);
    device half*        oo = (vrow < b1) ? q_out : ((vrow < b2) ? k_out : v_out);
    const uint          rr = (vrow < b1) ? vrow : ((vrow < b2) ? (vrow - b1) : (vrow - b2));
    const uint          nr = (vrow < b1) ? p.q_rows : p.kv_rows;

    #define SQF_AQKV_CASE(MF) { \
        float2 res[MF]; \
        sqf_q4_gemv2_rows_fixed<MF>(ww, ws, wb, x_in, rr, nr, p.cols, lane, res); \
        if (lane == 0u) sqf_store_gemv2<MF>(oo, nullptr, rr, nr, SQF_OUT_F16, res); \
        break; \
    }
    switch (SQF_CONST_M) {
        case 1u: SQF_AQKV_CASE(1u)
        case 2u: SQF_AQKV_CASE(2u)
        case 3u: SQF_AQKV_CASE(3u)
        case 4u: SQF_AQKV_CASE(4u)
        case 5u: SQF_AQKV_CASE(5u)
        case 6u: SQF_AQKV_CASE(6u)
        case 7u: SQF_AQKV_CASE(7u)
        case 8u: SQF_AQKV_CASE(8u)
        default: break;
    }
    #undef SQF_AQKV_CASE
}

// Zero-spill `2` rows/SG (`8` rows/TG) FP16 Dense GEMV (`half4x2` 16B loads + dual-rate FP16 ALU + `SQF_CONST_M`!)
template <uint M_FIX>
inline void sqf_f16_gemv2_fixed(
    device const half* w_f16,
    device const half* x_in,
    device half* y_f16,
    device float* y_f32,
    uint row0,
    uint rows,
    uint cols,
    uint out_mode,
    uint lane
) {
    const uint c8_cnt = cols >> 3u;
    const uint r0 = min(row0 + 0u, rows - 1u);
    const uint r1 = min(row0 + 1u, rows - 1u);

    device const half4* w0 = reinterpret_cast<device const half4*>(w_f16 + ulong(r0) * cols);
    device const half4* w1 = reinterpret_cast<device const half4*>(w_f16 + ulong(r1) * cols);

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint c8 = lane; c8 < c8_cnt; c8 += 32u) {
        const half4 hw00 = w0[c8 * 2u + 0u];
        const half4 hw01 = w0[c8 * 2u + 1u];
        const half4 hw10 = w1[c8 * 2u + 0u];
        const half4 hw11 = w1[c8 * 2u + 1u];
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols);
            const half4 x0 = x4[c8 * 2u + 0u];
            const half4 x1 = x4[c8 * 2u + 1u];
            const float4 p0 = float4(fma(hw00, x0, hw01 * x1));
            const float4 p1 = float4(fma(hw10, x0, hw11 * x1));
            acc[m].x += (p0.x + p0.y) + (p0.z + p0.w);
            acc[m].y += (p1.x + p1.y) + (p1.z + p1.w);
        }
    }
    if ((cols & 4u) != 0u && lane == 0u) {
        const uint c4 = cols >> 2u;
        const float4 fw0 = float4(w0[c4 - 1u]);
        const float4 fw1 = float4(w1[c4 - 1u]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            const float4 fx = float4(reinterpret_cast<device const half4*>(x_in + ulong(m) * cols)[c4 - 1u]);
            acc[m].x = fma(fw0.x, fx.x, fma(fw0.y, fx.y, fma(fw0.z, fx.z, fma(fw0.w, fx.w, acc[m].x))));
            acc[m].y = fma(fw1.x, fx.x, fma(fw1.y, fx.y, fma(fw1.z, fx.z, fma(fw1.w, fx.w, acc[m].y))));
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        acc[m] = float2(simd_sum(acc[m].x), simd_sum(acc[m].y));
    }
    if (lane == 0u) {
        sqf_store_gemv2<M_FIX>(y_f16, y_f32, row0, rows, out_mode, acc);
    }
}

kernel void sqf_f16_gemv_batch(
    device const half*           w_f16    [[buffer(0)]], // [rows, cols]
    device const half*           x_in     [[buffer(1)]], // [M, cols]
    device half*                 y_f16    [[buffer(2)]], // [M, rows]
    device float*                y_f32    [[buffer(3)]], // [M, rows]
    constant SqfGemvBatchParams& p        [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],        // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 4u + sgid) * 2u;
    if (row0 >= p.rows) return;

    #define SQF_F16_CASE(MF) { sqf_f16_gemv2_fixed<MF>(w_f16, x_in, y_f16, y_f32, row0, p.rows, p.cols, p.out_mode, lane); break; }
    switch (SQF_CONST_M) {
        case 1u: SQF_F16_CASE(1u)
        case 2u: SQF_F16_CASE(2u)
        case 3u: SQF_F16_CASE(3u)
        case 4u: SQF_F16_CASE(4u)
        case 5u: SQF_F16_CASE(5u)
        case 6u: SQF_F16_CASE(6u)
        case 7u: SQF_F16_CASE(7u)
        case 8u: SQF_F16_CASE(8u)
        default: break;
    }
    #undef SQF_F16_CASE
}

// High-Occupancy `1` Row/TG (`128` threads = `4` SGs, `half4x2` 16B loads) FP16 GEMV
// Used for tall-K / medium-N matrices (`moe_router_gate [512, 2560]`, `indexer_qk_proj`, etc.)!
template <uint M_FIX>
inline void sqf_f16_row_tg128_impl(
    device const half* w_f16,
    device const half* x_in,
    device half* y_f16,
    device float* y_f32,
    threadgroup float (&tg_red)[8][4],
    uint row,
    uint rows,
    uint cols,
    uint out_mode,
    uint tid,
    uint sgid,
    uint lane
) {
    const uint c8_cnt = cols >> 3u;
    device const half4* w4 = reinterpret_cast<device const half4*>(w_f16 + ulong(row) * cols);

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint c8 = tid; c8 < c8_cnt; c8 += 128u) {
        const half4 hw0 = w4[c8 * 2u + 0u];
        const half4 hw1 = w4[c8 * 2u + 1u];
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols);
            const float4 p = float4(fma(hw0, x4[c8 * 2u + 0u], hw1 * x4[c8 * 2u + 1u]));
            acc[m] += (p.x + p.y) + (p.z + p.w);
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            float tot = simd_sum((lane < 4u) ? tg_red[m][lane] : 0.0f);
            if (lane == 0u) {
                const ulong idx = ulong(m) * rows + row;
                if (out_mode == SQF_OUT_F16)      y_f16[idx] = sqf_clamp_f16(tot);
                else if (out_mode == SQF_OUT_F32) y_f32[idx] = tot;
                else                              y_f32[idx] += tot;
            }
        }
    }
}

kernel void sqf_f16_gemv_row_tg128(
    device const half*           w_f16    [[buffer(0)]],
    device const half*           x_in     [[buffer(1)]],
    device half*                 y_f16    [[buffer(2)]],
    device float*                y_f32    [[buffer(3)]],
    constant SqfGemvBatchParams& p        [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][4];
    if (row >= p.rows) return;
    #define SQF_R128_CASE(MF) { sqf_f16_row_tg128_impl<MF>(w_f16, x_in, y_f16, y_f32, tg_red, row, p.rows, p.cols, p.out_mode, tid, sgid, lane); break; }
    switch (SQF_CONST_M) {
        case 1u: SQF_R128_CASE(1u)
        case 2u: SQF_R128_CASE(2u)
        case 3u: SQF_R128_CASE(3u)
        case 4u: SQF_R128_CASE(4u)
        case 5u: SQF_R128_CASE(5u)
        case 6u: SQF_R128_CASE(6u)
        case 7u: SQF_R128_CASE(7u)
        case 8u: SQF_R128_CASE(8u)
        default: break;
    }
    #undef SQF_R128_CASE
}

// High-Occupancy 2D-Tiled Prefill FP16 Kernel for Small-N Matrices (`rows <= 64`: `block_inject [4,10240]`, `lin_in_proj_a/b [48,2560]`)
// Grid: `(rows, (M + 7) / 8, 1)`, `128` threads/TG -> 128..1536 TGs across 40 cores (replaces 4-TG starved NAX dispatch!)
kernel void sqf_f16_small_n_prefill(
    device const half*           w_f16    [[buffer(0)]],
    device const half*           x_in     [[buffer(1)]],
    device half*                 y_f16    [[buffer(2)]],
    device float*                y_f32    [[buffer(3)]],
    constant SqfGemvBatchParams& p        [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][4];
    const uint row = tgid.x;
    const uint m0  = tgid.y * 8u;
    if (row >= p.rows || m0 >= p.M) return;

    device const half* xb  = x_in + ulong(m0) * p.cols;
    device half*       y16 = y_f16 + ulong(m0) * p.rows;
    device float*      y32 = y_f32 + ulong(m0) * p.rows;

    if (m0 + 8u <= p.M) {
        sqf_f16_row_tg128_impl<8u>(w_f16, xb, y16, y32, tg_red, row, p.rows, p.cols, p.out_mode, tid, sgid, lane);
    } else {
        const uint mb = p.M - m0;
        for (uint m = 0u; m < mb; ++m) {
            sqf_f16_row_tg128_impl<1u>(w_f16, xb + ulong(m) * p.cols,
                                       y16 + ulong(m) * p.rows,
                                       y32 + ulong(m) * p.rows,
                                       tg_red, row, p.rows, p.cols, p.out_mode, tid, sgid, lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Full-Occupancy `8`-Lane/Row (`4` rows/SG, `16` rows/TG, `half4x2` 16B loads) FP16 GEMV
// Zero idle lanes for `cols = 320` (`5` full waves: `hc.input_mix_weight_up`) and `cols = 640` (`10` full waves: `shared_down_proj`)!
template <uint M_FIX>
inline void sqf_f16_8lane_impl(
    device const half* w_f16,
    device const half* x_in,
    device half* y_f16,
    device float* y_f32,
    uint row,
    uint rows,
    uint cols,
    uint out_mode,
    uint lane8
) {
    const uint c8_cnt = cols >> 3u;
    device const half4* w4 = reinterpret_cast<device const half4*>(w_f16 + ulong(row) * cols);

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint c8 = lane8; c8 < c8_cnt; c8 += 8u) {
        const float4 fw0 = float4(w4[c8 * 2u + 0u]);
        const float4 fw1 = float4(w4[c8 * 2u + 1u]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols);
            const float4 fx0 = float4(x4[c8 * 2u + 0u]);
            const float4 fx1 = float4(x4[c8 * 2u + 1u]);
            float a = acc[m];
            a = fma(fw0.x, fx0.x, fma(fw0.y, fx0.y, fma(fw0.z, fx0.z, fma(fw0.w, fx0.w, a))));
            a = fma(fw1.x, fx1.x, fma(fw1.y, fx1.y, fma(fw1.z, fx1.z, fma(fw1.w, fx1.w, a))));
            acc[m] = a;
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float v = acc[m];
        v += simd_shuffle_xor(v, 4u);
        v += simd_shuffle_xor(v, 2u);
        v += simd_shuffle_xor(v, 1u);
        if (lane8 == 0u) {
            const ulong idx = ulong(m) * rows + row;
            if (out_mode == SQF_OUT_F16)      y_f16[idx] = sqf_clamp_f16(v);
            else if (out_mode == SQF_OUT_F32) y_f32[idx] = v;
            else                              y_f32[idx] += v;
        }
    }
}

kernel void sqf_f16_gemv_8lane(
    device const half*           w_f16    [[buffer(0)]],
    device const half*           x_in     [[buffer(1)]],
    device half*                 y_f16    [[buffer(2)]],
    device float*                y_f32    [[buffer(3)]],
    constant SqfGemvBatchParams& p        [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = (tgid * 4u + sgid) * 4u + (lane >> 3u);
    if (row >= p.rows) return;
    const uint lane8 = lane & 7u;
    #define SQF_8L_CASE(MF) { sqf_f16_8lane_impl<MF>(w_f16, x_in, y_f16, y_f32, row, p.rows, p.cols, p.out_mode, lane8); break; }
    switch (p.M) {
        case 1u: SQF_8L_CASE(1u)
        case 2u: SQF_8L_CASE(2u)
        case 3u: SQF_8L_CASE(3u)
        case 4u: SQF_8L_CASE(4u)
        case 5u: SQF_8L_CASE(5u)
        case 6u: SQF_8L_CASE(6u)
        case 7u: SQF_8L_CASE(7u)
        case 8u: SQF_8L_CASE(8u)
        default: break;
    }
    #undef SQF_8L_CASE
}

// Single-Dispatch Fused `hc.input_mix_weight_down [R, 4*D]` + `hc.block_inject_weight [H, 4*D]` + `silu/2*sigmoid`!
// Grid: `(lowrank + (has_inject ? hc_count : 0), 1, 1)`, `256` threads/TG (`8` SGs) -> 324 TGs saturating all 40 cores!
struct SqfHcDownInjParams {
    uint  M_valid;
    uint  lowrank;
    uint  hc_count;
    uint  cols;       // hc_count * D = 10240
    float inv_hc;     // 0.25f
    uint  has_inject; // 1 for pre-attn/pre-mlp, 0 for final mixer
    uint  pad0;
    uint  pad1;
};

template <uint M_FIX>
inline void sqf_hc_down_inj_impl(
    device const half* w_down,
    device const half* w_inj,
    device const half* x_in,
    device half* u_out,
    device float* inj_out,
    threadgroup float (&tg_red)[8][8],
    constant SqfHcDownInjParams& p,
    uint row,
    uint tid,
    uint sgid,
    uint lane
) {
    const bool is_inj  = (row >= p.lowrank);
    const uint loc_row = is_inj ? (row - p.lowrank) : row;
    const uint c8_cnt  = p.cols >> 3u;
    device const half4* w4 = reinterpret_cast<device const half4*>(
        (is_inj ? w_inj : w_down) + ulong(loc_row) * p.cols);

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint c8 = tid; c8 < c8_cnt; c8 += 256u) {
        const float4 fw0 = float4(w4[c8 * 2u + 0u]);
        const float4 fw1 = float4(w4[c8 * 2u + 1u]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols);
            const float4 fx0 = float4(x4[c8 * 2u + 0u]);
            const float4 fx1 = float4(x4[c8 * 2u + 1u]);
            float a = acc[m];
            a = fma(fw0.x, fx0.x, fma(fw0.y, fx0.y, fma(fw0.z, fx0.z, fma(fw0.w, fx0.w, a))));
            a = fma(fw1.x, fx1.x, fma(fw1.y, fx1.y, fma(fw1.z, fx1.z, fma(fw1.w, fx1.w, a))));
            acc[m] = a;
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            float tot = simd_sum((lane < 8u) ? tg_red[m][lane] : 0.0f);
            if (lane == 0u) {
                if (is_inj) {
                    inj_out[ulong(m) * p.hc_count + loc_row] = 2.0f * sqf_sigmoid(tot * p.inv_hc);
                } else {
                    u_out[ulong(m) * p.lowrank + loc_row] = sqf_clamp_f16(sqf_silu(tot * p.inv_hc));
                }
            }
        }
    }
}

kernel void sqf_hc_down_inj_act_batch(
    device const half*           w_down  [[buffer(0)]],
    device const half*           w_inj   [[buffer(1)]],
    device const half*           x_in    [[buffer(2)]],
    device half*                 u_out   [[buffer(3)]],
    device float*                inj_out [[buffer(4)]],
    constant SqfHcDownInjParams& p       [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][8];
    const uint total_rows = p.lowrank + (p.has_inject != 0u ? p.hc_count : 0u);
    if (row >= total_rows) return;
    #define SQF_HCDI_CASE(MF) { sqf_hc_down_inj_impl<MF>(w_down, w_inj, x_in, u_out, inj_out, tg_red, p, row, tid, sgid, lane); break; }
    switch (p.M_valid) {
        case 1u: SQF_HCDI_CASE(1u)
        case 2u: SQF_HCDI_CASE(2u)
        case 3u: SQF_HCDI_CASE(3u)
        case 4u: SQF_HCDI_CASE(4u)
        case 5u: SQF_HCDI_CASE(5u)
        case 6u: SQF_HCDI_CASE(6u)
        case 7u: SQF_HCDI_CASE(7u)
        case 8u: SQF_HCDI_CASE(8u)
        default: break;
    }
    #undef SQF_HCDI_CASE
}

// Single-Dispatch Fused `lin_in_proj_a [n_v, D]` + `lin_in_proj_b [n_v, D]` FP16 GEMV (`1` row/TG, `128` th/TG -> 96 TGs!)
struct SqfLinAbParams {
    uint M;
    uint a_rows;
    uint b_rows;
    uint cols;
};

template <uint M_FIX>
inline void sqf_f16_lin_ab_impl(
    device const half* w_a,
    device const half* w_b,
    device const half* x_in,
    device float* out_a,
    device float* out_b,
    threadgroup float (&tg_red)[8][4],
    constant SqfLinAbParams& p,
    uint row,
    uint tid,
    uint sgid,
    uint lane
) {
    const bool is_b    = (row >= p.a_rows);
    const uint loc_row = is_b ? (row - p.a_rows) : row;
    const uint out_row = is_b ? p.b_rows : p.a_rows;
    const uint c8_cnt  = p.cols >> 3u;
    device const half4* w4 = reinterpret_cast<device const half4*>((is_b ? w_b : w_a) + ulong(loc_row) * p.cols);

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint c8 = tid; c8 < c8_cnt; c8 += 128u) {
        const half4 hw0 = w4[c8 * 2u + 0u];
        const half4 hw1 = w4[c8 * 2u + 1u];
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols);
            const float4 pr = float4(fma(hw0, x4[c8 * 2u + 0u], hw1 * x4[c8 * 2u + 1u]));
            acc[m] += (pr.x + pr.y) + (pr.z + pr.w);
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        device float* dst = is_b ? out_b : out_a;
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            float tot = simd_sum((lane < 4u) ? tg_red[m][lane] : 0.0f);
            if (lane == 0u) dst[ulong(m) * out_row + loc_row] = tot;
        }
    }
}

kernel void sqf_f16_lin_ab_batch(
    device const half*       w_a   [[buffer(0)]],
    device const half*       w_b   [[buffer(1)]],
    device const half*       x_in  [[buffer(2)]],
    device float*            out_a [[buffer(3)]],
    device float*            out_b [[buffer(4)]],
    constant SqfLinAbParams& p     [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][4];
    if (row >= p.a_rows + p.b_rows) return;
    #define SQF_LAB_CASE(MF) { sqf_f16_lin_ab_impl<MF>(w_a, w_b, x_in, out_a, out_b, tg_red, p, row, tid, sgid, lane); break; }
    switch (SQF_CONST_M) {
        case 1u: SQF_LAB_CASE(1u)
        case 2u: SQF_LAB_CASE(2u)
        case 3u: SQF_LAB_CASE(3u)
        case 4u: SQF_LAB_CASE(4u)
        case 5u: SQF_LAB_CASE(5u)
        case 6u: SQF_LAB_CASE(6u)
        case 7u: SQF_LAB_CASE(7u)
        case 8u: SQF_LAB_CASE(8u)
        default: break;
    }
    #undef SQF_LAB_CASE
}

// Single-Dispatch Fused `shared_gate_proj [d_ff, D]` + `shared_up_proj [d_ff, D]` + `SwiGLU` -> `mid_f16`!
// Grid: `(d_ff, 1, 1)` (`640` TGs = `16` TGs/core!), `128` threads/TG (`4` SGs), single `x_in` pass!
template <uint M_FIX>
inline void sqf_f16_shared_gu_silu_impl(
    device const half* w_gate,
    device const half* w_up,
    device const half* x_in,
    device half* mid_out,
    threadgroup float (&tg_g)[8][4],
    threadgroup float (&tg_u)[8][4],
    uint row,
    uint d_ff,
    uint cols,
    uint tid,
    uint sgid,
    uint lane
) {
    const uint c8_cnt = cols >> 3u;
    device const half4* wg4 = reinterpret_cast<device const half4*>(w_gate + ulong(row) * cols);
    device const half4* wu4 = reinterpret_cast<device const half4*>(w_up   + ulong(row) * cols);

    float ga[M_FIX];
    float ua[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) { ga[m] = 0.0f; ua[m] = 0.0f; }

    for (uint c8 = tid; c8 < c8_cnt; c8 += 128u) {
        const float4 fg0 = float4(wg4[c8 * 2u + 0u]);
        const float4 fg1 = float4(wg4[c8 * 2u + 1u]);
        const float4 fu0 = float4(wu4[c8 * 2u + 0u]);
        const float4 fu1 = float4(wu4[c8 * 2u + 1u]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols);
            const float4 fx0 = float4(x4[c8 * 2u + 0u]);
            const float4 fx1 = float4(x4[c8 * 2u + 1u]);
            float g = ga[m];
            g = fma(fg0.x, fx0.x, fma(fg0.y, fx0.y, fma(fg0.z, fx0.z, fma(fg0.w, fx0.w, g))));
            g = fma(fg1.x, fx1.x, fma(fg1.y, fx1.y, fma(fg1.z, fx1.z, fma(fg1.w, fx1.w, g))));
            ga[m] = g;
            float u = ua[m];
            u = fma(fu0.x, fx0.x, fma(fu0.y, fx0.y, fma(fu0.z, fx0.z, fma(fu0.w, fx0.w, u))));
            u = fma(fu1.x, fx1.x, fma(fu1.y, fx1.y, fma(fu1.z, fx1.z, fma(fu1.w, fx1.w, u))));
            ua[m] = u;
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float sg = simd_sum(ga[m]);
        float su = simd_sum(ua[m]);
        if (lane == 0u) { tg_g[m][sgid] = sg; tg_u[m][sgid] = su; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            float gv = simd_sum((lane < 4u) ? tg_g[m][lane] : 0.0f);
            float uv = simd_sum((lane < 4u) ? tg_u[m][lane] : 0.0f);
            if (lane == 0u) {
                mid_out[ulong(m) * d_ff + row] = sqf_clamp_f16(sqf_silu(gv) * uv);
            }
        }
    }
}

kernel void sqf_f16_shared_gate_up_silu_batch(
    device const half*           w_gate  [[buffer(0)]],
    device const half*           w_up    [[buffer(1)]],
    device const half*           x_in    [[buffer(2)]],
    device half*                 mid_out [[buffer(3)]],
    constant SqfGemvBatchParams& p       [[buffer(4)]], // rows = shared_dff, cols = D
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_g[8][4];
    threadgroup float tg_u[8][4];
    if (row >= p.rows) return;
    #define SQF_SHGU_CASE(MF) { sqf_f16_shared_gu_silu_impl<MF>(w_gate, w_up, x_in, mid_out, tg_g, tg_u, row, p.rows, p.cols, tid, sgid, lane); break; }
    switch (p.M) {
        case 1u: SQF_SHGU_CASE(1u)
        case 2u: SQF_SHGU_CASE(2u)
        case 3u: SQF_SHGU_CASE(3u)
        case 4u: SQF_SHGU_CASE(4u)
        case 5u: SQF_SHGU_CASE(5u)
        case 6u: SQF_SHGU_CASE(6u)
        case 7u: SQF_SHGU_CASE(7u)
        case 8u: SQF_SHGU_CASE(8u)
        default: break;
    }
    #undef SQF_SHGU_CASE
}

/* ============================================================================
 * 3b. Phase C: Planar `Q8_0` (gs=64) HC/PLE/MTP & Fused `Q4_A` Shared Expert Kernels
 * ============================================================================ */

// Zero-spill `2` rows/SG (`8` rows/TG) Planar `Q8_0` GEMV (`M = 1..8`, `uint2` 8B coalesced loads)
template <uint M_FIX>
inline void sqf_q8_gemv2_fixed(
    device const uint2* w_q8,
    device const half*  w_s,
    device const half*  x_in,
    device half*        y_f16,
    device float*       y_f32,
    uint row0,
    uint rows,
    uint cols,
    uint out_mode,
    uint lane
) {
    const uint c8_cnt      = cols >> 3u;
    const uint grp_per_row = cols >> 6u;
    const uint r0 = min(row0 + 0u, rows - 1u);
    const uint r1 = min(row0 + 1u, rows - 1u);

    device const uint2* w0 = w_q8 + ulong(r0) * c8_cnt;
    device const uint2* w1 = w_q8 + ulong(r1) * c8_cnt;
    device const half*  s0 = w_s  + ulong(r0) * grp_per_row;
    device const half*  s1 = w_s  + ulong(r1) * grp_per_row;

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint c8 = lane; c8 < c8_cnt; c8 += 32u) {
        const uint   grp  = c8 >> 3u;
        const float  sc0  = float(s0[grp]);
        const float  sc1  = float(s1[grp]);
        const uint2  q0   = w0[c8];
        const uint2  q1   = w1[c8];
        const float4 fw0a = float4(as_type<char4>(q0.x));
        const float4 fw0b = float4(as_type<char4>(q0.y));
        const float4 fw1a = float4(as_type<char4>(q1.x));
        const float4 fw1b = float4(as_type<char4>(q1.y));
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols);
            const float4 fx0 = float4(x4[c8 * 2u + 0u]);
            const float4 fx1 = float4(x4[c8 * 2u + 1u]);
            const float  d0  = dot(fw0a, fx0) + dot(fw0b, fx1);
            const float  d1  = dot(fw1a, fx0) + dot(fw1b, fx1);
            acc[m].x = fma(d0, sc0, acc[m].x);
            acc[m].y = fma(d1, sc1, acc[m].y);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        acc[m] = float2(simd_sum(acc[m].x), simd_sum(acc[m].y));
    }
    if (lane == 0u) {
        sqf_store_gemv2<M_FIX>(y_f16, y_f32, row0, rows, out_mode, acc);
    }
}

kernel void sqf_q8_gemv_batch(
    device const uint2*          w_q8  [[buffer(0)]], // [rows, cols / 8]
    device const half*           w_s   [[buffer(1)]], // [rows, cols / 64]
    device const half*           x_in  [[buffer(2)]], // [M, cols]
    device half*                 y_f16 [[buffer(3)]], // [M, rows]
    device float*                y_f32 [[buffer(4)]], // [M, rows]
    constant SqfGemvBatchParams& p     [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 4u + sgid) * 2u;
    if (row0 >= p.rows) return;
    #define SQF_Q8_CASE(MF) { sqf_q8_gemv2_fixed<MF>(w_q8, w_s, x_in, y_f16, y_f32, row0, p.rows, p.cols, p.out_mode, lane); break; }
    switch (p.M) {
        case 1u: SQF_Q8_CASE(1u)
        case 2u: SQF_Q8_CASE(2u)
        case 3u: SQF_Q8_CASE(3u)
        case 4u: SQF_Q8_CASE(4u)
        case 5u: SQF_Q8_CASE(5u)
        case 6u: SQF_Q8_CASE(6u)
        case 7u: SQF_Q8_CASE(7u)
        case 8u: SQF_Q8_CASE(8u)
        default: break;
    }
    #undef SQF_Q8_CASE
}

// Full-Occupancy `4`-Lane/Row (`8` rows/SG, `32` rows/TG, `16B` `uint4` loads!) Planar `Q8_0` GEMV for `hc.input_mix_weight_up [10240, 320]`
// Each group of 64 weights = exact 1 wave of 4 lanes x 16B `uint4` (64B coalesced) + 1 scale multiply + 2 shuffles!
template <uint M_FIX>
inline void sqf_q8_8lane_impl(
    device const uint4* w_q8,
    device const half*  w_s,
    device const half*  x_in,
    device half*        y_f16,
    device float*       y_f32,
    uint row,
    uint rows,
    uint cols,
    uint out_mode,
    uint lane4
) {
    const uint grp_per_row = cols >> 6u;
    device const uint4* w16 = w_q8 + ulong(row) * (cols >> 4u);
    device const half*  sr  = w_s  + ulong(row) * grp_per_row;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint g = 0u; g < grp_per_row; ++g) {
        const float  sc  = float(sr[g]);
        const uint   c16 = g * 4u + lane4;
        const uint4  q16 = w16[c16];
        const float4 fw0 = float4(as_type<char4>(q16.x));
        const float4 fw1 = float4(as_type<char4>(q16.y));
        const float4 fw2 = float4(as_type<char4>(q16.z));
        const float4 fw3 = float4(as_type<char4>(q16.w));
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + c16 * 16u);
            const float d = (dot(fw0, float4(x4[0])) + dot(fw1, float4(x4[1]))) +
                            (dot(fw2, float4(x4[2])) + dot(fw3, float4(x4[3])));
            acc[m] = fma(d, sc, acc[m]);
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float v = acc[m];
        v += simd_shuffle_xor(v, 2u);
        v += simd_shuffle_xor(v, 1u);
        if (lane4 == 0u) {
            const ulong idx = ulong(m) * rows + row;
            if (out_mode == SQF_OUT_F16)      y_f16[idx] = sqf_clamp_f16(v);
            else if (out_mode == SQF_OUT_F32) y_f32[idx] = v;
            else                              y_f32[idx] += v;
        }
    }
}

kernel void sqf_q8_gemv_8lane(
    device const uint4*          w_q8  [[buffer(0)]],
    device const half*           w_s   [[buffer(1)]],
    device const half*           x_in  [[buffer(2)]],
    device half*                 y_f16 [[buffer(3)]],
    device float*                y_f32 [[buffer(4)]],
    constant SqfGemvBatchParams& p     [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = (tgid * 4u + sgid) * 8u + (lane >> 2u);
    if (row >= p.rows) return;
    const uint lane4 = lane & 3u;
    #define SQF_Q8_8L_CASE(MF) { sqf_q8_8lane_impl<MF>(w_q8, w_s, x_in, y_f16, y_f32, row, p.rows, p.cols, p.out_mode, lane4); break; }
    switch (p.M) {
        case 1u: SQF_Q8_8L_CASE(1u)
        case 2u: SQF_Q8_8L_CASE(2u)
        case 3u: SQF_Q8_8L_CASE(3u)
        case 4u: SQF_Q8_8L_CASE(4u)
        case 5u: SQF_Q8_8L_CASE(5u)
        case 6u: SQF_Q8_8L_CASE(6u)
        case 7u: SQF_Q8_8L_CASE(7u)
        case 8u: SQF_Q8_8L_CASE(8u)
        default: break;
    }
    #undef SQF_Q8_8L_CASE
}

// Single-Dispatch Fused `Q8_0` `hc.input_mix_weight_down [R, 4*D]` + `FP16` `hc.block_inject_weight [H, 4*D]` + `silu/2*sigmoid`
// Grid: `(R + H, 1, 1)`, `128` threads/TG (`4` SGs, `16B` `uint4` coalesced loads -> `10240 / (128*16) = 5` exact full waves!)
template <uint M_FIX>
inline void sqf_q8_hc_down_inj_impl(
    device const uint4* w_down_q8,
    device const half*  w_down_s,
    device const half*  w_inj,
    device const half*  x_in,
    device half*        u_out,
    device float*       inj_out,
    threadgroup float (&tg_red)[8][4],
    constant SqfHcDownInjParams& p,
    uint row,
    uint tid,
    uint sgid,
    uint lane
) {
    const bool is_inj  = (row >= p.lowrank);
    const uint loc_row = is_inj ? (row - p.lowrank) : row;
    const uint c16_cnt = p.cols >> 4u;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    if (is_inj) {
        device const half4* w4 = reinterpret_cast<device const half4*>(w_inj + ulong(loc_row) * p.cols);
        for (uint c16 = tid; c16 < c16_cnt; c16 += 128u) {
            const half4 hw0 = w4[c16 * 4u + 0u];
            const half4 hw1 = w4[c16 * 4u + 1u];
            const half4 hw2 = w4[c16 * 4u + 2u];
            const half4 hw3 = w4[c16 * 4u + 3u];
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + c16 * 16u);
                const float4 st = float4(fma(hw0, x4[0], hw1 * x4[1])) + float4(fma(hw2, x4[2], hw3 * x4[3]));
                acc[m] += (st.x + st.y) + (st.z + st.w);
            }
        }
    } else {
        const uint grp_per_row = p.cols >> 6u;
        device const uint4* w16 = w_down_q8 + ulong(loc_row) * c16_cnt;
        device const half*  sr  = w_down_s  + ulong(loc_row) * grp_per_row;
        for (uint c16 = tid; c16 < c16_cnt; c16 += 128u) {
            const float sc  = float(sr[c16 >> 2u]);
            const uint4 q16 = w16[c16];
            const half4 hw0 = half4(as_type<char4>(q16.x));
            const half4 hw1 = half4(as_type<char4>(q16.y));
            const half4 hw2 = half4(as_type<char4>(q16.z));
            const half4 hw3 = half4(as_type<char4>(q16.w));
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + c16 * 16u);
                const float4 st = float4(fma(hw0, x4[0], hw1 * x4[1])) + float4(fma(hw2, x4[2], hw3 * x4[3]));
                acc[m] = fma((st.x + st.y) + (st.z + st.w), sc, acc[m]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            float tot = simd_sum((lane < 4u) ? tg_red[m][lane] : 0.0f);
            if (lane == 0u) {
                if (is_inj) {
                    inj_out[ulong(m) * p.hc_count + loc_row] = 2.0f * sqf_sigmoid(tot * p.inv_hc);
                } else {
                    u_out[ulong(m) * p.lowrank + loc_row] = sqf_clamp_f16(sqf_silu(tot * p.inv_hc));
                }
            }
        }
    }
}

kernel void sqf_q8_hc_down_inj_act_batch(
    device const uint4*          w_down_q8 [[buffer(0)]],
    device const half*           w_down_s  [[buffer(1)]],
    device const half*           w_inj     [[buffer(2)]],
    device const half*           x_in      [[buffer(3)]],
    device half*                 u_out     [[buffer(4)]],
    device float*                inj_out   [[buffer(5)]],
    constant SqfHcDownInjParams& p         [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][4];
    const uint total_rows = p.lowrank + (p.has_inject != 0u ? p.hc_count : 0u);
    if (row >= total_rows) return;
    #define SQF_Q8_HCDI_CASE(MF) { sqf_q8_hc_down_inj_impl<MF>(w_down_q8, w_down_s, w_inj, x_in, u_out, inj_out, tg_red, p, row, tid, sgid, lane); break; }
    switch (SQF_CONST_M) {
        case 1u: SQF_Q8_HCDI_CASE(1u)
        case 2u: SQF_Q8_HCDI_CASE(2u)
        case 3u: SQF_Q8_HCDI_CASE(3u)
        case 4u: SQF_Q8_HCDI_CASE(4u)
        case 5u: SQF_Q8_HCDI_CASE(5u)
        case 6u: SQF_Q8_HCDI_CASE(6u)
        case 7u: SQF_Q8_HCDI_CASE(7u)
        case 8u: SQF_Q8_HCDI_CASE(8u)
        default: break;
    }
    #undef SQF_Q8_HCDI_CASE
}

// Single-Dispatch Zero-Barrier Fused `Q8_0` `hc.input_mix_weight_up [4*D, R]` + `hc_blend`!
// Eliminates 97 `sqf_hc_blend` dispatches AND `b_hc_up_f32` read/write traffic per step (100% bit-exact!).
// Grid: `((D + 7) / 8, 1, 1)` (`320` TGs x `128` th = `8` d/TG, `16` lanes/d = `4` streams x `4` lanes/Q8 row)
struct SqfQ8HcUpBlendParams {
    uint  M_valid;
    uint  D;
    uint  lowrank;
    float inv_hc;
    uint  write_f32;
    uint  pad0;
};

template <uint M_FIX>
inline void sqf_q8_hc_up_blend_impl(
    device const uint4*            w_q8,
    device const half*             w_s,
    device const half*             u_in,
    device const float*            h_norm_f32,
    device half*                   x_out_f16,
    device float*                  x_out_f32,
    constant SqfQ8HcUpBlendParams& p,
    uint d,
    uint s,
    uint lane4,
    uint lane
) {
    const uint row         = s * p.D + d;
    const uint grp_per_row = p.lowrank >> 6u;
    device const uint4* w16 = w_q8 + ulong(row) * (p.lowrank >> 4u);
    device const half*  sr  = w_s  + ulong(row) * grp_per_row;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint g = 0u; g < grp_per_row; ++g) {
        const float sc  = float(sr[g]);
        const uint  c16 = g * 4u + lane4;
        const uint4 q16 = w16[c16];
        const half4 hw0 = half4(as_type<char4>(q16.x));
        const half4 hw1 = half4(as_type<char4>(q16.y));
        const half4 hw2 = half4(as_type<char4>(q16.z));
        const half4 hw3 = half4(as_type<char4>(q16.w));
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(u_in + ulong(m) * p.lowrank + c16 * 16u);
            const float4 st = float4(fma(hw0, x4[0], hw1 * x4[1])) + float4(fma(hw2, x4[2], hw3 * x4[3]));
            acc[m] = fma((st.x + st.y) + (st.z + st.w), sc, acc[m]);
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float v = acc[m];
        v += simd_shuffle_xor(v, 2u);
        v += simd_shuffle_xor(v, 1u);

        float mw = 0.0f;
        float hn = 0.0f;
        if (lane4 == 0u) {
            mw = 1.0f / (1.0f + exp(-v));
            hn = h_norm_f32[(ulong(m) * 4ul + ulong(s)) * p.D + d];
        }
        const float w0 = mw, h0 = hn;
        const float w1 = simd_shuffle_down(mw, 4u),  h1 = simd_shuffle_down(hn, 4u);
        const float w2 = simd_shuffle_down(mw, 8u),  h2 = simd_shuffle_down(hn, 8u);
        const float w3 = simd_shuffle_down(mw, 12u), h3 = simd_shuffle_down(hn, 12u);

        if ((lane & 15u) == 0u) {
            float blend = fma(w0, h0, 0.0f);
            blend = fma(w1, h1, blend);
            blend = fma(w2, h2, blend);
            blend = fma(w3, h3, blend);
            const float xv      = blend * p.inv_hc;
            const ulong out_idx = ulong(m) * p.D + d;
            x_out_f16[out_idx]  = sqf_clamp_f16(xv);
            if (p.write_f32 != 0u) x_out_f32[out_idx] = xv;
        }
    }
}

kernel void sqf_q8_hc_up_blend(
    device const uint4*            w_q8       [[buffer(0)]],
    device const half*             w_s        [[buffer(1)]],
    device const half*             u_in       [[buffer(2)]],
    device const float*            h_norm_f32 [[buffer(3)]],
    device half*                   x_out_f16  [[buffer(4)]],
    device float*                  x_out_f32  [[buffer(5)]],
    constant SqfQ8HcUpBlendParams& p          [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint half_sg = lane >> 4u;
    const uint d       = (tgid * 4u + sgid) * 2u + half_sg;
    if (d >= p.D) return;
    const uint s     = (lane >> 2u) & 3u;
    const uint lane4 = lane & 3u;
    #define SQF_Q8_HCUB_CASE(MF) { sqf_q8_hc_up_blend_impl<MF>(w_q8, w_s, u_in, h_norm_f32, x_out_f16, x_out_f32, p, d, s, lane4, lane); break; }
    switch (SQF_CONST_M) {
        case 1u: SQF_Q8_HCUB_CASE(1u)
        case 2u: SQF_Q8_HCUB_CASE(2u)
        case 3u: SQF_Q8_HCUB_CASE(3u)
        case 4u: SQF_Q8_HCUB_CASE(4u)
        case 5u: SQF_Q8_HCUB_CASE(5u)
        case 6u: SQF_Q8_HCUB_CASE(6u)
        case 7u: SQF_Q8_HCUB_CASE(7u)
        case 8u: SQF_Q8_HCUB_CASE(8u)
        default: break;
    }
    #undef SQF_Q8_HCUB_CASE
}

// Single-Dispatch Zero-Barrier Fused `Q4_A` `shared_gate_proj` + `shared_up_proj` + `SwiGLU` -> `mid_f16`!
// Grid: `((d_ff + 3) / 4, 1, 1)` (`160` TGs = `640` SGs across 40 cores), `128` threads/TG (`4` SGs, `1` row/SG)
template <uint M_FIX>
inline void sqf_q4_shared_gu_silu_impl(
    device const uint2* wg_u2,
    device const half*  wg_s,
    device const half*  wg_b,
    device const uint2* wu_u2,
    device const half*  wu_s,
    device const half*  wu_b,
    device const half*  x_in,
    device half*        mid_out,
    uint row,
    uint d_ff,
    uint cols,
    uint lane
) {
    const uint u2_per_row  = cols >> 4u;
    const uint grp_per_row = cols >> 6u;

    float ga[M_FIX];
    float ua[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) { ga[m] = 0.0f; ua[m] = 0.0f; }

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp = u2_idx >> 2u;
        const uint2 qg  = wg_u2[ulong(row) * u2_per_row + u2_idx];
        const uint2 qu  = wu_u2[ulong(row) * u2_per_row + u2_idx];

        const float sg = float(wg_s[ulong(row) * grp_per_row + grp]);
        const float bg = float(wg_b[ulong(row) * grp_per_row + grp]);
        const float su = float(wu_s[ulong(row) * grp_per_row + grp]);
        const float bu = float(wu_b[ulong(row) * grp_per_row + grp]);

        half4 g0, g1, g2, g3;
        half4 u0, u1, u2, u3;
        sqf_unpack_u32_nibbles_h4(qg.x, g0, g1); sqf_unpack_u32_nibbles_h4(qg.y, g2, g3);
        sqf_unpack_u32_nibbles_h4(qu.x, u0, u1); sqf_unpack_u32_nibbles_h4(qu.y, u2, u3);

        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + u2_idx * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float xs = sqf_sum_h4x4(x0, x1, x2, x3);

            ga[m] = fma(sg, sqf_dot_h4x4(g0, g1, g2, g3, x0, x1, x2, x3), fma(bg, xs, ga[m]));
            ua[m] = fma(su, sqf_dot_h4x4(u0, u1, u2, u3, x0, x1, x2, x3), fma(bu, xs, ua[m]));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float gv = simd_sum(ga[m]);
        const float uv = simd_sum(ua[m]);
        if (lane == 0u) {
            mid_out[ulong(m) * d_ff + row] = sqf_clamp_f16(sqf_silu(gv) * uv);
        }
    }
}

kernel void sqf_q4_shared_gate_up_silu_batch(
    device const uint2*          wg_u2   [[buffer(0)]],
    device const half*           wg_s    [[buffer(1)]],
    device const half*           wg_b    [[buffer(2)]],
    device const uint2*          wu_u2   [[buffer(3)]],
    device const half*           wu_s    [[buffer(4)]],
    device const half*           wu_b    [[buffer(5)]],
    device const half*           x_in    [[buffer(6)]],
    device half*                 mid_out [[buffer(7)]],
    constant SqfGemvBatchParams& p       [[buffer(8)]], // rows = shared_dff, cols = D
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid * 4u + sgid;
    if (row >= p.rows) return;
    #define SQF_Q4_SHGU_CASE(MF) { sqf_q4_shared_gu_silu_impl<MF>(wg_u2, wg_s, wg_b, wu_u2, wu_s, wu_b, x_in, mid_out, row, p.rows, p.cols, lane); break; }
    switch (SQF_CONST_M) {
        case 1u: SQF_Q4_SHGU_CASE(1u)
        case 2u: SQF_Q4_SHGU_CASE(2u)
        case 3u: SQF_Q4_SHGU_CASE(3u)
        case 4u: SQF_Q4_SHGU_CASE(4u)
        case 5u: SQF_Q4_SHGU_CASE(5u)
        case 6u: SQF_Q4_SHGU_CASE(6u)
        case 7u: SQF_Q4_SHGU_CASE(7u)
        case 8u: SQF_Q4_SHGU_CASE(8u)
        default: break;
    }
    #undef SQF_Q4_SHGU_CASE
}

/* ============================================================================
 * 4. Batched Prefill Metal 4.1 `NAX` (`mpp::tensor_ops::matmul2d`) Q4, Q8, FP16
 *    & Fused Hyper-Connection / Shared-Expert Kernels
 * ============================================================================ */

struct SqfNaxGemmParams {
    uint M_valid;
    uint M_padded;
    uint N_logical;
    uint N_padded;  // Multiple of 64/128
    uint K;         // Multiple of 64 (fast BK=128 path when K % 128 == 0)
    uint out_mode;  // SQF_OUT_*
    uint pad0;
    uint pad1;
};

struct SqfHcDownNaxParams {
    uint  M_valid;
    uint  M_padded;
    uint  lowrank;
    uint  cols;
    float inv_hc;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

struct SqfHcUpBlendNaxParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    uint  lowrank;
    uint  hc_count;
    float inv_hc;
    uint  write_f32;
    uint  pad0;
};

struct SqfHcInjectPrefillParams {
    uint  M_valid;
    uint  hc_count;
    uint  cols;
    float inv_hc;
};

struct SqfSharedGuNaxParams {
    uint M_valid;
    uint M_padded;
    uint d_ff;
    uint K;
};

#if defined(__HAVE_TENSOR__)
// ---- Direct-Device `static_slice` `K=dynamic_extent` FP16 NAX (`58.6 TF`, 0 TG mem, 0 barriers!) ----
template <int BM, int BN, int SG>
inline void sqf_f16_gemm_nax_direct_impl(
    device const half*         w_f16,
    device const half*         x_in,
    device half*               y_f16,
    device float*              y_f32,
    constant SqfNaxGemmParams& p,
    uint2                      tgid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * uint(BN);
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(const_cast<device half*>(x_in),  dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tW(const_cast<device half*>(w_f16), dextents<int, 2>(int(p.K), int(p.N_logical)));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX = tX.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW = tW.template slice<dynamic_extent, BN>(0, int(n0));

    if (p.out_mode == SQF_OUT_F32 && (p.N_logical % uint(BN)) == 0u) {
        tensor<device float, dextents<int, 2>, tensor_inline> tY(y_f32, dextents<int, 2>(int(p.N_logical), int(p.M_padded)));
        auto sY = tY.template slice<BN, BM>(int(n0), int(m0));
        op.run(sX, sW, sY);
        return;
    }

    auto cT = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sW)>, float>();
    op.run(sX, sW, cT);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_valid && n_glob < p.N_logical) {
                ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                if (p.out_mode == SQF_OUT_F16)      y_f16[out_idx]  = sqf_clamp_f16(cT[i]);
                else if (p.out_mode == SQF_OUT_F32) y_f32[out_idx]  = cT[i];
                else                                y_f32[out_idx] += cT[i];
            }
        }
    }
}

// ---- Q4_A (gs=64) Prefill NAX: `uint2` 8B loads + Dual-Rate FP16 unpack + `BK=128` (dodges K=64 cliff!) ----
template <int BM, int BK, int SG>
inline void sqf_q4_gemm_nax_impl(
    device const uint2*        w_u2,
    device const half*         scales,
    device const half*         biases,
    device const half*         x_in,
    device half*               y_f16,
    device float*              y_f32,
    constant SqfNaxGemmParams& p,
    threadgroup half           (&W_tile)[64][BK],
    uint2                      tgid,
    uint                       tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(const_cast<device half*>(x_in), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup half, extents<int, BK, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, BK, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<BK, BM>(0, int(m0));
    auto cT  = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX0)>, sqf_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint u2_per_row    = p.K >> 4u;
    const uint grp_per_row   = p.K >> 6u;
    constexpr uint u2_per_kt = uint(BK) >> 4u;          // 8 for BK=128, 4 for BK=64
    constexpr uint sub_shift = (BK == 128) ? 3u : 2u;   // log2(u2_per_kt)
    constexpr uint r_step    = (uint(SG) * 32u) >> sub_shift;
    constexpr uint q_iters   = 64u / r_step;
    const uint r_grp         = tid >> sub_shift;
    const uint sub           = tid & (u2_per_kt - 1u);
    const uint kt_cnt        = p.K / uint(BK);

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        const uint col_u2 = kt * u2_per_kt + sub;
        const uint grp    = col_u2 >> 2u;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint  r_loc  = q * r_step + r_grp;
            const uint  r_glob = n0 + r_loc;
            const uint2 qv     = w_u2[ulong(r_glob) * u2_per_row + col_u2];
            const half4 vsc    = half4(scales[ulong(r_glob) * grp_per_row + grp]);
            const half4 vbi    = half4(biases[ulong(r_glob) * grp_per_row + grp]);

            half4 n0h, n1h, n2h, n3h;
            sqf_unpack_u32_nibbles_h4(qv.x, n0h, n1h);
            sqf_unpack_u32_nibbles_h4(qv.y, n2h, n3h);

            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            dst[0] = fma(n0h, vsc, vbi);
            dst[1] = fma(n1h, vsc, vbi);
            dst[2] = fma(n2h, vsc, vbi);
            dst[3] = fma(n3h, vsc, vbi);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX.template slice<BK, BM>(int(kt * uint(BK)), int(m0));
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.out_mode == SQF_OUT_F32 && (p.N_logical % 64u) == 0u) {
        tensor<device float, dextents<int, 2>, tensor_inline> tY(y_f32, dextents<int, 2>(int(p.N_logical), int(p.M_padded)));
        auto sY = tY.template slice<64, BM>(int(n0), int(m0));
        cT.store(sY);
        return;
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_valid && n_glob < p.N_logical) {
                ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                if (p.out_mode == SQF_OUT_F16)      y_f16[out_idx]  = sqf_clamp_f16(cT[i]);
                else if (p.out_mode == SQF_OUT_F32) y_f32[out_idx]  = cT[i];
                else                                y_f32[out_idx] += cT[i];
            }
        }
    }
}

// ---- Q8_0 (gs=64) Prefill NAX: `uint4` 16B loads + `BK=128` + `cT.store` ----
template <int BM, int BK, int SG>
inline void sqf_q8_gemm_nax_impl(
    device const uint4*        w_u4,
    device const half*         scales,
    device const half*         x_in,
    device half*               y_f16,
    device float*              y_f32,
    constant SqfNaxGemmParams& p,
    threadgroup half           (&W_tile)[64][BK],
    uint2                      tgid,
    uint                       tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(const_cast<device half*>(x_in), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup half, extents<int, BK, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, BK, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<BK, BM>(0, int(m0));
    auto cT  = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX0)>, sqf_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint u4_per_row  = p.K >> 4u;
    const uint grp_per_row = p.K >> 6u;
    constexpr uint u4_per_kt = uint(BK) >> 4u;          // 8 for BK=128, 4 for BK=64
    constexpr uint sub_shift = (BK == 128) ? 3u : 2u;
    constexpr uint r_step    = (uint(SG) * 32u) >> sub_shift;
    constexpr uint q_iters   = 64u / r_step;
    const uint r_grp         = tid >> sub_shift;
    const uint sub           = tid & (u4_per_kt - 1u);
    const uint kt_cnt        = p.K / uint(BK);

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        const uint col_u4 = kt * u4_per_kt + sub;
        const uint grp    = col_u4 >> 2u;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint r_loc  = q * r_step + r_grp;
            const uint r_glob = n0 + r_loc;
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            if (r_glob < p.N_logical) {
                const half  sc  = scales[ulong(r_glob) * grp_per_row + grp];
                const uint4 q16 = w_u4[ulong(r_glob) * u4_per_row + col_u4];
                dst[0] = half4(as_type<char4>(q16.x)) * sc;
                dst[1] = half4(as_type<char4>(q16.y)) * sc;
                dst[2] = half4(as_type<char4>(q16.z)) * sc;
                dst[3] = half4(as_type<char4>(q16.w)) * sc;
            } else {
                dst[0] = half4(0.0h); dst[1] = half4(0.0h);
                dst[2] = half4(0.0h); dst[3] = half4(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX.template slice<BK, BM>(int(kt * uint(BK)), int(m0));
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.out_mode == SQF_OUT_F32 && (p.N_logical % 64u) == 0u) {
        tensor<device float, dextents<int, 2>, tensor_inline> tY(y_f32, dextents<int, 2>(int(p.N_logical), int(p.M_padded)));
        auto sY = tY.template slice<64, BM>(int(n0), int(m0));
        cT.store(sY);
        return;
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_valid && n_glob < p.N_logical) {
                ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                if (p.out_mode == SQF_OUT_F16)      y_f16[out_idx]  = sqf_clamp_f16(cT[i]);
                else if (p.out_mode == SQF_OUT_F32) y_f32[out_idx]  = cT[i];
                else                                y_f32[out_idx] += cT[i];
            }
        }
    }
}

// ---- Fused Q4_A Shared Expert `gate_proj + up_proj + SwiGLU` -> `mid_f16` (`BK=128`, 1 dispatch!) ----
template <int BM, int SG>
inline void sqf_q4_shared_gu_silu_nax_impl(
    device const uint2*            wg_u2,
    device const half*             wg_s,
    device const half*             wg_b,
    device const uint2*            wu_u2,
    device const half*             wu_s,
    device const half*             wu_b,
    device const half*             x_in,
    device half*                   mid_out,
    constant SqfSharedGuNaxParams& p,
    threadgroup half               (&W_tile)[64][128],
    uint2                          tgid,
    uint                           tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.d_ff) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(const_cast<device half*>(x_in), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0   = tX.template slice<128, BM>(0, int(m0));
    auto cGate = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX0)>, sqf_strip_as<decltype(tW)>, float>();
    auto cUp   = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX0)>, sqf_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cGate.get_capacity(); ++i) {
        if (cGate.is_valid_element(i)) { cGate[i] = 0.0f; cUp[i] = 0.0f; }
    }

    const uint u2_per_row  = p.K >> 4u;
    const uint grp_per_row = p.K >> 6u;
    constexpr uint r_step  = (uint(SG) * 32u) >> 3u;
    constexpr uint q_iters = 64u / r_step;
    const uint r_grp       = tid >> 3u;
    const uint sub         = tid & 7u;
    const uint kt_cnt      = p.K >> 7u;

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        const uint col_u2 = kt * 8u + sub;
        const uint grp    = col_u2 >> 2u;
        auto sX           = tX.template slice<128, BM>(int(kt * 128u), int(m0));

        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint  r_loc  = q * r_step + r_grp;
            const uint  r_glob = n0 + r_loc;
            const uint2 qv     = wg_u2[ulong(r_glob) * u2_per_row + col_u2];
            const half4 vsc    = half4(wg_s[ulong(r_glob) * grp_per_row + grp]);
            const half4 vbi    = half4(wg_b[ulong(r_glob) * grp_per_row + grp]);
            half4 n0h, n1h, n2h, n3h;
            sqf_unpack_u32_nibbles_h4(qv.x, n0h, n1h);
            sqf_unpack_u32_nibbles_h4(qv.y, n2h, n3h);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            dst[0] = fma(n0h, vsc, vbi);
            dst[1] = fma(n1h, vsc, vbi);
            dst[2] = fma(n2h, vsc, vbi);
            dst[3] = fma(n3h, vsc, vbi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(sX, tW, cGate);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint  r_loc  = q * r_step + r_grp;
            const uint  r_glob = n0 + r_loc;
            const uint2 qv     = wu_u2[ulong(r_glob) * u2_per_row + col_u2];
            const half4 vsc    = half4(wu_s[ulong(r_glob) * grp_per_row + grp]);
            const half4 vbi    = half4(wu_b[ulong(r_glob) * grp_per_row + grp]);
            half4 n0h, n1h, n2h, n3h;
            sqf_unpack_u32_nibbles_h4(qv.x, n0h, n1h);
            sqf_unpack_u32_nibbles_h4(qv.y, n2h, n3h);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            dst[0] = fma(n0h, vsc, vbi);
            dst[1] = fma(n1h, vsc, vbi);
            dst[2] = fma(n2h, vsc, vbi);
            dst[3] = fma(n3h, vsc, vbi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(sX, tW, cUp);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cGate.get_capacity(); ++i) {
        if (cGate.is_valid_element(i)) {
            auto idx    = cGate.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_padded && n_glob < p.d_ff) {
                mid_out[ulong(m_glob) * p.d_ff + n_glob] =
                    (m_glob < p.M_valid) ? sqf_clamp_f16(sqf_silu(cGate[i]) * cUp[i]) : half(0.0h);
            }
        }
    }
}

// ---- Fused HC Prefill `Q8_0` `input_mix_weight_down [R, 4*D]` + `silu(x * inv_hc)` -> `u_f16` (`BK=128`!) ----
template <int BM, int SG>
inline void sqf_q8_hc_down_act_nax_impl(
    device const uint4*          w_u4,
    device const half*           scales,
    device const half*           x_in,
    device half*                 u_out,
    constant SqfHcDownNaxParams& p,
    threadgroup half             (&W_tile)[64][128],
    uint2                        tgid,
    uint                         tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.lowrank) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(const_cast<device half*>(x_in), dextents<int, 2>(int(p.cols), int(p.M_padded)));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<128, BM>(0, int(m0));
    auto cT  = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX0)>, sqf_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint u4_per_row  = p.cols >> 4u;
    const uint grp_per_row = p.cols >> 6u;
    constexpr uint r_step  = (uint(SG) * 32u) >> 3u;
    constexpr uint q_iters = 64u / r_step;
    const uint r_grp       = tid >> 3u;
    const uint sub         = tid & 7u;
    const uint kt_cnt      = p.cols >> 7u;

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        const uint col_u4 = kt * 8u + sub;
        const uint grp    = col_u4 >> 2u;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint r_loc  = q * r_step + r_grp;
            const uint r_glob = n0 + r_loc;
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            const half  sc  = scales[ulong(r_glob) * grp_per_row + grp];
            const uint4 q16 = w_u4[ulong(r_glob) * u4_per_row + col_u4];
            dst[0] = half4(as_type<char4>(q16.x)) * sc;
            dst[1] = half4(as_type<char4>(q16.y)) * sc;
            dst[2] = half4(as_type<char4>(q16.z)) * sc;
            dst[3] = half4(as_type<char4>(q16.w)) * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX.template slice<128, BM>(int(kt * 128u), int(m0));
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_padded && n_glob < p.lowrank) {
                u_out[ulong(m_glob) * p.lowrank + n_glob] =
                    (m_glob < p.M_valid) ? sqf_clamp_f16(sqf_silu(cT[i] * p.inv_hc)) : half(0.0h);
            }
        }
    }
}

// ---- Fused HC Prefill `F16` Direct-Device `input_mix_weight_down [R, 4*D]` + `silu(x * inv_hc)` -> `u_f16` ----
template <int BM, int SG>
inline void sqf_f16_hc_down_act_nax_impl(
    device const half*           w_f16,
    device const half*           x_in,
    device half*                 u_out,
    constant SqfHcDownNaxParams& p,
    uint2                        tgid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.lowrank) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(const_cast<device half*>(x_in),  dextents<int, 2>(int(p.cols), int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tW(const_cast<device half*>(w_f16), dextents<int, 2>(int(p.cols), int(p.lowrank)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX = tX.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW = tW.template slice<dynamic_extent, 64>(0, int(n0));
    auto cT = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sW)>, float>();
    op.run(sX, sW, cT);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_padded && n_glob < p.lowrank) {
                u_out[ulong(m_glob) * p.lowrank + n_glob] =
                    (m_glob < p.M_valid) ? sqf_clamp_f16(sqf_silu(cT[i] * p.inv_hc)) : half(0.0h);
            }
        }
    }
}

// ---- Fused HC Prefill `Q8_0` `input_mix_weight_up [4*D, R]` + `hc_blend` -> `x_out` (ELIMINATES `b_hc_up_f32`!) ----
// Uses `BK=128` (zero-padded tail on final K-tile when R=320 -> 3 tiles instead of 5!) and loops all 4 streams inside TG!
template <int BM, int SG>
inline void sqf_q8_hc_up_blend_nax_impl(
    device const uint4*             w_u4,
    device const half*              scales,
    device const half*              u_in,
    device const half*              h_norm_f16,
    device half*                    x_out_f16,
    device float*                   x_out_f32,
    constant SqfHcUpBlendNaxParams& p,
    threadgroup half                (&W_tile)[64][128],
    uint2                           tgid,
    uint                            tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint d0 = tgid.x * 64u;
    if (m0 >= p.M_padded || d0 >= p.D) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tU(const_cast<device half*>(u_in), dextents<int, 2>(int(p.lowrank), int(p.M_padded)));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sU0  = tU.template slice<128, BM>(0, int(m0));
    auto cAcc = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sU0)>, sqf_strip_as<decltype(tW)>, float>();
    auto cUp  = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sU0)>, sqf_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cAcc.get_capacity(); ++i) {
        if (cAcc.is_valid_element(i)) cAcc[i] = 0.0f;
    }

    const uint u4_per_row  = p.lowrank >> 4u;
    const uint grp_per_row = p.lowrank >> 6u;
    const uint kt_cnt      = (p.lowrank + 127u) >> 7u;
    constexpr uint r_step  = (uint(SG) * 32u) >> 3u;
    constexpr uint q_iters = 64u / r_step;
    const uint r_grp       = tid >> 3u;
    const uint sub         = tid & 7u;
    const ulong hc_dim     = ulong(p.hc_count) * p.D;

    for (uint s = 0u; s < p.hc_count; ++s) {
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cUp.get_capacity(); ++i) {
            if (cUp.is_valid_element(i)) cUp[i] = 0.0f;
        }
        const uint row_base = s * p.D + d0;
        for (uint kt = 0u; kt < kt_cnt; ++kt) {
            const uint col_u4 = kt * 8u + sub;
            const bool in_k   = (col_u4 < u4_per_row);
            const uint grp    = col_u4 >> 2u;
            #pragma clang loop unroll(full)
            for (uint q = 0u; q < q_iters; ++q) {
                const uint r_loc  = q * r_step + r_grp;
                const uint r_glob = row_base + r_loc;
                threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
                if (in_k) {
                    const half  sc  = scales[ulong(r_glob) * grp_per_row + grp];
                    const uint4 q16 = w_u4[ulong(r_glob) * u4_per_row + col_u4];
                    dst[0] = half4(as_type<char4>(q16.x)) * sc;
                    dst[1] = half4(as_type<char4>(q16.y)) * sc;
                    dst[2] = half4(as_type<char4>(q16.z)) * sc;
                    dst[3] = half4(as_type<char4>(q16.w)) * sc;
                } else {
                    dst[0] = half4(0.0h); dst[1] = half4(0.0h);
                    dst[2] = half4(0.0h); dst[3] = half4(0.0h);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto sU = tU.template slice<128, BM>(int(kt * 128u), int(m0));
            op.run(sU, tW, cUp);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cUp.get_capacity(); ++i) {
            if (cUp.is_valid_element(i)) {
                auto idx    = cUp.get_multidimensional_index(i);
                uint m_glob = m0 + uint(idx[1]);
                uint d_glob = d0 + uint(idx[0]);
                if (m_glob < p.M_valid && d_glob < p.D) {
                    const float hn = float(h_norm_f16[ulong(m_glob) * hc_dim + ulong(s) * p.D + d_glob]);
                    cAcc[i] = fma(sqf_sigmoid(cUp[i]), hn, cAcc[i]);
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cAcc.get_capacity(); ++i) {
        if (cAcc.is_valid_element(i)) {
            auto idx    = cAcc.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint d_glob = d0 + uint(idx[0]);
            if (m_glob < p.M_padded && d_glob < p.D) {
                const ulong out_idx = ulong(m_glob) * p.D + d_glob;
                if (m_glob < p.M_valid) {
                    const float xv = cAcc[i] * p.inv_hc;
                    x_out_f16[out_idx] = sqf_clamp_f16(xv);
                    if (p.write_f32 != 0u) x_out_f32[out_idx] = xv;
                } else {
                    x_out_f16[out_idx] = half(0.0h);
                }
            }
        }
    }
}

// ---- Fused HC Prefill `F16` Direct-Device `input_mix_weight_up [4*D, R]` + `hc_blend` (0 TG mem, 0 barriers!) ----
template <int BM, int SG>
inline void sqf_f16_hc_up_blend_nax_impl(
    device const half*              w_f16,
    device const half*              u_in,
    device const half*              h_norm_f16,
    device half*                    x_out_f16,
    device float*                   x_out_f32,
    constant SqfHcUpBlendNaxParams& p,
    uint2                           tgid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint d0 = tgid.x * 64u;
    if (m0 >= p.M_padded || d0 >= p.D) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tU(const_cast<device half*>(u_in),  dextents<int, 2>(int(p.lowrank), int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tW(const_cast<device half*>(w_f16), dextents<int, 2>(int(p.lowrank), int(p.hc_count * p.D)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sU   = tU.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW0  = tW.template slice<dynamic_extent, 64>(0, int(d0));
    auto cAcc = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sU)>, sqf_strip_as<decltype(sW0)>, float>();
    auto cUp  = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sU)>, sqf_strip_as<decltype(sW0)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cAcc.get_capacity(); ++i) {
        if (cAcc.is_valid_element(i)) cAcc[i] = 0.0f;
    }

    const ulong hc_dim = ulong(p.hc_count) * p.D;
    for (uint s = 0u; s < p.hc_count; ++s) {
        auto sW = tW.template slice<dynamic_extent, 64>(0, int(s * p.D + d0));
        op.run(sU, sW, cUp);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cUp.get_capacity(); ++i) {
            if (cUp.is_valid_element(i)) {
                auto idx    = cUp.get_multidimensional_index(i);
                uint m_glob = m0 + uint(idx[1]);
                uint d_glob = d0 + uint(idx[0]);
                if (m_glob < p.M_valid && d_glob < p.D) {
                    const float hn = float(h_norm_f16[ulong(m_glob) * hc_dim + ulong(s) * p.D + d_glob]);
                    cAcc[i] = fma(sqf_sigmoid(cUp[i]), hn, cAcc[i]);
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cAcc.get_capacity(); ++i) {
        if (cAcc.is_valid_element(i)) {
            auto idx    = cAcc.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint d_glob = d0 + uint(idx[0]);
            if (m_glob < p.M_padded && d_glob < p.D) {
                const ulong out_idx = ulong(m_glob) * p.D + d_glob;
                if (m_glob < p.M_valid) {
                    const float xv = cAcc[i] * p.inv_hc;
                    x_out_f16[out_idx] = sqf_clamp_f16(xv);
                    if (p.write_f32 != 0u) x_out_f32[out_idx] = xv;
                } else {
                    x_out_f16[out_idx] = half(0.0h);
                }
            }
        }
    }
}
#endif

// Fused 4-Row Prefill `block_inject [hc_count, 10240]` + `2*sigmoid(x * inv_hc)` -> `inj_out`!
// Reads `x_in [M_valid, 10240]` ONCE for all `4` streams (4x less DRAM traffic vs per-row!), FP32 exact dot!
// Grid: `(1, (M_valid + 3) / 4, 1)`, `256` threads/TG (`8` SGs)
kernel void sqf_f16_hc_inject_prefill(
    device const half*                 w_inj   [[buffer(0)]],
    device const half*                 x_in    [[buffer(1)]],
    device float*                      inj_out [[buffer(2)]],
    constant SqfHcInjectPrefillParams& p       [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]], // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]], // 0..7
    uint  lane [[thread_index_in_simdgroup]]       // 0..31
) {
    const uint m0 = tgid.y * 4u;
    if (m0 >= p.M_valid) return;

    const uint c8_cnt = p.cols >> 3u;
    const uint mb     = min(4u, p.M_valid - m0);
    threadgroup float tg_red[4][4][8];

    if (p.hc_count == 4u) {
        device const half4* w0 = reinterpret_cast<device const half4*>(w_inj + 0ul * p.cols);
        device const half4* w1 = reinterpret_cast<device const half4*>(w_inj + 1ul * p.cols);
        device const half4* w2 = reinterpret_cast<device const half4*>(w_inj + 2ul * p.cols);
        device const half4* w3 = reinterpret_cast<device const half4*>(w_inj + 3ul * p.cols);

        float acc[4][4];
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < 4u; ++m) {
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 4u; ++s) acc[m][s] = 0.0f;
        }

        for (uint c8 = tid; c8 < c8_cnt; c8 += 256u) {
            const float4 w0a = float4(w0[c8 * 2u + 0u]), w0b = float4(w0[c8 * 2u + 1u]);
            const float4 w1a = float4(w1[c8 * 2u + 0u]), w1b = float4(w1[c8 * 2u + 1u]);
            const float4 w2a = float4(w2[c8 * 2u + 0u]), w2b = float4(w2[c8 * 2u + 1u]);
            const float4 w3a = float4(w3[c8 * 2u + 0u]), w3b = float4(w3[c8 * 2u + 1u]);
            for (uint m = 0u; m < mb; ++m) {
                device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m0 + m) * p.cols);
                const float4 xa = float4(x4[c8 * 2u + 0u]);
                const float4 xb = float4(x4[c8 * 2u + 1u]);
                acc[m][0] += dot(w0a, xa) + dot(w0b, xb);
                acc[m][1] += dot(w1a, xa) + dot(w1b, xb);
                acc[m][2] += dot(w2a, xa) + dot(w2b, xb);
                acc[m][3] += dot(w3a, xa) + dot(w3b, xb);
            }
        }

        for (uint m = 0u; m < mb; ++m) {
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 4u; ++s) {
                const float sv = simd_sum(acc[m][s]);
                if (lane == 0u) tg_red[m][s][sgid] = sv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            for (uint m = 0u; m < mb; ++m) {
                #pragma clang loop unroll(full)
                for (uint s = 0u; s < 4u; ++s) {
                    const float tot = simd_sum((lane < 8u) ? tg_red[m][s][lane] : 0.0f);
                    if (lane == 0u) inj_out[ulong(m0 + m) * 4ul + s] = 2.0f * sqf_sigmoid(tot * p.inv_hc);
                }
            }
        }
        return;
    }

    // Generic fallback for hc_count != 4
    for (uint s = 0u; s < p.hc_count; ++s) {
        device const half4* ws = reinterpret_cast<device const half4*>(w_inj + ulong(s) * p.cols);
        for (uint m = 0u; m < mb; ++m) {
            float acc = 0.0f;
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m0 + m) * p.cols);
            for (uint c8 = tid; c8 < c8_cnt; c8 += 256u) {
                acc += dot(float4(ws[c8 * 2u + 0u]), float4(x4[c8 * 2u + 0u]))
                     + dot(float4(ws[c8 * 2u + 1u]), float4(x4[c8 * 2u + 1u]));
            }
            const float sv = simd_sum(acc);
            if (lane == 0u) tg_red[0][0][sgid] = sv;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                const float tot = simd_sum((lane < 8u) ? tg_red[0][0][lane] : 0.0f);
                if (lane == 0u) inj_out[ulong(m0 + m) * p.hc_count + s] = 2.0f * sqf_sigmoid(tot * p.inv_hc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Fused 2D Prefill `lin_in_proj_a [48, 2560]` + `lin_in_proj_b [48, 2560]` (Grid: `(a_rows + b_rows, (M+7)/8, 1)`, 128 th/TG)
kernel void sqf_f16_lin_ab_prefill(
    device const half*       w_a   [[buffer(0)]],
    device const half*       w_b   [[buffer(1)]],
    device const half*       x_in  [[buffer(2)]],
    device float*            out_a [[buffer(3)]],
    device float*            out_b [[buffer(4)]],
    constant SqfLinAbParams& p     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][4];
    const uint row = tgid.x;
    const uint m0  = tgid.y * 8u;
    if (row >= p.a_rows + p.b_rows || m0 >= p.M) return;

    device const half* xb = x_in  + ulong(m0) * p.cols;
    device float*      oa = out_a + ulong(m0) * p.a_rows;
    device float*      ob = out_b + ulong(m0) * p.b_rows;

    if (m0 + 8u <= p.M) {
        sqf_f16_lin_ab_impl<8u>(w_a, w_b, xb, oa, ob, tg_red, p, row, tid, sgid, lane);
    } else {
        const uint mb = p.M - m0;
        for (uint m = 0u; m < mb; ++m) {
            sqf_f16_lin_ab_impl<1u>(w_a, w_b, xb + ulong(m) * p.cols,
                                    oa + ulong(m) * p.a_rows,
                                    ob + ulong(m) * p.b_rows,
                                    tg_red, p, row, tid, sgid, lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// ---- NAX GEMM Kernel Entry Points ----
kernel void sqf_f16_gemm_nax_64x128_sg4(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* y16 [[buffer(2)]], device float* y32 [[buffer(3)]], constant SqfNaxGemmParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sqf_f16_gemm_nax_direct_impl<64, 128, 4>(w, x, y16, y32, p, tgid);
#else
    (void)w;(void)x;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sqf_f16_gemm_nax_64x64_sg4(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* y16 [[buffer(2)]], device float* y32 [[buffer(3)]], constant SqfNaxGemmParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sqf_f16_gemm_nax_direct_impl<64, 64, 4>(w, x, y16, y32, p, tgid);
#else
    (void)w;(void)x;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sqf_f16_gemm_nax_32x64_sg2(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* y16 [[buffer(2)]], device float* y32 [[buffer(3)]], constant SqfNaxGemmParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sqf_f16_gemm_nax_direct_impl<32, 64, 2>(w, x, y16, y32, p, tgid);
#else
    (void)w;(void)x;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sqf_f16_gemm_nax_16x32_sg2(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* y16 [[buffer(2)]], device float* y32 [[buffer(3)]], constant SqfNaxGemmParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sqf_f16_gemm_nax_direct_impl<16, 32, 2>(w, x, y16, y32, p, tgid);
#else
    (void)w;(void)x;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sqf_q4_gemm_nax_64x64_k128_sg4(device const uint2* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* b [[buffer(2)]], device const half* x [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SqfNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q4_gemm_nax_impl<64, 128, 4>(w, s, b, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)b;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q4_gemm_nax_32x64_k128_sg2(device const uint2* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* b [[buffer(2)]], device const half* x [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SqfNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q4_gemm_nax_impl<32, 128, 2>(w, s, b, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)b;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q4_gemm_nax_16x64_k128_sg2(device const uint2* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* b [[buffer(2)]], device const half* x [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SqfNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q4_gemm_nax_impl<16, 128, 2>(w, s, b, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)b;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q4_gemm_nax_m16(device const uint2* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* b [[buffer(2)]], device const half* x [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SqfNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sqf_q4_gemm_nax_impl<16, 64, 2>(w, s, b, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)b;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q4_gemm_nax_m64(device const uint2* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* b [[buffer(2)]], device const half* x [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SqfNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sqf_q4_gemm_nax_impl<64, 64, 4>(w, s, b, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)b;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q8_gemm_nax_64x64_k128_sg4(device const uint4* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* x [[buffer(2)]], device half* y16 [[buffer(3)]], device float* y32 [[buffer(4)]], constant SqfNaxGemmParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q8_gemm_nax_impl<64, 128, 4>(w, s, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q8_gemm_nax_32x64_k128_sg2(device const uint4* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* x [[buffer(2)]], device half* y16 [[buffer(3)]], device float* y32 [[buffer(4)]], constant SqfNaxGemmParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q8_gemm_nax_impl<32, 128, 2>(w, s, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q8_gemm_nax_16x64_k128_sg2(device const uint4* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* x [[buffer(2)]], device half* y16 [[buffer(3)]], device float* y32 [[buffer(4)]], constant SqfNaxGemmParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q8_gemm_nax_impl<16, 128, 2>(w, s, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q8_gemm_nax_m16(device const uint4* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* x [[buffer(2)]], device half* y16 [[buffer(3)]], device float* y32 [[buffer(4)]], constant SqfNaxGemmParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sqf_q8_gemm_nax_impl<16, 64, 2>(w, s, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q8_gemm_nax_m64(device const uint4* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* x [[buffer(2)]], device half* y16 [[buffer(3)]], device float* y32 [[buffer(4)]], constant SqfNaxGemmParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sqf_q8_gemm_nax_impl<64, 64, 4>(w, s, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q4_shared_gu_silu_nax_m64(device const uint2* wg [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]], device const uint2* wu [[buffer(3)]], device const half* us [[buffer(4)]], device const half* ub [[buffer(5)]], device const half* x [[buffer(6)]], device half* mo [[buffer(7)]], constant SqfSharedGuNaxParams& p [[buffer(8)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q4_shared_gu_silu_nax_impl<64, 4>(wg, ws, wb, wu, us, ub, x, mo, p, W_tile, tgid, tid);
#else
    (void)wg;(void)ws;(void)wb;(void)wu;(void)us;(void)ub;(void)x;(void)mo;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q4_shared_gu_silu_nax_m32(device const uint2* wg [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]], device const uint2* wu [[buffer(3)]], device const half* us [[buffer(4)]], device const half* ub [[buffer(5)]], device const half* x [[buffer(6)]], device half* mo [[buffer(7)]], constant SqfSharedGuNaxParams& p [[buffer(8)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q4_shared_gu_silu_nax_impl<32, 2>(wg, ws, wb, wu, us, ub, x, mo, p, W_tile, tgid, tid);
#else
    (void)wg;(void)ws;(void)wb;(void)wu;(void)us;(void)ub;(void)x;(void)mo;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q8_hc_down_act_nax_m64(device const uint4* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* x [[buffer(2)]], device half* u [[buffer(3)]], constant SqfHcDownNaxParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q8_hc_down_act_nax_impl<64, 4>(w, s, x, u, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)u;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q8_hc_down_act_nax_m32(device const uint4* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* x [[buffer(2)]], device half* u [[buffer(3)]], constant SqfHcDownNaxParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q8_hc_down_act_nax_impl<32, 2>(w, s, x, u, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)u;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_f16_hc_down_act_nax_m64(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* u [[buffer(2)]], constant SqfHcDownNaxParams& p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sqf_f16_hc_down_act_nax_impl<64, 4>(w, x, u, p, tgid);
#else
    (void)w;(void)x;(void)u;(void)p;(void)tgid;
#endif
}

kernel void sqf_f16_hc_down_act_nax_m32(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* u [[buffer(2)]], constant SqfHcDownNaxParams& p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sqf_f16_hc_down_act_nax_impl<32, 2>(w, x, u, p, tgid);
#else
    (void)w;(void)x;(void)u;(void)p;(void)tgid;
#endif
}

kernel void sqf_q8_hc_up_blend_nax_m64(device const uint4* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* u [[buffer(2)]], device const half* hn [[buffer(3)]], device half* x16 [[buffer(4)]], device float* x32 [[buffer(5)]], constant SqfHcUpBlendNaxParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q8_hc_up_blend_nax_impl<64, 4>(w, s, u, hn, x16, x32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)u;(void)hn;(void)x16;(void)x32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_q8_hc_up_blend_nax_m32(device const uint4* w [[buffer(0)]], device const half* s [[buffer(1)]], device const half* u [[buffer(2)]], device const half* hn [[buffer(3)]], device half* x16 [[buffer(4)]], device float* x32 [[buffer(5)]], constant SqfHcUpBlendNaxParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sqf_q8_hc_up_blend_nax_impl<32, 2>(w, s, u, hn, x16, x32, p, W_tile, tgid, tid);
#else
    (void)w;(void)s;(void)u;(void)hn;(void)x16;(void)x32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sqf_f16_hc_up_blend_nax_m64(device const half* w [[buffer(0)]], device const half* u [[buffer(1)]], device const half* hn [[buffer(2)]], device half* x16 [[buffer(3)]], device float* x32 [[buffer(4)]], constant SqfHcUpBlendNaxParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sqf_f16_hc_up_blend_nax_impl<64, 4>(w, u, hn, x16, x32, p, tgid);
#else
    (void)w;(void)u;(void)hn;(void)x16;(void)x32;(void)p;(void)tgid;
#endif
}

kernel void sqf_f16_hc_up_blend_nax_m32(device const half* w [[buffer(0)]], device const half* u [[buffer(1)]], device const half* hn [[buffer(2)]], device half* x16 [[buffer(3)]], device float* x32 [[buffer(4)]], constant SqfHcUpBlendNaxParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sqf_f16_hc_up_blend_nax_impl<32, 2>(w, u, hn, x16, x32, p, tgid);
#else
    (void)w;(void)u;(void)hn;(void)x16;(void)x32;(void)p;(void)tgid;
#endif
}

/* ============================================================================
 * 4b. Opt-In Metal 4.1 INT8xINT8->INT32 NAX Prefill GEMMs (`--int8-prefill`)
 *     Active ONLY during prefill (`M_valid >= 32`); decode `M <= 8` untouched!
 * ============================================================================ */

struct SqfI8QuantParams {
    uint M_valid;
    uint M_padded;
    uint N_logical;
    uint N_padded;
    uint K;
    uint pad0;
    uint pad1;
    uint pad2;
};

// Zero-barrier 1-warp/row F16 -> per-row symmetric INT8 activation quantizer (`K % 16 == 0`).
// Grid: `(M_padded, 1, 1)`, `32` threads/TG
kernel void sqf_quant_f16_to_i8_row(
    device const half4*       x_in  [[buffer(0)]],
    device uint4*             x_out [[buffer(1)]],
    device float*             x_sc  [[buffer(2)]],
    constant SqfI8QuantParams& p    [[buffer(3)]],
    uint m    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_padded) return;
    const uint k16 = p.K >> 4u;
    device uint4* dst = x_out + ulong(m) * k16;
    if (m >= p.M_valid) {
        if (lane == 0u) x_sc[m] = 0.0f;
        for (uint i = lane; i < k16; i += 32u) dst[i] = uint4(0u);
        return;
    }
    device const half4* row = x_in + ulong(m) * (p.K >> 2u);
    float ma = 0.0f;
    for (uint i = lane; i < k16; i += 32u) {
        const half4 a0 = abs(row[i * 4u + 0u]);
        const half4 a1 = abs(row[i * 4u + 1u]);
        const half4 a2 = abs(row[i * 4u + 2u]);
        const half4 a3 = abs(row[i * 4u + 3u]);
        const half4 mx = max(max(a0, a1), max(a2, a3));
        ma = max(ma, float(max(max(mx.x, mx.y), max(mx.z, mx.w))));
    }
    const float rmax  = simd_max(ma);
    const float inv_s = (rmax > 1e-12f) ? (127.0f / rmax) : 0.0f;
    if (lane == 0u) x_sc[m] = (rmax > 1e-12f) ? (rmax / 127.0f) : 0.0f;
    for (uint i = lane; i < k16; i += 32u) {
        const char4 c0 = char4(clamp(rint(float4(row[i * 4u + 0u]) * inv_s), float4(-127.0f), float4(127.0f)));
        const char4 c1 = char4(clamp(rint(float4(row[i * 4u + 1u]) * inv_s), float4(-127.0f), float4(127.0f)));
        const char4 c2 = char4(clamp(rint(float4(row[i * 4u + 2u]) * inv_s), float4(-127.0f), float4(127.0f)));
        const char4 c3 = char4(clamp(rint(float4(row[i * 4u + 3u]) * inv_s), float4(-127.0f), float4(127.0f)));
        dst[i] = uint4(as_type<uint>(c0), as_type<uint>(c1), as_type<uint>(c2), as_type<uint>(c3));
    }
}

// Q8_0 (group-64) -> per-row INT8 SLC staging (`128` threads/TG, for large `K` like `HC_DOWN K=10240`)
// Grid: `(N_padded, 1, 1)`, `128` threads/TG
kernel void sqf_q8_requant_row_i8(
    device const uint4*        w_u4   [[buffer(0)]],
    device const half*         w_sc   [[buffer(1)]],
    device uint4*              out_u4 [[buffer(2)]],
    device float*              out_sc [[buffer(3)]],
    constant SqfI8QuantParams& p      [[buffer(4)]],
    uint n    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_m[4];
    if (n >= p.N_padded) return;
    const uint u4_r = p.K >> 4u;
    device uint4* dst = out_u4 + ulong(n) * u4_r;
    if (n >= p.N_logical) {
        if (tid == 0u) out_sc[n] = 0.0f;
        for (uint u = tid; u < u4_r; u += 128u) dst[u] = uint4(0u);
        return;
    }
    const uint ng = p.K >> 6u;
    device const half* sr = w_sc + ulong(n) * ng;
    float ms = 0.0f;
    for (uint g = tid; g < ng; g += 128u) {
        ms = max(ms, fabs(float(sr[g])));
    }
    ms = simd_max(ms);
    if (lane == 0u) tg_m[sgid] = ms;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot = simd_max((lane < 4u) ? tg_m[lane] : 0.0f);
        if (lane == 0u) tg_m[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rsc   = tg_m[0];
    const float inv_s = (rsc > 1e-12f) ? (1.0f / rsc) : 0.0f;
    if (tid == 0u) out_sc[n] = rsc;
    device const uint4* src = w_u4 + ulong(n) * u4_r;
    for (uint u = tid; u < u4_r; u += 128u) {
        const float ratio = float(sr[u >> 2u]) * inv_s;
        const uint4 q     = src[u];
        const char4 c0 = char4(clamp(rint(float4(as_type<char4>(q.x)) * ratio), float4(-127.0f), float4(127.0f)));
        const char4 c1 = char4(clamp(rint(float4(as_type<char4>(q.y)) * ratio), float4(-127.0f), float4(127.0f)));
        const char4 c2 = char4(clamp(rint(float4(as_type<char4>(q.z)) * ratio), float4(-127.0f), float4(127.0f)));
        const char4 c3 = char4(clamp(rint(float4(as_type<char4>(q.w)) * ratio), float4(-127.0f), float4(127.0f)));
        dst[u] = uint4(as_type<uint>(c0), as_type<uint>(c1), as_type<uint>(c2), as_type<uint>(c3));
    }
}

// Q8_0 (group-64) -> per-row INT8 SLC staging (`32` threads/TG, zero barriers, for small `K <= 512` like `HC_UP K=320`)
// Grid: `(N_padded, 1, 1)`, `32` threads/TG
kernel void sqf_q8_requant_row_i8_w32(
    device const uint4*        w_u4   [[buffer(0)]],
    device const half*         w_sc   [[buffer(1)]],
    device uint4*              out_u4 [[buffer(2)]],
    device float*              out_sc [[buffer(3)]],
    constant SqfI8QuantParams& p      [[buffer(4)]],
    uint n    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (n >= p.N_padded) return;
    const uint u4_r = p.K >> 4u;
    device uint4* dst = out_u4 + ulong(n) * u4_r;
    if (n >= p.N_logical) {
        if (lane == 0u) out_sc[n] = 0.0f;
        for (uint u = lane; u < u4_r; u += 32u) dst[u] = uint4(0u);
        return;
    }
    const uint ng = p.K >> 6u;
    device const half* sr = w_sc + ulong(n) * ng;
    float ms = 0.0f;
    for (uint g = lane; g < ng; g += 32u) {
        ms = max(ms, fabs(float(sr[g])));
    }
    const float rsc   = simd_max(ms);
    const float inv_s = (rsc > 1e-12f) ? (1.0f / rsc) : 0.0f;
    if (lane == 0u) out_sc[n] = rsc;
    device const uint4* src = w_u4 + ulong(n) * u4_r;
    for (uint u = lane; u < u4_r; u += 32u) {
        const float ratio = float(sr[u >> 2u]) * inv_s;
        const uint4 q     = src[u];
        const char4 c0 = char4(clamp(rint(float4(as_type<char4>(q.x)) * ratio), float4(-127.0f), float4(127.0f)));
        const char4 c1 = char4(clamp(rint(float4(as_type<char4>(q.y)) * ratio), float4(-127.0f), float4(127.0f)));
        const char4 c2 = char4(clamp(rint(float4(as_type<char4>(q.z)) * ratio), float4(-127.0f), float4(127.0f)));
        const char4 c3 = char4(clamp(rint(float4(as_type<char4>(q.w)) * ratio), float4(-127.0f), float4(127.0f)));
        dst[u] = uint4(as_type<uint>(c0), as_type<uint>(c1), as_type<uint>(c2), as_type<uint>(c3));
    }
}

// Q4_A (group-64 affine) -> per-row symmetric INT8 SLC staging (`128` threads/TG)
// Grid: `(N_padded, 1, 1)`, `128` threads/TG
kernel void sqf_q4_requant_row_i8(
    device const uint2*        w_u2   [[buffer(0)]],
    device const half*         w_sc   [[buffer(1)]],
    device const half*         w_bi   [[buffer(2)]],
    device uint4*              out_u4 [[buffer(3)]],
    device float*              out_sc [[buffer(4)]],
    constant SqfI8QuantParams& p      [[buffer(5)]],
    uint n    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_m[4];
    if (n >= p.N_padded) return;
    const uint u2_r = p.K >> 4u;
    device uint4* dst = out_u4 + ulong(n) * u2_r;
    if (n >= p.N_logical) {
        if (tid == 0u) out_sc[n] = 0.0f;
        for (uint u = tid; u < u2_r; u += 128u) dst[u] = uint4(0u);
        return;
    }
    const uint ng = p.K >> 6u;
    device const half* sr = w_sc + ulong(n) * ng;
    device const half* br = w_bi + ulong(n) * ng;
    float ma = 0.0f;
    for (uint g = tid; g < ng; g += 128u) {
        const float s = float(sr[g]);
        const float b = float(br[g]);
        ma = max(ma, max(fabs(b), fabs(fma(15.0f, s, b))));
    }
    ma = simd_max(ma);
    if (lane == 0u) tg_m[sgid] = ma;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot = simd_max((lane < 4u) ? tg_m[lane] : 0.0f);
        if (lane == 0u) tg_m[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rmax  = tg_m[0];
    const float inv_s = (rmax > 1e-12f) ? (127.0f / rmax) : 0.0f;
    if (tid == 0u) out_sc[n] = (rmax > 1e-12f) ? (rmax / 127.0f) : 0.0f;
    device const uint2* src = w_u2 + ulong(n) * u2_r;
    for (uint u = tid; u < u2_r; u += 128u) {
        const uint  g  = u >> 2u;
        const float sc = float(sr[g]) * inv_s;
        const float bi = float(br[g]) * inv_s;
        const uint2 qv = src[u];
        half4 n0, n1, n2, n3;
        sqf_unpack_u32_nibbles_h4(qv.x, n0, n1);
        sqf_unpack_u32_nibbles_h4(qv.y, n2, n3);
        const char4 c0 = char4(clamp(rint(fma(float4(n0), float4(sc), float4(bi))), float4(-127.0f), float4(127.0f)));
        const char4 c1 = char4(clamp(rint(fma(float4(n1), float4(sc), float4(bi))), float4(-127.0f), float4(127.0f)));
        const char4 c2 = char4(clamp(rint(fma(float4(n2), float4(sc), float4(bi))), float4(-127.0f), float4(127.0f)));
        const char4 c3 = char4(clamp(rint(fma(float4(n3), float4(sc), float4(bi))), float4(-127.0f), float4(127.0f)));
        dst[u] = uint4(as_type<uint>(c0), as_type<uint>(c1), as_type<uint>(c2), as_type<uint>(c3));
    }
}

#if defined(__HAVE_TENSOR__)
// Direct-Device INT8xINT8->INT32 NAX GEMM (`K = dynamic_extent`, 0 TG weight mem, 0 K-loop barriers, ~97.7 TOPS!)
template <int BM, int BN, int SG>
inline void sqf_i8_gemm_nax_impl(
    device const int8_t*       w_i8,
    device const float*        w_sc,
    device const int8_t*       x_i8,
    device const float*        x_sc,
    device half*               y_f16,
    device float*              y_f32,
    constant SqfNaxGemmParams& p,
    threadgroup float          (&sw_tg)[BN],
    threadgroup float          (&sx_tg)[BM],
    uint2                      tgid,
    uint                       tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * uint(BN);
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    if (tid < uint(BN)) sw_tg[tid] = w_sc[n0 + tid];
    if (tid < uint(BM)) sx_tg[tid] = x_sc[m0 + tid];

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tX(
        const_cast<device int8_t*>(x_i8), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<device int8_t, dextents<int, 2>, tensor_inline> tW(
        const_cast<device int8_t*>(w_i8), dextents<int, 2>(int(p.K), int(p.N_padded)));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX   = tX.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW   = tW.template slice<dynamic_extent, BN>(0, int(n0));
    auto cI32 = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sW)>, int32_t>();
    op.run(sX, sW, cI32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool full_tile_ok = (p.N_logical == p.N_padded) && (n0 + uint(BN) <= p.N_logical) && (m0 + uint(BM) <= p.M_padded);
    if (p.out_mode == SQF_OUT_F16 && full_tile_ok) {
        auto cOut = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sW)>, half>();
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cI32.get_capacity(); ++i) {
            if (cI32.is_valid_element(i)) {
                auto idx = cI32.get_multidimensional_index(i);
                cOut[i]  = sqf_clamp_f16(float(cI32[i]) * (sx_tg[uint(idx[1])] * sw_tg[uint(idx[0])]));
            }
        }
        tensor<device half, dextents<int, 2>, tensor_inline> tY(y_f16, dextents<int, 2>(int(p.N_logical), int(p.M_padded)));
        auto sY = tY.template slice<BN, BM>(int(n0), int(m0));
        cOut.store(sY);
    } else if (p.out_mode == SQF_OUT_F32 && full_tile_ok) {
        auto cOut = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sW)>, float>();
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cI32.get_capacity(); ++i) {
            if (cI32.is_valid_element(i)) {
                auto idx = cI32.get_multidimensional_index(i);
                cOut[i]  = float(cI32[i]) * (sx_tg[uint(idx[1])] * sw_tg[uint(idx[0])]);
            }
        }
        tensor<device float, dextents<int, 2>, tensor_inline> tY(y_f32, dextents<int, 2>(int(p.N_logical), int(p.M_padded)));
        auto sY = tY.template slice<BN, BM>(int(n0), int(m0));
        cOut.store(sY);
    } else {
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cI32.get_capacity(); ++i) {
            if (cI32.is_valid_element(i)) {
                auto idx      = cI32.get_multidimensional_index(i);
                const uint mg = m0 + uint(idx[1]);
                const uint ng = n0 + uint(idx[0]);
                if (mg < p.M_padded && ng < p.N_logical) {
                    const float val = (mg < p.M_valid)
                        ? (float(cI32[i]) * (sx_tg[uint(idx[1])] * sw_tg[uint(idx[0])]))
                        : 0.0f;
                    const ulong oidx = ulong(mg) * p.N_logical + ng;
                    if (p.out_mode == SQF_OUT_F16)      y_f16[oidx] = sqf_clamp_f16(val);
                    else if (p.out_mode == SQF_OUT_F32) y_f32[oidx] = val;
                    else if (mg < p.M_valid)            y_f32[oidx] += val;
                }
            }
        }
    }
}

// Fused INT8 NAX HC_DOWN `[M, 10240] x [320, 10240]^T` + `silu(x * inv_hc)` -> `u_out [M, 320]`
template <int BM, int SG>
inline void sqf_i8_hc_down_act_nax_impl(
    device const int8_t*         w_i8,
    device const float*          w_sc,
    device const int8_t*         x_i8,
    device const float*          x_sc,
    device half*                 u_out,
    constant SqfHcDownNaxParams& p,
    threadgroup float            (&sw_tg)[64],
    threadgroup float            (&sx_tg)[BM],
    uint2                        tgid,
    uint                         tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.lowrank) return;

    if (tid < 64u)      sw_tg[tid] = w_sc[n0 + tid];
    if (tid < uint(BM)) sx_tg[tid] = x_sc[m0 + tid];

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tX(
        const_cast<device int8_t*>(x_i8), dextents<int, 2>(int(p.cols), int(p.M_padded)));
    tensor<device int8_t, dextents<int, 2>, tensor_inline> tW(
        const_cast<device int8_t*>(w_i8), dextents<int, 2>(int(p.cols), int(p.lowrank)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX   = tX.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW   = tW.template slice<dynamic_extent, 64>(0, int(n0));
    auto cI32 = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sW)>, int32_t>();
    auto cOut = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sW)>, half>();
    op.run(sX, sW, cI32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cI32.get_capacity(); ++i) {
        if (cI32.is_valid_element(i)) {
            auto idx      = cI32.get_multidimensional_index(i);
            const uint ml = uint(idx[1]);
            const uint nl = uint(idx[0]);
            const float v = float(cI32[i]) * (sx_tg[ml] * sw_tg[nl]);
            cOut[i]       = (m0 + ml < p.M_valid) ? sqf_clamp_f16(sqf_silu(v * p.inv_hc)) : half(0.0h);
        }
    }
    tensor<device half, dextents<int, 2>, tensor_inline> tU(u_out, dextents<int, 2>(int(p.lowrank), int(p.M_padded)));
    auto sU = tU.template slice<64, BM>(int(n0), int(m0));
    cOut.store(sU);
}

// Fused INT8 NAX HC_UP `[M, 320] x [4*D, 320]^T` + 4-stream weighted mean blend -> `x_out_f16` (+ optional `x_out_f32`)
template <int BM, int SG>
inline void sqf_i8_hc_up_blend_nax_impl(
    device const int8_t*            w_i8,
    device const float*             w_sc,
    device const int8_t*            u_i8,
    device const float*             u_sc,
    device const half*              h_norm_f16,
    device half*                    x_out_f16,
    device float*                   x_out_f32,
    constant SqfHcUpBlendNaxParams& p,
    threadgroup float               (&sw_tg)[64],
    threadgroup float               (&su_tg)[BM],
    uint2                           tgid,
    uint                            tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint d0 = tgid.x * 64u;
    if (m0 >= p.M_padded || d0 >= p.D) return;

    if (tid < uint(BM)) su_tg[tid] = u_sc[m0 + tid];

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tU(
        const_cast<device int8_t*>(u_i8), dextents<int, 2>(int(p.lowrank), int(p.M_padded)));
    tensor<device int8_t, dextents<int, 2>, tensor_inline> tW(
        const_cast<device int8_t*>(w_i8), dextents<int, 2>(int(p.lowrank), int(p.hc_count * p.D)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sU   = tU.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW0  = tW.template slice<dynamic_extent, 64>(0, int(d0));
    auto cI32 = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sU)>, sqf_strip_as<decltype(sW0)>, int32_t>();
    auto cAcc = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sU)>, sqf_strip_as<decltype(sW0)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cAcc.get_capacity(); ++i) {
        if (cAcc.is_valid_element(i)) cAcc[i] = 0.0f;
    }

    const ulong hc_dim = ulong(p.hc_count) * p.D;
    for (uint s = 0u; s < p.hc_count; ++s) {
        if (tid < 64u) sw_tg[tid] = w_sc[s * p.D + d0 + tid];
        auto sW = tW.template slice<dynamic_extent, 64>(0, int(s * p.D + d0));
        op.run(sU, sW, cI32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cI32.get_capacity(); ++i) {
            if (cI32.is_valid_element(i)) {
                auto idx      = cI32.get_multidimensional_index(i);
                const uint ml = uint(idx[1]);
                const uint dl = uint(idx[0]);
                const uint mg = m0 + ml;
                if (mg < p.M_valid) {
                    const float upv = float(cI32[i]) * (su_tg[ml] * sw_tg[dl]);
                    const float hnv = float(h_norm_f16[ulong(mg) * hc_dim + ulong(s) * p.D + d0 + dl]);
                    cAcc[i] = fma(sqf_sigmoid(upv), hnv, cAcc[i]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto cOut = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sU)>, sqf_strip_as<decltype(sW0)>, half>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cAcc.get_capacity(); ++i) {
        if (cAcc.is_valid_element(i)) {
            auto idx      = cAcc.get_multidimensional_index(i);
            const uint mg = m0 + uint(idx[1]);
            const float xv = (mg < p.M_valid) ? (cAcc[i] * p.inv_hc) : 0.0f;
            cOut[i] = sqf_clamp_f16(xv);
            if (p.write_f32 != 0u && mg < p.M_valid) {
                x_out_f32[ulong(mg) * p.D + d0 + uint(idx[0])] = xv;
            }
        }
    }
    tensor<device half, dextents<int, 2>, tensor_inline> tY(x_out_f16, dextents<int, 2>(int(p.D), int(p.M_padded)));
    auto sY = tY.template slice<64, BM>(int(d0), int(m0));
    cOut.store(sY);
}

// Fused INT8 NAX Shared Expert `gate + up + SwiGLU` -> `mid_f16` (0 intermediate FP32 buffers!)
template <int BM, int SG>
inline void sqf_i8_shared_gu_silu_nax_impl(
    device const int8_t*           wg_i8,
    device const float*            wg_sc,
    device const int8_t*           wu_i8,
    device const float*            wu_sc,
    device const int8_t*           x_i8,
    device const float*            x_sc,
    device half*                   mid_out,
    constant SqfSharedGuNaxParams& p,
    threadgroup float              (&swg_tg)[64],
    threadgroup float              (&swu_tg)[64],
    threadgroup float              (&sx_tg)[BM],
    uint2                          tgid,
    uint                           tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.d_ff) return;

    if (tid < 64u) {
        swg_tg[tid] = wg_sc[n0 + tid];
        swu_tg[tid] = wu_sc[n0 + tid];
    }
    if (tid < uint(BM)) sx_tg[tid] = x_sc[m0 + tid];

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tX(
        const_cast<device int8_t*>(x_i8), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<device int8_t, dextents<int, 2>, tensor_inline> tWg(
        const_cast<device int8_t*>(wg_i8), dextents<int, 2>(int(p.K), int(p.d_ff)));
    tensor<device int8_t, dextents<int, 2>, tensor_inline> tWu(
        const_cast<device int8_t*>(wu_i8), dextents<int, 2>(int(p.K), int(p.d_ff)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX  = tX.template slice<dynamic_extent, BM>(0, int(m0));
    auto sWg = tWg.template slice<dynamic_extent, 64>(0, int(n0));
    auto sWu = tWu.template slice<dynamic_extent, 64>(0, int(n0));
    auto cG  = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sWg)>, int32_t>();
    auto cU  = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sWu)>, int32_t>();
    auto cOut = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX)>, sqf_strip_as<decltype(sWg)>, half>();

    op.run(sX, sWg, cG);
    op.run(sX, sWu, cU);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cG.get_capacity(); ++i) {
        if (cG.is_valid_element(i)) {
            auto idx      = cG.get_multidimensional_index(i);
            const uint ml = uint(idx[1]);
            const uint nl = uint(idx[0]);
            const float sx = sx_tg[ml];
            const float gv = float(cG[i]) * (sx * swg_tg[nl]);
            const float uv = float(cU[i]) * (sx * swu_tg[nl]);
            cOut[i] = (m0 + ml < p.M_valid) ? sqf_clamp_f16(sqf_silu(gv) * uv) : half(0.0h);
        }
    }
    tensor<device half, dextents<int, 2>, tensor_inline> tY(mid_out, dextents<int, 2>(int(p.d_ff), int(p.M_padded)));
    auto sY = tY.template slice<64, BM>(int(n0), int(m0));
    cOut.store(sY);
}
#endif

kernel void sqf_i8_gemm_nax_64x128_sg4(device const int8_t* w [[buffer(0)]], device const float* ws [[buffer(1)]], device const int8_t* x [[buffer(2)]], device const float* xs [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SqfNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sw_tg[128];
    threadgroup float sx_tg[64];
#if defined(__HAVE_TENSOR__)
    sqf_i8_gemm_nax_impl<64, 128, 4>(w, ws, x, xs, y16, y32, p, sw_tg, sx_tg, tgid, tid);
#else
    (void)w;(void)ws;(void)x;(void)xs;(void)y16;(void)y32;(void)p;(void)sw_tg;(void)sx_tg;(void)tgid;(void)tid;
#endif
}

kernel void sqf_i8_gemm_nax_64x64_sg4(device const int8_t* w [[buffer(0)]], device const float* ws [[buffer(1)]], device const int8_t* x [[buffer(2)]], device const float* xs [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SqfNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sw_tg[64];
    threadgroup float sx_tg[64];
#if defined(__HAVE_TENSOR__)
    sqf_i8_gemm_nax_impl<64, 64, 4>(w, ws, x, xs, y16, y32, p, sw_tg, sx_tg, tgid, tid);
#else
    (void)w;(void)ws;(void)x;(void)xs;(void)y16;(void)y32;(void)p;(void)sw_tg;(void)sx_tg;(void)tgid;(void)tid;
#endif
}

kernel void sqf_i8_gemm_nax_32x64_sg2(device const int8_t* w [[buffer(0)]], device const float* ws [[buffer(1)]], device const int8_t* x [[buffer(2)]], device const float* xs [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SqfNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sw_tg[64];
    threadgroup float sx_tg[32];
#if defined(__HAVE_TENSOR__)
    sqf_i8_gemm_nax_impl<32, 64, 2>(w, ws, x, xs, y16, y32, p, sw_tg, sx_tg, tgid, tid);
#else
    (void)w;(void)ws;(void)x;(void)xs;(void)y16;(void)y32;(void)p;(void)sw_tg;(void)sx_tg;(void)tgid;(void)tid;
#endif
}

kernel void sqf_i8_hc_down_act_nax_m64(device const int8_t* w [[buffer(0)]], device const float* ws [[buffer(1)]], device const int8_t* x [[buffer(2)]], device const float* xs [[buffer(3)]], device half* u [[buffer(4)]], constant SqfHcDownNaxParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sw_tg[64];
    threadgroup float sx_tg[64];
#if defined(__HAVE_TENSOR__)
    sqf_i8_hc_down_act_nax_impl<64, 4>(w, ws, x, xs, u, p, sw_tg, sx_tg, tgid, tid);
#else
    (void)w;(void)ws;(void)x;(void)xs;(void)u;(void)p;(void)sw_tg;(void)sx_tg;(void)tgid;(void)tid;
#endif
}

kernel void sqf_i8_hc_down_act_nax_m32(device const int8_t* w [[buffer(0)]], device const float* ws [[buffer(1)]], device const int8_t* x [[buffer(2)]], device const float* xs [[buffer(3)]], device half* u [[buffer(4)]], constant SqfHcDownNaxParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sw_tg[64];
    threadgroup float sx_tg[32];
#if defined(__HAVE_TENSOR__)
    sqf_i8_hc_down_act_nax_impl<32, 2>(w, ws, x, xs, u, p, sw_tg, sx_tg, tgid, tid);
#else
    (void)w;(void)ws;(void)x;(void)xs;(void)u;(void)p;(void)sw_tg;(void)sx_tg;(void)tgid;(void)tid;
#endif
}

kernel void sqf_i8_hc_up_blend_nax_m64(device const int8_t* w [[buffer(0)]], device const float* ws [[buffer(1)]], device const int8_t* u [[buffer(2)]], device const float* us [[buffer(3)]], device const half* hn [[buffer(4)]], device half* x16 [[buffer(5)]], device float* x32 [[buffer(6)]], constant SqfHcUpBlendNaxParams& p [[buffer(7)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sw_tg[64];
    threadgroup float su_tg[64];
#if defined(__HAVE_TENSOR__)
    sqf_i8_hc_up_blend_nax_impl<64, 4>(w, ws, u, us, hn, x16, x32, p, sw_tg, su_tg, tgid, tid);
#else
    (void)w;(void)ws;(void)u;(void)us;(void)hn;(void)x16;(void)x32;(void)p;(void)sw_tg;(void)su_tg;(void)tgid;(void)tid;
#endif
}

kernel void sqf_i8_hc_up_blend_nax_m32(device const int8_t* w [[buffer(0)]], device const float* ws [[buffer(1)]], device const int8_t* u [[buffer(2)]], device const float* us [[buffer(3)]], device const half* hn [[buffer(4)]], device half* x16 [[buffer(5)]], device float* x32 [[buffer(6)]], constant SqfHcUpBlendNaxParams& p [[buffer(7)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sw_tg[64];
    threadgroup float su_tg[32];
#if defined(__HAVE_TENSOR__)
    sqf_i8_hc_up_blend_nax_impl<32, 2>(w, ws, u, us, hn, x16, x32, p, sw_tg, su_tg, tgid, tid);
#else
    (void)w;(void)ws;(void)u;(void)us;(void)hn;(void)x16;(void)x32;(void)p;(void)sw_tg;(void)su_tg;(void)tgid;(void)tid;
#endif
}

kernel void sqf_i8_shared_gu_silu_nax_m64(device const int8_t* wg [[buffer(0)]], device const float* wgs [[buffer(1)]], device const int8_t* wu [[buffer(2)]], device const float* wus [[buffer(3)]], device const int8_t* x [[buffer(4)]], device const float* xs [[buffer(5)]], device half* mo [[buffer(6)]], constant SqfSharedGuNaxParams& p [[buffer(7)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float swg_tg[64];
    threadgroup float swu_tg[64];
    threadgroup float sx_tg[64];
#if defined(__HAVE_TENSOR__)
    sqf_i8_shared_gu_silu_nax_impl<64, 4>(wg, wgs, wu, wus, x, xs, mo, p, swg_tg, swu_tg, sx_tg, tgid, tid);
#else
    (void)wg;(void)wgs;(void)wu;(void)wus;(void)x;(void)xs;(void)mo;(void)p;(void)swg_tg;(void)swu_tg;(void)sx_tg;(void)tgid;(void)tid;
#endif
}

kernel void sqf_i8_shared_gu_silu_nax_m32(device const int8_t* wg [[buffer(0)]], device const float* wgs [[buffer(1)]], device const int8_t* wu [[buffer(2)]], device const float* wus [[buffer(3)]], device const int8_t* x [[buffer(4)]], device const float* xs [[buffer(5)]], device half* mo [[buffer(6)]], constant SqfSharedGuNaxParams& p [[buffer(7)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float swg_tg[64];
    threadgroup float swu_tg[64];
    threadgroup float sx_tg[32];
#if defined(__HAVE_TENSOR__)
    sqf_i8_shared_gu_silu_nax_impl<32, 2>(wg, wgs, wu, wus, x, xs, mo, p, swg_tg, swu_tg, sx_tg, tgid, tid);
#else
    (void)wg;(void)wgs;(void)wu;(void)wus;(void)x;(void)xs;(void)mo;(void)p;(void)swg_tg;(void)swu_tg;(void)sx_tg;(void)tgid;(void)tid;
#endif
}

/* ============================================================================
 * 5. 4-Stream Hyper-Connections (`hc_count = 4`) & Final Mixer Kernels
 * ============================================================================ */

struct SqfGroupedNormParams {
    uint  M_valid;
    uint  M_padded;
    uint  groups;     // 4 for grouped `hc_norm`, 1 for ungrouped (`mtp.pre_fc_norm_embedding`)
    uint  group_dim;  // D (2560)
    float eps;
    uint  write_f32;  // 1 => also write `out_f32 [M_valid, groups * group_dim]`
    uint  pad0;
    uint  pad1;
};

// Grid: `(groups, M_padded, 1)`, 256 threads/TG
kernel void sqf_grouped_rmsnorm(
    device const float*         x_in_f32  [[buffer(0)]], // [M_valid, groups * group_dim]
    device const float*         norm_w    [[buffer(1)]], // [groups * group_dim] (+1.0 pre-folded)
    device half*                out_f16   [[buffer(2)]], // [M_padded, groups * group_dim]
    device float*               out_f32   [[buffer(3)]], // optional [M_valid, groups * group_dim]
    constant SqfGroupedNormParams& p      [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],          // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],       // 0..7
    uint  lane [[thread_index_in_simdgroup]]             // 0..31
) {
    const uint g = tgid.x;
    const uint m = tgid.y;
    if (g >= p.groups || m >= p.M_padded) return;

    const uint  total_dim = p.groups * p.group_dim;
    const uint  d4_cnt    = p.group_dim >> 2u;
    const ulong row_off   = ulong(m) * total_dim + ulong(g) * p.group_dim;

    if (m >= p.M_valid) {
        device half4* dst_h = reinterpret_cast<device half4*>(out_f16 + row_off);
        for (uint i = tid; i < d4_cnt; i += 256u) {
            dst_h[i] = half4(0.0h);
        }
        return;
    }

    threadgroup float tg_sq[8];
    device const float4* src4 = reinterpret_cast<device const float4*>(x_in_f32 + row_off);
    device const float4* w4   = reinterpret_cast<device const float4*>(norm_w + ulong(g) * p.group_dim);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 v = src4[i];
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }

    float sg_sum = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float v = (lane < 8u) ? tg_sq[lane] : 0.0f;
        float tot = simd_sum(v);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.group_dim) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    device half4*  dst_h = reinterpret_cast<device half4*>(out_f16 + row_off);
    device float4* dst_f = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + row_off) : nullptr;

    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 normed = (src4[i] * inv_rms) * w4[i];
        dst_h[i] = sqf_clamp_f16x4(normed);
        if (dst_f) dst_f[i] = normed;
    }
}

struct SqfHcDownActParams {
    uint  M_valid;
    uint  M_padded;
    uint  lowrank;      // 320
    uint  hc_count;     // 4
    float inv_hc;       // 0.25f
    uint  has_inject;   // 1 for pre-attn/pre-mlp, 0 for final mixer
    uint  pad0;
    uint  pad1;
};

// `u = silu(down_raw * 0.25f)` -> `u_f16 [M_padded, lowrank]` + optional `inj = 2.0f * sigmoid(inj_raw * 0.25f)`
kernel void sqf_hc_down_act(
    device const float*         down_raw [[buffer(0)]], // [M_valid, lowrank]
    device const float*         inj_raw  [[buffer(1)]], // optional [M_valid, hc_count]
    device half*                u_out    [[buffer(2)]], // [M_padded, lowrank]
    device float*               inj_out  [[buffer(3)]], // optional [M_valid, hc_count]
    constant SqfHcDownActParams& p       [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]          // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_padded) return;

    const uint idx = tgid.x * 256u + tid;
    if (idx < p.lowrank) {
        const ulong off = ulong(m) * p.lowrank + idx;
        if (m >= p.M_valid) {
            u_out[off] = half(0.0h);
        } else {
            u_out[off] = sqf_clamp_f16(sqf_silu(down_raw[off] * p.inv_hc));
        }
    }

    if (p.has_inject != 0u && m < p.M_valid && tgid.x == 0u && tid < p.hc_count) {
        const ulong ioff = ulong(m) * p.hc_count + tid;
        inj_out[ioff] = 2.0f * sqf_sigmoid(inj_raw[ioff] * p.inv_hc);
    }
}

struct SqfHcBlendParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    uint  hc_count;
    float inv_hc;
    uint  write_f32;
    uint  pad0;
    uint  pad1;
};

// `x_in[m, d] = inv_hc * sum_{s=0..hc_count-1} (sigmoid(up_raw[m, s, d]) * h_norm_f32[m, s, d])`
kernel void sqf_hc_blend(
    device const float*        up_raw     [[buffer(0)]], // [M_valid, hc_count * D]
    device const float*        h_norm_f32 [[buffer(1)]], // [M_valid, hc_count * D]
    device half*               x_out_f16  [[buffer(2)]], // [M_padded, D]
    device float*              x_out_f32  [[buffer(3)]], // optional [M_valid, D]
    constant SqfHcBlendParams& p          [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]           // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_padded) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    const ulong out_off4 = ulong(m) * d4_cnt + idx4;
    if (m >= p.M_valid) {
        reinterpret_cast<device half4*>(x_out_f16)[out_off4] = half4(0.0h);
        return;
    }

    const uint hc_d4 = p.hc_count * d4_cnt;
    float4 acc(0.0f);
    for (uint s = 0u; s < p.hc_count; ++s) {
        const ulong in_off4 = ulong(m) * hc_d4 + ulong(s) * d4_cnt + idx4;
        const float4 u4     = reinterpret_cast<device const float4*>(up_raw)[in_off4];
        const float4 hn4    = reinterpret_cast<device const float4*>(h_norm_f32)[in_off4];
        const float4 mix_w  = float4(1.0f) / (float4(1.0f) + exp(-u4));
        acc = fma(mix_w, hn4, acc);
    }
    const float4 x_val = acc * p.inv_hc;
    reinterpret_cast<device half4*>(x_out_f16)[out_off4] = sqf_clamp_f16x4(x_val);
    if (p.write_f32 != 0u) {
        reinterpret_cast<device float4*>(x_out_f32)[out_off4] = x_val;
    }
}

struct SqfHcResAddParams {
    uint M_valid;
    uint D;
    uint hc_count;
    uint pad;
};

// Updates UN-NORMED `b_hc[m, s, d] += inj[m, s] * y_f32[m, d]`
kernel void sqf_hc_residual_add(
    device const float*         y_f32  [[buffer(0)]], // [M_valid, D]
    device const float*         inj    [[buffer(1)]], // [M_valid, hc_count]
    device float*               hc_buf [[buffer(2)]], // [M_valid, hc_count * D]
    constant SqfHcResAddParams& p      [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]        // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    const float4 y4   = reinterpret_cast<device const float4*>(y_f32)[ulong(m) * d4_cnt + idx4];
    const uint  hc_d4 = p.hc_count * d4_cnt;

    for (uint s = 0u; s < p.hc_count; ++s) {
        const float  inj_s  = inj[ulong(m) * p.hc_count + s];
        const ulong  hc_off = ulong(m) * hc_d4 + ulong(s) * d4_cnt + idx4;
        device float4* dst  = reinterpret_cast<device float4*>(hc_buf) + hc_off;
        *dst = fma(float4(inj_s), y4, *dst);
    }
}

// Single-Dispatch Fused Post-Attention `hc_buf += inj * y_f32` + Pre-MLP `grouped_rmsnorm(hc_buf, mlp_hc.hc_norm)`!
// Caches updated `hc` in 3 `float4` registers (`D <= 3072`) -> ZERO re-read of `hc_buf` & -48 dispatches/step (100% bit-exact!).
// Grid: `(groups, M_padded, 1)`, 256 threads/TG
kernel void sqf_hc_res_add_grouped_rmsnorm(
    device const float*            y_f32     [[buffer(0)]], // [M_valid, group_dim]
    device const float*            inj       [[buffer(1)]], // [M_valid, groups]
    device float*                  hc_buf    [[buffer(2)]], // [M_valid, groups * group_dim] (in-place updated!)
    device const float*            norm_w    [[buffer(3)]], // [groups * group_dim] (+1.0 folded)
    device half*                   out_f16   [[buffer(4)]], // [M_padded, groups * group_dim]
    device float*                  out_f32   [[buffer(5)]], // optional [M_valid, groups * group_dim]
    constant SqfGroupedNormParams& p         [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],             // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],          // 0..7
    uint  lane [[thread_index_in_simdgroup]]                // 0..31
) {
    const uint g = tgid.x;
    const uint m = tgid.y;
    if (g >= p.groups || m >= p.M_padded) return;

    const uint  total_dim = p.groups * p.group_dim;
    const uint  d4_cnt    = p.group_dim >> 2u;
    const ulong row_off   = ulong(m) * total_dim + ulong(g) * p.group_dim;

    if (m >= p.M_valid) {
        device half4* dst_h = reinterpret_cast<device half4*>(out_f16 + row_off);
        for (uint i = tid; i < d4_cnt; i += 256u) {
            dst_h[i] = half4(0.0h);
        }
        return;
    }

    threadgroup float tg_sq[8];
    device const float4* y4   = reinterpret_cast<device const float4*>(y_f32 + ulong(m) * p.group_dim);
    device float4*       hc4  = reinterpret_cast<device float4*>(hc_buf + row_off);
    device const float4* w4   = reinterpret_cast<device const float4*>(norm_w + ulong(g) * p.group_dim);
    const float          injs = inj[ulong(m) * p.groups + g];

    float sq_acc = 0.0f;
    if (d4_cnt <= 768u) {
        float4 v_reg[3];
        #pragma clang loop unroll(full)
        for (uint k = 0u; k < 3u; ++k) {
            const uint i = k * 256u + tid;
            if (i < d4_cnt) {
                float4 v = fma(float4(injs), y4[i], hc4[i]);
                hc4[i]   = v;
                v_reg[k] = v;
                sq_acc   = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
            }
        }
        float sg_sum = simd_sum(sq_acc);
        if (lane == 0u) tg_sq[sgid] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.group_dim) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inv_rms  = tg_sq[0];
        device half4*  dst_h = reinterpret_cast<device half4*>(out_f16 + row_off);
        device float4* dst_f = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + row_off) : nullptr;
        #pragma clang loop unroll(full)
        for (uint k = 0u; k < 3u; ++k) {
            const uint i = k * 256u + tid;
            if (i < d4_cnt) {
                float4 normed = (v_reg[k] * inv_rms) * w4[i];
                dst_h[i] = sqf_clamp_f16x4(normed);
                if (dst_f) dst_f[i] = normed;
            }
        }
    } else {
        for (uint i = tid; i < d4_cnt; i += 256u) {
            float4 v = fma(float4(injs), y4[i], hc4[i]);
            hc4[i]   = v;
            sq_acc   = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
        }
        float sg_sum = simd_sum(sq_acc);
        if (lane == 0u) tg_sq[sgid] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.group_dim) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inv_rms  = tg_sq[0];
        device half4*  dst_h = reinterpret_cast<device half4*>(out_f16 + row_off);
        device float4* dst_f = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + row_off) : nullptr;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            float4 normed = (hc4[i] * inv_rms) * w4[i];
            dst_h[i] = sqf_clamp_f16x4(normed);
            if (dst_f) dst_f[i] = normed;
        }
    }
}

/* ============================================================================
 * 6. Per-Layer Embedding (`PLE`, `layer_idx == 1`) Gating & Dilated Conv1d
 * ============================================================================ */

struct SqfPleGateParams {
    uint  M_valid;
    uint  D;
    uint  hc_count;
    float eps;
};

// Fuses `norm_key` + `norm_query` + `dot/sqrt(D)` + `sqrt(max(|dot|, 1e-6))*sign(dot)` + `gv` + `norm_conv`!
// Grid: `(hc_count, M_valid, 1)`, 256 threads/TG
kernel void sqf_ple_gate_gv(
    device const float*       k_raw       [[buffer(0)]], // [M_valid, hc_count * D]
    device const float*       val_in      [[buffer(1)]], // [M_valid, D]
    device const float*       hc_in       [[buffer(2)]], // [M_valid, hc_count * D]
    device const float*       norm_key    [[buffer(3)]], // [hc_count * D]
    device const float*       norm_query  [[buffer(4)]], // [hc_count * D]
    device const float*       norm_conv   [[buffer(5)]], // [hc_count * D]
    device float*             gv_out      [[buffer(6)]], // [M_valid, hc_count * D]
    device float*             gv_norm_out [[buffer(7)]], // [M_valid, hc_count * D]
    constant SqfPleGateParams& p          [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],          // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],       // 0..7
    uint  lane [[thread_index_in_simdgroup]]             // 0..31
) {
    const uint s = tgid.x;
    const uint m = tgid.y;
    if (s >= p.hc_count || m >= p.M_valid) return;

    const uint  hc_dim  = p.hc_count * p.D;
    const uint  d4_cnt  = p.D >> 2u;
    const ulong st_off  = ulong(s) * p.D;
    const ulong row_off = ulong(m) * hc_dim + st_off;

    device const float4* kr4  = reinterpret_cast<device const float4*>(k_raw + row_off);
    device const float4* qr4  = reinterpret_cast<device const float4*>(hc_in + row_off);
    device const float4* val4 = reinterpret_cast<device const float4*>(val_in + ulong(m) * p.D);
    device const float4* nk4  = reinterpret_cast<device const float4*>(norm_key + st_off);
    device const float4* nq4  = reinterpret_cast<device const float4*>(norm_query + st_off);
    device const float4* nc4  = reinterpret_cast<device const float4*>(norm_conv + st_off);

    threadgroup float tg_a[8];
    threadgroup float tg_b[8];
    threadgroup float tg_c[8];
    threadgroup float tg_d[8];
    threadgroup float tg_s[4];

    // 1. RMS denominators for `k_raw[m, s]` and `hc_in[m, s]`
    float k_sq = 0.0f;
    float q_sq = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 kv = kr4[i];
        float4 qv = qr4[i];
        k_sq = fma(kv.x, kv.x, fma(kv.y, kv.y, fma(kv.z, kv.z, fma(kv.w, kv.w, k_sq))));
        q_sq = fma(qv.x, qv.x, fma(qv.y, qv.y, fma(qv.z, qv.z, fma(qv.w, qv.w, q_sq))));
    }
    float sk = simd_sum(k_sq);
    float sq = simd_sum(q_sq);
    if (lane == 0u) {
        tg_a[sgid] = sk;
        tg_b[sgid] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tk = simd_sum((lane < 8u) ? tg_a[lane] : 0.0f);
        float tq = simd_sum((lane < 8u) ? tg_b[lane] : 0.0f);
        if (lane == 0u) {
            tg_s[0] = rsqrt(tk / float(p.D) + p.eps);
            tg_s[1] = rsqrt(tq / float(p.D) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_k = tg_s[0];
    const float inv_q = tg_s[1];

    // 2. Dot product `dot(k_norm[s], q_norm[s]) / sqrt(D)` + signed-sqrt gate + `gv` RMS norm
    float dot_acc = 0.0f;
    float val_sq  = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 kn = (kr4[i] * inv_k) * nk4[i];
        float4 qn = (qr4[i] * inv_q) * nq4[i];
        float4 vv = val4[i];
        dot_acc = fma(kn.x, qn.x, fma(kn.y, qn.y, fma(kn.z, qn.z, fma(kn.w, qn.w, dot_acc))));
        val_sq  = fma(vv.x, vv.x, fma(vv.y, vv.y, fma(vv.z, vv.z, fma(vv.w, vv.w, val_sq))));
    }
    float s_dot = simd_sum(dot_acc);
    float s_vsq = simd_sum(val_sq);
    if (lane == 0u) {
        tg_c[sgid] = s_dot;
        tg_d[sgid] = s_vsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot_dot = simd_sum((lane < 8u) ? tg_c[lane] : 0.0f);
        float tot_vsq = simd_sum((lane < 8u) ? tg_d[lane] : 0.0f);
        if (lane == 0u) {
            float dot_s  = tot_dot * rsqrt(float(p.D));
            float gate_s = 0.0f;
            if (dot_s != 0.0f) {
                float sgn = (dot_s > 0.0f) ? 1.0f : -1.0f;
                gate_s    = sqrt(max(fabs(dot_s), 1e-6f)) * sgn;
            }
            float sig_g  = sqf_sigmoid(gate_s);
            float gv_msq = (sig_g * sig_g) * (tot_vsq / float(p.D));
            tg_s[2] = sig_g;
            tg_s[3] = rsqrt(gv_msq + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sig_g  = tg_s[2];
    const float inv_gv = tg_s[3];

    device float4* dst_gv  = reinterpret_cast<device float4*>(gv_out + row_off);
    device float4* dst_gvn = reinterpret_cast<device float4*>(gv_norm_out + row_off);
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 gv4  = val4[i] * sig_g;
        dst_gv[i]   = gv4;
        dst_gvn[i]  = (gv4 * inv_gv) * nc4[i];
    }
}

struct SqfPleConvParams {
    uint M;
    uint channels;       // hc_count * D = 10240
    uint dilation;       // ngram_size = 3 (`state_len = 3 * dilation = 9`)
    uint save_snap_mask; // bit `t` => save `ple_state` after token `t` into `snap_t`
    uint tree_branch_t;  // 0 for chain; >0 => reload `hist` from `snap0` right before `t == tree_branch_t`
    uint pad0;
    uint pad1;
    uint pad2;
};

// Dilated Causal Depthwise `conv1d(k=4, dilation=3) + SiLU` + `b_hc += gv + conv_out`
kernel void sqf_ple_dilated_conv1d_add(
    device const float*       gv_in       [[buffer(0)]], // [M, channels]
    device const float*       gv_norm_in  [[buffer(1)]], // [M, channels]
    device const float4*      conv_w      [[buffer(2)]], // [channels] (taps [t-3d, t-2d, t-d, t])
    device float*             ple_state   [[buffer(3)]], // [state_len, channels] (`state_len <= 12`)
    device float*             hc_inout    [[buffer(4)]], // [M, channels]
    device float*             snap0       [[buffer(5)]],
    device float*             snap1       [[buffer(6)]],
    device float*             snap2       [[buffer(7)]],
    device float*             snap3       [[buffer(8)]],
    device float*             snap4       [[buffer(9)]],
    device float*             snap5       [[buffer(10)]],
    constant SqfPleConvParams& p          [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.channels) return;

    const uint C    = p.channels;
    const uint dil  = max(p.dilation, 1u);
    const uint slen = 3u * dil; // 9 for dilation=3
    const float4 w  = conv_w[gid];

    float hist[12];
    for (uint i = 0u; i < slen; ++i) {
        hist[i] = ple_state[ulong(i) * C + gid];
    }

    const uint tap0 = 0u;           // t - 3*dil (0 for dil=3)
    const uint tap1 = dil;          // t - 2*dil (3 for dil=3)
    const uint tap2 = 2u * dil;     // t - 1*dil (6 for dil=3)

    for (uint t = 0u; t < p.M; ++t) {
        if (p.tree_branch_t != 0u && t == p.tree_branch_t) {
            for (uint i = 0u; i < slen; ++i) {
                hist[i] = snap0[ulong(i) * C + gid];
            }
        }
        const ulong idx   = ulong(t) * C + gid;
        const float x_t   = gv_norm_in[idx];
        const float raw   = fma(w.x, hist[tap0], fma(w.y, hist[tap1], fma(w.z, hist[tap2], w.w * x_t)));
        const float convc = sqf_silu(raw);
        hc_inout[idx]    += gv_in[idx] + convc;

        for (uint i = 0u; i + 1u < slen; ++i) {
            hist[i] = hist[i + 1u];
        }
        hist[slen - 1u] = x_t;

        if ((p.save_snap_mask & (1u << t)) != 0u) {
            device float* snap = (t == 0u) ? snap0
                               : (t == 1u) ? snap1
                               : (t == 2u) ? snap2
                               : (t == 3u) ? snap3
                               : (t == 4u) ? snap4 : snap5;
            for (uint i = 0u; i < slen; ++i) {
                snap[ulong(i) * C + gid] = hist[i];
            }
        }
    }

    for (uint i = 0u; i < slen; ++i) {
        ple_state[ulong(i) * C + gid] = hist[i];
    }
}

/* ============================================================================
 * 7. 36-Layer `GatedDeltaNet` (`linear_attn`, `output_gate_type = "sigmoid"`)
 * ============================================================================ */

struct SqfDeltaConvParams {
    uint M;
    uint channels;
    uint save_snap_mask;
    uint snap_layer_idx;
    uint num_lin_layers;
    uint tree_branch_t; // 0 for chain; >0 => reload `(h0,h1,h2)` from `snap0` before `t == tree_branch_t`
    uint pad1;
    uint pad2;
};

kernel void sqf_deltanet_conv1d_silu(
    device const half*          qkv_in     [[buffer(0)]], // [M, channels]
    device const float4*        conv_w     [[buffer(1)]], // [channels]
    device half*                conv_state [[buffer(2)]], // [3, channels] for this linear layer
    device half*                qkv_out    [[buffer(3)]], // [M, channels]
    device half*                snap0      [[buffer(4)]], // [3, channels] pre-offset to lin_slot
    device half*                snap1      [[buffer(5)]],
    device half*                snap2      [[buffer(6)]],
    device half*                snap3      [[buffer(7)]],
    device half*                snap4      [[buffer(8)]],
    device half*                snap5      [[buffer(9)]],
    constant SqfDeltaConvParams& p         [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.channels) return;

    const uint   C = p.channels;
    const float4 w = conv_w[gid];

    half h0 = conv_state[0u * C + gid];
    half h1 = conv_state[1u * C + gid];
    half h2 = conv_state[2u * C + gid];

    for (uint t = 0u; t < p.M; ++t) {
        if (p.tree_branch_t != 0u && t == p.tree_branch_t) {
            h0 = snap0[0u * C + gid];
            h1 = snap0[1u * C + gid];
            h2 = snap0[2u * C + gid];
        }
        const ulong idx = ulong(t) * C + gid;
        const half  h3  = qkv_in[idx];
        const float raw = fma(w.x, float(h0), fma(w.y, float(h1), fma(w.z, float(h2), w.w * float(h3))));
        qkv_out[idx]    = sqf_clamp_f16(sqf_silu(raw));

        h0 = h1;
        h1 = h2;
        h2 = h3;

        if ((p.save_snap_mask & (1u << t)) != 0u) {
            device half* snap = (t == 0u) ? snap0
                              : (t == 1u) ? snap1
                              : (t == 2u) ? snap2
                              : (t == 3u) ? snap3
                              : (t == 4u) ? snap4 : snap5;
            snap[0u * C + gid] = h0;
            snap[1u * C + gid] = h1;
            snap[2u * C + gid] = h2;
        }
    }

    conv_state[0u * C + gid] = h0;
    conv_state[1u * C + gid] = h1;
    conv_state[2u * C + gid] = h2;
}

// 2D Parallel Prefill Depthwise Causal Conv1d (FIR-4) + SiLU (`M > 8`, `save_snap_mask == 0`, `tree_branch_t == 0`):
// Replaces 40-TG serial loop with `40 * ceil(M/16)` = up to `2,560` TGs (100% bit-exact vs `sqf_deltanet_conv1d_silu`!).
// Reads `conv_state` without mutating it; follow with `sqf_deltanet_conv1d_save_tail` to commit final 3 taps.
// Grid: `((channels + 255) / 256, (M + 15) / 16, 1)`, `256` threads/TG
kernel void sqf_deltanet_conv1d_silu_prefill(
    device const half*          qkv_in     [[buffer(0)]], // [M, channels]
    device const float4*        conv_w     [[buffer(1)]], // [channels]
    device const half*          conv_state [[buffer(2)]], // [3, channels]
    device half*                qkv_out    [[buffer(3)]], // [M, channels]
    constant SqfDeltaConvParams& p         [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]            // 0..255
) {
    const uint c = tgid.x * 256u + tid;
    const uint t0 = tgid.y * 16u;
    if (c >= p.channels || t0 >= p.M) return;

    const uint   C     = p.channels;
    const float4 w     = conv_w[c];
    const uint   t_end = min(t0 + 16u, p.M);

    half h0 = (t0 >= 3u) ? qkv_in[ulong(t0 - 3u) * C + c] : conv_state[ulong(t0)      * C + c];
    half h1 = (t0 >= 2u) ? qkv_in[ulong(t0 - 2u) * C + c] : conv_state[ulong(t0 + 1u) * C + c];
    half h2 = (t0 >= 1u) ? qkv_in[ulong(t0 - 1u) * C + c] : conv_state[ulong(t0 + 2u) * C + c];

    for (uint t = t0; t < t_end; ++t) {
        const ulong idx = ulong(t) * C + c;
        const half  h3  = qkv_in[idx];
        const float raw = fma(w.x, float(h0), fma(w.y, float(h1), fma(w.z, float(h2), w.w * float(h3))));
        qkv_out[idx]    = sqf_clamp_f16(sqf_silu(raw));
        h0 = h1;
        h1 = h2;
        h2 = h3;
    }
}

// Commits final 3 taps of `[M, channels]` into `conv_state [3, channels]` after `sqf_deltanet_conv1d_silu_prefill`.
// Grid: `((channels + 255) / 256, 1, 1)`, `256` threads/TG
kernel void sqf_deltanet_conv1d_save_tail(
    device const half*          qkv_in     [[buffer(0)]], // [M, channels]
    device half*                conv_state [[buffer(1)]], // [3, channels]
    constant SqfDeltaConvParams& p         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.channels || p.M == 0u) return;
    const uint C = p.channels;
    half h0, h1, h2;
    if (p.M >= 3u) {
        h0 = qkv_in[ulong(p.M - 3u) * C + gid];
        h1 = qkv_in[ulong(p.M - 2u) * C + gid];
        h2 = qkv_in[ulong(p.M - 1u) * C + gid];
    } else if (p.M == 2u) {
        h0 = conv_state[2u * C + gid];
        h1 = qkv_in[0u * C + gid];
        h2 = qkv_in[1u * C + gid];
    } else {
        h0 = conv_state[1u * C + gid];
        h1 = conv_state[2u * C + gid];
        h2 = qkv_in[0u * C + gid];
    }
    conv_state[0u * C + gid] = h0;
    conv_state[1u * C + gid] = h1;
    conv_state[2u * C + gid] = h2;
}

struct SqfDeltaNetParams {
    uint  M_valid;
    uint  M_padded;
    uint  n_k;          // 16
    uint  n_v;          // 48
    float eps;          // 1e-6f
    uint  gate_sigmoid; // 1 => sigmoid(z), 0 => silu(z)
    uint  save_snap_mask;
    uint  snap_layer_idx;
    uint  num_lin_layers;
    uint  tree_branch_t; // 0 for chain; >0 => reload `s_reg` from `snap0` before `t == tree_branch_t`
    uint  pad1;
    uint  pad2;
};

// Exact Recurrent Update for `M = 1 .. 8` (100% bit-identical per-token math between K=0 & MTP!)
// Grid: `(n_v, 1, 1)`, `512` threads/TG (`qtr = tid >> 7` in 0..3, `dv = tid & 127`)
kernel void sqf_deltanet_recurrent_gated_norm(
    device const half*         qkv_conv  [[buffer(0)]], // [M_valid, 2*k_dim + v_dim]
    device const half*         z_in      [[buffer(1)]], // [M_valid, v_dim]
    device const float*        a_in      [[buffer(2)]], // [M_valid, n_v]
    device const float*        b_in      [[buffer(3)]], // [M_valid, n_v]
    device const float*        A_log     [[buffer(4)]], // [n_v]
    device const float*        dt_bias   [[buffer(5)]], // [n_v]
    device const float*        lin_norm  [[buffer(6)]], // [128] (NO +1.0)
    device float*              rec_state [[buffer(7)]], // [n_v, 128, 128] for this linear layer
    device half*               y_out     [[buffer(8)]], // [M_padded, v_dim]
    device float*              rec_snaps [[buffer(9)]], // optional [4, num_lin, n_v, 128, 128]
    constant SqfDeltaNetParams& p        [[buffer(10)]],
    uint hv   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],          // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],       // 0..15
    uint lane [[thread_index_in_simdgroup]]             // 0..31
) {
    if (hv >= p.n_v) return;

    const uint v_per_k  = p.n_v / p.n_k;
    const uint hk       = hv / v_per_k;
    const uint k_dim    = p.n_k * 128u;
    const uint v_dim    = p.n_v * 128u;
    const uint conv_dim = 2u * k_dim + v_dim;

    const uint dv      = tid & 127u;
    const uint qtr     = tid >> 7u;
    const uint dk_base = qtr * 32u;

    threadgroup float q_hat[128];
    threadgroup float k_hat[128];
    threadgroup float o_buf[128];
    threadgroup float tg_kv_red[4][128];
    threadgroup float tg_out_red[4][128];
    threadgroup float tg_qk_sq[8];
    threadgroup float tg_o_sq[4];
    threadgroup float tg_scalars[5];

    device float* state_hv = rec_state + ulong(hv) * 16384ul;
    const float a_log_hv   = A_log[hv];
    const float dt_bias_hv = dt_bias[hv];
    const float norm_w_dv  = lin_norm[dv];

    float s_reg[32];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 32u; ++i) {
        s_reg[i] = state_hv[(dk_base + i) * 128u + dv];
    }

    constexpr float kInvSqrt128 = 0.08838834764831845f;

    for (uint t = 0u; t < p.M_valid; ++t) {
        device const half* qkv_t = qkv_conv + ulong(t) * conv_dim;
        device const half* q_raw = qkv_t + hk * 128u;
        device const half* k_raw = qkv_t + k_dim + hk * 128u;
        device const half* v_raw = qkv_t + 2u * k_dim + hv * 128u;
        device const half* z_raw = z_in + ulong(t) * v_dim + hv * 128u;

        if (sgid < 4u) {
            float q_v  = float(q_raw[tid]);
            q_hat[tid] = q_v;
            float q_sq = simd_sum(q_v * q_v);
            if (lane == 0u) tg_qk_sq[sgid] = q_sq;
        } else if (sgid < 8u) {
            uint  idx  = tid - 128u;
            float k_v  = float(k_raw[idx]);
            k_hat[idx] = k_v;
            float k_sq = simd_sum(k_v * k_v);
            if (lane == 0u) tg_qk_sq[sgid] = k_sq;
        }
        if (tid == 256u) {
            float a_val = a_in[ulong(t) * p.n_v + hv];
            float b_val = b_in[ulong(t) * p.n_v + hv];
            float g     = -exp(a_log_hv) * sqf_softplus(a_val + dt_bias_hv);
            tg_scalars[0] = exp(g);
            tg_scalars[1] = sqf_sigmoid(b_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float q_tot = simd_sum((lane < 4u) ? tg_qk_sq[lane] : 0.0f);
            float k_tot = simd_sum((lane < 4u) ? tg_qk_sq[4u + lane] : 0.0f);
            if (lane == 0u) {
                tg_scalars[2] = rsqrt(q_tot + p.eps) * kInvSqrt128;
                tg_scalars[3] = rsqrt(k_tot + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 128u) {
            q_hat[tid] *= tg_scalars[2];
        } else if (tid < 256u) {
            k_hat[tid - 128u] *= tg_scalars[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = tg_scalars[0];
        const float beta  = tg_scalars[1];

        float kv_part = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 32u; ++i) {
            float sv = s_reg[i] * decay;
            s_reg[i] = sv;
            kv_part  = fma(sv, k_hat[dk_base + i], kv_part);
        }
        tg_kv_red[qtr][dv] = kv_part;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float kv_mem = (tg_kv_red[0][dv] + tg_kv_red[1][dv]) + (tg_kv_red[2][dv] + tg_kv_red[3][dv]);
        const float delta  = beta * (float(v_raw[dv]) - kv_mem);

        float out_part = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 32u; ++i) {
            float sv = fma(k_hat[dk_base + i], delta, s_reg[i]);
            s_reg[i] = sv;
            out_part = fma(sv, q_hat[dk_base + i], out_part);
        }
        tg_out_red[qtr][dv] = out_part;

        if ((p.save_snap_mask & (1u << t)) != 0u) {
            const ulong layer_stride = ulong(p.n_v) * 16384ul;
            const ulong slot_stride  = ulong(p.num_lin_layers) * layer_stride;
            device float* snap_hv = rec_snaps + ulong(t) * slot_stride +
                                    ulong(p.snap_layer_idx) * layer_stride + ulong(hv) * 16384ul;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 32u; ++i) {
                snap_hv[(dk_base + i) * 128u + dv] = s_reg[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid < 4u) {
            float o_val = (tg_out_red[0][dv] + tg_out_red[1][dv]) + (tg_out_red[2][dv] + tg_out_red[3][dv]);
            o_buf[dv]   = o_val;
            float o_sq  = simd_sum(o_val * o_val);
            if (lane == 0u) tg_o_sq[sgid] = o_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float o_tot = simd_sum((lane < 4u) ? tg_o_sq[lane] : 0.0f);
            if (lane == 0u) {
                tg_scalars[4] = rsqrt(o_tot * (1.0f / 128.0f) + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 128u) {
            float normed = o_buf[dv] * tg_scalars[4] * norm_w_dv;
            float z_val  = float(z_raw[dv]);
            float gate   = (p.gate_sigmoid != 0u) ? sqf_sigmoid(z_val) : sqf_silu(z_val);
            y_out[ulong(t) * v_dim + hv * 128u + dv] = sqf_clamp_f16(normed * gate);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 128u) {
        for (uint t = p.M_valid; t < p.M_padded; ++t) {
            y_out[ulong(t) * v_dim + hv * 128u + dv] = half(0.0h);
        }
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 32u; ++i) {
        state_hv[(dk_base + i) * 128u + dv] = s_reg[i];
    }
}

// Fast 192-TG Zero-Barrier Batched Prefill DeltaNet (`M > 8`)
kernel void sqf_deltanet_prep_batch(
    device const half*         qkv_conv   [[buffer(0)]], // [M_valid, conv_dim]
    device const float*        a_in       [[buffer(1)]], // [M_valid, n_v]
    device const float*        b_in       [[buffer(2)]], // [M_valid, n_v]
    device const float*        A_log      [[buffer(3)]], // [n_v]
    device const float*        dt_bias    [[buffer(4)]], // [n_v]
    device float*              qk_hat_out [[buffer(5)]], // [M_valid, 2 * k_dim]
    device float*              gates_out  [[buffer(6)]], // [M_valid, 2 * n_v]
    constant SqfDeltaNetParams& p         [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],         // (0..2*n_k, 0..M_valid-1)
    uint  tid  [[thread_index_in_threadgroup]],          // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]],       // 0..3
    uint  lane [[thread_index_in_simdgroup]]            // 0..31
) {
    const uint item = tgid.x;
    const uint t    = tgid.y;
    if (item > 2u * p.n_k || t >= p.M_valid) return;

    if (item == 2u * p.n_k) {
        for (uint hv = tid; hv < p.n_v; hv += 128u) {
            float a_val = a_in[ulong(t) * p.n_v + hv];
            float b_val = b_in[ulong(t) * p.n_v + hv];
            float g     = -exp(A_log[hv]) * sqf_softplus(a_val + dt_bias[hv]);
            gates_out[ulong(t) * (2u * p.n_v) + hv]         = exp(g);
            gates_out[ulong(t) * (2u * p.n_v) + p.n_v + hv] = sqf_sigmoid(b_val);
        }
        return;
    }

    const uint k_dim    = p.n_k * 128u;
    const uint v_dim    = p.n_v * 128u;
    const uint conv_dim = 2u * k_dim + v_dim;

    threadgroup float tg_sq[4];
    const bool  is_q = (item < p.n_k);
    const uint  hk   = is_q ? item : (item - p.n_k);
    const float v    = float(qkv_conv[ulong(t) * conv_dim + (is_q ? 0ul : ulong(k_dim)) + hk * 128u + tid]);

    float sg_sq = simd_sum(v * v);
    if (lane == 0u) tg_sq[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            constexpr float kInvSqrt128 = 0.08838834764831845f;
            float inv = rsqrt(tot + p.eps);
            tg_sq[0]  = is_q ? (inv * kInvSqrt128) : inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    qk_hat_out[ulong(t) * (2u * k_dim) + (is_q ? 0ul : ulong(k_dim)) + hk * 128u + tid] = v * tg_sq[0];
}

// Fast 192-TG (`4 * n_v`) 100%-Coalesced Recurrent DeltaNet for ALL `M = 1..256` (Decode, Spec Chain/Tree & Prefill!)
// Every SIMD lane (`lane = 0..31`) owns one contiguous `dv = col_blk * 32 + lane` (exact 128B cache-line loads/stores!),
// and each of `4` SGs (`128` th/TG) holds `32` `dk` rows (`s_reg[32]`) in registers with `1 KiB` TG reduction memory.
// Grid: `(4, n_v, 1)`, `128` threads/TG (`4` SGs)
kernel void sqf_deltanet_rec_batch(
    device const half*         qkv_conv  [[buffer(0)]],
    device const float*        qk_hat    [[buffer(1)]],
    device const float*        gates_in  [[buffer(2)]],
    device float*              rec_state [[buffer(3)]], // [n_v, 128, 128] pre-offset to lin_slot
    device float*              raw_out   [[buffer(4)]], // [M_valid, v_dim]
    constant SqfDeltaNetParams& p        [[buffer(5)]],
    device float*              snap0     [[buffer(6)]], // [n_v, 128, 128] pre-offset to lin_slot
    device float*              snap1     [[buffer(7)]],
    device float*              snap2     [[buffer(8)]],
    device float*              snap3     [[buffer(9)]],
    device float*              snap4     [[buffer(10)]],
    device float*              snap5     [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],        // (0..3, 0..n_v-1)
    uint  sgid [[simdgroup_index_in_threadgroup]],      // 0..3
    uint  lane [[thread_index_in_simdgroup]]            // 0..31
) {
    const uint col_blk = tgid.x;
    const uint hv      = tgid.y;
    if (col_blk >= 4u || hv >= p.n_v) return;

    const uint v_per_k  = p.n_v / p.n_k;
    const uint hk       = hv / v_per_k;
    const uint k_dim    = p.n_k * 128u;
    const uint v_dim    = p.n_v * 128u;
    const uint conv_dim = 2u * k_dim + v_dim;

    const uint dv      = col_blk * 32u + lane;
    const uint dk_base = sgid * 32u;

    threadgroup float tg_kv_red[4][32];
    threadgroup float tg_out_red[4][32];

    const ulong hv_off     = ulong(hv) * 16384ul;
    device float* state_hv = rec_state + hv_off;

    float s_reg[32];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 32u; ++i) {
        s_reg[i] = state_hv[(dk_base + i) * 128u + dv];
    }

    for (uint t = 0u; t < p.M_valid; ++t) {
        if (p.tree_branch_t != 0u && t == p.tree_branch_t) {
            device const float* s0_hv = snap0 + hv_off;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 32u; ++i) {
                s_reg[i] = s0_hv[(dk_base + i) * 128u + dv];
            }
        }

        const ulong qk_base = ulong(t) * (2ul * k_dim);
        device const float* q_ptr = qk_hat + qk_base + ulong(hk) * 128ul + dk_base;
        device const float* k_ptr = qk_hat + qk_base + ulong(k_dim) + ulong(hk) * 128ul + dk_base;
        const float decay = gates_in[ulong(t) * (2u * p.n_v) + hv];
        const float beta  = gates_in[ulong(t) * (2u * p.n_v) + p.n_v + hv];
        const float v_val = float(qkv_conv[ulong(t) * conv_dim + 2ul * k_dim + ulong(hv) * 128ul + dv]);

        float kv_part = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 32u; ++i) {
            float sv = s_reg[i] * decay;
            s_reg[i] = sv;
            kv_part  = fma(sv, k_ptr[i], kv_part);
        }
        tg_kv_red[sgid][lane] = kv_part;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float kv_mem = (tg_kv_red[0][lane] + tg_kv_red[1][lane]) +
                             (tg_kv_red[2][lane] + tg_kv_red[3][lane]);
        const float delta  = beta * (v_val - kv_mem);

        float out_part = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 32u; ++i) {
            float sv = fma(k_ptr[i], delta, s_reg[i]);
            s_reg[i] = sv;
            out_part = fma(sv, q_ptr[i], out_part);
        }
        tg_out_red[sgid][lane] = out_part;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            raw_out[ulong(t) * v_dim + ulong(hv) * 128ul + dv] =
                (tg_out_red[0][lane] + tg_out_red[1][lane]) +
                (tg_out_red[2][lane] + tg_out_red[3][lane]);
        }

        if ((p.save_snap_mask & (1u << t)) != 0u) {
            device float* snap_hv = ((t == 0u) ? snap0
                                   : (t == 1u) ? snap1
                                   : (t == 2u) ? snap2
                                   : (t == 3u) ? snap3
                                   : (t == 4u) ? snap4 : snap5) + hv_off;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 32u; ++i) {
                snap_hv[(dk_base + i) * 128u + dv] = s_reg[i];
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 32u; ++i) {
        state_hv[(dk_base + i) * 128u + dv] = s_reg[i];
    }
}

kernel void sqf_deltanet_post_batch(
    device const float*        raw_out  [[buffer(0)]],
    device const half*         z_in     [[buffer(1)]],
    device const float*        lin_norm [[buffer(2)]],
    device half*               y_out    [[buffer(3)]],
    constant SqfDeltaNetParams& p       [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],       // (0..n_v-1, 0..M_padded-1)
    uint  tid  [[thread_index_in_threadgroup]],        // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]],     // 0..3
    uint  lane [[thread_index_in_simdgroup]]           // 0..31
) {
    const uint hv = tgid.x;
    const uint t  = tgid.y;
    if (hv >= p.n_v || t >= p.M_padded) return;

    const uint  v_dim = p.n_v * 128u;
    const ulong idx   = ulong(t) * v_dim + hv * 128u + tid;
    if (t >= p.M_valid) {
        y_out[idx] = half(0.0h);
        return;
    }

    threadgroup float tg_sq[4];
    const float ov  = raw_out[idx];
    const float ssq = simd_sum(ov * ov);
    if (lane == 0u) tg_sq[sgid] = ssq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot * (1.0f / 128.0f) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float normed = ov * tg_sq[0] * lin_norm[tid];
    const float z_val  = float(z_in[idx]);
    const float gate   = (p.gate_sigmoid != 0u) ? sqf_sigmoid(z_val) : sqf_silu(z_val);
    y_out[idx]         = sqf_clamp_f16(normed * gate);
}

// ============================================================================
// Metal 4.1 SLC-Resident 3-Kernel Chunk-32 WY NAX GatedDeltaNet (36L!)
//   1. `sqf_deltanet_wy_qk_chunk32`      (`num_chunks x n_k`, `64` th/TG, `16 KiB` TG):
//      Computes L2-norm `Q_norm`, `K_hat` (`n_k=16`, 3x smaller -> L2-hot!) and
//      shared `KK_dot = K_hat @ K_hat^T`, `QK_dot = Q_norm @ K_hat^T` (`BK=128` NAX, `3x` less work!).
//   2. `sqf_deltanet_wy_prep_hv_chunk32` (`num_chunks x n_v`, `64` th/TG, `12.3 KiB` TG):
//      1-SG zero-TG-barrier `32x32` pure-FP32 `T = (I+M)^{-1}` overlapped with `P` + `K_sc`,
//      then `W = T @ K_sc` & `U = T @ V_sc` via `32x64x32 sg2` NAX (coalesced `c.store`).
//   3. `sqf_deltanet_wy_recur_chunk32`   (`4 x n_v = 192` TGs, `64` th/TG, `14.1 KiB` TG, 0 spill!):
//      Pure-FP32 `S_sub [128, 32]` pinned in registers; Direct-Device `tW/tQ/tP/tK` NAX (`BK=128`),
//      eliminating `K_tilde` & `Q_sc` completely via `K_hat^T @ (exp(g_31-g_r)*Delta)`!
//      Total intermediate footprint = `40.9 MB` (fits 100% in M5 Max `48 MiB` SLC & `b_moe_down_part`, 0 extra VRAM!).
// ============================================================================

struct SqfDeltaWyParams {
    uint  M_valid;
    uint  M_padded;   // Multiple of 32
    uint  num_chunks; // M_padded / 32
    uint  n_k;        // 16
    uint  n_v;        // 48
    float eps;        // 1e-6f
    uint  gate_sigmoid;
    uint  pad0;
};

// Kernel 1: `(num_chunks, n_k, 1)` = `512` TGs, `64` th/TG (`sg2`, `16.0 KiB` TG mem, `0` spill)
kernel void sqf_deltanet_wy_qk_chunk32(
    device const half*         qkv_conv [[buffer(0)]], // [M_valid, conv_dim]
    device half*               q_norm   [[buffer(1)]], // [n_k, M_padded, 128]
    device half*               k_hat    [[buffer(2)]], // [n_k, M_padded, 128]
    device float*              kk_dot   [[buffer(3)]], // [n_k, num_chunks, 32, 32]
    device float*              qk_dot   [[buffer(4)]], // [n_k, num_chunks, 32, 32]
    constant SqfDeltaWyParams& p        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],       // (c, hk)
    uint  sgid [[simdgroup_index_in_threadgroup]],     // 0..1
    uint  lane [[thread_index_in_simdgroup]]           // 0..31
) {
    const uint c  = tgid.x;
    const uint hk = tgid.y;
    if (c >= p.num_chunks || hk >= p.n_k) return;

#if defined(__HAVE_TENSOR__)
    const uint t_base   = c * 32u;
    const uint k_dim    = p.n_k * 128u;
    const uint v_dim    = p.n_v * 128u;
    const uint conv_dim = 2u * k_dim + v_dim;

    threadgroup half Q_tg[32][128]; // 8 KiB
    threadgroup half K_tg[32][128]; // 8 KiB

    device half* q_out_hk = q_norm + ulong(hk) * p.M_padded * 128ul;
    device half* k_out_hk = k_hat  + ulong(hk) * p.M_padded * 128ul;

    for (uint j = 0u; j < 16u; ++j) {
        const uint r = sgid * 16u + j;
        const uint t = t_base + r;
        half4 qh = half4(0.0h);
        half4 kh = half4(0.0h);
        if (t < p.M_valid) {
            device const half4* q4 = reinterpret_cast<device const half4*>(
                qkv_conv + ulong(t) * conv_dim + ulong(hk) * 128ul);
            device const half4* k4 = reinterpret_cast<device const half4*>(
                qkv_conv + ulong(t) * conv_dim + ulong(k_dim) + ulong(hk) * 128ul);
            const float4 qv = float4(q4[lane]);
            const float4 kv = float4(k4[lane]);
            const float inv_q = rsqrt(simd_sum(dot(qv, qv)) + p.eps);
            const float inv_k = rsqrt(simd_sum(dot(kv, kv)) + p.eps);
            qh = sqf_clamp_f16x4(qv * inv_q);
            kh = sqf_clamp_f16x4(kv * inv_k);
        }
        reinterpret_cast<threadgroup half4*>(&Q_tg[r][0])[lane] = qh;
        reinterpret_cast<threadgroup half4*>(&K_tg[r][0])[lane] = kh;
        reinterpret_cast<device half4*>(q_out_hk + ulong(t) * 128ul)[lane] = qh;
        reinterpret_cast<device half4*>(k_out_hk + ulong(t) * 128ul)[lane] = kh;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup half, extents<int, 128, 32>, tensor_inline> tQ(&Q_tg[0][0], extents<int, 128, 32>());
    tensor<threadgroup half, extents<int, 128, 32>, tensor_inline> tK(&K_tg[0][0], extents<int, 128, 32>());
    const ulong dot_off = (ulong(hk) * p.num_chunks + c) * 1024ul;
    tensor<device float, extents<int, 32, 32>, tensor_inline> tKK(kk_dot + dot_off, extents<int, 32, 32>());
    tensor<device float, extents<int, 32, 32>, tensor_inline> tQK(qk_dot + dot_off, extents<int, 32, 32>());

    constexpr auto desc_kk = matmul2d_descriptor(
        32, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_kk, execution_simdgroups<2>> op_kk;
    auto cOut = op_kk.template get_destination_cooperative_tensor<sqf_strip_as<decltype(tK)>, sqf_strip_as<decltype(tK)>, float>();

    op_kk.run(tK, tK, cOut);
    cOut.store(tKK);
    op_kk.run(tQ, tK, cOut);
    cOut.store(tQK);
#else
    (void)qkv_conv;(void)q_norm;(void)k_hat;(void)kk_dot;(void)qk_dot;(void)p;(void)sgid;(void)lane;
#endif
}

// Kernel 2: `(num_chunks, n_v, 1)` = `1,536` TGs, `64` th/TG (`sg2`, `12.3 KiB` TG mem, `0` spill)
kernel void sqf_deltanet_wy_prep_hv_chunk32(
    device const half*         qkv_conv [[buffer(0)]], // [M_valid, conv_dim]
    device const float*        a_in     [[buffer(1)]], // [M_valid, n_v]
    device const float*        b_in     [[buffer(2)]], // [M_valid, n_v]
    device const float*        A_log    [[buffer(3)]], // [n_v]
    device const float*        dt_bias  [[buffer(4)]], // [n_v]
    device const half*         k_hat    [[buffer(5)]], // [n_k, M_padded, 128]
    device const float*        kk_dot   [[buffer(6)]], // [n_k, num_chunks, 32, 32]
    device const float*        qk_dot   [[buffer(7)]], // [n_k, num_chunks, 32, 32]
    device float*              g_out    [[buffer(8)]], // [n_v, M_padded]
    device half*               w_out    [[buffer(9)]], // [n_v, M_padded, 128]
    device half*               u_out    [[buffer(10)]],// [n_v, M_padded, 128]
    device half*               p_out    [[buffer(11)]],// [n_v, num_chunks, 32, 32]
    constant SqfDeltaWyParams& p        [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],       // (c, hv)
    uint  tid  [[thread_index_in_threadgroup]],        // 0..63
    uint  sgid [[simdgroup_index_in_threadgroup]],     // 0..1
    uint  lane [[thread_index_in_simdgroup]]           // 0..31
) {
    const uint c  = tgid.x;
    const uint hv = tgid.y;
    if (c >= p.num_chunks || hv >= p.n_v) return;

#if defined(__HAVE_TENSOR__)
    const uint t_base   = c * 32u;
    const uint v_per_k  = p.n_v / p.n_k;
    const uint hk       = hv / v_per_k;
    const uint k_dim    = p.n_k * 128u;
    const uint v_dim    = p.n_v * 128u;
    const uint conv_dim = 2u * k_dim + v_dim;
    const float neg_expA = -exp(A_log[hv]);
    const float dt_b_hv  = dt_bias[hv];
    constexpr float kInvSqrt128 = 0.08838834764831845f;

    threadgroup float T_tg[32][32];   // 4 KiB -> overlaid half T_h[32][32]
    threadgroup half  KV_tg[32][128]; // 8 KiB: K_sc -> V_sc
    threadgroup float g_tg[32];       // 128 B
    threadgroup float b_tg[32];       // 128 B

    if (sgid == 0u) {
        const uint t = t_base + lane;
        float gv = 0.0f, bv = 0.0f;
        if (t < p.M_valid) {
            gv = neg_expA * sqf_softplus(a_in[ulong(t) * p.n_v + hv] + dt_b_hv);
            bv = sqf_sigmoid(b_in[ulong(t) * p.n_v + hv]);
        }
        const float gc = simd_prefix_inclusive_sum(gv);
        g_tg[lane] = gc;
        b_tg[lane] = bv;
        g_out[ulong(hv) * p.M_padded + t] = gc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup half* T_h_ptr = reinterpret_cast<threadgroup half*>(&T_tg[0][0]);
    const ulong hk_dot_off    = (ulong(hk) * p.num_chunks + c) * 1024ul;

    // Zero-TG-Barrier Overlap: `sgid==0` inverts `32x32` `T = (I+M)^{-1}` in pure FP32
    // while `sgid==1` materializes `P` and scales `K_hat` -> `K_sc` into `KV_tg`!
    if (sgid == 0u) {
        device const float* kk_ptr = kk_dot + hk_dot_off;
        const float g_col = g_tg[lane];
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 32u; ++r) {
            T_tg[r][lane] = (r > lane)
                ? (kk_ptr[r * 32u + lane] * b_tg[r] * exp(g_tg[r] - g_col))
                : ((r == lane) ? 1.0f : 0.0f);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 1u; i < 32u; ++i) {
            const float m_ij = T_tg[i][lane];
            float acc = -m_ij;
            for (uint k = 1u; k < i; ++k) {
                const float m_ik = simd_broadcast(m_ij, ushort(k));
                if (lane < k) {
                    acc = fma(-m_ik, T_tg[k][lane], acc);
                }
            }
            if (lane < i) {
                T_tg[i][lane] = acc;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        half t_col[32];
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 32u; ++r) t_col[r] = sqf_clamp_f16(T_tg[r][lane]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 32u; ++r) T_h_ptr[r * 32u + lane] = t_col[r];
    } else {
        device const float* qk_ptr = qk_dot + hk_dot_off;
        device half* p_dst         = p_out + (ulong(hv) * p.num_chunks + c) * 1024ul;
        const float g_col          = g_tg[lane];
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 32u; ++r) {
            p_dst[r * 32u + lane] = (r >= lane)
                ? sqf_clamp_f16(qk_ptr[r * 32u + lane] * kInvSqrt128 * exp(g_tg[r] - g_col))
                : half(0.0h);
        }

        device const half* kh_hk = k_hat + ulong(hk) * p.M_padded * 128ul;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 32u; ++r) {
            const float ksc = b_tg[r] * exp(g_tg[r]);
            device const half4* k4 = reinterpret_cast<device const half4*>(kh_hk + ulong(t_base + r) * 128ul);
            reinterpret_cast<threadgroup half4*>(&KV_tg[r][0])[lane] = sqf_clamp_f16x4(float4(k4[lane]) * ksc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup half, extents<int, 32, 32>, tensor_inline>  tTh(T_h_ptr, extents<int, 32, 32>());
    tensor<threadgroup half, extents<int, 128, 32>, tensor_inline> tKV(&KV_tg[0][0], extents<int, 128, 32>());
    tensor<device half, dextents<int, 2>, tensor_inline> tW(w_out + ulong(hv) * p.M_padded * 128ul, dextents<int, 2>(128, int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tU(u_out + ulong(hv) * p.M_padded * 128ul, dextents<int, 2>(128, int(p.M_padded)));

    constexpr auto desc_32x64 = matmul2d_descriptor(
        32, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_32x64, execution_simdgroups<2>> op_32x64;
    auto sKV0 = tKV.slice<64, 32>(0, 0);
    auto sKV1 = tKV.slice<64, 32>(64, 0);
    auto c64  = op_32x64.template get_destination_cooperative_tensor<sqf_strip_as<decltype(tTh)>, sqf_strip_as<decltype(sKV0)>, half>();

    // `W = T_h @ K_sc` (coalesced direct store to `tW`, ZERO TG writeback!)
    op_32x64.run(tTh, sKV0, c64);
    c64.store(tW.slice<64, 32>(0, int(t_base)));
    op_32x64.run(tTh, sKV1, c64);
    c64.store(tW.slice<64, 32>(64, int(t_base)));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load `V_sc = beta * V` (`64` threads cooperatively, 16 iters x 2 rows)
    const uint r_load2 = tid >> 5u;
    const uint c4_load = tid & 31u;
    #pragma clang loop unroll(full)
    for (uint rp = 0u; rp < 16u; ++rp) {
        const uint r = rp * 2u + r_load2;
        const uint t = t_base + r;
        if (t < p.M_valid) {
            device const half4* v4 = reinterpret_cast<device const half4*>(
                qkv_conv + ulong(t) * conv_dim + 2ul * k_dim + ulong(hv) * 128ul);
            reinterpret_cast<threadgroup half4*>(&KV_tg[r][0])[c4_load] = sqf_clamp_f16x4(float4(v4[c4_load]) * b_tg[r]);
        } else {
            reinterpret_cast<threadgroup half4*>(&KV_tg[r][0])[c4_load] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // `U = T_h @ V_sc` (coalesced direct store to `tU`!)
    op_32x64.run(tTh, sKV0, c64);
    c64.store(tU.slice<64, 32>(0, int(t_base)));
    op_32x64.run(tTh, sKV1, c64);
    c64.store(tU.slice<64, 32>(64, int(t_base)));
#else
    (void)qkv_conv;(void)a_in;(void)b_in;(void)A_log;(void)dt_bias;(void)k_hat;(void)kk_dot;(void)qk_dot;
    (void)g_out;(void)w_out;(void)u_out;(void)p_out;(void)p;(void)tid;(void)sgid;(void)lane;
#endif
}

// Kernel 3: `(4, n_v, 1)` = `192` TGs, `64` th/TG (`sg2`, `14.1 KiB` TG mem <= 16 KiB -> 2x occupancy, `0` spill!)
kernel void sqf_deltanet_wy_recur_chunk32(
    device const float*        g_in      [[buffer(0)]], // [n_v, M_padded]
    device const half*         q_norm    [[buffer(1)]], // [n_k, M_padded, 128]
    device const half*         k_hat     [[buffer(2)]], // [n_k, M_padded, 128]
    device const half*         w_in      [[buffer(3)]], // [n_v, M_padded, 128]
    device const half*         u_in      [[buffer(4)]], // [n_v, M_padded, 128]
    device const half*         p_in      [[buffer(5)]], // [n_v, num_chunks, 32, 32]
    device float*              rec_state [[buffer(6)]], // [n_v, 128, 128] pre-offset to lin_slot
    device float*              raw_out   [[buffer(7)]], // [M_padded, v_dim]
    constant SqfDeltaWyParams& p         [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],        // (0..3, 0..n_v-1)
    uint  tid  [[thread_index_in_threadgroup]],         // 0..63
    uint  sgid [[simdgroup_index_in_threadgroup]],      // 0..1
    uint  lane [[thread_index_in_simdgroup]]            // 0..31
) {
    const uint col_blk = tgid.x;
    const uint hv      = tgid.y;
    if (col_blk >= 4u || hv >= p.n_v) return;

#if defined(__HAVE_TENSOR__)
    const uint dv0     = col_blk * 32u;
    const uint v_per_k = p.n_v / p.n_k;
    const uint hk      = hv / v_per_k;
    const uint v_dim   = p.n_v * 128u;
    constexpr float kInvSqrt128 = 0.08838834764831845f;

    threadgroup half  S_tg[128][32]; // 8 KiB
    threadgroup float O_tg[32][32];  // 4 KiB
    threadgroup half  D_tg[32][32];  // 2 KiB
    threadgroup float g_tg[32];      // 128 B -> Total = 14.125 KiB <= 16 KiB!

    device float* state_hv = rec_state + ulong(hv) * 16384ul;
    tensor<device float, dextents<int, 2>, tensor_inline> tRec(state_hv, dextents<int, 2>(128, 128));
    auto sRec0 = tRec.slice<32, 64>(int(dv0), 0);
    auto sRec1 = tRec.slice<32, 64>(int(dv0), 64);

    tensor<device half, dextents<int, 2>, tensor_inline> tQ(const_cast<device half*>(q_norm + ulong(hk) * p.M_padded * 128ul), dextents<int, 2>(128, int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tK(const_cast<device half*>(k_hat  + ulong(hk) * p.M_padded * 128ul), dextents<int, 2>(128, int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tW(const_cast<device half*>(w_in   + ulong(hv) * p.M_padded * 128ul), dextents<int, 2>(128, int(p.M_padded)));
    device const half* u_hv = u_in + ulong(hv) * p.M_padded * 128ul;
    device const half* p_hv = p_in + ulong(hv) * p.num_chunks * 1024ul;
    device const float* g_hv = g_in + ulong(hv) * p.M_padded;
    tensor<device float, dextents<int, 2>, tensor_inline> tRaw(raw_out, dextents<int, 2>(int(v_dim), int(p.M_padded)));

    tensor<threadgroup half, extents<int, 32, 128>, tensor_inline> tSh(&S_tg[0][0], extents<int, 32, 128>());
    tensor<threadgroup float, extents<int, 32, 32>, tensor_inline> tO(&O_tg[0][0], extents<int, 32, 32>());
    tensor<threadgroup half, extents<int, 32, 32>, tensor_inline>  tD(&D_tg[0][0], extents<int, 32, 32>());

    constexpr auto desc_ws = matmul2d_descriptor(
        32, 32, 128, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_ws, execution_simdgroups<2>> op_ws;

    constexpr auto desc_32_acc = matmul2d_descriptor(
        32, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_32_acc, execution_simdgroups<2>> op_32_acc;

    constexpr auto desc_upd = matmul2d_descriptor(
        64, 32, 32, true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_upd, execution_simdgroups<2>> op_upd;

    auto sK0_init = tK.slice<64, 32>(0, 0);
    auto cS0 = op_upd.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sK0_init)>, sqf_strip_as<decltype(tD)>, float>();
    auto cS1 = op_upd.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sK0_init)>, sqf_strip_as<decltype(tD)>, float>();
    cS0.load(sRec0);
    cS1.load(sRec1);

    auto sW_init = tW.slice<128, 32>(0, 0);
    auto cWS = op_ws.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sW_init)>, sqf_strip_as<decltype(tSh)>, float>();

    const uint d_row2 = tid >> 4u;        // 0..3 (`4` rows at a time, 16 threads/row x half2 = 32 cols!)
    const uint d_col2 = (tid & 15u) * 2u; // 0..30 step 2

    for (uint c = 0u; c < p.num_chunks; ++c) {
        const uint t_base = c * 32u;

        // 1. Cast pure-FP32 `cS0/cS1` -> `S_tg[128][32]` (`half`) + load `g_tg[0..31]`
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cS0.get_capacity(); ++i) {
            if (cS0.is_valid_element(i)) {
                auto idx      = cS0.get_multidimensional_index(i);
                const uint dk = uint(idx[1]);
                const uint dv = uint(idx[0]);
                S_tg[dk][dv]       = sqf_clamp_f16(cS0[i]);
                S_tg[dk + 64u][dv] = sqf_clamp_f16(cS1[i]);
            }
        }
        if (sgid == 0u) g_tg[lane] = g_hv[t_base + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Direct-Device `Delta = U - W @ S_h` (`BK=128` NAX!) -> `D_tg[32][32]`
        auto sW = tW.slice<128, 32>(0, int(t_base));
        op_ws.run(sW, tSh, cWS);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cWS.get_capacity(); ++i) {
            if (cWS.is_valid_element(i)) {
                auto idx     = cWS.get_multidimensional_index(i);
                const uint r = uint(idx[1]);
                const uint d = uint(idx[0]);
                D_tg[r][d]   = sqf_clamp_f16(float(u_hv[ulong(t_base + r) * 128ul + dv0 + d]) - cWS[i]);
            }
        }

        // 3. Direct-Device `O_inter = (Q_norm @ S_h) * exp(g_r) * kInvSqrt128` (`BK=128` NAX, `S_tg` still valid!)
        auto sQ = tQ.slice<128, 32>(0, int(t_base));
        op_ws.run(sQ, tSh, cWS);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cWS.get_capacity(); ++i) {
            if (cWS.is_valid_element(i)) {
                auto idx = cWS.get_multidimensional_index(i);
                cWS[i] *= exp(g_tg[uint(idx[1])]) * kInvSqrt128;
            }
        }
        cWS.store(tO);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Direct-Device `O = O_inter + P @ D_tg` -> coalesced `cO.store(sRaw)`!
        tensor<device half, extents<int, 32, 32>, tensor_inline> tP(const_cast<device half*>(p_hv + ulong(c) * 1024ul), extents<int, 32, 32>());
        auto cO = op_32_acc.template get_destination_cooperative_tensor<sqf_strip_as<decltype(tP)>, sqf_strip_as<decltype(tD)>, float>();
        cO.load(tO);
        op_32_acc.run(tP, tD, cO);
        cO.store(tRaw.slice<32, 32>(int(hv * 128u + dv0), int(t_base)));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. In-place scale `D_tg[r][d] *= exp(g_31 - g_r)` (eliminates `K_tilde` completely!)
        const float g_last = g_tg[31];
        #pragma clang loop unroll(full)
        for (uint rp = 0u; rp < 8u; ++rp) {
            const uint r   = rp * 4u + d_row2;
            const float sc = exp(g_last - g_tg[r]);
            threadgroup half2* dst = reinterpret_cast<threadgroup half2*>(&D_tg[r][d_col2]);
            *dst = half2(sqf_clamp_f16(float((*dst).x) * sc), sqf_clamp_f16(float((*dst).y) * sc));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 6. Direct-Device Pure-FP32 `S = exp(g_31)*S + K_hat^T @ D_scaled`!
        const float decay_c = exp(g_last);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cS0.get_capacity(); ++i) {
            if (cS0.is_valid_element(i)) {
                cS0[i] *= decay_c;
                cS1[i] *= decay_c;
            }
        }
        auto sK0 = tK.slice<64, 32>(0, int(t_base));
        auto sK1 = tK.slice<64, 32>(64, int(t_base));
        op_upd.run(sK0, tD, cS0);
        op_upd.run(sK1, tD, cS1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    cS0.store(sRec0);
    cS1.store(sRec1);
#else
    (void)g_in;(void)q_norm;(void)k_hat;(void)w_in;(void)u_in;(void)p_in;(void)rec_state;(void)raw_out;(void)p;(void)tid;(void)sgid;(void)lane;
#endif
}

/* ============================================================================
 * 8. 12-Layer `QSA` (`self_attn` + `QSAIndexer`) & `mtp.layers.0` Kernels
 * ============================================================================ */

/* ---- Phase B: Width-2 x Depth-3 Speculative Tree Coordinate Helpers ----
 *   tree_mode = 0: standard linear chain (`rope = S + m`, `store = S + m`, `kv_slot = pos`)
 *   tree_mode = 1: M=7 verify tree (`S = P`):
 *                  m=0 (root) -> P, m=1..3 (Branch A) -> P+1..P+3 @ slots P+1..P+3,
 *                  m=4..6 (Branch B) -> P+1..P+3 @ slots P+4..P+6
 *   tree_mode = 2: M=2 MTP draft depth 1 (`S = Q`): m=0 (a0) -> rope Q @ Q; m=1 (b0) -> rope Q @ Q+1
 *   tree_mode = 3: M=2 MTP draft depth 2 (`S = Q+1`): m=0 (a1) -> rope Q+1 @ Q+2; m=1 (b1) -> rope Q+1 @ Q+3 */
inline uint sqf_tree_rope_pos(uint tree_mode, uint S, uint m) {
    if (tree_mode == 1u) return (m >= 4u) ? (S + 1u + (m - 4u)) : (S + m);
    if (tree_mode == 2u || tree_mode == 3u) return S;
    return S + m;
}

inline uint sqf_tree_store_pos(uint tree_mode, uint S, uint m) {
    if (tree_mode == 3u) return S + 1u + m;
    return S + m;
}

inline uint sqf_tree_kv_slot(uint tree_mode, uint S, uint m, uint pos) {
    if (tree_mode == 1u) {
        if (m >= 4u && pos > S) return S + 4u + (pos - S - 1u);
        return pos;
    }
    if (tree_mode == 2u) {
        if (m == 1u && pos == S) return S + 1u;
        return pos;
    }
    if (tree_mode == 3u) {
        if (m == 0u) {
            return (pos == S) ? (S + 1u) : pos;
        } else {
            if (pos == S - 1u) return S;
            if (pos == S)      return S + 2u;
            return pos;
        }
    }
    return pos;
}

inline bool sqf_tree_is_branch_b(uint tree_mode, uint m) {
    if (tree_mode == 1u) return (m >= 4u);
    if (tree_mode == 2u || tree_mode == 3u) return (m == 1u);
    return false;
}

inline uint sqf_tree_max_rope_pos(uint tree_mode, uint S, uint M_valid) {
    if (tree_mode == 1u) return S + 3u;
    if (tree_mode == 2u || tree_mode == 3u) return S;
    return S + (M_valid - 1u);
}

struct SqfQsaPrepParams {
    uint  M_valid;
    uint  M_padded;
    uint  n_q;           // 24
    uint  n_kv;          // 2
    uint  head_dim;      // 256
    uint  rotary_dim;    // 64
    uint  seq_pos_start;
    uint  max_seq_alloc; // KV token stride: linear `max_seq_alloc`, or paged UMA pool tokens
    float eps;
    uint  kv_zero_end;
    uint  paged;         // 1 => translate logical block `pos >> 2` through `block_loc` (Tier 2 paged KV)
    uint  tree_mode;     // 0=chain, 1=M=7 verify tree, 2=M=2 draft d1, 3=M=2 draft d2
};

/* ---- Step 4 Tier 2: Global-`pos`-Invariant Virtual->Physical KV Block Indirection ----
 * `block_loc[b]` (`b = pos >> 2`): bit 31 set => UMA-resident at physical pool block `loc & 0x7FFFFFFF`.
 * RoPE + causal masks keep using the TRUE global `pos`; only the physical KV address is remapped. */
#define SQF_KV_LOC_UMA_BIT  0x80000000u
#define SQF_KV_LOC_IDX_MASK 0x7FFFFFFFu

inline uint sqf_kv_phys_pos(device const uint* block_loc, uint paged, uint pos) {
    if (paged == 0u) return pos;
    const uint loc = block_loc[pos >> 2];
    return ((loc & SQF_KV_LOC_IDX_MASK) << 2) | (pos & 3u);
}

inline bool sqf_kv_is_resident(device const uint* block_loc, uint paged, uint pos) {
    return (paged == 0u) || ((block_loc[pos >> 2] & SQF_KV_LOC_UMA_BIT) != 0u);
}

// Grid: `(n_q + 2 * n_kv, M_padded, 1)`, `head_dim` (`256`) threads/TG
kernel void sqf_qsa_qkv_prep(
    device const half*       q_gate_in  [[buffer(0)]], // [M_valid, n_q * 2 * head_dim]
    device const half*       k_raw_in   [[buffer(1)]], // [M_valid, n_kv * head_dim]
    device const half*       v_raw_in   [[buffer(2)]], // [M_valid, n_kv * head_dim]
    device const float*      q_norm_w   [[buffer(3)]], // [head_dim] (+1.0 folded)
    device const float*      k_norm_w   [[buffer(4)]], // [head_dim] (+1.0 folded)
    constant float*          inv_freq   [[buffer(5)]], // [rotary_dim / 2]
    device half*             q_rope_out [[buffer(6)]], // [n_q, M_padded, head_dim]
    device half*             k_cache    [[buffer(7)]], // [n_kv, max_seq_alloc, head_dim] (physical)
    device half*             v_cache    [[buffer(8)]], // [n_kv, max_seq_alloc, head_dim] (physical)
    constant SqfQsaPrepParams& p        [[buffer(9)]],
    device const uint*       block_loc  [[buffer(10)]], // [max_blocks] for this layer (read iff p.paged)
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint item       = tgid.x;
    const uint m          = tgid.y;
    const uint total_item = p.n_q + 2u * p.n_kv;
    if (item >= total_item || m >= p.M_padded || tid >= p.head_dim) return;

    const uint half_rot = p.rotary_dim >> 1u;

    if (m >= p.M_valid) {
        if (item < p.n_q) {
            q_rope_out[(ulong(item) * ulong(p.M_padded) + ulong(m)) * ulong(p.head_dim) + ulong(tid)] = half(0.0h);
        } else {
            const bool is_k = (item < p.n_q + p.n_kv);
            uint hkv = is_k ? (item - p.n_q) : (item - (p.n_q + p.n_kv));
            uint pos = p.seq_pos_start + m;
            device half* cache = is_k ? k_cache : v_cache;
            if (p.paged == 0u) {
                if (pos < p.kv_zero_end && pos < p.max_seq_alloc) {
                    cache[(ulong(hkv) * ulong(p.max_seq_alloc) + ulong(pos)) * ulong(p.head_dim) + ulong(tid)] = half(0.0h);
                }
            } else if (pos < p.kv_zero_end && sqf_kv_is_resident(block_loc, 1u, pos)) {
                const uint pp = sqf_kv_phys_pos(block_loc, 1u, pos);
                cache[(ulong(hkv) * ulong(p.max_seq_alloc) + ulong(pp)) * ulong(p.head_dim) + ulong(tid)] = half(0.0h);
            }
        }
        return;
    }

    const uint rope_pos = sqf_tree_rope_pos(p.tree_mode, p.seq_pos_start, m);
    const uint pos      = sqf_tree_store_pos(p.tree_mode, p.seq_pos_start, m);
    const uint ppos     = sqf_kv_phys_pos(block_loc, p.paged, pos);

    // V head copy
    if (item >= p.n_q + p.n_kv) {
        uint hkv = item - (p.n_q + p.n_kv);
        half v   = v_raw_in[(ulong(m) * ulong(p.n_kv) + ulong(hkv)) * ulong(p.head_dim) + ulong(tid)];
        v_cache[(ulong(hkv) * ulong(p.max_seq_alloc) + ulong(ppos)) * ulong(p.head_dim) + ulong(tid)] = v;
        if (m + 1u == p.M_valid) {
            if (p.paged == 0u) {
                for (uint zp = pos + 1u; zp < p.kv_zero_end && zp < p.max_seq_alloc; ++zp) {
                    v_cache[(ulong(hkv) * ulong(p.max_seq_alloc) + ulong(zp)) * ulong(p.head_dim) + ulong(tid)] = half(0.0h);
                }
            } else {
                for (uint zp = pos + 1u; zp < p.kv_zero_end; ++zp) {
                    if (!sqf_kv_is_resident(block_loc, 1u, zp)) continue;
                    const uint zpp = sqf_kv_phys_pos(block_loc, 1u, zp);
                    v_cache[(ulong(hkv) * ulong(p.max_seq_alloc) + ulong(zpp)) * ulong(p.head_dim) + ulong(tid)] = half(0.0h);
                }
            }
        }
        return;
    }

    threadgroup float tg_sq[8];
    threadgroup float tg_vec[128];

    const bool  is_q    = (item < p.n_q);
    const ulong q_str   = ulong(p.n_q) * (2ul * p.head_dim);
    const float raw_val = is_q ? float(q_gate_in[ulong(m) * q_str + ulong(item) * (2ul * p.head_dim) + tid])
                               : float(k_raw_in[(ulong(m) * p.n_kv + (item - p.n_q)) * p.head_dim + tid]);
    const float nw      = is_q ? q_norm_w[tid] : k_norm_w[tid];

    float sg_sq = simd_sum(raw_val * raw_val);
    if (lane == 0u) tg_sq[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_sgs = (p.head_dim + 31u) >> 5u;
    if (sgid == 0u) {
        float tot = simd_sum((lane < num_sgs) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.head_dim) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float normed = raw_val * tg_sq[0] * nw;
    if (tid < p.rotary_dim) {
        tg_vec[tid] = normed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_val = normed;
    if (tid < half_rot) {
        float angle = float(rope_pos) * inv_freq[tid];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_val     = tg_vec[tid] * c - tg_vec[tid + half_rot] * s;
    } else if (tid < p.rotary_dim) {
        uint  i     = tid - half_rot;
        float angle = float(rope_pos) * inv_freq[i];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_val     = tg_vec[tid] * c + tg_vec[i] * s;
    }

    half out_h = sqf_clamp_f16(out_val);
    if (is_q) {
        q_rope_out[(ulong(item) * ulong(p.M_padded) + ulong(m)) * ulong(p.head_dim) + ulong(tid)] = out_h;
    } else {
        uint hkv = item - p.n_q;
        k_cache[(ulong(hkv) * ulong(p.max_seq_alloc) + ulong(ppos)) * ulong(p.head_dim) + ulong(tid)] = out_h;
        if (m + 1u == p.M_valid) {
            if (p.paged == 0u) {
                for (uint zp = pos + 1u; zp < p.kv_zero_end && zp < p.max_seq_alloc; ++zp) {
                    k_cache[(ulong(hkv) * ulong(p.max_seq_alloc) + ulong(zp)) * ulong(p.head_dim) + ulong(tid)] = half(0.0h);
                }
            } else {
                for (uint zp = pos + 1u; zp < p.kv_zero_end; ++zp) {
                    if (!sqf_kv_is_resident(block_loc, 1u, zp)) continue;
                    const uint zpp = sqf_kv_phys_pos(block_loc, 1u, zp);
                    k_cache[(ulong(hkv) * ulong(p.max_seq_alloc) + ulong(zpp)) * ulong(p.head_dim) + ulong(tid)] = half(0.0h);
                }
            }
        }
    }
}

struct SqfIndexerParams {
    uint  M_valid;
    uint  idx_n_heads;    // 4
    uint  idx_head_dim;   // 128
    uint  rotary_dim;     // 64
    uint  compress_ratio; // 4
    uint  block_topk;     // 512 (`budget / compress_ratio`)
    uint  seq_pos_start;
    uint  first_block;    // `seq_pos_start / compress_ratio`
    uint  num_new_blocks; // newly completed blocks
    float eps;
    uint  max_blocks_stride;
    uint  tree_mode;      // 0=chain, 1=M=7 verify tree, 2=M=2 draft d1, 3=M=2 draft d2
    uint  M_padded;       // Multiple of 64 for prefill NAX indexer score
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

// Writes `raw_k [M_valid, 128]` into `idx_raw_k_cache` AND computes `q_idx [M_valid, 4, 128]` (`q_norm` + `rope_64`)!
// Grid: `(idx_n_heads + 1, M_valid, 1)` (`5 x M_valid`), `128` threads/TG
kernel void sqf_qsa_indexer_prep(
    device const float*       idx_qk_in   [[buffer(0)]], // [M_valid, (idx_n_heads + 1) * 128]
    device const float*       idx_q_norm  [[buffer(1)]], // [128] (+1.0 folded)
    constant float*           inv_freq    [[buffer(2)]], // [32]
    device half*              raw_k_cache [[buffer(3)]], // [max_seq_alloc, 128] for this full layer
    device float*             q_idx_out   [[buffer(4)]], // [M_valid, idx_n_heads, 128]
    constant SqfIndexerParams& p          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],          // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]],       // 0..3
    uint  lane [[thread_index_in_simdgroup]]             // 0..31
) {
    const uint item = tgid.x;
    const uint m    = tgid.y;
    if (item > p.idx_n_heads || m >= p.M_valid || tid >= p.idx_head_dim) return;

    const uint row_stride = (p.idx_n_heads + 1u) * p.idx_head_dim;
    const uint rope_pos   = sqf_tree_rope_pos(p.tree_mode, p.seq_pos_start, m);
    const uint store_pos  = sqf_tree_store_pos(p.tree_mode, p.seq_pos_start, m);

    if (item == p.idx_n_heads) {
        float rk = idx_qk_in[ulong(m) * row_stride + ulong(p.idx_n_heads) * p.idx_head_dim + tid];
        raw_k_cache[ulong(store_pos) * p.idx_head_dim + tid] = sqf_clamp_f16(rk);
        return;
    }

    threadgroup float tg_sq[4];
    threadgroup float tg_vec[64];

    const float qv = idx_qk_in[ulong(m) * row_stride + ulong(item) * p.idx_head_dim + tid];
    float sg_sq    = simd_sum(qv * qv);
    if (lane == 0u) tg_sq[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.idx_head_dim) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint  half_rot = p.rotary_dim >> 1u;
    const float normed   = qv * tg_sq[0] * idx_q_norm[tid];
    if (tid < p.rotary_dim) {
        tg_vec[tid] = normed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_v = normed;
    if (tid < half_rot) {
        float angle = float(rope_pos) * inv_freq[tid];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_v       = tg_vec[tid] * c - tg_vec[tid + half_rot] * s;
    } else if (tid < p.rotary_dim) {
        uint  i     = tid - half_rot;
        float angle = float(rope_pos) * inv_freq[i];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_v       = tg_vec[tid] * c + tg_vec[i] * s;
    }

    q_idx_out[(ulong(m) * p.idx_n_heads + item) * p.idx_head_dim + tid] = out_v;
}

// Prefill-Only Indexer Prep: writes `raw_k_cache [store_pos, 128]` AND head-major `[4, M_padded, 128]` FP16 `q_idx`
// (zero-padded for `m >= M_valid`) for zero-TG-mem Direct-Device NAX `sqf_qsa_indexer_score_nax`!
// Grid: `(idx_n_heads + 1, M_padded, 1)`, `128` threads/TG
kernel void sqf_qsa_indexer_prep_prefill(
    device const float*       idx_qk_in    [[buffer(0)]], // [M_valid, (idx_n_heads + 1) * 128]
    device const float*       idx_q_norm   [[buffer(1)]], // [128] (+1.0 folded)
    constant float*           inv_freq     [[buffer(2)]], // [32]
    device half*              raw_k_cache  [[buffer(3)]], // [max_seq_alloc, 128]
    device half*              q_idx_hm_f16 [[buffer(4)]], // [idx_n_heads, M_padded, 128]
    constant SqfIndexerParams& p           [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],           // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]],        // 0..3
    uint  lane [[thread_index_in_simdgroup]]              // 0..31
) {
    const uint item = tgid.x;
    const uint m    = tgid.y;
    if (item > p.idx_n_heads || m >= p.M_padded || tid >= p.idx_head_dim) return;

    if (m >= p.M_valid) {
        if (item < p.idx_n_heads) {
            q_idx_hm_f16[(ulong(item) * ulong(p.M_padded) + ulong(m)) * ulong(p.idx_head_dim) + ulong(tid)] = half(0.0h);
        }
        return;
    }

    const uint row_stride = (p.idx_n_heads + 1u) * p.idx_head_dim;
    const uint pos        = p.seq_pos_start + m;

    if (item == p.idx_n_heads) {
        const float rk = idx_qk_in[ulong(m) * row_stride + ulong(p.idx_n_heads) * p.idx_head_dim + tid];
        raw_k_cache[ulong(pos) * p.idx_head_dim + tid] = sqf_clamp_f16(rk);
        return;
    }

    threadgroup float tg_sq[4];
    threadgroup float tg_vec[64];

    const float qv = idx_qk_in[ulong(m) * row_stride + ulong(item) * p.idx_head_dim + tid];
    const float sg_sq = simd_sum(qv * qv);
    if (lane == 0u) tg_sq[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.idx_head_dim) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint  half_rot = p.rotary_dim >> 1u;
    const float normed   = qv * tg_sq[0] * idx_q_norm[tid];
    if (tid < p.rotary_dim) {
        tg_vec[tid] = normed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_v = normed;
    if (tid < half_rot) {
        const float angle = float(pos) * inv_freq[tid];
        const float c     = precise::cos(angle);
        const float s     = precise::sin(angle);
        out_v             = tg_vec[tid] * c - tg_vec[tid + half_rot] * s;
    } else if (tid < p.rotary_dim) {
        const uint  i     = tid - half_rot;
        const float angle = float(pos) * inv_freq[i];
        const float c     = precise::cos(angle);
        const float s     = precise::sin(angle);
        out_v             = tg_vec[tid] * c + tg_vec[i] * s;
    }

    q_idx_hm_f16[(ulong(item) * ulong(p.M_padded) + ulong(m)) * ulong(p.idx_head_dim) + ulong(tid)] = sqf_clamp_f16(out_v);
}

// Builds `block_k[b] = rope_64(rms_norm_128(mean_{j=0..C-1}(raw_k[slot(C*b + j)]), idx_k_norm), pos = C * b)`
// Grid: `(num_new_blocks, 1, 1)`, `128` threads/TG
kernel void sqf_qsa_indexer_build_blocks(
    device const half*        raw_k_cache   [[buffer(0)]], // [max_seq_alloc, 128]
    device const float*       idx_k_norm    [[buffer(1)]], // [128] (+1.0 folded)
    constant float*           inv_freq      [[buffer(2)]], // [32]
    device half*              block_k_cache [[buffer(3)]], // [max_blocks_alloc, 128]
    constant SqfIndexerParams& p            [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],             // 0..127
    uint sgid [[simdgroup_index_in_threadgroup]],          // 0..3
    uint lane [[thread_index_in_simdgroup]]                // 0..31
) {
    if (tgid >= p.num_new_blocks || tid >= p.idx_head_dim) return;

    const uint b        = p.first_block + tgid;
    const uint base_pos = b * p.compress_ratio;
    const uint br_a_m   = (p.tree_mode == 1u) ? 3u : 0u;

    float acc = 0.0f;
    for (uint j = 0u; j < p.compress_ratio; ++j) {
        const uint sp = sqf_tree_kv_slot(p.tree_mode, p.seq_pos_start, br_a_m, base_pos + j);
        acc += float(raw_k_cache[ulong(sp) * p.idx_head_dim + tid]);
    }
    const float mean_k = acc / float(p.compress_ratio);

    threadgroup float tg_sq[4];
    threadgroup float tg_vec[64];

    float sg_sq = simd_sum(mean_k * mean_k);
    if (lane == 0u) tg_sq[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.idx_head_dim) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint  half_rot = p.rotary_dim >> 1u;
    const float normed   = mean_k * tg_sq[0] * idx_k_norm[tid];
    if (tid < p.rotary_dim) {
        tg_vec[tid] = normed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_v = normed;
    if (tid < half_rot) {
        float angle = float(base_pos) * inv_freq[tid];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_v       = tg_vec[tid] * c - tg_vec[tid + half_rot] * s;
    } else if (tid < p.rotary_dim) {
        uint  i     = tid - half_rot;
        float angle = float(base_pos) * inv_freq[i];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_v       = tg_vec[tid] * c + tg_vec[i] * s;
    }

    block_k_cache[ulong(b) * p.idx_head_dim + tid] = sqf_clamp_f16(out_v);
}

// Builds Branch B's single boundary `block_k` (`b = p.first_block`) into scratch `branch_b_blk [128]`!
// Grid: `(1, 1, 1)`, `128` threads/TG
kernel void sqf_qsa_indexer_build_branch_b_block(
    device const half*        raw_k_cache  [[buffer(0)]],
    device const float*       idx_k_norm   [[buffer(1)]],
    constant float*           inv_freq     [[buffer(2)]],
    device half*              branch_b_blk [[buffer(3)]], // [128]
    constant SqfIndexerParams& p           [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (tid >= p.idx_head_dim) return;

    const uint b        = p.first_block;
    const uint base_pos = b * p.compress_ratio;
    const uint br_b_m   = (p.tree_mode == 1u) ? 6u : 1u;

    float acc = 0.0f;
    for (uint j = 0u; j < p.compress_ratio; ++j) {
        const uint sp = sqf_tree_kv_slot(p.tree_mode, p.seq_pos_start, br_b_m, base_pos + j);
        acc += float(raw_k_cache[ulong(sp) * p.idx_head_dim + tid]);
    }
    const float mean_k = acc / float(p.compress_ratio);

    threadgroup float tg_sq[4];
    threadgroup float tg_vec[64];

    float sg_sq = simd_sum(mean_k * mean_k);
    if (lane == 0u) tg_sq[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.idx_head_dim) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint  half_rot = p.rotary_dim >> 1u;
    const float normed   = mean_k * tg_sq[0] * idx_k_norm[tid];
    if (tid < p.rotary_dim) {
        tg_vec[tid] = normed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_v = normed;
    if (tid < half_rot) {
        float angle = float(base_pos) * inv_freq[tid];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_v       = tg_vec[tid] * c - tg_vec[tid + half_rot] * s;
    } else if (tid < p.rotary_dim) {
        uint  i     = tid - half_rot;
        float angle = float(base_pos) * inv_freq[i];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_v       = tg_vec[tid] * c + tg_vec[i] * s;
    }

    branch_b_blk[tid] = sqf_clamp_f16(out_v);
}

// Scores complete blocks `b = 0 .. num_complete - 1` (`score[b] = sum_{h=0..3} max(0, dot(q_idx[h], block_k[b])) / sqrt(128)`)
// Grid: `(ceil(max_complete / 8), M_valid, 1)`, `256` threads/TG (`8` SIMD-groups = `8` blocks/TG, `4` floats/lane)
kernel void sqf_qsa_indexer_score(
    device const float4*      q_idx_in      [[buffer(0)]], // [M_valid, idx_n_heads, 32]
    device const half4*       block_k_cache [[buffer(1)]], // [max_blocks_alloc, 32]
    device float*             scores_out    [[buffer(2)]], // [M_valid, max_blocks_stride]
    constant SqfIndexerParams& p            [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],         // 0..7
    uint  lane [[thread_index_in_simdgroup]]               // 0..31
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;

    const uint q_pos        = p.seq_pos_start + m;
    const uint num_complete = (q_pos + 1u) / p.compress_ratio;
    if (num_complete <= p.block_topk) return; // Fast dense path needs no scoring!

    const uint b = tgid.x * 8u + sgid;
    if (b >= num_complete) return;

    const float4 bk = float4(block_k_cache[ulong(b) * 32ul + lane]);
    float sum_relu  = 0.0f;
    for (uint h = 0u; h < p.idx_n_heads; ++h) {
        const float4 qh = q_idx_in[(ulong(m) * p.idx_n_heads + h) * 32ul + lane];
        float d = simd_sum(dot(qh, bk));
        sum_relu += max(0.0f, d);
    }

    if (lane == 0u) {
        scores_out[ulong(m) * ulong(p.max_blocks_stride) + ulong(b)] = sum_relu * rsqrt(float(p.idx_head_dim));
    }
}

// Shared-K Batched Speculative Decode Block Scorer (`M = 1..8`, compile-time unrolled, tree-aware!):
// Loads each `block_k_cache[b]` row ONCE and scores all `M` queries in registers (`4x-7x` less DRAM traffic at `64K-256K`!)
// Grid: `((max_q_complete + 7) / 8, 1, 1)`, `256` threads/TG (`8` SGs = `8` blocks/TG)
template <uint M_FIX>
inline void sqf_qsa_indexer_score_batch_impl(
    device const float4*      q_idx_in,
    device const half4*       block_k_cache,
    device float*             scores_out,
    device const half4*       branch_b_blk,
    constant SqfIndexerParams& p,
    uint b,
    uint lane
) {
    const uint max_q_pos    = sqf_tree_max_rope_pos(p.tree_mode, p.seq_pos_start, M_FIX);
    const uint max_complete = (max_q_pos + 1u) / p.compress_ratio;
    if (max_complete <= p.block_topk || b >= max_complete) return;

    const float4 bk          = float4(block_k_cache[ulong(b) * 32ul + lane]);
    const bool   is_b_patch  = (p.tree_mode != 0u) && (b == p.first_block);
    const float4 bk_b        = is_b_patch ? float4(branch_b_blk[lane]) : bk;
    const float  inv_sqrt_hd = rsqrt(float(p.idx_head_dim));

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const uint q_pos        = sqf_tree_rope_pos(p.tree_mode, p.seq_pos_start, m);
        const uint num_complete = (q_pos + 1u) / p.compress_ratio;
        if (num_complete > p.block_topk && b < num_complete) {
            const float4 use_bk = (is_b_patch && sqf_tree_is_branch_b(p.tree_mode, m)) ? bk_b : bk;
            float sum_relu = 0.0f;
            for (uint h = 0u; h < p.idx_n_heads; ++h) {
                const float4 qh = q_idx_in[(ulong(m) * p.idx_n_heads + h) * 32ul + lane];
                float d = simd_sum(dot(qh, use_bk));
                sum_relu += max(0.0f, d);
            }
            if (lane == 0u) {
                scores_out[ulong(m) * ulong(p.max_blocks_stride) + ulong(b)] = sum_relu * inv_sqrt_hd;
            }
        }
    }
}

kernel void sqf_qsa_indexer_score_batch(
    device const float4*      q_idx_in      [[buffer(0)]], // [M_valid, idx_n_heads, 32]
    device const half4*       block_k_cache [[buffer(1)]], // [max_blocks_alloc, 32]
    device float*             scores_out    [[buffer(2)]], // [M_valid, max_blocks_stride]
    constant SqfIndexerParams& p            [[buffer(3)]],
    device const half4*       branch_b_blk  [[buffer(4)]], // [32] Branch B boundary block (tree_mode != 0)
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],          // 0..7
    uint lane [[thread_index_in_simdgroup]]                // 0..31
) {
    const uint b = tgid * 8u + sgid;
    #define SQF_IDX_SB_CASE(MF) { sqf_qsa_indexer_score_batch_impl<MF>(q_idx_in, block_k_cache, scores_out, branch_b_blk, p, b, lane); break; }
    switch (p.M_valid) {
        case 1u: SQF_IDX_SB_CASE(1u)
        case 2u: SQF_IDX_SB_CASE(2u)
        case 3u: SQF_IDX_SB_CASE(3u)
        case 4u: SQF_IDX_SB_CASE(4u)
        case 5u: SQF_IDX_SB_CASE(5u)
        case 6u: SQF_IDX_SB_CASE(6u)
        case 7u: SQF_IDX_SB_CASE(7u)
        case 8u: SQF_IDX_SB_CASE(8u)
        default: break;
    }
    #undef SQF_IDX_SB_CASE
}

// Tiled Batched-Prefill Block Scorer (`M_TILE = 32` queries x `B_TILE = 64` blocks per TG, `256` threads = `8` SGs):
//   - Each SG (`sgid = 0..7`) loads `4` queries x `4` heads (`float4 q_reg[4][4]`) into registers ONCE per TG.
//   - Across `8` steps of `8` blocks (`b_step = b_tg + s * 8`), all `8` SGs cooperatively stage `8` `block_k_cache`
//     rows (`2 KiB`) into `threadgroup half4 tg_bk[8][32]`, cutting `block_k_cache` DRAM traffic by `32x` (`28.6 TiB -> 0.89 TiB`
//     at `256K` prefill) while keeping 100.0000% bit-identical IEEE-754 scores (`simd_sum(dot(qh, bk))` across `h = 0..3`)!
// Grid: `((max_q_complete + 63) / 64, (M_valid + 31) / 32, 1)`, `256` threads/TG (`8` SIMD-groups)
kernel void sqf_qsa_indexer_score_tiled(
    device const float4*      q_idx_in      [[buffer(0)]], // [M_valid, 4, 32]
    device const half4*       block_k_cache [[buffer(1)]], // [max_blocks_alloc, 32]
    device float*             scores_out    [[buffer(2)]], // [M_valid, max_blocks_stride]
    constant SqfIndexerParams& p            [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],         // 0..7
    uint  lane [[thread_index_in_simdgroup]]               // 0..31
) {
    const uint m_tg = tgid.y * 32u;
    if (m_tg >= p.M_valid) return;

    const uint m_tg_last       = min(m_tg + 31u, p.M_valid - 1u);
    const uint max_tg_q_pos    = p.seq_pos_start + m_tg_last;
    const uint max_tg_complete = (max_tg_q_pos + 1u) / p.compress_ratio;
    const uint b_tg            = tgid.x * 64u;
    if (max_tg_complete <= p.block_topk || b_tg >= max_tg_complete) return;

    const uint m_sg = m_tg + sgid * 4u;
    float4 q_reg[4][4];
    uint   q_complete[4];

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 4u; ++r) {
        const uint m = m_sg + r;
        if (m < p.M_valid) {
            const uint qp = p.seq_pos_start + m;
            const uint nc = (qp + 1u) / p.compress_ratio;
            q_complete[r] = (nc > p.block_topk) ? nc : 0u;
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 4u; ++h) {
                q_reg[r][h] = q_idx_in[(ulong(m) * 4ul + ulong(h)) * 32ul + ulong(lane)];
            }
        } else {
            q_complete[r] = 0u;
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 4u; ++h) {
                q_reg[r][h] = float4(0.0f);
            }
        }
    }

    threadgroup half4 tg_bk[8][32];
    threadgroup float tg_sc[8][4][8];
    const float inv_sqrt_hd = rsqrt(float(p.idx_head_dim));

    for (uint s = 0u; s < 8u; ++s) {
        const uint b_step = b_tg + s * 8u;
        if (b_step >= max_tg_complete) break;

        const uint b_load = b_step + sgid;
        tg_bk[sgid][lane] = (b_load < max_tg_complete)
            ? block_k_cache[ulong(b_load) * 32ul + ulong(lane)]
            : half4(0.0h);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint bi = 0u; bi < 8u; ++bi) {
            const uint b = b_step + bi;
            if (b >= max_tg_complete) break;
            const float4 bk = float4(tg_bk[bi][lane]);

            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                if (b < q_complete[r]) {
                    float sum_relu = 0.0f;
                    #pragma clang loop unroll(full)
                    for (uint h = 0u; h < 4u; ++h) {
                        float d = simd_sum(dot(q_reg[r][h], bk));
                        sum_relu += max(0.0f, d);
                    }
                    if (lane == 0u) {
                        tg_sc[sgid][r][bi] = sum_relu * inv_sqrt_hd;
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < 8u) {
            const uint b = b_step + lane;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                if (b < q_complete[r]) {
                    const uint m = m_sg + r;
                    scores_out[ulong(m) * ulong(p.max_blocks_stride) + ulong(b)] = tg_sc[sgid][r][lane];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Metal 4.1 Direct-Device NAX Indexer Block Scorer (`64x64x128 sg4`, 0 TG mem, 0 barriers, ~50 TF!):
// Replaces scalar FP32 `sqf_qsa_indexer_score_tiled` (`+43 ms` at 32K -> `< 1 ms`!) during prefill.
// Grid: `((max_q_complete + 63) / 64, M_padded / 64, 1)`, `128` threads/TG (`4` SGs)
kernel void sqf_qsa_indexer_score_nax(
    device const half*        q_idx_hm_f16  [[buffer(0)]], // [4, M_padded, 128]
    device const half*        block_k_cache [[buffer(1)]], // [max_blocks_stride, 128] (`max_blocks_stride % 64 == 0`)
    device float*             scores_out    [[buffer(2)]], // [M_valid, max_blocks_stride]
    constant SqfIndexerParams& p            [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint b_tg = tgid.x * 64u;
    const uint m_tg = tgid.y * 64u;
    if (m_tg >= p.M_valid) return;

    const uint m_tg_last       = min(m_tg + 63u, p.M_valid - 1u);
    const uint max_tg_complete = (p.seq_pos_start + m_tg_last + 1u) / p.compress_ratio;
    if (max_tg_complete <= p.block_topk || b_tg >= max_tg_complete) return;

#if defined(__HAVE_TENSOR__)
    tensor<device half, dextents<int, 2>, tensor_inline> tB(
        const_cast<device half*>(block_k_cache), dextents<int, 2>(128, int(p.max_blocks_stride)));
    auto sB = tB.slice<128, 64>(0, int(b_tg));

    constexpr auto desc = matmul2d_descriptor(
        64, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    const ulong head_stride = ulong(p.M_padded) * 128ul;
    tensor<device half, dextents<int, 2>, tensor_inline> tQ0(
        const_cast<device half*>(q_idx_hm_f16), dextents<int, 2>(128, int(p.M_padded)));
    auto sQ0 = tQ0.slice<128, 64>(0, int(m_tg));

    auto cSum = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sQ0)>, sqf_strip_as<decltype(sB)>, float>();
    auto cH   = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sQ0)>, sqf_strip_as<decltype(sB)>, float>();

    op.run(sQ0, sB, cSum);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cSum.get_capacity(); ++i) {
        if (cSum.is_valid_element(i)) cSum[i] = max(0.0f, cSum[i]);
    }

    #pragma clang loop unroll(full)
    for (uint h = 1u; h < 4u; ++h) {
        tensor<device half, dextents<int, 2>, tensor_inline> tQh(
            const_cast<device half*>(q_idx_hm_f16) + ulong(h) * head_stride,
            dextents<int, 2>(128, int(p.M_padded)));
        auto sQh = tQh.slice<128, 64>(0, int(m_tg));
        op.run(sQh, sB, cH);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cSum.get_capacity(); ++i) {
            if (cSum.is_valid_element(i)) cSum[i] += max(0.0f, cH[i]);
        }
    }

    const float inv_sqrt_hd = rsqrt(float(p.idx_head_dim));
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cSum.get_capacity(); ++i) {
        if (cSum.is_valid_element(i)) {
            auto idx     = cSum.get_multidimensional_index(i);
            const uint m = m_tg + uint(idx[1]);
            const uint b = b_tg + uint(idx[0]);
            if (m < p.M_valid) {
                const uint nc = (p.seq_pos_start + m + 1u) / p.compress_ratio;
                if (nc > p.block_topk && b < nc) {
                    scores_out[ulong(m) * ulong(p.max_blocks_stride) + ulong(b)] = cSum[i] * inv_sqrt_hd;
                }
            }
        }
    }
#else
    (void)q_idx_hm_f16; (void)block_k_cache; (void)scores_out; (void)p;
#endif
}

#define SQF_INDEXER_SEL_WORDS 4096u // 16 KiB -> 131,072 blocks = 524,288 tokens (512K)!

// 4-Pass 8-Bit IEEE-754 Radix Select + `16 KiB` `sel_bits[4096]` Causal-Ascending Block Selector!
// Supports up to `4096 * 32 = 131,072` blocks = `524,288` tokens (`512K`, 2x full Qwen3.8-Flash-Next 256K context)!
// Grid: `(M_valid, 1, 1)`, `1024` threads/TG (`32` SGs)
kernel void sqf_qsa_indexer_topk(
    device const float*       scores_in      [[buffer(0)]], // [M_valid, max_blocks_stride]
    device uint*              sel_blocks_out [[buffer(1)]], // [M_valid, block_topk]
    constant SqfIndexerParams& p             [[buffer(2)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],              // 0..1023
    uint sgid [[simdgroup_index_in_threadgroup]],           // 0..31
    uint lane [[thread_index_in_simdgroup]]                 // 0..31
) {
    if (m >= p.M_valid) return;

    const uint q_pos        = sqf_tree_rope_pos(p.tree_mode, p.seq_pos_start, m);
    const uint raw_complete = (q_pos + 1u) / p.compress_ratio;
    const uint num_complete = min(raw_complete, SQF_INDEXER_SEL_WORDS * 32u);
    if (num_complete <= p.block_topk) return;

    threadgroup atomic_uint hist[8][256];                   // 8 KiB -> 8x lower pass-0 atomic contention!
    threadgroup uint        hist_sum[256];                  // 1 KiB
    threadgroup atomic_uint sel_bits[SQF_INDEXER_SEL_WORDS]; // 16 KiB -> up to 131,072 blocks = 524,288 tokens!
    threadgroup uint        tg_state[4];
    threadgroup uint        tg_sg_cnt[32];

    device const float* row_scores = scores_in + ulong(m) * ulong(p.max_blocks_stride);
    device uint*        row_out    = sel_blocks_out + ulong(m) * ulong(p.block_topk);

    const uint bank  = sgid & 7u;
    uint prefix_val  = 0u;
    uint prefix_mask = 0u;
    uint rem_k       = p.block_topk;

    // 1. 4-pass 8-bit MSB->LSB radix select to find exact `(thresh_u, rem_k)`
    for (uint pass = 0u; pass < 4u; ++pass) {
        const uint shift = 24u - pass * 8u;
        for (uint i = tid; i < 2048u; i += 1024u) {
            atomic_store_explicit(&hist[i >> 8u][i & 255u], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = tid; b < num_complete; b += 1024u) {
            uint u = sqf_nonneg_f32_to_u32(row_scores[b]);
            if ((u & prefix_mask) == prefix_val) {
                uint bkt = (u >> shift) & 0xFFu;
                atomic_fetch_add_explicit(&hist[bank][bkt], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 256u) {
            uint tot = 0u;
            #pragma clang loop unroll(full)
            for (uint bk = 0u; bk < 8u; ++bk) {
                tot += atomic_load_explicit(&hist[bk][tid], memory_order_relaxed);
            }
            hist_sum[tid] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint acc    = 0u;
            uint chosen = 0u;
            for (int bkt = 255; bkt >= 0; --bkt) {
                uint c = hist_sum[uint(bkt)];
                if (acc + c >= rem_k) {
                    chosen = uint(bkt);
                    rem_k -= acc;
                    break;
                }
                acc += c;
            }
            tg_state[0] = prefix_val | (chosen << shift);
            tg_state[1] = prefix_mask | (0xFFu << shift);
            tg_state[2] = rem_k;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        prefix_val  = tg_state[0];
        prefix_mask = tg_state[1];
        rem_k       = tg_state[2];
    }

    const uint thresh_u  = prefix_val;
    const uint num_words = min(SQF_INDEXER_SEL_WORDS, (num_complete + 31u) >> 5u);

    for (uint w = tid; w < num_words; w += 1024u) {
        atomic_store_explicit(&sel_bits[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Mark all blocks strictly `> thresh_u`
    for (uint b = tid; b < num_complete; b += 1024u) {
        uint u = sqf_nonneg_f32_to_u32(row_scores[b]);
        if (u > thresh_u && (b >> 5u) < SQF_INDEXER_SEL_WORDS) {
            atomic_fetch_or_explicit(&sel_bits[b >> 5u], 1u << (b & 31u), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Mark first `rem_k` ties (`u == thresh_u`) in deterministic lowest-`b`-first order
    uint ties_taken = 0u;
    for (uint base = 0u; base < num_complete && ties_taken < rem_k; base += 1024u) {
        const uint b      = base + tid;
        const bool is_tie = (b < num_complete) && (sqf_nonneg_f32_to_u32(row_scores[b]) == thresh_u);
        const uint t_val  = is_tie ? 1u : 0u;
        const uint sg_pre = simd_prefix_exclusive_sum(t_val);
        const uint sg_tot = simd_sum(t_val);
        if (lane == 0u) tg_sg_cnt[sgid] = sg_tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            uint c   = tg_sg_cnt[lane];
            uint pre = simd_prefix_exclusive_sum(c);
            uint tot = simd_sum(c);
            tg_sg_cnt[lane] = pre;
            if (lane == 0u) tg_state[3] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint rank = ties_taken + tg_sg_cnt[sgid] + sg_pre;
        if (is_tie && rank < rem_k && (b >> 5u) < SQF_INDEXER_SEL_WORDS) {
            atomic_fetch_or_explicit(&sel_bits[b >> 5u], 1u << (b & 31u), memory_order_relaxed);
        }
        ties_taken += tg_state[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Emit the `block_topk` selected blocks in strictly ascending order (`0 .. num_complete - 1`)
    uint written = 0u;
    for (uint w_base = 0u; w_base < num_words; w_base += 1024u) {
        const uint w_idx  = w_base + tid;
        uint       word   = (w_idx < num_words) ? atomic_load_explicit(&sel_bits[w_idx], memory_order_relaxed) : 0u;
        const uint pc     = popcount(word);
        const uint sg_pre = simd_prefix_exclusive_sum(pc);
        const uint sg_tot = simd_sum(pc);
        if (lane == 0u) tg_sg_cnt[sgid] = sg_tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            uint c   = tg_sg_cnt[lane];
            uint pre = simd_prefix_exclusive_sum(c);
            uint tot = simd_sum(c);
            tg_sg_cnt[lane] = pre;
            if (lane == 0u) tg_state[3] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint out_pos = written + tg_sg_cnt[sgid] + sg_pre;
        while (word != 0u) {
            uint bit = ctz(word);
            if (out_pos < p.block_topk) {
                row_out[out_pos] = w_idx * 32u + bit;
            }
            ++out_pos;
            word &= (word - 1u);
        }
        written += tg_state[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct SqfQsaDecodeParams {
    uint  M_valid;
    uint  M_padded;
    uint  n_q;            // 24
    uint  n_kv;           // 2
    uint  head_dim;       // 256
    uint  q_per_tg;       // e.g. 6 (so 4 groups for 24Q / 2KV!)
    uint  seq_pos_start;
    uint  max_seq_alloc;
    uint  max_splits;     // e.g. 64
    uint  compress_ratio; // 4
    uint  block_topk;     // 512
    float attn_scale;     // 1.0f / sqrt(head_dim) = 0.0625f
    uint  m_base;         // First query row of this dispatch (paged group-split; 0 otherwise)
    uint  paged;          // 1 => translate logical KV blocks through `block_loc`
    uint  tree_mode;      // 0=chain, 1=M=7 verify tree, 2=M=2 draft d1, 3=M=2 draft d2
    uint  m_end;          // Exclusive upper query row of this group (`0` => `M_valid`)
};

inline void sqf_qsa_query_split_plan(
    uint q_pos,
    uint compress_ratio,
    uint block_topk,
    uint max_splits,
    thread bool& is_sparse,
    thread uint& n_tok,
    thread uint& my_splits,
    thread uint& my_kv_per_split
) {
    const uint num_complete = (q_pos + 1u) / compress_ratio;
    is_sparse = (num_complete > block_topk);
    if (!is_sparse) {
        n_tok = q_pos + 1u;
    } else {
        const uint tail = (q_pos + 1u) - num_complete * compress_ratio;
        n_tok = block_topk * compress_ratio + tail;
    }
    uint target     = clamp((n_tok + 63u) >> 6u, 1u, max_splits);
    my_kv_per_split = ((n_tok + target - 1u) / target + 7u) & ~7u;
    my_splits       = max(1u, (n_tok + my_kv_per_split - 1u) / my_kv_per_split);
}

// Per-`m` Deterministic GQA-Grouped Split-KV Attention (Dense + QSA Sparse, `M = 1..8` AND Sparse Prefill!)
// Grid: `(max_splits, num_q_groups, M_valid)`, `256` threads/TG (`8` SIMD-groups)
kernel void sqf_qsa_decode_splitkv(
    device const half*        q_rope      [[buffer(0)]], // [n_q, M_padded, head_dim]
    device const half*        k_cache     [[buffer(1)]], // [n_kv, max_seq_alloc, head_dim]
    device const half*        v_cache     [[buffer(2)]], // [n_kv, max_seq_alloc, head_dim]
    device const uint*        sel_blocks  [[buffer(3)]], // [M_valid, block_topk]
    device float*             part_m      [[buffer(4)]], // [M_valid, max_splits, n_q]
    device float*             part_l      [[buffer(5)]], // [M_valid, max_splits, n_q]
    device float*             part_o      [[buffer(6)]], // [M_valid, max_splits, n_q, head_dim]
    constant SqfQsaDecodeParams& p        [[buffer(7)]],
    device const uint*        block_loc   [[buffer(8)]], // [max_blocks] for this layer (read iff p.paged)
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],          // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],       // 0..7
    uint  lane [[thread_index_in_simdgroup]]             // 0..31
) {
    const uint split_idx = tgid.x;
    const uint q_grp     = tgid.y;
    const uint m         = tgid.z + p.m_base;
    const uint hq_base   = q_grp * p.q_per_tg;
    if (m >= p.M_valid || hq_base >= p.n_q) return;

    const uint g_cnt     = min(p.q_per_tg, p.n_q - hq_base);
    const uint gqa_ratio = p.n_q / p.n_kv;
    const uint hkv       = hq_base / gqa_ratio;
    const uint q_pos     = sqf_tree_rope_pos(p.tree_mode, p.seq_pos_start, m);

    bool is_sparse = false;
    uint n_tok = 0u, my_splits = 1u, my_kv_per_split = 8u;
    sqf_qsa_query_split_plan(q_pos, p.compress_ratio, p.block_topk, p.max_splits,
                             is_sparse, n_tok, my_splits, my_kv_per_split);
    if (split_idx >= my_splits) return;

    const uint kv_beg       = min(n_tok, split_idx * my_kv_per_split);
    const uint kv_end       = min(n_tok, kv_beg + my_kv_per_split);
    const uint num_complete = (q_pos + 1u) / p.compress_ratio;
    const uint blk_tokens   = p.block_topk * p.compress_ratio;

    threadgroup float tg_q[6][256];
    threadgroup float tg_m[6][8];
    threadgroup float tg_l[6][8];
    threadgroup float tg_o[8][256];

    if (tid < p.head_dim) {
        for (uint g = 0u; g < g_cnt; ++g) {
            tg_q[g][tid] = float(q_rope[(ulong(hq_base + g) * p.M_padded + m) * p.head_dim + tid]) * p.attn_scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint dims_per_lane = p.head_dim >> 5u; // 8 for head_dim=256, 4 for head_dim=128
    const uint d_base        = lane * dims_per_lane;

    float q_reg[6][8];
    for (uint g = 0u; g < g_cnt; ++g) {
        for (uint j = 0u; j < dims_per_lane; ++j) {
            q_reg[g][j] = tg_q[g][d_base + j];
        }
    }

    float m_sg[6];
    float l_sg[6];
    float o_sg[6][8];
    for (uint g = 0u; g < 6u; ++g) {
        m_sg[g] = -INFINITY;
        l_sg[g] = 0.0f;
        for (uint j = 0u; j < 8u; ++j) o_sg[g][j] = 0.0f;
    }

    device const half* k_head = k_cache + ulong(hkv) * ulong(p.max_seq_alloc) * ulong(p.head_dim);
    device const half* v_head = v_cache + ulong(hkv) * ulong(p.max_seq_alloc) * ulong(p.head_dim);
    device const uint* m_sel  = sel_blocks + ulong(m) * ulong(p.block_topk);

    for (uint t_idx = kv_beg + sgid; t_idx < kv_end; t_idx += 8u) {
        uint pos;
        if (!is_sparse) {
            pos = t_idx;
        } else if (t_idx < blk_tokens) {
            uint blk = m_sel[t_idx / p.compress_ratio];
            pos      = blk * p.compress_ratio + (t_idx % p.compress_ratio);
        } else {
            pos = num_complete * p.compress_ratio + (t_idx - blk_tokens);
        }

        // Tree remap (no-op when `tree_mode == 0`) + paged logical->physical translation
        pos = sqf_tree_kv_slot(p.tree_mode, p.seq_pos_start, m, pos);
        const uint ppos = sqf_kv_phys_pos(block_loc, p.paged, pos);

        float kf[8];
        float vf[8];
        device const half* kp = k_head + ulong(ppos) * p.head_dim + d_base;
        device const half* vp = v_head + ulong(ppos) * p.head_dim + d_base;
        for (uint j = 0u; j < dims_per_lane; ++j) {
            kf[j] = float(kp[j]);
            vf[j] = float(vp[j]);
        }

        for (uint g = 0u; g < g_cnt; ++g) {
            float dot_part = 0.0f;
            for (uint j = 0u; j < dims_per_lane; ++j) {
                dot_part = fma(q_reg[g][j], kf[j], dot_part);
            }
            float score  = simd_sum(dot_part);
            float m_prev = m_sg[g];
            float m_new  = max(m_prev, score);
            float alpha  = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);
            float weight = exp(score - m_new);
            m_sg[g]      = m_new;
            l_sg[g]      = fma(l_sg[g], alpha, weight);
            for (uint j = 0u; j < dims_per_lane; ++j) {
                o_sg[g][j] = fma(o_sg[g][j], alpha, weight * vf[j]);
            }
        }
    }

    if (lane == 0u) {
        for (uint g = 0u; g < g_cnt; ++g) {
            tg_m[g][sgid] = m_sg[g];
            tg_l[g][sgid] = l_sg[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong ml_base = (ulong(m) * p.max_splits + split_idx) * p.n_q;
    for (uint g = 0u; g < g_cnt; ++g) {
        const uint hq = hq_base + g;
        for (uint j = 0u; j < dims_per_lane; ++j) {
            tg_o[sgid][d_base + j] = o_sg[g][j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < p.head_dim) {
            float m_max = -INFINITY;
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 8u; ++s) {
                m_max = max(m_max, tg_m[g][s]);
            }

            float l_tot = 0.0f;
            float o_tot = 0.0f;
            if (m_max > -INFINITY) {
                #pragma clang loop unroll(full)
                for (uint s = 0u; s < 8u; ++s) {
                    float ms = tg_m[g][s];
                    if (ms > -INFINITY) {
                        float w = exp(ms - m_max);
                        l_tot = fma(tg_l[g][s], w, l_tot);
                        o_tot = fma(tg_o[s][tid], w, o_tot);
                    }
                }
            }

            if (tid == 0u) {
                part_m[ml_base + hq] = m_max;
                part_l[ml_base + hq] = l_tot;
            }
            part_o[(ml_base + hq) * p.head_dim + tid] = o_tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Zero-Spill High-Occupancy GQA=12 (`head_dim = 256`, `compress_ratio = 4`) Split-KV Attention:
// `6` SIMD-groups (`192` threads/TG), `2` Q heads/SG -> ALL `12` Q heads of `1` KV head in ONE TG!
//   - Cuts register usage from `~135+` (spilling to stack) -> `~44` regs/thread (ZERO spill!).
//   - Reads each KV head ONCE per split (2x less KV DRAM traffic) via coalesced `half4` staging into `6 KiB` TG memory.
//   - Eliminates the end-of-TG cross-SG `tg_o` reduction barrier completely (`12x-17x` faster than baseline!).
//   - Strictly ascending per-split `t_base` loop => 100.0000% bit-identical `K>0 == K=0`!
// Grid: `(launch_splits, n_kv, g_cnt_m)`, `192` threads/TG (`6` SGs)
kernel void sqf_qsa_decode_splitkv_gqa12_hd256(
    device const half4*       q_rope      [[buffer(0)]], // [n_q, M_padded, 64]
    device const half4*       k_cache     [[buffer(1)]], // [n_kv, max_seq_alloc, 64]
    device const half4*       v_cache     [[buffer(2)]], // [n_kv, max_seq_alloc, 64]
    device const uint*        sel_blocks  [[buffer(3)]], // [M_valid, block_topk]
    device float*             part_m      [[buffer(4)]], // [M_valid, max_splits, n_q]
    device float*             part_l      [[buffer(5)]], // [M_valid, max_splits, n_q]
    device float4*            part_o      [[buffer(6)]], // [M_valid, max_splits, n_q, 64]
    constant SqfQsaDecodeParams& p        [[buffer(7)]],
    device const uint*        block_loc   [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],       // 0..5
    uint  lane [[thread_index_in_simdgroup]]             // 0..31
) {
    const uint split_idx = tgid.x;
    const uint hkv       = tgid.y;
    const uint m         = tgid.z + p.m_base;
    if (m >= p.M_valid || hkv >= p.n_kv) return;

    const uint hq0   = hkv * 12u + sgid * 2u;
    const uint hq1   = hq0 + 1u;
    const uint q_pos = sqf_tree_rope_pos(p.tree_mode, p.seq_pos_start, m);

    bool is_sparse = false;
    uint n_tok = 0u, my_splits = 1u, my_kv_per_split = 8u;
    sqf_qsa_query_split_plan(q_pos, p.compress_ratio, p.block_topk, p.max_splits,
                             is_sparse, n_tok, my_splits, my_kv_per_split);
    if (split_idx >= my_splits) return;

    const uint kv_beg       = min(n_tok, split_idx * my_kv_per_split);
    const uint kv_end       = min(n_tok, kv_beg + my_kv_per_split);
    const uint num_complete = (q_pos + 1u) >> 2u;
    const uint blk_tokens   = p.block_topk << 2u;

    const ulong qb0  = (ulong(hq0) * p.M_padded + m) * 64ul;
    const ulong qb1  = (ulong(hq1) * p.M_padded + m) * 64ul;
    const float4 q00 = float4(q_rope[qb0 + lane * 2u + 0u]) * p.attn_scale;
    const float4 q01 = float4(q_rope[qb0 + lane * 2u + 1u]) * p.attn_scale;
    const float4 q10 = float4(q_rope[qb1 + lane * 2u + 0u]) * p.attn_scale;
    const float4 q11 = float4(q_rope[qb1 + lane * 2u + 1u]) * p.attn_scale;

    float  m0 = -INFINITY, m1 = -INFINITY;
    float  l0 = 0.0f, l1 = 0.0f;
    float4 o00 = float4(0.0f), o01 = float4(0.0f);
    float4 o10 = float4(0.0f), o11 = float4(0.0f);

    device const half4* k_head = k_cache + ulong(hkv) * ulong(p.max_seq_alloc) * 64ul;
    device const half4* v_head = v_cache + ulong(hkv) * ulong(p.max_seq_alloc) * 64ul;
    device const uint*  m_sel  = sel_blocks + ulong(m) * ulong(p.block_topk);

    threadgroup half4 tg_k[6][64];
    threadgroup half4 tg_v[6][64];

    for (uint t_base = kv_beg; t_base < kv_end; t_base += 6u) {
        const uint t_cur = t_base + sgid;
        if (t_cur < kv_end) {
            uint pos;
            if (!is_sparse) {
                pos = t_cur;
            } else if (t_cur < blk_tokens) {
                uint blk = m_sel[t_cur >> 2u];
                pos      = (blk << 2u) | (t_cur & 3u);
            } else {
                pos = (num_complete << 2u) + (t_cur - blk_tokens);
            }
            pos = sqf_tree_kv_slot(p.tree_mode, p.seq_pos_start, m, pos);
            const uint ppos = sqf_kv_phys_pos(block_loc, p.paged, pos);
            const ulong off = ulong(ppos) * 64ul + lane * 2u;
            tg_k[sgid][lane * 2u + 0u] = k_head[off + 0u];
            tg_k[sgid][lane * 2u + 1u] = k_head[off + 1u];
            tg_v[sgid][lane * 2u + 0u] = v_head[off + 0u];
            tg_v[sgid][lane * 2u + 1u] = v_head[off + 1u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_cnt = min(6u, kv_end - t_base);
        for (uint ti = 0u; ti < tile_cnt; ++ti) {
            const float4 k0 = float4(tg_k[ti][lane * 2u + 0u]);
            const float4 k1 = float4(tg_k[ti][lane * 2u + 1u]);
            const float4 v0 = float4(tg_v[ti][lane * 2u + 0u]);
            const float4 v1 = float4(tg_v[ti][lane * 2u + 1u]);

            const float s0  = simd_sum(dot(q00, k0) + dot(q01, k1));
            const float s1  = simd_sum(dot(q10, k0) + dot(q11, k1));

            const float mn0 = max(m0, s0);
            const float a0  = (m0 == -INFINITY) ? 0.0f : exp(m0 - mn0);
            const float w0  = exp(s0 - mn0);
            m0 = mn0; l0 = fma(l0, a0, w0);
            o00 = fma(o00, float4(a0), float4(w0) * v0);
            o01 = fma(o01, float4(a0), float4(w0) * v1);

            const float mn1 = max(m1, s1);
            const float a1  = (m1 == -INFINITY) ? 0.0f : exp(m1 - mn1);
            const float w1  = exp(s1 - mn1);
            m1 = mn1; l1 = fma(l1, a1, w1);
            o10 = fma(o10, float4(a1), float4(w1) * v0);
            o11 = fma(o11, float4(a1), float4(w1) * v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong ml0 = (ulong(m) * p.max_splits + split_idx) * p.n_q + hq0;
    const ulong ml1 = ml0 + 1ul;
    if (lane == 0u) {
        part_m[ml0] = m0; part_l[ml0] = l0;
        part_m[ml1] = m1; part_l[ml1] = l1;
    }
    part_o[ml0 * 64ul + lane * 2u + 0u] = o00;
    part_o[ml0 * 64ul + lane * 2u + 1u] = o01;
    part_o[ml1 * 64ul + lane * 2u + 0u] = o10;
    part_o[ml1 * 64ul + lane * 2u + 1u] = o11;
}

// Reduces `my_splits` partitions per query `m` AND fuses `* sigmoid(gate)`!
// Grid: `(n_q, M_padded, 1)`, `head_dim` (`256`) threads/TG
kernel void sqf_qsa_decode_reduce(
    device const float*       part_m         [[buffer(0)]], // [M_valid, max_splits, n_q]
    device const float*       part_l         [[buffer(1)]], // [M_valid, max_splits, n_q]
    device const float*       part_o         [[buffer(2)]], // [M_valid, max_splits, n_q, head_dim]
    device const half*        q_gate_in      [[buffer(3)]], // [M_valid, n_q * 2 * head_dim]
    device half*              gated_attn_out [[buffer(4)]], // [M_padded, n_q * head_dim]
    constant SqfQsaDecodeParams& p           [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint hq = tgid.x;
    const uint m  = tgid.y;
    if (hq >= p.n_q || m >= p.M_padded || tid >= p.head_dim) return;

    const ulong out_stride = ulong(p.n_q) * p.head_dim;
    if (m >= p.M_valid) {
        gated_attn_out[ulong(m) * out_stride + ulong(hq) * p.head_dim + tid] = half(0.0h);
        return;
    }

    const uint q_pos = sqf_tree_rope_pos(p.tree_mode, p.seq_pos_start, m);
    bool is_sparse = false;
    uint n_tok = 0u, my_splits = 1u, my_kv_per_split = 8u;
    sqf_qsa_query_split_plan(q_pos, p.compress_ratio, p.block_topk, p.max_splits,
                             is_sparse, n_tok, my_splits, my_kv_per_split);

    float m_max = -INFINITY;
    for (uint s = 0u; s < my_splits; ++s) {
        m_max = max(m_max, part_m[(ulong(m) * p.max_splits + s) * p.n_q + hq]);
    }

    float l_tot = 0.0f;
    float o_tot = 0.0f;
    if (m_max > -INFINITY) {
        for (uint s = 0u; s < my_splits; ++s) {
            const ulong base = (ulong(m) * p.max_splits + s) * p.n_q + hq;
            float ms = part_m[base];
            if (ms > -INFINITY) {
                float w = exp(ms - m_max);
                l_tot = fma(part_l[base], w, l_tot);
                o_tot = fma(part_o[base * p.head_dim + tid], w, o_tot);
            }
        }
    }

    const float attn_val = (l_tot > 0.0f) ? (o_tot / l_tot) : 0.0f;
    const ulong q_str    = ulong(p.n_q) * (2ul * p.head_dim);
    const float g_val    = float(q_gate_in[ulong(m) * q_str + ulong(hq) * (2ul * p.head_dim) + p.head_dim + tid]);
    gated_attn_out[ulong(m) * out_stride + ulong(hq) * p.head_dim + tid] =
        sqf_clamp_f16(attn_val * sqf_sigmoid(g_val));
}

// Phase B Tree Branch-B Compaction: copies `1..3` winning Branch B tokens from `[src_pos..src_pos+cnt-1]`
// to `[dst_pos..dst_pos+cnt-1]` across `k_cache`, `v_cache`, and `idx_raw_k` for one full-attn layer!
// Grid: `(2 * n_kv + 1, cnt, 1)`, `head_dim` (`256`) threads/TG
struct SqfTreeCompactParams {
    uint dst_pos;
    uint src_pos;
    uint count;
    uint n_kv;
    uint head_dim;
    uint idx_head_dim;
    uint max_seq_alloc;
    uint pad0;
};

kernel void sqf_tree_compact_branch_b(
    device half*                 k_cache     [[buffer(0)]],
    device half*                 v_cache     [[buffer(1)]],
    device half*                 raw_k_cache [[buffer(2)]],
    constant SqfTreeCompactParams& p         [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint item = tgid.x;
    const uint k    = tgid.y;
    if (k >= p.count) return;

    const ulong dst = ulong(p.dst_pos + k);
    const ulong src = ulong(p.src_pos + k);
    if (item < p.n_kv) {
        if (tid < p.head_dim) {
            const ulong h_off = ulong(item) * ulong(p.max_seq_alloc);
            k_cache[(h_off + dst) * ulong(p.head_dim) + tid] = k_cache[(h_off + src) * ulong(p.head_dim) + tid];
        }
    } else if (item < 2u * p.n_kv) {
        if (tid < p.head_dim) {
            const ulong h_off = ulong(item - p.n_kv) * ulong(p.max_seq_alloc);
            v_cache[(h_off + dst) * ulong(p.head_dim) + tid] = v_cache[(h_off + src) * ulong(p.head_dim) + tid];
        }
    } else if (item == 2u * p.n_kv) {
        if (tid < p.idx_head_dim) {
            raw_k_cache[dst * ulong(p.idx_head_dim) + tid] = raw_k_cache[src * ulong(p.idx_head_dim) + tid];
        }
    }
}

struct SqfQsaPrefillNaxParams {
    uint M_valid;
    uint M_padded;
    uint n_q;
    uint n_kv;
    uint seq_pos_start;
    uint total_kv;
    uint max_seq_alloc;
    uint paged;    // 1 => tile base `kv_pos` mapped through `block_loc` (host guarantees 8-block tile contiguity)
};

// Metal 4 `NAX` Cooperative-Tensor Dense Causal FlashAttention (`head_dim = 256`, `q_pos < 2051`)!
kernel void sqf_qsa_prefill_dense_nax(
    device const half*              q_rope         [[buffer(0)]], // [n_q, M_padded, 256]
    device const half*              k_cache        [[buffer(1)]], // [n_kv, max_seq_alloc, 256]
    device const half*              v_cache        [[buffer(2)]], // [n_kv, max_seq_alloc, 256]
    device const half*              q_gate_in      [[buffer(3)]], // [M_valid, n_q * 512]
    device half*                    gated_attn_out [[buffer(4)]], // [M_padded, n_q * 256]
    constant SqfQsaPrefillNaxParams& p             [[buffer(5)]],
    device const uint*              block_loc      [[buffer(6)]], // [max_blocks] for this layer (read iff p.paged)
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],                // 0..1
    uint  lane [[thread_index_in_simdgroup]]                      // 0..31
) {
    const uint hq = tgid.y;
    const uint m0 = tgid.x * 32u + sgid * 16u;
    if (hq >= p.n_q || m0 >= p.M_padded) return;

    const ulong o_str = ulong(p.n_q) * 256ul;
    const ulong g_str = ulong(p.n_q) * 512ul;

    if (m0 >= p.M_valid) {
        for (uint r = 0u; r < 16u && (m0 + r) < p.M_padded; ++r) {
            for (uint d = lane; d < 256u; d += 32u) {
                gated_attn_out[ulong(m0 + r) * o_str + ulong(hq) * 256ul + d] = half(0.0h);
            }
        }
        return;
    }

    const uint gqa_ratio = p.n_q / p.n_kv;
    const uint hkv       = hq / gqa_ratio;
    const uint max_q_pos = p.seq_pos_start + min(m0 + 15u, p.M_valid - 1u);
    const uint kv_limit  = min(p.total_kv, max_q_pos + 1u);

#if defined(__HAVE_TENSOR__)
    threadgroup float S_tg[2][16][32];
    threadgroup half  P_tg[2][16][32];
    threadgroup float M_tg[2][16];
    threadgroup float L_tg[2][16];
    threadgroup float A_tg[2][16];

    if (lane < 16u) {
        M_tg[sgid][lane] = -INFINITY;
        L_tg[sgid][lane] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device half* q_head = const_cast<device half*>(q_rope) + ulong(hq) * p.M_padded * 256ul;
    device half* k_head = const_cast<device half*>(k_cache) + ulong(hkv) * p.max_seq_alloc * 256ul;
    device half* v_head = const_cast<device half*>(v_cache) + ulong(hkv) * p.max_seq_alloc * 256ul;

    tensor<device half, dextents<int, 2>, tensor_inline> tQ(q_head, dextents<int, 2>(256, int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tK(k_head, dextents<int, 2>(256, int(p.max_seq_alloc)));
    tensor<device half, dextents<int, 2>, tensor_inline> tV(v_head, dextents<int, 2>(256, int(p.max_seq_alloc)));
    tensor<threadgroup half, extents<int, 32, 16>, tensor_inline> tP(&P_tg[sgid][0][0], extents<int, 32, 16>());

    constexpr auto desc_qk = matmul2d_descriptor(
        16, 32, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;

    constexpr auto desc_pv = matmul2d_descriptor(
        16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    auto sQ   = tQ.slice<256, 16>(0, int(m0));
    auto qCT  = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(sQ);

    auto sV0  = tV.slice<64, 32>(0, 0);
    auto oCT0 = op_pv.get_destination_cooperative_tensor<sqf_strip_as<decltype(tP)>, sqf_strip_as<decltype(sV0)>, float>();
    auto oCT1 = op_pv.get_destination_cooperative_tensor<sqf_strip_as<decltype(tP)>, sqf_strip_as<decltype(sV0)>, float>();
    auto oCT2 = op_pv.get_destination_cooperative_tensor<sqf_strip_as<decltype(tP)>, sqf_strip_as<decltype(sV0)>, float>();
    auto oCT3 = op_pv.get_destination_cooperative_tensor<sqf_strip_as<decltype(tP)>, sqf_strip_as<decltype(sV0)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            oCT0[i] = 0.0f; oCT1[i] = 0.0f; oCT2[i] = 0.0f; oCT3[i] = 0.0f;
        }
    }

    for (uint kv_pos = 0u; kv_pos < kv_limit; kv_pos += 32u) {
        const uint kv_phys = sqf_kv_phys_pos(block_loc, p.paged, kv_pos); // == kv_pos when unpaged
        auto sK  = tK.slice<256, 32>(0, int(kv_phys));
        auto sCT = op_qk.get_destination_cooperative_tensor<sqf_strip_as<decltype(qCT)>, sqf_strip_as<decltype(sK)>, float>();
        op_qk.run(qCT, sK, sCT);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < sCT.get_capacity(); ++i) {
            if (sCT.is_valid_element(i)) {
                auto idx = sCT.get_multidimensional_index(i);
                S_tg[sgid][uint(idx[1])][uint(idx[0])] = sCT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint kv_col_pos = kv_pos + lane;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 16u; ++r) {
            const uint m_glob = m0 + r;
            const uint q_pos  = p.seq_pos_start + m_glob;
            const bool valid  = (m_glob < p.M_valid) && (kv_col_pos <= q_pos) && (kv_col_pos < p.total_kv);
            const float s     = valid ? (S_tg[sgid][r][lane] * 0.0625f) : -INFINITY;

            const float rmax   = simd_max(s);
            const float m_prev = M_tg[sgid][r];
            const float m_next = max(m_prev, rmax);

            float alpha = 0.0f;
            float p_val = 0.0f;
            if (m_next > -INFINITY) {
                alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_next);
                p_val = valid ? exp(s - m_next) : 0.0f;
            }
            const float p_sum = simd_sum(p_val);

            P_tg[sgid][r][lane] = half(p_val);
            if (lane == 0u) {
                M_tg[sgid][r] = m_next;
                L_tg[sgid][r] = fma(L_tg[sgid][r], alpha, p_sum);
                A_tg[sgid][r] = alpha;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
            if (oCT0.is_valid_element(i)) {
                auto idx = oCT0.get_multidimensional_index(i);
                float a  = A_tg[sgid][uint(idx[1])];
                oCT0[i] *= a; oCT1[i] *= a; oCT2[i] *= a; oCT3[i] *= a;
            }
        }

        auto sV_0 = tV.slice<64, 32>(0,   int(kv_phys));
        auto sV_1 = tV.slice<64, 32>(64,  int(kv_phys));
        auto sV_2 = tV.slice<64, 32>(128, int(kv_phys));
        auto sV_3 = tV.slice<64, 32>(192, int(kv_phys));
        op_pv.run(tP, sV_0, oCT0);
        op_pv.run(tP, sV_1, oCT1);
        op_pv.run(tP, sV_2, oCT2);
        op_pv.run(tP, sV_3, oCT3);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            auto idx    = oCT0.get_multidimensional_index(i);
            uint r      = uint(idx[1]);
            uint c      = uint(idx[0]);
            uint m_glob = m0 + r;
            if (m_glob < p.M_valid) {
                float inv_l  = (L_tg[sgid][r] > 0.0f) ? (1.0f / L_tg[sgid][r]) : 0.0f;
                ulong g_base = ulong(m_glob) * g_str + ulong(hq) * 512ul + 256ul;
                ulong o_base = ulong(m_glob) * o_str + ulong(hq) * 256ul;

                gated_attn_out[o_base + 0u   + c] = sqf_clamp_f16((oCT0[i] * inv_l) * sqf_sigmoid(float(q_gate_in[g_base + 0u   + c])));
                gated_attn_out[o_base + 64u  + c] = sqf_clamp_f16((oCT1[i] * inv_l) * sqf_sigmoid(float(q_gate_in[g_base + 64u  + c])));
                gated_attn_out[o_base + 128u + c] = sqf_clamp_f16((oCT2[i] * inv_l) * sqf_sigmoid(float(q_gate_in[g_base + 128u + c])));
                gated_attn_out[o_base + 192u + c] = sqf_clamp_f16((oCT3[i] * inv_l) * sqf_sigmoid(float(q_gate_in[g_base + 192u + c])));
            } else if (m_glob < p.M_padded) {
                ulong o_base = ulong(m_glob) * o_str + ulong(hq) * 256ul;
                gated_attn_out[o_base + 0u   + c] = half(0.0h);
                gated_attn_out[o_base + 64u  + c] = half(0.0h);
                gated_attn_out[o_base + 128u + c] = half(0.0h);
                gated_attn_out[o_base + 192u + c] = half(0.0h);
            }
        }
    }
#else
    (void)q_rope; (void)k_cache; (void)v_cache; (void)q_gate_in; (void)gated_attn_out; (void)p; (void)lane; (void)hkv; (void)kv_limit; (void)g_str; (void)block_loc;
#endif
}

// Metal 4.1 16-Query SIMD-Merge Block-Union NAX Sparse/Paged-Dense Prefill Attention (`head_dim=256`, `compress_ratio=4`)!
//   - Groups 16 consecutive queries per TG (`100%` NAX utilization, 10x FEWER KV reads via on-the-fly 16-way SIMD merge!)
//   - Works up to infinite context length with ZERO bitmap memory, exact per-query top-512 mask & causal tail mask!
//   - 2-SG Split-D (`sgid=0`: `d=0..127`, `sgid=1`: `d=128..255`), `~96` regs -> ZERO spills, `16B` `uint4` burst loads!
//   - Single-pass fused `* sigmoid(gate)` + zero-padding tail, eliminating `b_part_m/l/o` (384 MiB!) & `reduce`!
// Grid: `(num_q_tiles_in_group, n_q, 1)`, `64` threads/TG (`2` SGs)
kernel void sqf_qsa_sparse_prefill_nax(
    device const uint4*          q_rope         [[buffer(0)]], // [n_q, M_padded, 32] (each uint4 = 8 halfs)
    device const uint4*          k_cache        [[buffer(1)]], // [n_kv, max_seq_alloc, 32]
    device const uint4*          v_cache        [[buffer(2)]], // [n_kv, max_seq_alloc, 32]
    device const uint*           sel_blocks     [[buffer(3)]], // [M_valid, block_topk]
    device const half*           q_gate_in      [[buffer(4)]], // [M_valid, n_q * 512]
    device half*                 gated_attn_out [[buffer(5)]], // [M_padded, n_q * 256]
    constant SqfQsaDecodeParams& p              [[buffer(6)]],
    device const uint*           block_loc      [[buffer(7)]], // [max_blocks] (read iff p.paged)
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],             // 0..1
    uint  lane [[thread_index_in_simdgroup]]                   // 0..31
) {
    const uint m0 = p.m_base + tgid.x * 16u;
    const uint hq = tgid.y;
    if (m0 >= p.M_padded || hq >= p.n_q) return;

    const ulong o_str  = ulong(p.n_q) * 256ul;
    const ulong g_str  = ulong(p.n_q) * 512ul;
    const uint  m_end  = (p.m_end > 0u) ? min(p.m_end, p.M_valid) : p.M_valid;
    const uint  r_pair = lane >> 4u; // 0..1
    const uint  c8     = lane & 15u; // 0..15 (`16 * 8 = 128` halfs)
    const ulong sg_c8  = ulong(sgid) * 16ul;

    if (m0 >= m_end) {
        if (m0 >= p.M_valid) {
            const uint pad_rows = min(16u, p.M_padded - m0);
            for (uint r = r_pair; r < pad_rows; r += 2u) {
                device uint4* out8 = reinterpret_cast<device uint4*>(
                    gated_attn_out + ulong(m0 + r) * o_str + ulong(hq) * 256ul);
                out8[sg_c8 + c8] = uint4(0u);
            }
        }
        return;
    }

#if defined(__HAVE_TENSOR__)
    const uint q_cnt     = min(16u, m_end - m0);
    const uint gqa_ratio = p.n_q / p.n_kv;
    const uint hkv       = hq / gqa_ratio;

    threadgroup half  KV_tg[2][32][128]; // 16 KiB (sgid-private!)
    threadgroup float S_tg[2][16][32];   // 4 KiB
    threadgroup half  P_tg[16][32];      // 1 KiB
    threadgroup float M_tg[16];
    threadgroup float L_tg[16];
    threadgroup float A_tg[16];

    if (sgid == 0u && lane < 16u) {
        M_tg[lane] = -INFINITY;
        L_tg[lane] = 0.0f;
    }

    // 1. Coalesced 16B `uint4` load of `q_cnt <= 16` queries (`d = sgid*128 .. sgid*128+127`), pin in NAX `qCT`!
    const ulong q_head_off = ulong(hq) * ulong(p.M_padded);
    #pragma clang loop unroll(full)
    for (uint rp = 0u; rp < 8u; ++rp) {
        const uint r = rp * 2u + r_pair;
        reinterpret_cast<threadgroup uint4*>(&KV_tg[sgid][r][0])[c8] = (r < q_cnt)
            ? q_rope[(q_head_off + ulong(m0 + r)) * 32ul + sg_c8 + c8]
            : uint4(0u);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup half, extents<int, 128, 16>, tensor_inline> tQ_tmp(&KV_tg[sgid][0][0], extents<int, 128, 16>());
    tensor<threadgroup half, extents<int, 128, 32>, tensor_inline> tKV(&KV_tg[sgid][0][0], extents<int, 128, 32>());
    tensor<threadgroup half, extents<int, 32, 16>, tensor_inline>  tP(&P_tg[0][0], extents<int, 32, 16>());

    constexpr auto desc_qk = matmul2d_descriptor(
        16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;

    constexpr auto desc_pv = matmul2d_descriptor(
        16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    auto qCT = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(tQ_tmp);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    auto sV0   = tKV.slice<64, 32>(0, 0);
    auto sV1   = tKV.slice<64, 32>(64, 0);
    auto oCT_a = op_pv.get_destination_cooperative_tensor<sqf_strip_as<decltype(tP)>, sqf_strip_as<decltype(sV0)>, float>();
    auto oCT_b = op_pv.get_destination_cooperative_tensor<sqf_strip_as<decltype(tP)>, sqf_strip_as<decltype(sV1)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT_a.get_capacity(); ++i) {
        if (oCT_a.is_valid_element(i)) {
            oCT_a[i] = 0.0f;
            oCT_b[i] = 0.0f;
        }
    }

    // 2. Initialize 16-way ascending sorted block-union merge cursors (lanes 0..15 track queries m0..m0+q_cnt-1!)
    bool is_sparse_q = false;
    bool has_tail_q  = false;
    uint ptr_q       = 0u;
    uint end_ptr_q   = 0u;
    uint tail_blk_q  = 0u;
    uint cur_b       = 0xFFFFFFFFu;
    device const uint* m_sel_q = sel_blocks;

    if (lane < q_cnt) {
        const uint m_q   = m0 + lane;
        const uint q_pos = p.seq_pos_start + m_q;
        const uint nc    = (q_pos + 1u) >> 2u;
        is_sparse_q      = (nc > p.block_topk);
        has_tail_q       = (((q_pos + 1u) & 3u) != 0u);
        tail_blk_q       = nc;
        m_sel_q          = sel_blocks + ulong(m_q) * ulong(p.block_topk);
        end_ptr_q        = is_sparse_q ? p.block_topk : nc;
        if (ptr_q < end_ptr_q) {
            cur_b = is_sparse_q ? m_sel_q[0] : 0u;
        } else if (has_tail_q) {
            cur_b      = tail_blk_q;
            has_tail_q = false;
        }
    }

    device const uint4* k_head = k_cache + ulong(hkv) * ulong(p.max_seq_alloc) * 32ul;
    device const uint4* v_head = v_cache + ulong(hkv) * ulong(p.max_seq_alloc) * 32ul;
    const uint lane_bi = lane >> 2u;
    const uint lane_sub = lane & 3u;

    // 3. Main Block-Union Loop (`8` union blocks = `32` KV tokens per iter, `10x` less KV reads!)
    for (;;) {
        uint u_ppos[8];
        uint my_col_blk = 0u;
        uint my_col_msk = 0u;
        uint n_blks     = 0u;

        #pragma clang loop unroll(full)
        for (uint bi = 0u; bi < 8u; ++bi) {
            const uint min_b = simd_min(cur_b);
            if (min_b == 0xFFFFFFFFu) {
                u_ppos[bi] = 0u;
                continue;
            }
            const uint msk = simd_or((lane < 16u && cur_b == min_b) ? (1u << lane) : 0u);
            if (lane_bi == bi) {
                my_col_blk = min_b;
                my_col_msk = msk;
            }
            const uint phys_blk = (p.paged != 0u) ? (block_loc[min_b] & SQF_KV_LOC_IDX_MASK) : min_b;
            u_ppos[bi] = phys_blk << 2u;
            ++n_blks;
            if (lane < 16u && cur_b == min_b) {
                ++ptr_q;
                if (ptr_q < end_ptr_q) {
                    cur_b = is_sparse_q ? m_sel_q[ptr_q] : ptr_q;
                } else if (has_tail_q) {
                    cur_b      = tail_blk_q;
                    has_tail_q = false;
                } else {
                    cur_b = 0xFFFFFFFFu;
                }
            }
        }
        if (n_blks == 0u) break;

        // Burst-load `K` (`16B` `uint4`, `2` rows per instruction -> ONLY `16` loads for 32 tokens!)
        #pragma clang loop unroll(full)
        for (uint bi = 0u; bi < 8u; ++bi) {
            const uint r0 = bi * 4u;
            if (bi < n_blks) {
                const ulong pp = ulong(u_ppos[bi]);
                reinterpret_cast<threadgroup uint4*>(&KV_tg[sgid][r0 + r_pair][0])[c8] =
                    k_head[(pp + ulong(r_pair)) * 32ul + sg_c8 + c8];
                reinterpret_cast<threadgroup uint4*>(&KV_tg[sgid][r0 + 2u + r_pair][0])[c8] =
                    k_head[(pp + ulong(2u + r_pair)) * 32ul + sg_c8 + c8];
            } else {
                reinterpret_cast<threadgroup uint4*>(&KV_tg[sgid][r0 + r_pair][0])[c8]      = uint4(0u);
                reinterpret_cast<threadgroup uint4*>(&KV_tg[sgid][r0 + 2u + r_pair][0])[c8] = uint4(0u);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        auto sCT = op_qk.get_destination_cooperative_tensor<sqf_strip_as<decltype(qCT)>, sqf_strip_as<decltype(tKV)>, float>();
        op_qk.run(qCT, tKV, sCT);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < sCT.get_capacity(); ++i) {
            if (sCT.is_valid_element(i)) {
                auto idx = sCT.get_multidimensional_index(i);
                S_tg[sgid][uint(idx[1])][uint(idx[0])] = sCT[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #1: S_tg ready + prev P_tg/A_tg consumed!

        // Split online softmax across both SGs (`sgid 0`: rows 0..7, `sgid 1`: rows 8..15)
        const bool in_blks = (lane_bi < n_blks);
        const uint col_tok = (my_col_blk << 2u) | lane_sub;
        const uint r_start = sgid * 8u;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 8u; ++r) {
            const uint  row    = r_start + r;
            const uint  q_pos  = p.seq_pos_start + m0 + row;
            const bool  valid  = (row < q_cnt) && in_blks && (((my_col_msk >> row) & 1u) != 0u) && (col_tok <= q_pos);
            const float s      = valid ? ((S_tg[0][row][lane] + S_tg[1][row][lane]) * p.attn_scale) : -INFINITY;
            const float rmax   = simd_max(s);
            const float m_prev = M_tg[row];
            const float m_next = max(m_prev, rmax);
            float alpha = 0.0f;
            float p_val = 0.0f;
            if (m_next > -INFINITY) {
                alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_next);
                p_val = valid ? exp(s - m_next) : 0.0f;
            }
            const float p_sum = simd_sum(p_val);
            P_tg[row][lane]   = half(p_val);
            if (lane == 0u) {
                M_tg[row] = m_next;
                L_tg[row] = fma(L_tg[row], alpha, p_sum);
                A_tg[row] = alpha;
            }
        }

        // Burst-load `V` BEFORE Barrier #2 (KV_tg dead after op_qk, `u_ppos` already in regs!)
        #pragma clang loop unroll(full)
        for (uint bi = 0u; bi < 8u; ++bi) {
            const uint r0 = bi * 4u;
            if (bi < n_blks) {
                const ulong pp = ulong(u_ppos[bi]);
                reinterpret_cast<threadgroup uint4*>(&KV_tg[sgid][r0 + r_pair][0])[c8] =
                    v_head[(pp + ulong(r_pair)) * 32ul + sg_c8 + c8];
                reinterpret_cast<threadgroup uint4*>(&KV_tg[sgid][r0 + 2u + r_pair][0])[c8] =
                    v_head[(pp + ulong(2u + r_pair)) * 32ul + sg_c8 + c8];
            } else {
                reinterpret_cast<threadgroup uint4*>(&KV_tg[sgid][r0 + r_pair][0])[c8]      = uint4(0u);
                reinterpret_cast<threadgroup uint4*>(&KV_tg[sgid][r0 + 2u + r_pair][0])[c8] = uint4(0u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #2: P_tg, A_tg, L_tg, KV_tg(V) ready!

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT_a.get_capacity(); ++i) {
            if (oCT_a.is_valid_element(i)) {
                auto idx    = oCT_a.get_multidimensional_index(i);
                const float a = A_tg[uint(idx[1])];
                oCT_a[i] *= a;
                oCT_b[i] *= a;
            }
        }

        op_pv.run(tP, sV0, oCT_a);
        op_pv.run(tP, sV1, oCT_b);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Final normalize + fused `* sigmoid(gate)` + safe padding tail zeroing
    const uint d_sg = sgid * 128u;
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT_a.get_capacity(); ++i) {
        if (oCT_a.is_valid_element(i)) {
            auto idx       = oCT_a.get_multidimensional_index(i);
            const uint r   = uint(idx[1]);
            const uint c   = uint(idx[0]);
            const uint m_g = m0 + r;
            if (r < q_cnt) {
                const float inv_l  = (L_tg[r] > 0.0f) ? (1.0f / L_tg[r]) : 0.0f;
                const ulong g_base = ulong(m_g) * g_str + ulong(hq) * 512ul + 256ul + ulong(d_sg);
                const ulong o_base = ulong(m_g) * o_str + ulong(hq) * 256ul + ulong(d_sg);
                gated_attn_out[o_base + c] =
                    sqf_clamp_f16((oCT_a[i] * inv_l) * sqf_sigmoid(float(q_gate_in[g_base + c])));
                gated_attn_out[o_base + 64u + c] =
                    sqf_clamp_f16((oCT_b[i] * inv_l) * sqf_sigmoid(float(q_gate_in[g_base + 64u + c])));
            } else if (m_g >= p.M_valid && m_g < p.M_padded) {
                const ulong o_base = ulong(m_g) * o_str + ulong(hq) * 256ul + ulong(d_sg);
                gated_attn_out[o_base + c]       = half(0.0h);
                gated_attn_out[o_base + 64u + c] = half(0.0h);
            }
        }
    }
#else
    (void)q_rope; (void)k_cache; (void)v_cache; (void)sel_blocks; (void)q_gate_in; (void)gated_attn_out; (void)p; (void)block_loc; (void)o_str; (void)g_str;
#endif
}

/* ============================================================================
 * 9. 512-Expert Top-10 SwiGLU MoE + Gated Shared Expert Kernels
 * ============================================================================ */

struct SqfMoeRouterParams {
    uint M_valid;
    uint num_experts;    // 512
    uint top_k;          // 10 (<= 16)
    uint norm_topk_prob; // 1
};

// Grid: `(M_valid, 1, 1)`, `256` threads/TG (`8` SGs)
kernel void sqf_moe_topk_softmax(
    device const float*         logits_in  [[buffer(0)]], // [M_valid, num_experts]
    device uint*                topk_idx   [[buffer(1)]], // [M_valid, top_k]
    device float*               topk_w     [[buffer(2)]], // [M_valid, top_k]
    constant SqfMoeRouterParams& p         [[buffer(3)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],            // 0..255
    uint sgid [[simdgroup_index_in_threadgroup]],         // 0..7
    uint lane [[thread_index_in_simdgroup]]               // 0..31
) {
    if (m >= p.M_valid) return;

    threadgroup float tg_val[8];
    threadgroup uint  tg_idx[8];
    threadgroup float sel_vals[16];
    threadgroup uint  sel_ids[16];

    device const float* row = logits_in + ulong(m) * p.num_experts;
    const uint K = min(p.top_k, 16u);

    for (uint k = 0u; k < K; ++k) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;

        for (uint e = tid; e < p.num_experts; e += 256u) {
            bool taken = false;
            for (uint prev = 0u; prev < k; ++prev) {
                if (sel_ids[prev] == e) {
                    taken = true;
                    break;
                }
            }
            if (!taken) {
                float v = row[e];
                if (v > best_v || (v == best_v && e < best_i)) {
                    best_v = v;
                    best_i = e;
                }
            }
        }

        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) {
            tg_val[sgid] = best_v;
            tg_idx[sgid] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_val[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_idx[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(v, off);
                uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) {
                    v   = ov;
                    idx = oi;
                }
            }
            if (lane == 0u) {
                sel_vals[k] = v;
                sel_ids[k]  = (idx < p.num_experts) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.norm_topk_prob != 0u) {
        if (tid == 0u) {
            float m_max = sel_vals[0];
            float sum_e = 0.0f;
            float w_tmp[16];
            for (uint k = 0u; k < K; ++k) {
                float ew = exp(sel_vals[k] - m_max);
                w_tmp[k] = ew;
                sum_e   += ew;
            }
            float inv_s = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
            for (uint k = 0u; k < K; ++k) {
                topk_idx[ulong(m) * p.top_k + k] = sel_ids[k];
                topk_w[ulong(m) * p.top_k + k]   = w_tmp[k] * inv_s;
            }
        }
    } else {
        float m_max   = sel_vals[0];
        float loc_sum = 0.0f;
        for (uint e = tid; e < p.num_experts; e += 256u) {
            loc_sum += exp(row[e] - m_max);
        }
        float sg_s = simd_sum(loc_sum);
        if (lane == 0u) tg_val[sgid] = sg_s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float tot = simd_sum((lane < 8u) ? tg_val[lane] : 0.0f);
            if (lane == 0u) {
                float inv_s = (tot > 0.0f) ? (1.0f / tot) : 0.0f;
                for (uint k = 0u; k < K; ++k) {
                    topk_idx[ulong(m) * p.top_k + k] = sel_ids[k];
                    topk_w[ulong(m) * p.top_k + k]   = exp(sel_vals[k] - m_max) * inv_s;
                }
            }
        }
    }
}

struct SqfSwigluParams {
    uint M_valid;
    uint M_padded;
    uint d_ff;
    uint pad;
};

kernel void sqf_swiglu_f32_to_f16(
    device const float*      gate_f32 [[buffer(0)]], // [M_valid, d_ff]
    device const float*      up_f32   [[buffer(1)]], // [M_valid, d_ff]
    device half*             mid_f16  [[buffer(2)]], // [M_padded, d_ff]
    constant SqfSwigluParams& p       [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]       // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_padded) return;
    const uint idx4 = tgid.x * 256u + tid;
    const uint d4   = p.d_ff >> 2u;
    if (idx4 >= d4) return;

    const ulong off4 = ulong(m) * d4 + idx4;
    if (m >= p.M_valid) {
        reinterpret_cast<device half4*>(mid_f16)[off4] = half4(0.0h);
        return;
    }
    float4 gv   = reinterpret_cast<device const float4*>(gate_f32)[off4];
    float4 uv   = reinterpret_cast<device const float4*>(up_f32)[off4];
    float4 prod = (gv / (float4(1.0f) + exp(-gv))) * uv;
    reinterpret_cast<device half4*>(mid_f16)[off4] = sqf_clamp_f16x4(prod);
}

struct SqfSharedGateParams {
    uint M_valid;
    uint D;
};

// Computes `gate_scalar[m] = sigmoid(dot(shared_expert_gate_f32, x_in_f32[m]))`
// Grid: `(M_valid, 1, 1)`, `256` threads/TG
kernel void sqf_shared_gate_dot(
    device const float*         x_in_f32    [[buffer(0)]], // [M_valid, D]
    device const float*         gate_w_f32  [[buffer(1)]], // [D]
    device float*               gate_scalar [[buffer(2)]], // [M_valid]
    constant SqfSharedGateParams& p         [[buffer(3)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],             // 0..255
    uint sgid [[simdgroup_index_in_threadgroup]],          // 0..7
    uint lane [[thread_index_in_simdgroup]]                // 0..31
) {
    if (m >= p.M_valid) return;
    threadgroup float tg_sum[8];

    const uint d4_cnt = p.D >> 2u;
    device const float4* x4 = reinterpret_cast<device const float4*>(x_in_f32 + ulong(m) * p.D);
    device const float4* w4 = reinterpret_cast<device const float4*>(gate_w_f32);

    float acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 xv = x4[i];
        float4 wv = w4[i];
        acc = fma(xv.x, wv.x, fma(xv.y, wv.y, fma(xv.z, wv.z, fma(xv.w, wv.w, acc))));
    }
    float s = simd_sum(acc);
    if (lane == 0u) tg_sum[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 8u) ? tg_sum[lane] : 0.0f);
        if (lane == 0u) {
            gate_scalar[m] = sqf_sigmoid(tot);
        }
    }
}

struct SqfMoeTopkSharedGateParams {
    uint M_valid;
    uint num_experts; // <= 512
    uint top_k;       // <= 16
    uint D;           // 2560
};

// Zero-Barrier 2-SG (`64`-thread) Fused `MoE Top-K Softmax (E<=512, norm_topk_prob=1)` + `shared_gate_dot`!
// SG0 holds all 512 logits in 16 registers/lane (0 TG barriers, 10x less reads vs 20-barrier baseline!);
// SG1 runs `sigmoid(dot(x_in_f32, shared_gate_w))` concurrently (-48 dispatches + -960 barriers/step, 100% bit-exact!).
// Grid: `(M_valid, 1, 1)`, `64` threads/TG (`2` SGs)
kernel void sqf_moe_topk_and_shared_gate(
    device const float*                 logits_in   [[buffer(0)]], // [M_valid, num_experts]
    device uint*                        topk_idx    [[buffer(1)]], // [M_valid, top_k]
    device float*                       topk_w      [[buffer(2)]], // [M_valid, top_k]
    device const float*                 x_in_f32    [[buffer(3)]], // [M_valid, D]
    device const float*                 gate_w_f32  [[buffer(4)]], // [D]
    device float*                       gate_scalar [[buffer(5)]], // [M_valid]
    constant SqfMoeTopkSharedGateParams& p          [[buffer(6)]],
    uint m    [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],                  // 0 or 1
    uint lane [[thread_index_in_simdgroup]]                        // 0..31
) {
    if (m >= p.M_valid) return;

    if (sgid == 0u) {
        threadgroup float tg_sv[16];
        device const float* row = logits_in + ulong(m) * p.num_experts;
        const uint K = min(p.top_k, 16u);

        float vals[16];
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 16u; ++j) {
            const uint e = j * 32u + lane;
            vals[j] = (e < p.num_experts) ? row[e] : -INFINITY;
        }

        for (uint k = 0u; k < K; ++k) {
            float best_v = -INFINITY;
            uint  best_i = 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 16u; ++j) {
                const uint  e = j * 32u + lane;
                const float v = vals[j];
                if (v > best_v || (v == best_v && e < best_i)) {
                    best_v = v;
                    best_i = e;
                }
            }
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(best_v, off);
                const uint  oi = simd_shuffle_xor(best_i, off);
                if (ov > best_v || (ov == best_v && oi < best_i)) {
                    best_v = ov;
                    best_i = oi;
                }
            }
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 16u; ++j) {
                if (j * 32u + lane == best_i) vals[j] = -INFINITY;
            }
            if (lane == 0u) {
                topk_idx[ulong(m) * p.top_k + k] = (best_i < p.num_experts) ? best_i : 0u;
                tg_sv[k] = best_v;
            }
        }

        if (lane == 0u) {
            const float m_max = tg_sv[0];
            float sum_e = 0.0f;
            float w_tmp[16];
            for (uint k = 0u; k < K; ++k) {
                const float ew = exp(tg_sv[k] - m_max);
                w_tmp[k] = ew;
                sum_e   += ew;
            }
            const float inv_s = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
            for (uint k = 0u; k < K; ++k) {
                topk_w[ulong(m) * p.top_k + k] = w_tmp[k] * inv_s;
            }
        }
    } else {
        const uint d4_cnt = p.D >> 2u;
        device const float4* x4 = reinterpret_cast<device const float4*>(x_in_f32 + ulong(m) * p.D);
        device const float4* w4 = reinterpret_cast<device const float4*>(gate_w_f32);
        float acc = 0.0f;
        for (uint i = lane; i < d4_cnt; i += 32u) {
            const float4 xv = x4[i];
            const float4 wv = w4[i];
            acc = fma(xv.x, wv.x, fma(xv.y, wv.y, fma(xv.z, wv.z, fma(xv.w, wv.w, acc))));
        }
        const float tot = simd_sum(acc);
        if (lane == 0u) {
            gate_scalar[m] = sqf_sigmoid(tot);
        }
    }
}

struct SqfMoeGatherParams {
    uint M_valid;
    uint top_k;       // 10
    uint D;           // 2560
    uint moe_d_ff;    // 640
    uint num_experts; // 512
    uint pad0;
    uint pad1;
    uint pad2;
};

// Active-Expert Gather `gate_up_proj` + SwiGLU (`1` row/SG, `4` SGs/TG = `128` th/TG, single `x_in` pass -> ~540 GB/s!)
// Grid: `((moe_d_ff + 3) / 4, top_k, M_valid)`, `128` threads/TG
kernel void sqf_moe_gather_gate_up_silu(
    device const uint2*        gu_w     [[buffer(0)]], // [E * 2 * moe_d_ff, D / 16]
    device const half*         gu_s     [[buffer(1)]], // [E * 2 * moe_d_ff, D / 64]
    device const half*         gu_b     [[buffer(2)]], // [E * 2 * moe_d_ff, D / 64]
    device const half*         x_in     [[buffer(3)]], // [M_valid, D]
    device const uint*         topk_idx [[buffer(4)]], // [M_valid, top_k]
    device half*               mid_out  [[buffer(5)]], // [M_valid * top_k, moe_d_ff]
    constant SqfMoeGatherParams& p      [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],     // 0..3
    uint  lane [[thread_index_in_simdgroup]]           // 0..31
) {
    const uint row = tgid.x * 4u + sgid;
    const uint k   = tgid.y;
    const uint m   = tgid.z;
    if (row >= p.moe_d_ff || k >= p.top_k || m >= p.M_valid) return;

    const uint e           = topk_idx[ulong(m) * p.top_k + k];
    const uint u2_per_row  = p.D >> 4u;
    const uint grp_per_row = p.D >> 6u;
    const uint rg          = e * (2u * p.moe_d_ff) + row;
    const uint ru          = rg + p.moe_d_ff;

    device const half* xm = x_in + ulong(m) * p.D;
    float ga = 0.0f;
    float ua = 0.0f;

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 qg      = gu_w[ulong(rg) * u2_per_row + u2_idx];
        const uint2 qu      = gu_w[ulong(ru) * u2_per_row + u2_idx];
        const float gsc     = float(gu_s[ulong(rg) * grp_per_row + grp_idx]);
        const float gbi     = float(gu_b[ulong(rg) * grp_per_row + grp_idx]);
        const float usc     = float(gu_s[ulong(ru) * grp_per_row + grp_idx]);
        const float ubi     = float(gu_b[ulong(ru) * grp_per_row + grp_idx]);

        half4 g0, g1, g2, g3, u0, u1, u2, u3;
        sqf_unpack_u32_nibbles_h4(qg.x, g0, g1); sqf_unpack_u32_nibbles_h4(qg.y, g2, g3);
        sqf_unpack_u32_nibbles_h4(qu.x, u0, u1); sqf_unpack_u32_nibbles_h4(qu.y, u2, u3);

        device const half4* xp = reinterpret_cast<device const half4*>(xm + u2_idx * 16u);
        const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        const float xs = sqf_sum_h4x4(x0, x1, x2, x3);

        ga = fma(gsc, sqf_dot_h4x4(g0, g1, g2, g3, x0, x1, x2, x3), fma(gbi, xs, ga));
        ua = fma(usc, sqf_dot_h4x4(u0, u1, u2, u3, x0, x1, x2, x3), fma(ubi, xs, ua));
    }

    const float gv = simd_sum(ga);
    const float uv = simd_sum(ua);
    if (lane == 0u) {
        const ulong dst_off = (ulong(m) * p.top_k + k) * p.moe_d_ff + row;
        mid_out[dst_off]    = sqf_clamp_f16(sqf_silu(gv) * uv);
    }
}

// Zero-Waste `8`-Lane/Row Active-Expert Gather `down_proj` (`4` rows/SG = `16` rows/TG, `100%` SIMD occupancy on `moe_d_ff=640`!)
// Exact same 8-lane reduction order as `sqf_moe_grouped_down` -> 100.0000% bit-exact `K>0 == K=0`!
// Grid: `((D + 15) / 16, top_k, M_valid)`, `128` threads/TG (`4` SGs, `4` rows/SG)
kernel void sqf_moe_gather_down(
    device const uint2*        dn_w      [[buffer(0)]], // [E * D, moe_d_ff / 16]
    device const half*         dn_s      [[buffer(1)]], // [E * D, moe_d_ff / 64]
    device const half*         dn_b      [[buffer(2)]], // [E * D, moe_d_ff / 64]
    device const half*         mid_in    [[buffer(3)]], // [M_valid * top_k, moe_d_ff]
    device const uint*         topk_idx  [[buffer(4)]], // [M_valid, top_k]
    device float*              down_part [[buffer(5)]], // [M_valid * top_k, D]
    constant SqfMoeGatherParams& p       [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],      // 0..3
    uint  lane [[thread_index_in_simdgroup]]            // 0..31
) {
    const uint row   = (tgid.x * 4u + sgid) * 4u + (lane >> 3u);
    const uint k     = tgid.y;
    const uint m     = tgid.z;
    if (row >= p.D || k >= p.top_k || m >= p.M_valid) return;

    const uint  lane8       = lane & 7u;
    const ulong pair_idx    = ulong(m) * p.top_k + k;
    const uint  e           = topk_idx[pair_idx];
    const uint  r           = e * p.D + row;
    const uint  u2_per_row  = p.moe_d_ff >> 4u;
    const uint  grp_per_row = p.moe_d_ff >> 6u;

    device const uint2* rw  = dn_w + ulong(r) * u2_per_row;
    device const half*  rs  = dn_s + ulong(r) * grp_per_row;
    device const half*  rb  = dn_b + ulong(r) * grp_per_row;
    device const half*  xm  = mid_in + pair_idx * p.moe_d_ff;

    float acc = 0.0f;
    for (uint u2_idx = lane8; u2_idx < u2_per_row; u2_idx += 8u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 q0      = rw[u2_idx];
        const float sc      = float(rs[grp_idx]);
        const float bi      = float(rb[grp_idx]);

        half4 n0, n1, n2, n3;
        sqf_unpack_u32_nibbles_h4(q0.x, n0, n1);
        sqf_unpack_u32_nibbles_h4(q0.y, n2, n3);

        device const half4* xp = reinterpret_cast<device const half4*>(xm + u2_idx * 16u);
        const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        const float xs = sqf_sum_h4x4(x0, x1, x2, x3);

        acc = fma(sc, sqf_dot_h4x4(n0, n1, n2, n3, x0, x1, x2, x3), fma(bi, xs, acc));
    }

    acc += simd_shuffle_xor(acc, 4u);
    acc += simd_shuffle_xor(acc, 2u);
    acc += simd_shuffle_xor(acc, 1u);
    if (lane8 == 0u) {
        down_part[pair_idx * p.D + row] = acc;
    }
}

// Batched Expert-Grouped Token Binning (`1` TG of `512` threads in `< 5 us`!)
// Groups all `M_valid * top_k` assignments by expert `e` and builds compact worklist of `(expert, start, count <= 8)` tiles!
struct SqfMoeWorkTile {
    uint expert_id;
    uint pair_start;
    uint pair_count; // 1..8
    uint pad;
};

kernel void sqf_moe_bin_tokens(
    device const uint*         topk_idx     [[buffer(0)]], // [M_valid * top_k]
    device uint*               sorted_pairs [[buffer(1)]], // [M_valid * top_k] (`m * top_k + k`)
    device SqfMoeWorkTile*     work_tiles   [[buffer(2)]], // [<= M_valid * top_k]
    device uint*               num_tiles    [[buffer(3)]], // [1]
    constant SqfMoeGatherParams& p          [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],             // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],          // 0..15
    uint lane [[thread_index_in_simdgroup]]                // 0..31
) {
    threadgroup atomic_uint counts[512];
    threadgroup atomic_uint cursors[512];
    threadgroup uint        offs[512];
    threadgroup uint        tile_counts[512];
    threadgroup uint        tile_offs[512];
    threadgroup uint        tg_sg[16];

    if (tid < 512u) {
        atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pairs = p.M_valid * p.top_k;
    for (uint i = tid; i < total_pairs; i += 512u) {
        uint e = topk_idx[i];
        if (e < p.num_experts) {
            atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exclusive prefix sum of `counts[0..511]` AND `tile_counts[e] = (counts[e] + 7) / 8`
    const uint c_val = (tid < p.num_experts) ? atomic_load_explicit(&counts[tid], memory_order_relaxed) : 0u;
    const uint t_val = (c_val + 7u) >> 3u;

    const uint c_pre  = simd_prefix_exclusive_sum(c_val);
    const uint c_sum  = simd_sum(c_val);
    if (lane == 0u) tg_sg[sgid] = c_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        uint v   = (lane < 16u) ? tg_sg[lane] : 0u;
        uint pre = simd_prefix_exclusive_sum(v);
        if (lane < 16u) tg_sg[lane] = pre;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint exp_off = tg_sg[sgid] + c_pre;
    offs[tid]          = exp_off;
    atomic_store_explicit(&cursors[tid], exp_off, memory_order_relaxed);

    const uint t_pre = simd_prefix_exclusive_sum(t_val);
    const uint t_sum = simd_sum(t_val);
    if (lane == 0u) tg_sg[sgid] = t_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        uint v   = (lane < 16u) ? tg_sg[lane] : 0u;
        uint pre = simd_prefix_exclusive_sum(v);
        uint tot = simd_sum(v);
        if (lane < 16u) tg_sg[lane] = pre;
        if (lane == 0u) num_tiles[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tile_counts[tid] = t_val;
    tile_offs[tid]   = tg_sg[sgid] + t_pre;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter `pair_idx = m * top_k + k` into `sorted_pairs`
    for (uint i = tid; i < total_pairs; i += 512u) {
        uint e = topk_idx[i];
        if (e < p.num_experts) {
            uint dst = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
            sorted_pairs[dst] = i;
        }
    }

    // Emit compact `B <= 8` worklist tiles
    if (tid < p.num_experts && t_val > 0u) {
        const uint base_pair = offs[tid];
        const uint base_tile = tile_offs[tid];
        for (uint ti = 0u; ti < t_val; ++ti) {
            uint st  = ti * 8u;
            uint cnt = min(8u, c_val - st);
            work_tiles[base_tile + ti] = SqfMoeWorkTile{tid, base_pair + st, cnt, 0u};
        }
    }
}

// Expert-Grouped Register-Batched (`B <= 8`, ZERO SPILLS!) `gate_up_proj + SwiGLU`!
// Grid: `((moe_d_ff + 3) / 4, max_possible_tiles, 1)`, `128` threads/TG (`4` SGs)
template <uint B_FIX>
inline void sqf_moe_grouped_gu_fixed(
    device const uint2*          gu_w,
    device const half*           gu_s,
    device const half*           gu_b,
    device const half*           x_in,
    device const uint*           sorted_pairs,
    device half*                 mid_out,
    uint e,
    uint pair_start,
    uint row,
    uint D,
    uint moe_d_ff,
    uint top_k,
    uint lane
) {
    uint tok_m[B_FIX];
    uint pair_id[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        uint pid   = sorted_pairs[pair_start + b];
        pair_id[b] = pid;
        tok_m[b]   = pid / top_k;
    }

    const uint u2_per_row  = D >> 4u;
    const uint grp_per_row = D >> 6u;
    const uint rg          = e * (2u * moe_d_ff) + row;
    const uint ru          = rg + moe_d_ff;

    float g_acc[B_FIX];
    float u_acc[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        g_acc[b] = 0.0f;
        u_acc[b] = 0.0f;
    }

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 qg      = gu_w[ulong(rg) * u2_per_row + u2_idx];
        const uint2 qu      = gu_w[ulong(ru) * u2_per_row + u2_idx];
        const float gsc     = float(gu_s[ulong(rg) * grp_per_row + grp_idx]);
        const float gbi     = float(gu_b[ulong(rg) * grp_per_row + grp_idx]);
        const float usc     = float(gu_s[ulong(ru) * grp_per_row + grp_idx]);
        const float ubi     = float(gu_b[ulong(ru) * grp_per_row + grp_idx]);

        half4 g0, g1, g2, g3, u0, u1, u2, u3;
        sqf_unpack_u32_nibbles_h4(qg.x, g0, g1); sqf_unpack_u32_nibbles_h4(qg.y, g2, g3);
        sqf_unpack_u32_nibbles_h4(qu.x, u0, u1); sqf_unpack_u32_nibbles_h4(qu.y, u2, u3);

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) {
            device const half4* xp = reinterpret_cast<device const half4*>(
                x_in + ulong(tok_m[b]) * D + u2_idx * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float xs = sqf_sum_h4x4(x0, x1, x2, x3);

            g_acc[b] = fma(gsc, sqf_dot_h4x4(g0, g1, g2, g3, x0, x1, x2, x3), fma(gbi, xs, g_acc[b]));
            u_acc[b] = fma(usc, sqf_dot_h4x4(u0, u1, u2, u3, x0, x1, x2, x3), fma(ubi, xs, u_acc[b]));
        }
    }

    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        const float gv = simd_sum(g_acc[b]);
        const float uv = simd_sum(u_acc[b]);
        if (lane == 0u) {
            mid_out[ulong(pair_id[b]) * moe_d_ff + row] = sqf_clamp_f16(sqf_silu(gv) * uv);
        }
    }
}

kernel void sqf_moe_grouped_gate_up_silu(
    device const uint2*          gu_w         [[buffer(0)]],
    device const half*           gu_s         [[buffer(1)]],
    device const half*           gu_b         [[buffer(2)]],
    device const half*           x_in         [[buffer(3)]], // [M_padded, D]
    device const uint*           sorted_pairs [[buffer(4)]], // [M_valid * top_k]
    device const SqfMoeWorkTile* work_tiles   [[buffer(5)]],
    device const uint*           num_tiles    [[buffer(6)]],
    device half*                 mid_out      [[buffer(7)]], // [M_valid * top_k, moe_d_ff]
    constant SqfMoeGatherParams& p            [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],           // 0..3
    uint  lane [[thread_index_in_simdgroup]]                 // 0..31
) {
    const uint tile_idx = tgid.y;
    if (tile_idx >= num_tiles[0]) return;

    const uint row = tgid.x * 4u + sgid;
    if (row >= p.moe_d_ff) return;

    const SqfMoeWorkTile tile = work_tiles[tile_idx];
    #define SQF_GGU_CASE(BF) { \
        sqf_moe_grouped_gu_fixed<BF>(gu_w, gu_s, gu_b, x_in, sorted_pairs, mid_out, \
                                     tile.expert_id, tile.pair_start, row, p.D, p.moe_d_ff, p.top_k, lane); \
        break; \
    }
    switch (tile.pair_count) {
        case 1u: SQF_GGU_CASE(1u)
        case 2u: SQF_GGU_CASE(2u)
        case 3u: SQF_GGU_CASE(3u)
        case 4u: SQF_GGU_CASE(4u)
        case 5u: if (SQF_CONST_M >= 5u) { SQF_GGU_CASE(5u) } break;
        case 6u: if (SQF_CONST_M >= 6u) { SQF_GGU_CASE(6u) } break;
        case 7u: if (SQF_CONST_M >= 7u) { SQF_GGU_CASE(7u) } break;
        case 8u: if (SQF_CONST_M >= 8u) { SQF_GGU_CASE(8u) } break;
        default: break;
    }
    #undef SQF_GGU_CASE
}

// Zero-Waste `8`-Lane/Row Expert-Grouped `down_proj` (`4` rows/SG = `16` rows/TG, `100%` SIMD occupancy on `moe_d_ff=640`, HALF regs!)
// Specialized via `SQF_CONST_M` (`max_b` in `{4,5,6,8}`) + exact same 8-lane order as `sqf_moe_gather_down` (100% bit-exact!).
// Grid: `((D + 15) / 16, max_possible_tiles, 1)`, `128` threads/TG (`4` SGs)
template <uint B_FIX>
inline void sqf_moe_grouped_dn_8lane_fixed(
    device const uint2*          dn_w,
    device const half*           dn_s,
    device const half*           dn_b,
    device const half*           mid_in,
    device const uint*           sorted_pairs,
    device float*                down_part,
    uint e,
    uint pair_start,
    uint row,
    uint D,
    uint moe_d_ff,
    uint lane8
) {
    uint pair_id[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        pair_id[b] = sorted_pairs[pair_start + b];
    }

    const uint u2_per_row  = moe_d_ff >> 4u;
    const uint grp_per_row = moe_d_ff >> 6u;
    const uint r           = e * D + row;

    device const uint2* rw = dn_w + ulong(r) * u2_per_row;
    device const half*  rs = dn_s + ulong(r) * grp_per_row;
    device const half*  rb = dn_b + ulong(r) * grp_per_row;

    float acc[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) acc[b] = 0.0f;

    for (uint u2_idx = lane8; u2_idx < u2_per_row; u2_idx += 8u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 q0      = rw[u2_idx];
        const float sc      = float(rs[grp_idx]);
        const float bi      = float(rb[grp_idx]);

        half4 n0, n1, n2, n3;
        sqf_unpack_u32_nibbles_h4(q0.x, n0, n1);
        sqf_unpack_u32_nibbles_h4(q0.y, n2, n3);

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) {
            device const half4* xp = reinterpret_cast<device const half4*>(
                mid_in + ulong(pair_id[b]) * moe_d_ff + u2_idx * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float xs = sqf_sum_h4x4(x0, x1, x2, x3);

            acc[b] = fma(sc, sqf_dot_h4x4(n0, n1, n2, n3, x0, x1, x2, x3), fma(bi, xs, acc[b]));
        }
    }

    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        float s = acc[b];
        s += simd_shuffle_xor(s, 4u);
        s += simd_shuffle_xor(s, 2u);
        s += simd_shuffle_xor(s, 1u);
        if (lane8 == 0u) {
            down_part[ulong(pair_id[b]) * D + row] = s;
        }
    }
}

kernel void sqf_moe_grouped_down(
    device const uint2*          dn_w         [[buffer(0)]],
    device const half*           dn_s         [[buffer(1)]],
    device const half*           dn_b         [[buffer(2)]],
    device const half*           mid_in       [[buffer(3)]], // [M_valid * top_k, moe_d_ff]
    device const uint*           sorted_pairs [[buffer(4)]],
    device const SqfMoeWorkTile* work_tiles   [[buffer(5)]],
    device const uint*           num_tiles    [[buffer(6)]],
    device float*                down_part    [[buffer(7)]], // [M_valid * top_k, D]
    constant SqfMoeGatherParams& p            [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],           // 0..3
    uint  lane [[thread_index_in_simdgroup]]                 // 0..31
) {
    const uint tile_idx = tgid.y;
    if (tile_idx >= num_tiles[0]) return;

    const uint row = (tgid.x * 4u + sgid) * 4u + (lane >> 3u);
    if (row >= p.D) return;
    const uint lane8 = lane & 7u;

    const SqfMoeWorkTile tile = work_tiles[tile_idx];
    #define SQF_GDN_CASE(BF) { \
        sqf_moe_grouped_dn_8lane_fixed<BF>(dn_w, dn_s, dn_b, mid_in, sorted_pairs, down_part, \
                                           tile.expert_id, tile.pair_start, row, p.D, p.moe_d_ff, lane8); \
        break; \
    }
    switch (tile.pair_count) {
        case 1u: SQF_GDN_CASE(1u)
        case 2u: SQF_GDN_CASE(2u)
        case 3u: SQF_GDN_CASE(3u)
        case 4u: SQF_GDN_CASE(4u)
        case 5u: if (SQF_CONST_M >= 5u) { SQF_GDN_CASE(5u) } break;
        case 6u: if (SQF_CONST_M >= 6u) { SQF_GDN_CASE(6u) } break;
        case 7u: if (SQF_CONST_M >= 7u) { SQF_GDN_CASE(7u) } break;
        case 8u: if (SQF_CONST_M >= 8u) { SQF_GDN_CASE(8u) } break;
        default: break;
    }
    #undef SQF_GDN_CASE
}

// Prefill-Only Batched Expert Token Binning (`tile_cap = 16` for Metal 4 NAX TensorOps MoE!)
// Separate PSO from decode `sqf_moe_bin_tokens` so decode register footprint stays 100% untouched.
kernel void sqf_moe_prefill_bin_tokens(
    device const uint*           topk_idx     [[buffer(0)]], // [M_valid * top_k]
    device uint*                 sorted_pairs [[buffer(1)]], // [M_valid * top_k] (`m * top_k + k`)
    device SqfMoeWorkTile*       work_tiles   [[buffer(2)]], // [<= (M_valid * top_k)/16 + E]
    device uint*                 num_tiles    [[buffer(3)]], // [1]
    constant SqfMoeGatherParams& p            [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],               // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],            // 0..15
    uint lane [[thread_index_in_simdgroup]]                  // 0..31
) {
    threadgroup atomic_uint counts[512];
    threadgroup atomic_uint cursors[512];
    threadgroup uint        offs[512];
    threadgroup uint        tile_offs[512];
    threadgroup uint        tg_sg[16];

    if (tid < 512u) {
        atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pairs = p.M_valid * p.top_k;
    for (uint i = tid; i < total_pairs; i += 512u) {
        uint e = topk_idx[i];
        if (e < p.num_experts) {
            atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c_val = (tid < p.num_experts) ? atomic_load_explicit(&counts[tid], memory_order_relaxed) : 0u;
    const uint t_val = (c_val + 15u) >> 4u;

    const uint c_pre = simd_prefix_exclusive_sum(c_val);
    const uint c_sum = simd_sum(c_val);
    if (lane == 0u) tg_sg[sgid] = c_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        uint v   = (lane < 16u) ? tg_sg[lane] : 0u;
        uint pre = simd_prefix_exclusive_sum(v);
        if (lane < 16u) tg_sg[lane] = pre;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint exp_off = tg_sg[sgid] + c_pre;
    offs[tid]          = exp_off;
    atomic_store_explicit(&cursors[tid], exp_off, memory_order_relaxed);

    const uint t_pre = simd_prefix_exclusive_sum(t_val);
    const uint t_sum = simd_sum(t_val);
    if (lane == 0u) tg_sg[sgid] = t_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        uint v   = (lane < 16u) ? tg_sg[lane] : 0u;
        uint pre = simd_prefix_exclusive_sum(v);
        uint tot = simd_sum(v);
        if (lane < 16u) tg_sg[lane] = pre;
        if (lane == 0u) num_tiles[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tile_offs[tid] = tg_sg[sgid] + t_pre;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < total_pairs; i += 512u) {
        uint e = topk_idx[i];
        if (e < p.num_experts) {
            uint dst = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
            sorted_pairs[dst] = i;
        }
    }

    if (tid < p.num_experts && t_val > 0u) {
        const uint base_pair = offs[tid];
        const uint base_tile = tile_offs[tid];
        for (uint ti = 0u; ti < t_val; ++ti) {
            uint st  = ti * 16u;
            uint cnt = min(16u, c_val - st);
            work_tiles[base_tile + ti] = SqfMoeWorkTile{tid, base_pair + st, cnt, 0u};
        }
    }
}

// Prefill-Only Metal 4 NAX (`matmul2d<16, 64, 64>`) Expert-Grouped Fused `gate_up_proj + SwiGLU`!
// Reads each expert's weights 1x per 16 tokens, gathers `X_tile[16][64]` into 2 KiB TG memory, runs on NAX Tensor Cores!
// Grid: `(moe_d_ff / 64, max_tiles, 1)`, `128` threads/TG (`4` SGs, `10 KiB` TG mem, `16` floats/thread -> ZERO spill!)
kernel void sqf_moe_prefill_gu_nax(
    device const uint*           gu_w         [[buffer(0)]], // [E * 2 * moe_d_ff, D / 8]
    device const half*           gu_s         [[buffer(1)]], // [E * 2 * moe_d_ff, D / 64]
    device const half*           gu_b         [[buffer(2)]], // [E * 2 * moe_d_ff, D / 64]
    device const half*           x_in         [[buffer(3)]], // [M_padded, D]
    device const uint*           sorted_pairs [[buffer(4)]], // [M_valid * top_k]
    device const SqfMoeWorkTile* work_tiles   [[buffer(5)]],
    device const uint*           num_tiles    [[buffer(6)]],
    device half*                 mid_out      [[buffer(7)]], // [M_valid * top_k, moe_d_ff]
    constant SqfMoeGatherParams& p            [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]               // 0..127
) {
    const uint tile_idx = tgid.y;
    if (tile_idx >= num_tiles[0]) return;
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.moe_d_ff) return;

#if defined(__HAVE_TENSOR__)
    const SqfMoeWorkTile tile = work_tiles[tile_idx];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    threadgroup uint tg_pid[16];
    threadgroup uint tg_tok[16];
    threadgroup half X_tile[16][64];
    threadgroup half W_tile[64][64];

    if (tid < cnt) {
        const uint pid = sorted_pairs[tile.pair_start + tid];
        tg_pid[tid] = pid;
        tg_tok[tid] = pid / p.top_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup half, extents<int, 64, 16>, tensor_inline> tX(&X_tile[0][0], extents<int, 64, 16>());
    tensor<threadgroup half, extents<int, 64, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 64, 64>());

    constexpr auto desc = matmul2d_descriptor(
        16, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto cGate = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(tX)>, sqf_strip_as<decltype(tW)>, float>();
    auto cUp   = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(tX)>, sqf_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cGate.get_capacity(); ++i) {
        if (cGate.is_valid_element(i)) {
            cGate[i] = 0.0f;
            cUp[i]   = 0.0f;
        }
    }

    const uint u32_per_row = p.D >> 3u;
    const uint grp_per_row = p.D >> 6u;
    const uint r_grp       = tid >> 3u; // 0..15
    const uint sub         = tid & 7u;  // 0..7
    const bool valid_x     = (r_grp < cnt);
    const uint tm_x        = valid_x ? tg_tok[r_grp] : 0u;
    const uint rg_base     = tile.expert_id * (2u * p.moe_d_ff) + n0;
    const uint ru_base     = rg_base + p.moe_d_ff;

    for (uint grp = 0u; grp < grp_per_row; ++grp) {
        // 1. Coalesced gather of `X_tile[16][64]` (0 device loads for padded rows >= cnt)
        threadgroup half4* xdst = reinterpret_cast<threadgroup half4*>(&X_tile[r_grp][sub * 8u]);
        if (valid_x) {
            device const half4* xsrc = reinterpret_cast<device const half4*>(
                x_in + ulong(tm_x) * p.D + grp * 64u + sub * 8u);
            xdst[0] = xsrc[0];
            xdst[1] = xsrc[1];
        } else {
            xdst[0] = half4(0.0h);
            xdst[1] = half4(0.0h);
        }

        // 2. Unpack 64x64 `gate` Q4 tile -> NAX `cGate += tX * tW^T`
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 4u; ++q) {
            const uint  r_loc  = q * 16u + r_grp;
            const uint  r_glob = rg_base + r_loc;
            const uint  u32_v  = gu_w[ulong(r_glob) * u32_per_row + grp * 8u + sub];
            const float sc     = float(gu_s[ulong(r_glob) * grp_per_row + grp]);
            const float bi     = float(gu_b[ulong(r_glob) * grp_per_row + grp]);
            half4 n_lo, n_hi;
            sqf_unpack_u32_nibbles_h4(u32_v, n_lo, n_hi);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 8u]);
            dst[0] = sqf_clamp_f16x4(fma(float4(n_lo), float4(sc), float4(bi)));
            dst[1] = sqf_clamp_f16x4(fma(float4(n_hi), float4(sc), float4(bi)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tX, tW, cGate);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Unpack 64x64 `up` Q4 tile -> NAX `cUp += tX * tW^T` (reusing `tX` in TG memory!)
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 4u; ++q) {
            const uint  r_loc  = q * 16u + r_grp;
            const uint  r_glob = ru_base + r_loc;
            const uint  u32_v  = gu_w[ulong(r_glob) * u32_per_row + grp * 8u + sub];
            const float sc     = float(gu_s[ulong(r_glob) * grp_per_row + grp]);
            const float bi     = float(gu_b[ulong(r_glob) * grp_per_row + grp]);
            half4 n_lo, n_hi;
            sqf_unpack_u32_nibbles_h4(u32_v, n_lo, n_hi);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 8u]);
            dst[0] = sqf_clamp_f16x4(fma(float4(n_lo), float4(sc), float4(bi)));
            dst[1] = sqf_clamp_f16x4(fma(float4(n_hi), float4(sc), float4(bi)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tX, tW, cUp);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cGate.get_capacity(); ++i) {
        if (cGate.is_valid_element(i)) {
            auto idx = cGate.get_multidimensional_index(i);
            uint r   = uint(idx[1]);
            uint c   = uint(idx[0]);
            if (r < cnt) {
                mid_out[ulong(tg_pid[r]) * p.moe_d_ff + (n0 + c)] =
                    sqf_clamp_f16(sqf_silu(cGate[i]) * cUp[i]);
            }
        }
    }
#else
    (void)gu_w; (void)gu_s; (void)gu_b; (void)x_in; (void)sorted_pairs; (void)work_tiles; (void)mid_out; (void)tid;
#endif
}

// Prefill-Only Metal 4 NAX (`matmul2d<16, 64, 64>`) Expert-Grouped `down_proj`!
// Replaces 152 GB/s scalar down_proj (`75%` idle wave on cols=640) with 500+ GB/s NAX Matrix Cores!
// Grid: `(D / 64, max_tiles, 1)`, `128` threads/TG (`4` SGs)
kernel void sqf_moe_prefill_dn_nax(
    device const uint*           dn_w         [[buffer(0)]], // [E * D, moe_d_ff / 8]
    device const half*           dn_s         [[buffer(1)]], // [E * D, moe_d_ff / 64]
    device const half*           dn_b         [[buffer(2)]], // [E * D, moe_d_ff / 64]
    device const half*           mid_in       [[buffer(3)]], // [M_valid * top_k, moe_d_ff]
    device const uint*           sorted_pairs [[buffer(4)]], // [M_valid * top_k]
    device const SqfMoeWorkTile* work_tiles   [[buffer(5)]],
    device const uint*           num_tiles    [[buffer(6)]],
    device float*                down_part    [[buffer(7)]], // [M_valid * top_k, D]
    constant SqfMoeGatherParams& p            [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]               // 0..127
) {
    const uint tile_idx = tgid.y;
    if (tile_idx >= num_tiles[0]) return;
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.D) return;

#if defined(__HAVE_TENSOR__)
    const SqfMoeWorkTile tile = work_tiles[tile_idx];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    threadgroup uint tg_pid[16];
    threadgroup half X_tile[16][64];
    threadgroup half W_tile[64][64];

    if (tid < cnt) {
        tg_pid[tid] = sorted_pairs[tile.pair_start + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup half, extents<int, 64, 16>, tensor_inline> tX(&X_tile[0][0], extents<int, 64, 16>());
    tensor<threadgroup half, extents<int, 64, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 64, 64>());

    constexpr auto desc = matmul2d_descriptor(
        16, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto cT = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(tX)>, sqf_strip_as<decltype(tW)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint u32_per_row = p.moe_d_ff >> 3u;
    const uint grp_per_row = p.moe_d_ff >> 6u;
    const uint r_grp       = tid >> 3u; // 0..15
    const uint sub         = tid & 7u;  // 0..7
    const bool valid_x     = (r_grp < cnt);
    const uint pid_x       = valid_x ? tg_pid[r_grp] : 0u;
    const uint r_base      = tile.expert_id * p.D + n0;

    for (uint grp = 0u; grp < grp_per_row; ++grp) {
        threadgroup half4* xdst = reinterpret_cast<threadgroup half4*>(&X_tile[r_grp][sub * 8u]);
        if (valid_x) {
            device const half4* xsrc = reinterpret_cast<device const half4*>(
                mid_in + ulong(pid_x) * p.moe_d_ff + grp * 64u + sub * 8u);
            xdst[0] = xsrc[0];
            xdst[1] = xsrc[1];
        } else {
            xdst[0] = half4(0.0h);
            xdst[1] = half4(0.0h);
        }

        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 4u; ++q) {
            const uint  r_loc  = q * 16u + r_grp;
            const uint  r_glob = r_base + r_loc;
            const uint  u32_v  = dn_w[ulong(r_glob) * u32_per_row + grp * 8u + sub];
            const float sc     = float(dn_s[ulong(r_glob) * grp_per_row + grp]);
            const float bi     = float(dn_b[ulong(r_glob) * grp_per_row + grp]);
            half4 n_lo, n_hi;
            sqf_unpack_u32_nibbles_h4(u32_v, n_lo, n_hi);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 8u]);
            dst[0] = sqf_clamp_f16x4(fma(float4(n_lo), float4(sc), float4(bi)));
            dst[1] = sqf_clamp_f16x4(fma(float4(n_hi), float4(sc), float4(bi)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            uint r   = uint(idx[1]);
            uint c   = uint(idx[0]);
            if (r < cnt) {
                down_part[ulong(tg_pid[r]) * p.D + (n0 + c)] = cT[i];
            }
        }
    }
#else
    (void)dn_w; (void)dn_s; (void)dn_b; (void)mid_in; (void)sorted_pairs; (void)work_tiles; (void)down_part; (void)tid;
#endif
}

// ============================================================================
// Metal 4.1 Tri-Bin (`BM = 16 / 32 / 64`, `BK = 128`) Direct-tX NAX MoE Prefill
//   - Cuts MoE weight DRAM traffic 2x-4x at `chunk = 512/1024` (1 tile per expert up to 64 tok!)
//   - Eliminates `BK = 64` 2.17x NAX cliff (`BK = 128`, `uint2` 8B + dual-rate FP16 unpack!)
//   - Eliminates `X_tile` TG memory & per-N-tile scattered gathers (`x_sorted` aliases `b_moe_down_part`, 0 extra VRAM!)
//   - Zero wasted TGs via `dispatchThreadgroupsWithIndirectBuffer` (6 x 16B `SqfIndirectGridArgs`!)
// ============================================================================

struct SqfIndirectGridArgs {
    uint threadgroupsPerGridX;
    uint threadgroupsPerGridY;
    uint threadgroupsPerGridZ;
    uint pad;
};

// 512-thread Single-TG Quad-Bin Expert Worklist Builder (`BM = 16 / 32 / 64 / 128`)
// Writes `work_tiles` at offsets: `0` (`BM=16`), `512` (`BM=32`), `1024` (`BM=64`), `1536` (`BM=128`)
// Writes 8 indirect dispatch descriptors into `indirect_args[0..7]` (GU 16/32/64/128, DN 16/32/64/128).
// Grid: `(1, 1, 1)`, `512` threads/TG (`16` SGs)
kernel void sqf_moe_bin_tokens_tri(
    device const uint*           topk_idx      [[buffer(0)]], // [M_valid * top_k]
    device uint*                 sorted_pairs  [[buffer(1)]], // [M_valid * top_k]
    device SqfMoeWorkTile*       work_tiles    [[buffer(2)]], // [>= 2048]
    device SqfIndirectGridArgs*  indirect_args [[buffer(3)]], // [8]
    constant SqfMoeGatherParams& p             [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],                // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],             // 0..15
    uint lane [[thread_index_in_simdgroup]]                   // 0..31
) {
    threadgroup atomic_uint counts[512];
    threadgroup atomic_uint cursors[512];
    threadgroup uint        offs[512];
    threadgroup uint        off16[512];
    threadgroup uint        off32[512];
    threadgroup uint        off64[512];
    threadgroup uint        off128[512];
    threadgroup uint        tg_sg[16];

    if (tid < 512u) {
        atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pairs = p.M_valid * p.top_k;
    for (uint i = tid; i < total_pairs; i += 512u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) {
            atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c_val = (tid < p.num_experts) ? atomic_load_explicit(&counts[tid], memory_order_relaxed) : 0u;
    const uint rem   = c_val & 127u;
    const uint t128  = (c_val >> 7u) + ((rem > 64u) ? 1u : 0u);
    const uint t64   = (rem > 32u && rem <= 64u) ? 1u : 0u;
    const uint t32   = (rem > 16u && rem <= 32u) ? 1u : 0u;
    const uint t16   = (rem > 0u  && rem <= 16u) ? 1u : 0u;

    // 1. Prefix sum `counts` -> `offs` / `cursors`
    {
        const uint pre = simd_prefix_exclusive_sum(c_val);
        const uint sum = simd_sum(c_val);
        if (lane == 0u) tg_sg[sgid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const uint v = (lane < 16u) ? tg_sg[lane] : 0u;
            const uint sp = simd_prefix_exclusive_sum(v);
            if (lane < 16u) tg_sg[lane] = sp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint exp_off = tg_sg[sgid] + pre;
        offs[tid]          = exp_off;
        atomic_store_explicit(&cursors[tid], exp_off, memory_order_relaxed);
    }

    // 2. Prefix sum `t16` -> `off16` + indirect slots 0 (GU16) & 4 (DN16)
    {
        const uint pre = simd_prefix_exclusive_sum(t16);
        const uint sum = simd_sum(t16);
        if (lane == 0u) tg_sg[sgid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const uint v   = (lane < 16u) ? tg_sg[lane] : 0u;
            const uint sp  = simd_prefix_exclusive_sum(v);
            const uint tot = simd_sum(v);
            if (lane < 16u) tg_sg[lane] = sp;
            if (lane == 0u) {
                indirect_args[0] = SqfIndirectGridArgs{p.moe_d_ff >> 6u, tot, 1u, 0u};
                indirect_args[4] = SqfIndirectGridArgs{p.D >> 6u,        tot, 1u, 0u};
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        off16[tid] = tg_sg[sgid] + pre;
    }

    // 3. Prefix sum `t32` -> `off32` + indirect slots 1 (GU32) & 5 (DN32)
    {
        const uint pre = simd_prefix_exclusive_sum(t32);
        const uint sum = simd_sum(t32);
        if (lane == 0u) tg_sg[sgid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const uint v   = (lane < 16u) ? tg_sg[lane] : 0u;
            const uint sp  = simd_prefix_exclusive_sum(v);
            const uint tot = simd_sum(v);
            if (lane < 16u) tg_sg[lane] = sp;
            if (lane == 0u) {
                indirect_args[1] = SqfIndirectGridArgs{p.moe_d_ff >> 6u, tot, 1u, 0u};
                indirect_args[5] = SqfIndirectGridArgs{p.D >> 6u,        tot, 1u, 0u};
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        off32[tid] = tg_sg[sgid] + pre;
    }

    // 4. Prefix sum `t64` -> `off64` + indirect slots 2 (GU64) & 6 (DN64)
    {
        const uint pre = simd_prefix_exclusive_sum(t64);
        const uint sum = simd_sum(t64);
        if (lane == 0u) tg_sg[sgid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const uint v   = (lane < 16u) ? tg_sg[lane] : 0u;
            const uint sp  = simd_prefix_exclusive_sum(v);
            const uint tot = simd_sum(v);
            if (lane < 16u) tg_sg[lane] = sp;
            if (lane == 0u) {
                indirect_args[2] = SqfIndirectGridArgs{p.moe_d_ff >> 6u, tot, 1u, 0u};
                indirect_args[6] = SqfIndirectGridArgs{p.D >> 6u,        tot, 1u, 0u};
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        off64[tid] = tg_sg[sgid] + pre;
    }

    // 5. Prefix sum `t128` -> `off128` + indirect slots 3 (GU128) & 7 (DN128)
    {
        const uint pre = simd_prefix_exclusive_sum(t128);
        const uint sum = simd_sum(t128);
        if (lane == 0u) tg_sg[sgid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const uint v   = (lane < 16u) ? tg_sg[lane] : 0u;
            const uint sp  = simd_prefix_exclusive_sum(v);
            const uint tot = simd_sum(v);
            if (lane < 16u) tg_sg[lane] = sp;
            if (lane == 0u) {
                indirect_args[3] = SqfIndirectGridArgs{p.moe_d_ff >> 6u, tot, 1u, 0u};
                indirect_args[7] = SqfIndirectGridArgs{p.D >> 6u,        tot, 1u, 0u};
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        off128[tid] = tg_sg[sgid] + pre;
    }

    // 6. Scatter `pair_idx = m * top_k + k` into `sorted_pairs`
    for (uint i = tid; i < total_pairs; i += 512u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) {
            const uint dst = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
            sorted_pairs[dst] = i;
        }
    }

    // 7. Emit Quad-Bin work tiles (`BM=128` @ 1536, `BM=64` @ 1024, `BM=32` @ 512, `BM=16` @ 0)
    if (tid < p.num_experts && c_val > 0u) {
        const uint base_pair = offs[tid];
        uint st = 0u;
        const uint b128 = off128[tid];
        for (uint ti = 0u; ti < t128; ++ti) {
            const uint cnt = min(128u, c_val - st);
            work_tiles[1536u + b128 + ti] = SqfMoeWorkTile{tid, base_pair + st, cnt, 0u};
            st += cnt;
        }
        if (t64 != 0u) {
            const uint cnt = c_val - st;
            work_tiles[1024u + off64[tid]] = SqfMoeWorkTile{tid, base_pair + st, cnt, 0u};
            st += cnt;
        }
        if (t32 != 0u) {
            const uint cnt = c_val - st;
            work_tiles[512u + off32[tid]] = SqfMoeWorkTile{tid, base_pair + st, cnt, 0u};
            st += cnt;
        }
        if (t16 != 0u) {
            const uint cnt = c_val - st;
            work_tiles[off16[tid]] = SqfMoeWorkTile{tid, base_pair + st, cnt, 0u};
        }
    }
}

// One-Shot Expert-Sorted `x_in` Gather (`16B` coalesced `uint4`) + `+128` Tail Zeroing for `x_sorted` & `mid_sorted`!
// Grid: `(total_pairs + 128, 1, 1)`, `256` threads/TG
kernel void sqf_moe_gather_x_f16(
    device const uint4*          x_in         [[buffer(0)]], // [M_padded, D / 8]
    device const uint*           sorted_pairs [[buffer(1)]], // [total_pairs]
    device uint4*                x_sorted     [[buffer(2)]], // [total_pairs + 128, D / 8] (aliases `b_moe_down_part`!)
    device uint4*                mid_sorted   [[buffer(3)]], // [total_pairs + 128, moe_d_ff / 8]
    constant SqfMoeGatherParams& p            [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]                 // 0..255
) {
    const uint total_pairs = p.M_valid * p.top_k;
    if (row >= total_pairs + 128u) return;

    const uint d8 = p.D >> 3u;
    if (row < total_pairs) {
        const uint tok = sorted_pairs[row] / p.top_k;
        device const uint4* src = x_in + ulong(tok) * d8;
        device uint4*       dst = x_sorted + ulong(row) * d8;
        for (uint i = tid; i < d8; i += 256u) {
            dst[i] = src[i];
        }
    } else {
        device uint4* xdst = x_sorted + ulong(row) * d8;
        for (uint i = tid; i < d8; i += 256u) {
            xdst[i] = uint4(0u);
        }
        const uint m8 = p.moe_d_ff >> 3u;
        device uint4* mdst = mid_sorted + ulong(row) * m8;
        for (uint i = tid; i < m8; i += 256u) {
            mdst[i] = uint4(0u);
        }
    }
}

template <uint BM, uint SG, uint TILE_OFF>
inline void sqf_moe_prefill_gu_nax_impl(
    device const uint2*          gu_w,
    device const half*           gu_s,
    device const half*           gu_b,
    device const half*           x_sorted,
    device const SqfMoeWorkTile* work_tiles,
    device half*                 mid_sorted,
    constant SqfMoeGatherParams& p,
    threadgroup half             (&W_tile)[64][128],
    uint2 tgid,
    uint  tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.moe_d_ff) return;

#if defined(__HAVE_TENSOR__)
    const SqfMoeWorkTile tile = work_tiles[TILE_OFF + tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    const uint total_pairs = p.M_valid * p.top_k;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(
        const_cast<device half*>(x_sorted), dextents<int, 2>(int(p.D), int(total_pairs + 128u)));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0   = tX.slice<128, BM>(0, int(tile.pair_start));
    auto cGate = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX0)>, sqf_strip_as<decltype(tW)>, float>();
    auto cUp   = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX0)>, sqf_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cGate.get_capacity(); ++i) {
        if (cGate.is_valid_element(i)) {
            cGate[i] = 0.0f;
            cUp[i]   = 0.0f;
        }
    }

    constexpr uint N_TH          = SG * 32u;
    constexpr uint ROWS_PER_STEP = N_TH >> 3u; // 16 for SG=4, 32 for SG=8 (4 rows/SG = 64B cache-line aligned!)
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;

    const uint u2_per_row  = p.D >> 4u;
    const uint grp_per_row = p.D >> 6u;
    const uint r_grp       = tid >> 3u;
    const uint sub         = tid & 7u;
    const uint sub_g       = sub >> 2u;
    const uint rg_base     = tile.expert_id * (2u * p.moe_d_ff) + n0;
    const uint ru_base     = rg_base + p.moe_d_ff;

    for (uint k0 = 0u; k0 < p.D; k0 += 128u) {
        const uint k_u2 = k0 >> 4u;
        const uint grp  = (k0 >> 6u) + sub_g;
        auto sX = tX.slice<128, BM>(int(k0), int(tile.pair_start));

        // 1. Unpack `gate` [64 x 128] -> NAX `cGate += sX * tW^T`
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const uint  r_glob = rg_base + r_loc;
            const uint2 qv     = gu_w[ulong(r_glob) * u2_per_row + k_u2 + sub];
            const half4 vsc    = half4(gu_s[ulong(r_glob) * grp_per_row + grp]);
            const half4 vbi    = half4(gu_b[ulong(r_glob) * grp_per_row + grp]);
            half4 n0h, n1h, n2h, n3h;
            sqf_unpack_u32_nibbles_h4(qv.x, n0h, n1h);
            sqf_unpack_u32_nibbles_h4(qv.y, n2h, n3h);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            dst[0] = fma(n0h, vsc, vbi);
            dst[1] = fma(n1h, vsc, vbi);
            dst[2] = fma(n2h, vsc, vbi);
            dst[3] = fma(n3h, vsc, vbi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(sX, tW, cGate);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Unpack `up` [64 x 128] -> NAX `cUp += sX * tW^T` (reuses L1/L2-hot `sX`!)
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const uint  r_glob = ru_base + r_loc;
            const uint2 qv     = gu_w[ulong(r_glob) * u2_per_row + k_u2 + sub];
            const half4 vsc    = half4(gu_s[ulong(r_glob) * grp_per_row + grp]);
            const half4 vbi    = half4(gu_b[ulong(r_glob) * grp_per_row + grp]);
            half4 n0h, n1h, n2h, n3h;
            sqf_unpack_u32_nibbles_h4(qv.x, n0h, n1h);
            sqf_unpack_u32_nibbles_h4(qv.y, n2h, n3h);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            dst[0] = fma(n0h, vsc, vbi);
            dst[1] = fma(n1h, vsc, vbi);
            dst[2] = fma(n2h, vsc, vbi);
            dst[3] = fma(n3h, vsc, vbi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(sX, tW, cUp);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cGate.get_capacity(); ++i) {
        if (cGate.is_valid_element(i)) {
            auto idx     = cGate.get_multidimensional_index(i);
            const uint r = uint(idx[1]);
            const uint c = uint(idx[0]);
            if (r < cnt) {
                mid_sorted[ulong(tile.pair_start + r) * ulong(p.moe_d_ff) + ulong(n0 + c)] =
                    sqf_clamp_f16(sqf_silu(cGate[i]) * cUp[i]);
            }
        }
    }
#else
    (void)gu_w; (void)gu_s; (void)gu_b; (void)x_sorted; (void)work_tiles; (void)mid_sorted; (void)p; (void)W_tile; (void)tgid; (void)tid;
#endif
}

kernel void sqf_moe_prefill_gu_nax_m16(
    device const uint2*          gu_w       [[buffer(0)]],
    device const half*           gu_s       [[buffer(1)]],
    device const half*           gu_b       [[buffer(2)]],
    device const half*           x_sorted   [[buffer(3)]],
    device const SqfMoeWorkTile* work_tiles [[buffer(4)]],
    device half*                 mid_sorted [[buffer(5)]],
    constant SqfMoeGatherParams& p          [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][128];
    sqf_moe_prefill_gu_nax_impl<16u, 4u, 0u>(gu_w, gu_s, gu_b, x_sorted, work_tiles, mid_sorted, p, W_tile, tgid, tid);
}

kernel void sqf_moe_prefill_gu_nax_m32(
    device const uint2*          gu_w       [[buffer(0)]],
    device const half*           gu_s       [[buffer(1)]],
    device const half*           gu_b       [[buffer(2)]],
    device const half*           x_sorted   [[buffer(3)]],
    device const SqfMoeWorkTile* work_tiles [[buffer(4)]],
    device half*                 mid_sorted [[buffer(5)]],
    constant SqfMoeGatherParams& p          [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][128];
    sqf_moe_prefill_gu_nax_impl<32u, 4u, 512u>(gu_w, gu_s, gu_b, x_sorted, work_tiles, mid_sorted, p, W_tile, tgid, tid);
}

kernel void sqf_moe_prefill_gu_nax_m64(
    device const uint2*          gu_w       [[buffer(0)]],
    device const half*           gu_s       [[buffer(1)]],
    device const half*           gu_b       [[buffer(2)]],
    device const half*           x_sorted   [[buffer(3)]],
    device const SqfMoeWorkTile* work_tiles [[buffer(4)]],
    device half*                 mid_sorted [[buffer(5)]],
    constant SqfMoeGatherParams& p          [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][128];
    sqf_moe_prefill_gu_nax_impl<64u, 4u, 1024u>(gu_w, gu_s, gu_b, x_sorted, work_tiles, mid_sorted, p, W_tile, tgid, tid);
}

kernel void sqf_moe_prefill_gu_nax_m128(
    device const uint2*          gu_w       [[buffer(0)]],
    device const half*           gu_s       [[buffer(1)]],
    device const half*           gu_b       [[buffer(2)]],
    device const half*           x_sorted   [[buffer(3)]],
    device const SqfMoeWorkTile* work_tiles [[buffer(4)]],
    device half*                 mid_sorted [[buffer(5)]],
    constant SqfMoeGatherParams& p          [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][128];
    sqf_moe_prefill_gu_nax_impl<128u, 8u, 1536u>(gu_w, gu_s, gu_b, x_sorted, work_tiles, mid_sorted, p, W_tile, tgid, tid);
}

template <uint BM, uint SG, uint TILE_OFF>
inline void sqf_moe_prefill_dn_nax_impl(
    device const uint2*          dn_w,
    device const half*           dn_s,
    device const half*           dn_b,
    device const half*           mid_sorted,
    device const uint*           sorted_pairs,
    device const SqfMoeWorkTile* work_tiles,
    device half*                 down_part_f16,
    constant SqfMoeGatherParams& p,
    threadgroup uint             (&tg_pid)[BM],
    threadgroup half             (&W_tile)[64][128],
    uint2 tgid,
    uint  tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.D) return;

#if defined(__HAVE_TENSOR__)
    const SqfMoeWorkTile tile = work_tiles[TILE_OFF + tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    const uint total_pairs = p.M_valid * p.top_k;

    if (tid < cnt) {
        tg_pid[tid] = sorted_pairs[tile.pair_start + tid];
    }

    tensor<device half, dextents<int, 2>, tensor_inline> tX(
        const_cast<device half*>(mid_sorted), dextents<int, 2>(int(p.moe_d_ff), int(total_pairs + 128u)));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.slice<128, BM>(0, int(tile.pair_start));
    auto cT  = op.template get_destination_cooperative_tensor<sqf_strip_as<decltype(sX0)>, sqf_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    constexpr uint N_TH          = SG * 32u;
    constexpr uint ROWS_PER_STEP = N_TH >> 3u;
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;

    const uint u2_per_row  = p.moe_d_ff >> 4u;
    const uint grp_per_row = p.moe_d_ff >> 6u;
    const uint r_grp       = tid >> 3u;
    const uint sub         = tid & 7u;
    const uint sub_g       = sub >> 2u;
    const uint r_base      = tile.expert_id * p.D + n0;

    for (uint k0 = 0u; k0 < p.moe_d_ff; k0 += 128u) {
        const uint k_u2 = k0 >> 4u;
        const uint grp  = (k0 >> 6u) + sub_g;
        auto sX = tX.slice<128, BM>(int(k0), int(tile.pair_start));

        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const uint  r_glob = r_base + r_loc;
            const uint2 qv     = dn_w[ulong(r_glob) * u2_per_row + k_u2 + sub];
            const half4 vsc    = half4(dn_s[ulong(r_glob) * grp_per_row + grp]);
            const half4 vbi    = half4(dn_b[ulong(r_glob) * grp_per_row + grp]);
            half4 n0h, n1h, n2h, n3h;
            sqf_unpack_u32_nibbles_h4(qv.x, n0h, n1h);
            sqf_unpack_u32_nibbles_h4(qv.y, n2h, n3h);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            dst[0] = fma(n0h, vsc, vbi);
            dst[1] = fma(n1h, vsc, vbi);
            dst[2] = fma(n2h, vsc, vbi);
            dst[3] = fma(n3h, vsc, vbi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx     = cT.get_multidimensional_index(i);
            const uint r = uint(idx[1]);
            const uint c = uint(idx[0]);
            if (r < cnt) {
                down_part_f16[ulong(tg_pid[r]) * ulong(p.D) + ulong(n0 + c)] = sqf_clamp_f16(cT[i]);
            }
        }
    }
#else
    (void)dn_w; (void)dn_s; (void)dn_b; (void)mid_sorted; (void)sorted_pairs; (void)work_tiles; (void)down_part_f16; (void)p; (void)tg_pid; (void)W_tile; (void)tgid; (void)tid;
#endif
}

kernel void sqf_moe_prefill_dn_nax_m16(
    device const uint2*          dn_w         [[buffer(0)]],
    device const half*           dn_s         [[buffer(1)]],
    device const half*           dn_b         [[buffer(2)]],
    device const half*           mid_sorted   [[buffer(3)]],
    device const uint*           sorted_pairs [[buffer(4)]],
    device const SqfMoeWorkTile* work_tiles   [[buffer(5)]],
    device half*                 down_part    [[buffer(6)]],
    constant SqfMoeGatherParams& p            [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup uint tg_pid[16];
    threadgroup half W_tile[64][128];
    sqf_moe_prefill_dn_nax_impl<16u, 4u, 0u>(dn_w, dn_s, dn_b, mid_sorted, sorted_pairs, work_tiles, down_part, p, tg_pid, W_tile, tgid, tid);
}

kernel void sqf_moe_prefill_dn_nax_m32(
    device const uint2*          dn_w         [[buffer(0)]],
    device const half*           dn_s         [[buffer(1)]],
    device const half*           dn_b         [[buffer(2)]],
    device const half*           mid_sorted   [[buffer(3)]],
    device const uint*           sorted_pairs [[buffer(4)]],
    device const SqfMoeWorkTile* work_tiles   [[buffer(5)]],
    device half*                 down_part    [[buffer(6)]],
    constant SqfMoeGatherParams& p            [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup uint tg_pid[32];
    threadgroup half W_tile[64][128];
    sqf_moe_prefill_dn_nax_impl<32u, 4u, 512u>(dn_w, dn_s, dn_b, mid_sorted, sorted_pairs, work_tiles, down_part, p, tg_pid, W_tile, tgid, tid);
}

kernel void sqf_moe_prefill_dn_nax_m64(
    device const uint2*          dn_w         [[buffer(0)]],
    device const half*           dn_s         [[buffer(1)]],
    device const half*           dn_b         [[buffer(2)]],
    device const half*           mid_sorted   [[buffer(3)]],
    device const uint*           sorted_pairs [[buffer(4)]],
    device const SqfMoeWorkTile* work_tiles   [[buffer(5)]],
    device half*                 down_part    [[buffer(6)]],
    constant SqfMoeGatherParams& p            [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup uint tg_pid[64];
    threadgroup half W_tile[64][128];
    sqf_moe_prefill_dn_nax_impl<64u, 4u, 1024u>(dn_w, dn_s, dn_b, mid_sorted, sorted_pairs, work_tiles, down_part, p, tg_pid, W_tile, tgid, tid);
}

kernel void sqf_moe_prefill_dn_nax_m128(
    device const uint2*          dn_w         [[buffer(0)]],
    device const half*           dn_s         [[buffer(1)]],
    device const half*           dn_b         [[buffer(2)]],
    device const half*           mid_sorted   [[buffer(3)]],
    device const uint*           sorted_pairs [[buffer(4)]],
    device const SqfMoeWorkTile* work_tiles   [[buffer(5)]],
    device half*                 down_part    [[buffer(6)]],
    constant SqfMoeGatherParams& p            [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup uint tg_pid[128];
    threadgroup half W_tile[64][128];
    sqf_moe_prefill_dn_nax_impl<128u, 4u, 1536u>(dn_w, dn_s, dn_b, mid_sorted, sorted_pairs, work_tiles, down_part, p, tg_pid, W_tile, tgid, tid);
}

struct SqfMoeCombineParams {
    uint M_valid;
    uint top_k;
    uint D;
    uint hc_count;
};

// Prefill-only FP16 `down_part` combine (cuts `down_part` traffic 2x = -5.0 GB across 48 layers; decode keeps FP32!)
kernel void sqf_moe_combine_hc_res_f16(
    device const half*           down_part   [[buffer(0)]], // [M_valid * top_k, D] FP16
    device const float*          topk_w      [[buffer(1)]], // [M_valid, top_k]
    device const float*          shared_out  [[buffer(2)]], // [M_valid, D]
    device const float*          gate_scalar [[buffer(3)]], // [M_valid]
    device const float*          inj         [[buffer(4)]], // [M_valid, hc_count]
    device float*                hc_buf      [[buffer(5)]], // [M_valid, hc_count * D]
    constant SqfMoeCombineParams& p          [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]              // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    float4 y_moe = reinterpret_cast<device const float4*>(shared_out)[ulong(m) * d4_cnt + idx4] * gate_scalar[m];
    for (uint k = 0u; k < p.top_k; ++k) {
        const float  wk  = topk_w[ulong(m) * p.top_k + k];
        const float4 dk4 = float4(reinterpret_cast<device const half4*>(down_part)[(ulong(m) * p.top_k + k) * d4_cnt + idx4]);
        y_moe = fma(float4(wk), dk4, y_moe);
    }

    const uint hc_d4 = p.hc_count * d4_cnt;
    for (uint s = 0u; s < p.hc_count; ++s) {
        const float  inj_s  = inj[ulong(m) * p.hc_count + s];
        const ulong  hc_off = ulong(m) * hc_d4 + ulong(s) * d4_cnt + idx4;
        device float4* dst  = reinterpret_cast<device float4*>(hc_buf) + hc_off;
        *dst = fma(float4(inj_s), y_moe, *dst);
    }
}

// Combines `sum_{k=0..top_k-1} w[k]*routed[k] + gate_scalar*shared_out` AND writes back `hc_buf += inj * y_moe`!
kernel void sqf_moe_combine_hc_res(
    device const float*          down_part   [[buffer(0)]], // [M_valid * top_k, D]
    device const float*          topk_w      [[buffer(1)]], // [M_valid, top_k]
    device const float*          shared_out  [[buffer(2)]], // [M_valid, D]
    device const float*          gate_scalar [[buffer(3)]], // [M_valid]
    device const float*          inj         [[buffer(4)]], // [M_valid, hc_count]
    device float*                hc_buf      [[buffer(5)]], // [M_valid, hc_count * D]
    constant SqfMoeCombineParams& p          [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]              // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    float4 y_moe = reinterpret_cast<device const float4*>(shared_out)[ulong(m) * d4_cnt + idx4] * gate_scalar[m];
    for (uint k = 0u; k < p.top_k; ++k) {
        const float  wk  = topk_w[ulong(m) * p.top_k + k];
        const float4 dk4 = reinterpret_cast<device const float4*>(down_part)[(ulong(m) * p.top_k + k) * d4_cnt + idx4];
        y_moe = fma(float4(wk), dk4, y_moe);
    }

    const uint hc_d4 = p.hc_count * d4_cnt;
    for (uint s = 0u; s < p.hc_count; ++s) {
        const float  inj_s  = inj[ulong(m) * p.hc_count + s];
        const ulong  hc_off = ulong(m) * hc_d4 + ulong(s) * d4_cnt + idx4;
        device float4* dst  = reinterpret_cast<device float4*>(hc_buf) + hc_off;
        *dst = fma(float4(inj_s), y_moe, *dst);
    }
}

/* ============================================================================
 * 10. Native 1-Layer `MTP` (`mtp.*`) Fusion & Batched Vocab Sampler Kernels
 * ============================================================================ */

struct SqfMtpNormHiddenParams {
    uint  M_valid;
    uint  rows_padded; // Padded rows of `[M_valid * hc_count, D]` for NAX!
    uint  D;
    uint  hc_count;
    float eps;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

// UNGROUPED `10240`-dim RMSNorm (`mtp.pre_fc_norm_hidden`, `group_size = None`!)
// Output shape: `[rows_padded, D]` FP16 (`m * hc_count + s` rows, zero-padded to `rows_padded`)!
// Grid: `(1, max(M_valid, (rows_padded + hc_count - 1) / hc_count), 1)`, `256` threads/TG
kernel void sqf_mtp_ungrouped_norm_split(
    device const float*            hc_prev [[buffer(0)]], // [M_valid, hc_count * D]
    device const float*            norm_w  [[buffer(1)]], // [hc_count * D] (+1.0 folded)
    device half*                   out_f16 [[buffer(2)]], // [rows_padded, D]
    constant SqfMtpNormHiddenParams& p     [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],           // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],        // 0..7
    uint  lane [[thread_index_in_simdgroup]]              // 0..31
) {
    const uint m       = tgid.y;
    const uint hc_dim  = p.hc_count * p.D;
    const uint hc_d4   = hc_dim >> 2u;
    const uint d4_cnt  = p.D >> 2u;
    const uint r_start = m * p.hc_count;

    if (m >= p.M_valid) {
        for (uint s = 0u; s < p.hc_count && (r_start + s) < p.rows_padded; ++s) {
            device half4* dst = reinterpret_cast<device half4*>(out_f16 + ulong(r_start + s) * p.D);
            for (uint i = tid; i < d4_cnt; i += 256u) {
                dst[i] = half4(0.0h);
            }
        }
        return;
    }

    threadgroup float tg_sq[8];
    device const float4* src4 = reinterpret_cast<device const float4*>(hc_prev + ulong(m) * hc_dim);
    device const float4* w4   = reinterpret_cast<device const float4*>(norm_w);

    float sq_acc = 0.0f;
    for (uint i = tid; i < hc_d4; i += 256u) {
        float4 v = src4[i];
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }
    float s = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(hc_dim) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    device half4* dst4 = reinterpret_cast<device half4*>(out_f16 + ulong(m) * hc_dim);
    for (uint i = tid; i < hc_d4; i += 256u) {
        dst4[i] = sqf_clamp_f16x4((src4[i] * inv_rms) * w4[i]);
    }
}

struct SqfMtpAddParams {
    uint M_valid;
    uint D;
    uint hc_count;
    uint pad;
};

// `h_mtp[m, s, d] = e_proj[m, d] + h_fc[m * hc_count + s, d]`
kernel void sqf_mtp_add_broadcast(
    device const float*      e_proj [[buffer(0)]], // [M_valid, D]
    device const float*      h_fc   [[buffer(1)]], // [M_valid * hc_count, D]
    device float*            h_mtp  [[buffer(2)]], // [M_valid, hc_count * D]
    constant SqfMtpAddParams& p     [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]     // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    const float4 e4    = reinterpret_cast<device const float4*>(e_proj)[ulong(m) * d4_cnt + idx4];
    const uint   hc_d4 = p.hc_count * d4_cnt;
    for (uint s = 0u; s < p.hc_count; ++s) {
        const ulong off4  = ulong(m) * hc_d4 + ulong(s) * d4_cnt + idx4;
        const float4 hfc4 = reinterpret_cast<device const float4*>(h_fc)[off4];
        reinterpret_cast<device float4*>(h_mtp)[off4] = e4 + hfc4;
    }
}

struct SqfSampleParams {
    uint  M;              // Number of logit rows (1 for decode/draft, `1 + draft_k` for spec verify!)
    uint  vocab_size;     // Full vocab (248320) or FR-Spec 32K shortlist length (32768)
    uint  greedy;         // bit0 => argmax, bit1 => want_top2
    float inv_temp;
    uint  top_k;          // 0 => 64, else min(top_k, 64)
    float top_p;
    uint  rng_lo;
    uint  rng_hi;
    uint  use_token_map;  // 0 => identity, 1 => map via `token_map[idx]`
    uint  compute_pmax;   // 0 => off, 1 => write softmax p_max into `out_pmax[m]`
    uint  pad0;
    uint  pad1;
};

inline uint sqf_unmap_tok(uint idx, uint use_map, device const uint* token_map) {
    return (use_map != 0u) ? token_map[idx] : idx;
}

inline float sqf_rand_u01(uint lo, uint hi, uint salt) {
    ulong z = ((ulong(hi) << 32ul) | ulong(lo)) + ulong(salt) * 0x9E3779B97F4A7C15ul;
    z += 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30ul)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27ul)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31ul);
    uint u24 = uint(z >> 40ul) & 0xFFFFFFu;
    return (float(u24) + 0.5f) * (1.0f / 16777216.0f);
}

// Batched `1024`-Thread Vocab Argmax & Top-K/Top-P Sampler (`Grid: (M, 1, 1)`)
// Exports `out_tok[M]` AND (when `!greedy`) `out_top_ids[M, 64]`, `out_top_probs[M, 64]`, `out_keep_k[M]`
// for exact Leviathan stochastic speculative verification!
kernel void sqf_sample_vocab_batch(
    device const float*       logits        [[buffer(0)]], // [M, vocab_size]
    device uint*              out_tok       [[buffer(1)]], // [M]
    device uint*              out_top_ids   [[buffer(2)]], // optional [M, 64]
    device float*             out_top_probs [[buffer(3)]], // optional [M, 64]
    device uint*              out_keep_k    [[buffer(4)]], // optional [M]
    constant SqfSampleParams& p             [[buffer(5)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],             // 0..1023
    uint sgid [[simdgroup_index_in_threadgroup]],          // 0..31
    uint lane [[thread_index_in_simdgroup]]                // 0..31
) {
    if (m >= p.M) return;

    threadgroup float tg_val[32];
    threadgroup uint  tg_idx[32];
    threadgroup float top_vals[64];
    threadgroup uint  top_ids[64];

    device const float* row = logits + ulong(m) * p.vocab_size;

    if (p.greedy != 0u) {
        float best_v = -INFINITY;
        uint  best_i = 0u;
        for (uint i = tid; i < p.vocab_size; i += 1024u) {
            float v = row[i];
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }

        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) {
            tg_val[sgid] = best_v;
            tg_idx[sgid] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float v   = tg_val[lane];
            uint  idx = tg_idx[lane];
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(v, off);
                uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) {
                    v   = ov;
                    idx = oi;
                }
            }
            if (lane == 0u) {
                out_tok[m] = idx;
            }
        }
        return;
    }

    const uint K = min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
    for (uint k = 0u; k < K; ++k) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;

        for (uint i = tid; i < p.vocab_size; i += 1024u) {
            bool skip = false;
            for (uint prev = 0u; prev < k; ++prev) {
                if (top_ids[prev] == i) {
                    skip = true;
                    break;
                }
            }
            if (!skip) {
                float v = row[i];
                if (v > best_v || (v == best_v && i < best_i)) {
                    best_v = v;
                    best_i = i;
                }
            }
        }

        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) {
            tg_val[sgid] = best_v;
            tg_idx[sgid] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float v   = tg_val[lane];
            uint  idx = tg_idx[lane];
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(v, off);
                uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) {
                    v   = ov;
                    idx = oi;
                }
            }
            if (lane == 0u) {
                top_vals[k] = v;
                top_ids[k]  = (idx < p.vocab_size) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const bool is_greedy = (p.greedy & 1u) != 0u;
        const bool want_top2 = (p.greedy & 2u) != 0u;
        if (is_greedy) {
            out_tok[0] = top_ids[0];
            if (want_top2) {
                out_tok[1] = (K >= 2u) ? top_ids[1] : top_ids[0];
            }
            return;
        }
        const float max_v = top_vals[0];
        float probs[64];
        float sum_p = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float w = exp((top_vals[k] - max_v) * p.inv_temp);
            probs[k] = w;
            sum_p   += w;
        }
        const float inv_sum = (sum_p > 0.0f) ? (1.0f / sum_p) : 1.0f;
        for (uint k = 0u; k < K; ++k) probs[k] *= inv_sum;

        uint  keep_k    = K;
        float kept_mass = 1.0f;
        if (p.top_p > 0.0f && p.top_p < 1.0f) {
            float cum = 0.0f;
            for (uint k = 0u; k < K; ++k) {
                cum += probs[k];
                if (cum >= p.top_p) {
                    keep_k    = k + 1u;
                    kept_mass = cum;
                    break;
                }
            }
        }

        const float inv_kept = (kept_mass > 0.0f) ? (1.0f / kept_mass) : 1.0f;
        for (uint k = 0u; k < 64u; ++k) {
            float norm_p = (k < keep_k) ? (probs[k] * inv_kept) : 0.0f;
            if (out_top_ids)   out_top_ids[ulong(m) * 64ul + k]   = (k < K) ? top_ids[k] : 0u;
            if (out_top_probs) out_top_probs[ulong(m) * 64ul + k] = norm_p;
        }
        if (out_keep_k) out_keep_k[m] = keep_k;

        const float u          = sqf_rand_u01(p.rng_lo, p.rng_hi, m);
        float       cdf        = 0.0f;
        uint        chosen     = top_ids[0];
        uint        chosen_idx = 0u;
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += probs[k] * inv_kept;
            if (u <= cdf || k + 1u == keep_k) {
                chosen     = top_ids[k];
                chosen_idx = k;
                break;
            }
        }
        out_tok[m] = chosen;

        if (want_top2 && m == 0u) {
            const float p_first = probs[chosen_idx] * inv_kept;
            const float rem     = max(0.0f, 1.0f - p_first);
            uint        chosen2 = chosen;
            if (keep_k > 1u && rem > 1e-7f) {
                const float u2 = sqf_rand_u01(p.rng_lo ^ 0x9E3779B9u, p.rng_hi ^ 0x7F4A7C15u, 1u) * rem;
                float       cdf2 = 0.0f;
                for (uint k = 0u; k < keep_k; ++k) {
                    if (k == chosen_idx) continue;
                    cdf2   += probs[k] * inv_kept;
                    chosen2 = top_ids[k];
                    if (u2 <= cdf2) break;
                }
            }
            out_tok[1] = chosen2;
        }
    }
}

// 2-Stage Parallel Vocab Sampler (`64` TGs x `256` th -> `1` TG x `256` th):
// 100% bit-exact to `sqf_sample_vocab_batch` for both greedy AND top-K/top-P stochastic sampling,
// while running >50x faster (`4.35 ms -> 0.08 ms` for top_k=20) by saturating all 40 M5 Max cores!
kernel void sqf_sample_vocab_stage1(
    device const float*       logits    [[buffer(0)]], // [M, vocab_size]
    device float*             stage_v   [[buffer(1)]], // [M, 64, K_out]
    device uint*              stage_i   [[buffer(2)]], // [M, 64, K_out]
    device float*             stage_lse [[buffer(3)]], // [M, 64] (when compute_pmax != 0)
    constant SqfSampleParams& p         [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],       // (64, M)
    uint  tid  [[thread_index_in_threadgroup]],        // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],     // 0..7
    uint  lane [[thread_index_in_simdgroup]]           // 0..31
) {
    const uint blk = tgid.x;
    const uint m   = tgid.y;
    if (blk >= 64u || m >= p.M) return;

    const uint chunk = max(1u, (p.vocab_size + 63u) >> 6u);
    const uint beg   = blk * chunk;
    const uint end   = min(p.vocab_size, beg + chunk);
    device const float* row = logits + ulong(m) * p.vocab_size;

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    threadgroup float tg_blk_max;

    const bool is_greedy = (p.greedy & 1u) != 0u;
    const bool want_top2 = (p.greedy & 2u) != 0u;

    if (is_greedy && !want_top2) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;
        for (uint i = beg + tid; i < end; i += 256u) {
            float v = row[i];
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) {
            tg_v[sgid] = best_v;
            tg_i[sgid] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(v, off);
                uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) {
                    v   = ov;
                    idx = oi;
                }
            }
            if (lane == 0u) {
                tg_blk_max = v;
                stage_v[ulong(m) * 64ul + blk] = v;
                stage_i[ulong(m) * 64ul + blk] = idx;
            }
        }
        if (p.compute_pmax != 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float bmax = tg_blk_max;
            float lse = 0.0f;
            if (isfinite(bmax)) {
                for (uint i = beg + tid; i < end; i += 256u) {
                    lse += exp((row[i] - bmax) * p.inv_temp);
                }
            }
            lse = simd_sum(lse);
            if (lane == 0u) tg_v[sgid] = lse;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                const float tot = simd_sum((lane < 8u) ? tg_v[lane] : 0.0f);
                if (lane == 0u) stage_lse[ulong(m) * 64ul + blk] = tot;
            }
        }
        return;
    }

    const uint K = is_greedy ? min(p.vocab_size, 2u)
                             : min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
    const ulong base_out = (ulong(m) * 64ul + blk) * K;

    if (chunk <= 4096u) {
        // Fast single-read register path (handles all V <= 262,144 including 248,320!)
        float rv[16];
        uint  ri[16];
        uint  cnt = 0u;
        for (uint i = beg + tid; i < end && cnt < 16u; i += 256u) {
            rv[cnt] = row[i];
            ri[cnt] = i;
            ++cnt;
        }
        for (uint k = 0u; k < K; ++k) {
            float best_v = -INFINITY;
            uint  best_i = 0xFFFFFFFFu;
            uint  best_j = 16u;
            for (uint j = 0u; j < cnt; ++j) {
                float v   = rv[j];
                uint  idx = ri[j];
                if (v > best_v || (v == best_v && idx < best_i)) {
                    best_v = v;
                    best_i = idx;
                    best_j = j;
                }
            }
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(best_v, off);
                uint  oi = simd_shuffle_xor(best_i, off);
                if (ov > best_v || (ov == best_v && oi < best_i)) {
                    best_v = ov;
                    best_i = oi;
                }
            }
            if (lane == 0u) {
                tg_v[sgid] = best_v;
                tg_i[sgid] = best_i;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
                uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
                #pragma clang loop unroll(full)
                for (ushort off = 4u; off > 0u; off >>= 1u) {
                    float ov = simd_shuffle_xor(v, off);
                    uint  oi = simd_shuffle_xor(idx, off);
                    if (ov > v || (ov == v && oi < idx)) {
                        v   = ov;
                        idx = oi;
                    }
                }
                if (lane == 0u) {
                    if (k == 0u) tg_blk_max = v;
                    tg_i[0] = idx;
                    stage_v[base_out + k] = v;
                    stage_i[base_out + k] = idx;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (best_j < 16u && ri[best_j] == tg_i[0]) {
                rv[best_j] = -INFINITY;
                ri[best_j] = 0xFFFFFFFFu;
            }
        }
    } else {
        threadgroup uint picked[64];
        for (uint k = 0u; k < K; ++k) {
            float best_v = -INFINITY;
            uint  best_i = 0xFFFFFFFFu;
            for (uint i = beg + tid; i < end; i += 256u) {
                bool skip = false;
                for (uint prev = 0u; prev < k; ++prev) {
                    if (picked[prev] == i) { skip = true; break; }
                }
                if (!skip) {
                    float v = row[i];
                    if (v > best_v || (v == best_v && i < best_i)) {
                        best_v = v;
                        best_i = i;
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(best_v, off);
                uint  oi = simd_shuffle_xor(best_i, off);
                if (ov > best_v || (ov == best_v && oi < best_i)) {
                    best_v = ov;
                    best_i = oi;
                }
            }
            if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
                uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
                #pragma clang loop unroll(full)
                for (ushort off = 4u; off > 0u; off >>= 1u) {
                    float ov = simd_shuffle_xor(v, off);
                    uint  oi = simd_shuffle_xor(idx, off);
                    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
                }
                if (lane == 0u) {
                    if (k == 0u) tg_blk_max = v;
                    picked[k] = idx;
                    stage_v[base_out + k] = v;
                    stage_i[base_out + k] = idx;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (p.compute_pmax != 0u) {
        const float bmax = tg_blk_max;
        float lse = 0.0f;
        if (isfinite(bmax)) {
            for (uint i = beg + tid; i < end; i += 256u) {
                lse += exp((row[i] - bmax) * p.inv_temp);
            }
        }
        lse = simd_sum(lse);
        if (lane == 0u) tg_v[sgid] = lse;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float tot = simd_sum((lane < 8u) ? tg_v[lane] : 0.0f);
            if (lane == 0u) stage_lse[ulong(m) * 64ul + blk] = tot;
        }
    }
}

kernel void sqf_sample_vocab_stage2(
    device const float*       stage_v       [[buffer(0)]], // [M, 64 * K_out]
    device const uint*        stage_i       [[buffer(1)]], // [M, 64 * K_out]
    device uint*              out_tok       [[buffer(2)]], // [M]
    device uint*              out_top_ids   [[buffer(3)]], // optional [M, 64]
    device float*             out_top_probs [[buffer(4)]], // optional [M, 64]
    device uint*              out_keep_k    [[buffer(5)]], // optional [M]
    device const float*       stage_lse     [[buffer(6)]], // [M, 64] (when compute_pmax != 0)
    device const uint*        token_map     [[buffer(7)]], // [vocab_size] (when use_token_map != 0)
    device float*             out_pmax      [[buffer(8)]], // [M] (when compute_pmax != 0)
    constant SqfSampleParams& p             [[buffer(9)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],             // 0..255
    uint sgid [[simdgroup_index_in_threadgroup]],          // 0..7
    uint lane [[thread_index_in_simdgroup]]                // 0..31
) {
    if (m >= p.M) return;

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    threadgroup float tg_gmax;

    const bool is_greedy = (p.greedy & 1u) != 0u;
    const bool want_top2 = (p.greedy & 2u) != 0u;

    if (is_greedy && !want_top2) {
        float best_v = (tid < 64u) ? stage_v[ulong(m) * 64ul + tid] : -INFINITY;
        uint  best_i = (tid < 64u) ? stage_i[ulong(m) * 64ul + tid] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(v, off);
                uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_gmax = v;
                out_tok[m] = (idx < p.vocab_size) ? sqf_unmap_tok(idx, p.use_token_map, token_map) : 0u;
            }
        }
        if (p.compute_pmax != 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float gmax = tg_gmax;
            float term = 0.0f;
            if (tid < 64u && isfinite(gmax)) {
                const float bmax = stage_v[ulong(m) * 64ul + tid];
                const float blse = stage_lse[ulong(m) * 64ul + tid];
                if (blse > 0.0f && isfinite(bmax)) {
                    term = blse * exp((bmax - gmax) * p.inv_temp);
                }
            }
            term = simd_sum(term);
            if (lane == 0u) tg_v[sgid] = term;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                const float tot = simd_sum((lane < 2u) ? tg_v[lane] : 0.0f);
                if (lane == 0u) out_pmax[m] = (tot > 0.0f) ? clamp(1.0f / tot, 0.0f, 1.0f) : 0.0f;
            }
        }
        return;
    }

    const uint  K       = is_greedy ? min(p.vocab_size, 2u)
                                    : min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
    const uint  tot     = 64u * K;
    const ulong base_in = ulong(m) * tot;

    float rv[16];
    uint  ri[16];
    uint  cnt = 0u;
    for (uint i = tid; i < tot && cnt < 16u; i += 256u) {
        rv[cnt] = stage_v[base_in + i];
        ri[cnt] = stage_i[base_in + i];
        ++cnt;
    }

    threadgroup float top_vals[64];
    threadgroup uint  top_ids[64];

    for (uint k = 0u; k < K; ++k) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;
        uint  best_j = 16u;
        for (uint j = 0u; j < cnt; ++j) {
            float v   = rv[j];
            uint  idx = ri[j];
            if (v > best_v || (v == best_v && idx < best_i)) {
                best_v = v;
                best_i = idx;
                best_j = j;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(v, off);
                uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_i[0]     = idx;
                top_vals[k] = v;
                top_ids[k]  = (idx < p.vocab_size) ? sqf_unmap_tok(idx, p.use_token_map, token_map) : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (best_j < 16u && ri[best_j] == tg_i[0]) {
            rv[best_j] = -INFINITY;
            ri[best_j] = 0xFFFFFFFFu;
        }
    }

    if (p.compute_pmax != 0u) {
        const float gmax = top_vals[0];
        float term = 0.0f;
        if (tid < 64u && isfinite(gmax)) {
            const float bmax = stage_v[base_in + ulong(tid) * K];
            const float blse = stage_lse[ulong(m) * 64ul + tid];
            if (blse > 0.0f && isfinite(bmax)) {
                term = blse * exp((bmax - gmax) * p.inv_temp);
            }
        }
        term = simd_sum(term);
        if (lane == 0u) tg_v[sgid] = term;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float tot_lse = simd_sum((lane < 2u) ? tg_v[lane] : 0.0f);
            if (lane == 0u) out_pmax[m] = (tot_lse > 0.0f) ? clamp(1.0f / tot_lse, 0.0f, 1.0f) : 0.0f;
        }
    }

    if (tid == 0u) {
        if (is_greedy) {
            out_tok[0] = top_ids[0];
            if (want_top2) {
                out_tok[1] = (K >= 2u) ? top_ids[1] : top_ids[0];
            }
            return;
        }
        const float max_v = top_vals[0];
        float probs[64];
        float sum_p = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float w = exp((top_vals[k] - max_v) * p.inv_temp);
            probs[k] = w;
            sum_p   += w;
        }
        const float inv_sum = (sum_p > 0.0f) ? (1.0f / sum_p) : 1.0f;
        for (uint k = 0u; k < K; ++k) probs[k] *= inv_sum;

        uint  keep_k    = K;
        float kept_mass = 1.0f;
        if (p.top_p > 0.0f && p.top_p < 1.0f) {
            float cum = 0.0f;
            for (uint k = 0u; k < K; ++k) {
                cum += probs[k];
                if (cum >= p.top_p) {
                    keep_k    = k + 1u;
                    kept_mass = cum;
                    break;
                }
            }
        }

        const float inv_kept = (kept_mass > 0.0f) ? (1.0f / kept_mass) : 1.0f;
        for (uint k = 0u; k < 64u; ++k) {
            float norm_p = (k < keep_k) ? (probs[k] * inv_kept) : 0.0f;
            if (out_top_ids)   out_top_ids[ulong(m) * 64ul + k]   = (k < K) ? top_ids[k] : 0u;
            if (out_top_probs) out_top_probs[ulong(m) * 64ul + k] = norm_p;
        }
        if (out_keep_k) out_keep_k[m] = keep_k;

        const float u          = sqf_rand_u01(p.rng_lo, p.rng_hi, m);
        float       cdf        = 0.0f;
        uint        chosen     = top_ids[0];
        uint        chosen_idx = 0u;
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += probs[k] * inv_kept;
            if (u <= cdf || k + 1u == keep_k) {
                chosen     = top_ids[k];
                chosen_idx = k;
                break;
            }
        }
        out_tok[m] = chosen;

        if (want_top2 && m == 0u) {
            const float p_first = probs[chosen_idx] * inv_kept;
            const float rem     = max(0.0f, 1.0f - p_first);
            uint        chosen2 = chosen;
            if (keep_k > 1u && rem > 1e-7f) {
                const float u2 = sqf_rand_u01(p.rng_lo ^ 0x9E3779B9u, p.rng_hi ^ 0x7F4A7C15u, 1u) * rem;
                float       cdf2 = 0.0f;
                for (uint k = 0u; k < keep_k; ++k) {
                    if (k == chosen_idx) continue;
                    cdf2   += probs[k] * inv_kept;
                    chosen2 = top_ids[k];
                    if (u2 <= cdf2) break;
                }
            }
            out_tok[1] = chosen2;
        }
    }
}

struct SqfLogitPenaltyParams {
    uint  M;                  // Batch rows in `logits` (1 for decode/cb_draft step, 1..8 for cb_ver)
    uint  vocab_size;         // Full vocab V (248320)
    uint  logits_stride;      // Row stride in `logits` (`V` for full, `32768` for FR-Spec shortlist)
    uint  use_shortlist;      // 0 => full vocab; 1 => map via `full_to_short[t]`
    uint  pad_sl;             // reserved (keeps 80-byte struct layout)
    uint  num_hist_ids;       // Unique completion token IDs in `pen_ids[0..num_hist_ids-1]`
    uint  extra_mode;         // 0 => linear cb_ver (`m` sees `extra_toks[1..m]`), 1 => cb_draft (`m=0` sees `extra_toks[1..extra_count]`), 2 => tree cb_ver (M=7)
    uint  extra_count;        // `M - 1` for cb_ver; `k` (`0..eff_k-1`) for cb_draft
    float presence_penalty;
    float frequency_penalty;
    float repetition_penalty; // <= 0.0f or == 1.0f => disabled
    uint  pad0;
    uint  drop_toks[8];       // Sliding-window evictions for speculative steps +1..+8 (`0xFFFFFFFFu` => none)
};

inline float sqf_apply_pen_val(float v, uint c, float pres, float freq, float rep) {
    if (c == 0u) return v;
    if (rep > 0.0f && rep != 1.0f) {
        v = (v > 0.0f) ? (v / rep) : (v * rep);
    }
    return v - (pres + freq * float(c));
}

// Sparse O(U + M) completion-scoped & sliding-window OpenAI `presence_penalty` / `frequency_penalty` / `repetition_penalty`
// 100% bit-exact `K>0 == K=0` across speculative verification batches, tree verify, and FR-Spec 32K MTP drafts!
// Grid: `((num_hist_ids + extra_count + 255) / 256, 1, 1)`, 256 threads/TG
kernel void sqf_apply_logit_penalties(
    device float*                     logits        [[buffer(0)]], // [M, logits_stride] (in-place)
    device const uint*                pen_ids       [[buffer(1)]], // [num_hist_ids] unique token IDs in active window
    device const ushort*              pen_counts    [[buffer(2)]], // [vocab_size] occurrence counts in active window
    device const uint*                extra_toks    [[buffer(3)]], // [1 + extra_count] (`verify_tokens` or `b_mtp_token_ids`)
    device const uint*                full_to_short [[buffer(4)]], // [vocab_size] (when use_shortlist != 0)
    constant SqfLogitPenaltyParams&   p             [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_items = p.num_hist_ids + p.extra_count;
    if (gid >= total_items) return;

    uint t = 0u;
    uint base_c = 0u;
    if (gid < p.num_hist_ids) {
        t = pen_ids[gid];
        if (t >= p.vocab_size) return;
        base_c = uint(pen_counts[t]);
    } else {
        const uint j = 1u + (gid - p.num_hist_ids);
        t = extra_toks[j];
        if (t >= p.vocab_size || pen_counts[t] != 0u) return;
        for (uint pj = 1u; pj < j; ++pj) {
            if (extra_toks[pj] == t) return;
        }
        base_c = 0u;
    }

    const uint col = (p.use_shortlist != 0u) ? full_to_short[t] : t;
    if (col == 0xFFFFFFFFu || col >= p.logits_stride) return;

    if (p.extra_mode == 1u) {
        uint c = base_c;
        for (uint j = 1u; j <= p.extra_count; ++j) {
            if (extra_toks[j] == t) ++c;
            if ((j - 1u) < 8u && p.drop_toks[j - 1u] == t && c > 0u) --c;
        }
        if (c > 0u) {
            logits[col] = sqf_apply_pen_val(logits[col], c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
        }
    } else if (p.extra_mode == 2u) {
        // Tree verify (`M == 7`): m=0 (root), m=1..3 (branch A: extra[1..m]), m=4..6 (branch B: extra[4..m])
        if (base_c > 0u) {
            logits[col] = sqf_apply_pen_val(logits[col], base_c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
        }
        uint ca = base_c;
        for (uint m = 1u; m < min(p.M, 4u); ++m) {
            if (extra_toks[m] == t) ++ca;
            if ((m - 1u) < 8u && p.drop_toks[m - 1u] == t && ca > 0u) --ca;
            if (ca > 0u) {
                const ulong idx = ulong(m) * p.logits_stride + col;
                logits[idx] = sqf_apply_pen_val(logits[idx], ca, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
            }
        }
        uint cb = base_c;
        for (uint m = 4u; m < p.M; ++m) {
            if (extra_toks[m] == t) ++cb;
            if ((m - 4u) < 8u && p.drop_toks[m - 4u] == t && cb > 0u) --cb;
            if (cb > 0u) {
                const ulong idx = ulong(m) * p.logits_stride + col;
                logits[idx] = sqf_apply_pen_val(logits[idx], cb, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
            }
        }
    } else {
        uint c = base_c;
        for (uint m = 0u; m < p.M; ++m) {
            if (m > 0u && m <= p.extra_count) {
                if (extra_toks[m] == t) ++c;
                if ((m - 1u) < 8u && p.drop_toks[m - 1u] == t && c > 0u) --c;
            }
            if (c > 0u) {
                const ulong idx = ulong(m) * p.logits_stride + col;
                logits[idx] = sqf_apply_pen_val(logits[idx], c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
            }
        }
    }
}

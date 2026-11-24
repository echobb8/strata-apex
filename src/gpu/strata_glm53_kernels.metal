#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#if defined(__HAVE_TENSOR__)
#include <metal_tensor>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;
#if defined(__HAVE_TENSOR__)
using namespace mpp::tensor_ops;
#endif

// Inlined by strata_glm53_gpu.mm at runtime from ../../include/strata_glm53.h:
// @@STRATA_GLM53_HEADER@@
#ifndef STRATA_GLM53_H
#include "../../include/strata_glm53.h"
#endif

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 410)
template <typename T>
using sg53_strip_as = metal::remove_addrspace_t<T>;
#else
template <typename T>
using sg53_strip_as = T;
#endif

/* ============================================================================
 * 1. Numerically Safe IEEE-754 Helpers & Lossless 4-Bit Affine Unpack
 * ============================================================================ */

#define SG53_OUT_F16     0u
#define SG53_OUT_F32     1u
#define SG53_OUT_RES_F32 2u

inline half sg53_clamp_f16(float x) {
    return half(clamp(x, -65504.0f, 65504.0f));
}

inline half4 sg53_clamp_f16x4(float4 x) {
    return half4(clamp(x, float4(-65504.0f), float4(65504.0f)));
}

inline float sg53_silu(float x) {
    return x / (1.0f + exp(-x));
}

inline float sg53_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Clamped LimitedSwiGLU(limit = 10.0f): `silu(min(gate, limit)) * clamp(up, -limit, +limit)`
inline float sg53_limited_swiglu(float gate, float up, float limit) {
    float gc = min(gate, limit);
    float uc = clamp(up, -limit, limit);
    return (gc / (1.0f + exp(-gc))) * uc;
}

inline float4 sg53_limited_swiglu4(float4 gate, float4 up, float limit) {
    float4 gc = min(gate, float4(limit));
    float4 uc = clamp(up, float4(-limit), float4(limit));
    return (gc / (float4(1.0f) + exp(-gc))) * uc;
}

// Signed IEEE-754 float -> strictly monotonic uint32 order-preserving radix key
// (Required because Glm5NextIndexer `w_head` can be < 0, producing negative pool scores!)
inline uint sg53_f32_to_sortable_u32(float x) {
    if (isnan(x)) return 0u;
    float clean = (x == 0.0f) ? 0.0f : x;
    uint  bits  = as_type<uint>(clean);
    return ((bits & 0x80000000u) != 0u) ? (~bits) : (bits ^ 0x80000000u);
}

// Lossless IEEE-754 binary16 trick: `0x6400 | nib` == `1024.0h + half(nib)` for nib in 0..15
inline void sg53_unpack_u32_nibbles_h4(uint w, thread half4& lo, thread half4& hi) {
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

inline float sg53_dot_h4x4(
    half4 n0, half4 n1, half4 n2, half4 n3,
    half4 x0, half4 x1, half4 x2, half4 x3
) {
    float4 s0 = fma(float4(n0), float4(x0), float4(n1) * float4(x1));
    float4 s1 = fma(float4(n2), float4(x2), float4(n3) * float4(x3));
    float4 st = s0 + s1;
    return (st.x + st.y) + (st.z + st.w);
}

/* ============================================================================
 * 2. Zero-Fill, Buffer Copy & 4-Bit Affine Embedding Lookup (with MTP pos==0 mask)
 * ============================================================================ */

kernel void sg53_zero_u32(
    device uint4*  buf      [[buffer(0)]],
    constant uint& count_u4 [[buffer(1)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid < count_u4) {
        buf[gid] = uint4(0u);
    }
}

kernel void sg53_copy_u32(
    device const uint4* src      [[buffer(0)]],
    device uint4*       dst      [[buffer(1)]],
    constant uint&      count_u4 [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid < count_u4) {
        dst[gid] = src[gid];
    }
}

struct Sg53EmbedParams {
    uint M;
    uint cols;
    uint vocab_size;
    uint hc_tile_count; // 4 for main model entry `[M, 4, cols]`, 1 for MTP `[M, cols]`
    uint pos_start;     // MTP global position start
    uint zero_at_pos0;  // 1 => `e_tok = zeros(cols)` when `pos_start + m == 0`!
    uint pad0;
    uint pad1;
};

kernel void sg53_embed_lookup_q4(
    device const uint4*   weight_u4 [[buffer(0)]], // [vocab_size, cols / 32]
    device const half*    scales    [[buffer(1)]], // [vocab_size, cols / 64]
    device const half*    biases    [[buffer(2)]], // [vocab_size, cols / 64]
    device const uint*    token_ids [[buffer(3)]], // [M]
    device float*         out_f32   [[buffer(4)]], // [M, hc_tile_count * cols]
    constant Sg53EmbedParams& p     [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]      // 0..255
) {
    if (tgid >= p.M) return;
    const uint u4_per_row  = p.cols >> 5u;
    const uint grp_per_row = p.cols >> 6u;

    if (p.zero_at_pos0 != 0u && (p.pos_start + tgid) == 0u) {
        for (uint u = tid; u < u4_per_row; u += 256u) {
            for (uint s = 0u; s < p.hc_tile_count; ++s) {
                device float4* dst = reinterpret_cast<device float4*>(
                    out_f32 + (ulong(tgid) * p.hc_tile_count + s) * p.cols + u * 32u);
                dst[0] = float4(0.0f); dst[1] = float4(0.0f);
                dst[2] = float4(0.0f); dst[3] = float4(0.0f);
                dst[4] = float4(0.0f); dst[5] = float4(0.0f);
                dst[6] = float4(0.0f); dst[7] = float4(0.0f);
            }
        }
        return;
    }

    const uint tok = min(token_ids[tgid], p.vocab_size - 1u);

    for (uint u = tid; u < u4_per_row; u += 256u) {
        const uint  grp = u >> 1u;
        const uint4 q   = weight_u4[ulong(tok) * u4_per_row + u];
        const float sc  = float(scales[ulong(tok) * grp_per_row + grp]);
        const float bi  = float(biases[ulong(tok) * grp_per_row + grp]);

        half4 n0, n1, n2, n3, n4, n5, n6, n7;
        sg53_unpack_u32_nibbles_h4(q.x, n0, n1);
        sg53_unpack_u32_nibbles_h4(q.y, n2, n3);
        sg53_unpack_u32_nibbles_h4(q.z, n4, n5);
        sg53_unpack_u32_nibbles_h4(q.w, n6, n7);

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
 * 3. Multi-M (`M = 1 .. 8`) Coalesced Q4, FP16 & FP32 GEMV Suite
 * ============================================================================ */

inline void sg53_q4_gemv4_rows_batch(
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
        const uint grp_idx = u2_idx >> 2u;

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
        sg53_unpack_u32_nibbles_h4(q0.x, n00, n01); sg53_unpack_u32_nibbles_h4(q0.y, n02, n03);
        sg53_unpack_u32_nibbles_h4(q1.x, n10, n11); sg53_unpack_u32_nibbles_h4(q1.y, n12, n13);
        sg53_unpack_u32_nibbles_h4(q2.x, n20, n21); sg53_unpack_u32_nibbles_h4(q2.y, n22, n23);
        sg53_unpack_u32_nibbles_h4(q3.x, n30, n31); sg53_unpack_u32_nibbles_h4(q3.y, n32, n33);

        for (uint m = 0u; m < M; ++m) {
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + u2_idx * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float4 xs4 = (float4(x0) + float4(x1)) + (float4(x2) + float4(x3));
            const float  xs  = (xs4.x + xs4.y) + (xs4.z + xs4.w);

            out_m[m].x = fma(sc.x, sg53_dot_h4x4(n00, n01, n02, n03, x0, x1, x2, x3), fma(bi.x, xs, out_m[m].x));
            out_m[m].y = fma(sc.y, sg53_dot_h4x4(n10, n11, n12, n13, x0, x1, x2, x3), fma(bi.y, xs, out_m[m].y));
            out_m[m].z = fma(sc.z, sg53_dot_h4x4(n20, n21, n22, n23, x0, x1, x2, x3), fma(bi.z, xs, out_m[m].z));
            out_m[m].w = fma(sc.w, sg53_dot_h4x4(n30, n31, n32, n33, x0, x1, x2, x3), fma(bi.w, xs, out_m[m].w));
        }
    }

    for (uint m = 0u; m < M; ++m) {
        out_m[m] = float4(simd_sum(out_m[m].x), simd_sum(out_m[m].y), simd_sum(out_m[m].z), simd_sum(out_m[m].w));
    }
}

template <uint M_FIX>
inline void sg53_q4_gemv2_rows_fixed(
    device const uint2* w_u2,
    device const half*  scales,
    device const half*  biases,
    constant half*      x_in,
    uint row0,
    uint rows_logical,
    uint cols,
    uint lane,
    thread float2 (&out_m)[M_FIX]
) {
    const uint u2_per_row  = cols >> 4u;
    const uint grp_per_row = cols >> 6u;

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) out_m[m] = float2(0.0f);

    const uint r0 = min(row0 + 0u, rows_logical - 1u);
    const uint r1 = min(row0 + 1u, rows_logical - 1u);

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 q0      = w_u2[ulong(r0) * u2_per_row + u2_idx];
        const uint2 q1      = w_u2[ulong(r1) * u2_per_row + u2_idx];
        const float2 sc     = float2(float(scales[ulong(r0) * grp_per_row + grp_idx]),
                                     float(scales[ulong(r1) * grp_per_row + grp_idx]));
        const float2 bi     = float2(float(biases[ulong(r0) * grp_per_row + grp_idx]),
                                     float(biases[ulong(r1) * grp_per_row + grp_idx]));

        half4 n00, n01, n02, n03, n10, n11, n12, n13;
        sg53_unpack_u32_nibbles_h4(q0.x, n00, n01); sg53_unpack_u32_nibbles_h4(q0.y, n02, n03);
        sg53_unpack_u32_nibbles_h4(q1.x, n10, n11); sg53_unpack_u32_nibbles_h4(q1.y, n12, n13);

        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            constant half4* xp = reinterpret_cast<constant half4*>(x_in + ulong(m) * cols + u2_idx * 16u);
            const half4  x0  = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float4 xs4 = (float4(x0) + float4(x1)) + (float4(x2) + float4(x3));
            const float  xs  = (xs4.x + xs4.y) + (xs4.z + xs4.w);

            out_m[m].x = fma(sc.x, sg53_dot_h4x4(n00, n01, n02, n03, x0, x1, x2, x3), fma(bi.x, xs, out_m[m].x));
            out_m[m].y = fma(sc.y, sg53_dot_h4x4(n10, n11, n12, n13, x0, x1, x2, x3), fma(bi.y, xs, out_m[m].y));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        out_m[m] = float2(simd_sum(out_m[m].x), simd_sum(out_m[m].y));
    }
}

template <uint M_FIX>
inline void sg53_store_gemv2(
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
            if (out_mode == SG53_OUT_F16) {
                y_f16[base + 0u] = sg53_clamp_f16(res[m].x);
                y_f16[base + 1u] = sg53_clamp_f16(res[m].y);
            } else if (out_mode == SG53_OUT_F32) {
                *reinterpret_cast<device float2*>(y_f32 + base) = res[m];
            } else {
                device float2* dst = reinterpret_cast<device float2*>(y_f32 + base);
                *dst = *dst + res[m];
            }
        } else {
            if (out_mode == SG53_OUT_F16)      y_f16[base] = sg53_clamp_f16(res[m].x);
            else if (out_mode == SG53_OUT_F32) y_f32[base] = res[m].x;
            else                               y_f32[base] += res[m].x;
        }
    }
}

struct Sg53GemvBatchParams {
    uint M;
    uint rows;
    uint cols;
    uint out_mode; // SG53_OUT_*
};

// Compile-time M-specialized decode GEMV (`M = 1..5`) via Metal function_constant(0):
// Eliminates `case 6..8` register bloat, switches `x_in` to `device const` (avoids >64KiB constant cache thrash
// on `cols=8192/16384` at `M=5`), and uses dual-rate FP16 `half4` (`sg53_dot_h4x4`) -> half the registers at M=5!
constant uint kSg53GemvM [[function_constant(0)]];

kernel void sg53_q4_gemv_spec(
    device const uint2*           w_u2   [[buffer(0)]],
    device const half*            scales [[buffer(1)]],
    device const half*            biases [[buffer(2)]],
    device const half*            x_in   [[buffer(3)]], // [M, cols]
    device half*                  y_f16  [[buffer(4)]], // [M, rows]
    device float*                 y_f32  [[buffer(5)]], // [M, rows]
    constant Sg53GemvBatchParams& p      [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],       // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 4u + sgid) * 2u;
    if (row0 >= p.rows) return;

    const uint u2_per_row  = p.cols >> 4u;
    const uint grp_per_row = p.cols >> 6u;
    const uint r0 = min(row0 + 0u, p.rows - 1u);
    const uint r1 = min(row0 + 1u, p.rows - 1u);

    float2 acc[8];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < kSg53GemvM; ++m) acc[m] = float2(0.0f);

    device const uint2* r0_w = w_u2 + ulong(r0) * u2_per_row;
    device const uint2* r1_w = w_u2 + ulong(r1) * u2_per_row;
    device const half*  r0_s = scales + ulong(r0) * grp_per_row;
    device const half*  r1_s = scales + ulong(r1) * grp_per_row;
    device const half*  r0_b = biases + ulong(r0) * grp_per_row;
    device const half*  r1_b = biases + ulong(r1) * grp_per_row;

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint   grp = u2_idx >> 2u;
        const uint2  q0  = r0_w[u2_idx];
        const uint2  q1  = r1_w[u2_idx];
        const float2 sc  = float2(float(r0_s[grp]), float(r1_s[grp]));
        const float2 bi  = float2(float(r0_b[grp]), float(r1_b[grp]));

        half4 n00, n01, n02, n03, n10, n11, n12, n13;
        sg53_unpack_u32_nibbles_h4(q0.x, n00, n01); sg53_unpack_u32_nibbles_h4(q0.y, n02, n03);
        sg53_unpack_u32_nibbles_h4(q1.x, n10, n11); sg53_unpack_u32_nibbles_h4(q1.y, n12, n13);

        #pragma clang loop unroll(full)
        for (uint m = 0u; m < kSg53GemvM; ++m) {
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + u2_idx * 16u);
            const half4  x0  = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float4 xs4 = (float4(x0) + float4(x1)) + (float4(x2) + float4(x3));
            const float  xs  = (xs4.x + xs4.y) + (xs4.z + xs4.w);

            acc[m].x = fma(sc.x, sg53_dot_h4x4(n00, n01, n02, n03, x0, x1, x2, x3), fma(bi.x, xs, acc[m].x));
            acc[m].y = fma(sc.y, sg53_dot_h4x4(n10, n11, n12, n13, x0, x1, x2, x3), fma(bi.y, xs, acc[m].y));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < kSg53GemvM; ++m) {
        const float2 res(simd_sum(acc[m].x), simd_sum(acc[m].y));
        if (lane == 0u) {
            const ulong base = ulong(m) * p.rows + row0;
            if (row0 + 2u <= p.rows) {
                if (p.out_mode == SG53_OUT_F16) {
                    y_f16[base + 0u] = sg53_clamp_f16(res.x);
                    y_f16[base + 1u] = sg53_clamp_f16(res.y);
                } else if (p.out_mode == SG53_OUT_F32) {
                    *reinterpret_cast<device float2*>(y_f32 + base) = res;
                } else {
                    device float2* dst = reinterpret_cast<device float2*>(y_f32 + base);
                    *dst = *dst + res;
                }
            } else {
                if (p.out_mode == SG53_OUT_F16)      y_f16[base] = sg53_clamp_f16(res.x);
                else if (p.out_mode == SG53_OUT_F32) y_f32[base] = res.x;
                else                                 y_f32[base] += res.x;
            }
        }
    }
}

// 100%-Lane-Utilization Q4_A Decode GEMV for `cols == 128` (`lin_f_b_proj`, `lin_g_b_proj`: 68x/tok!):
// 4 quarter-warps of 8 lanes per SG -> 8 rows/SG = 32 rows/TG (0 loop iters, 4x fewer TGs, 0 idle lanes!).
kernel void sg53_q4_gemv_k128_spec(
    device const uint2*           w_u2   [[buffer(0)]],
    device const half*            scales [[buffer(1)]],
    device const half*            biases [[buffer(2)]],
    device const half*            x_in   [[buffer(3)]], // [M, 128]
    device half*                  y_f16  [[buffer(4)]], // [M, rows]
    device float*                 y_f32  [[buffer(5)]], // [M, rows]
    constant Sg53GemvBatchParams& p      [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],       // 0..3
    uint lane [[thread_index_in_simdgroup]]             // 0..31
) {
    const uint qtr  = lane >> 3u;                       // 0..3 (4 row-pairs per SG!)
    const uint sub  = lane & 7u;                        // 0..7 (8 uint2 = 128 cols!)
    const uint row0 = (tgid * 16u + sgid * 4u + qtr) * 2u;
    if (row0 >= p.rows) return;

    const uint grp = sub >> 2u;                         // 0..1
    const uint r0  = min(row0 + 0u, p.rows - 1u);
    const uint r1  = min(row0 + 1u, p.rows - 1u);

    const uint2  q0 = w_u2[ulong(r0) * 8ul + sub];
    const uint2  q1 = w_u2[ulong(r1) * 8ul + sub];
    const float2 sc = float2(float(scales[ulong(r0) * 2ul + grp]), float(scales[ulong(r1) * 2ul + grp]));
    const float2 bi = float2(float(biases[ulong(r0) * 2ul + grp]), float(biases[ulong(r1) * 2ul + grp]));

    half4 n00, n01, n02, n03, n10, n11, n12, n13;
    sg53_unpack_u32_nibbles_h4(q0.x, n00, n01); sg53_unpack_u32_nibbles_h4(q0.y, n02, n03);
    sg53_unpack_u32_nibbles_h4(q1.x, n10, n11); sg53_unpack_u32_nibbles_h4(q1.y, n12, n13);

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < kSg53GemvM; ++m) {
        device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(m) * 128ul + sub * 16u);
        const half4  x0  = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        const float4 xs4 = (float4(x0) + float4(x1)) + (float4(x2) + float4(x3));
        const float  xs  = (xs4.x + xs4.y) + (xs4.z + xs4.w);

        float ax = fma(sc.x, sg53_dot_h4x4(n00, n01, n02, n03, x0, x1, x2, x3), bi.x * xs);
        float ay = fma(sc.y, sg53_dot_h4x4(n10, n11, n12, n13, x0, x1, x2, x3), bi.y * xs);

        // Exact 8-lane butterfly reduction within quarter-warp (deterministic across M=1..5!)
        ax += simd_shuffle_xor(ax, 4u); ay += simd_shuffle_xor(ay, 4u);
        ax += simd_shuffle_xor(ax, 2u); ay += simd_shuffle_xor(ay, 2u);
        ax += simd_shuffle_xor(ax, 1u); ay += simd_shuffle_xor(ay, 1u);

        if (sub == 0u) {
            const ulong base = ulong(m) * p.rows + row0;
            if (row0 + 2u <= p.rows) {
                if (p.out_mode == SG53_OUT_F16) {
                    y_f16[base + 0u] = sg53_clamp_f16(ax);
                    y_f16[base + 1u] = sg53_clamp_f16(ay);
                } else if (p.out_mode == SG53_OUT_F32) {
                    *reinterpret_cast<device float2*>(y_f32 + base) = float2(ax, ay);
                } else {
                    device float2* dst = reinterpret_cast<device float2*>(y_f32 + base);
                    *dst = *dst + float2(ax, ay);
                }
            } else {
                if (p.out_mode == SG53_OUT_F16)      y_f16[base] = sg53_clamp_f16(ax);
                else if (p.out_mode == SG53_OUT_F32) y_f32[base] = ax;
                else                                 y_f32[base] += ax;
            }
        }
    }
}

// Compile-time M-specialized FP16 GEMV (`M = 1..5`)
kernel void sg53_f16_gemv_spec(
    device const half*            w_f16 [[buffer(0)]],
    device const half*            x_in  [[buffer(1)]],
    device half*                  y_f16 [[buffer(2)]],
    device float*                 y_f32 [[buffer(3)]],
    constant Sg53GemvBatchParams& p     [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 4u + sgid) * 2u;
    if (row0 >= p.rows) return;
    const uint c4_cnt = p.cols >> 2u;
    const uint r0 = min(row0 + 0u, p.rows - 1u);
    const uint r1 = min(row0 + 1u, p.rows - 1u);
    device const half4* w0 = reinterpret_cast<device const half4*>(w_f16 + ulong(r0) * p.cols);
    device const half4* w1 = reinterpret_cast<device const half4*>(w_f16 + ulong(r1) * p.cols);
    float2 acc[8];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < kSg53GemvM; ++m) acc[m] = float2(0.0f);
    for (uint c4 = lane; c4 < c4_cnt; c4 += 32u) {
        const float4 fw0 = float4(w0[c4]);
        const float4 fw1 = float4(w1[c4]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < kSg53GemvM; ++m) {
            const float4 fx = float4(reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols)[c4]);
            acc[m].x = fma(fw0.x, fx.x, fma(fw0.y, fx.y, fma(fw0.z, fx.z, fma(fw0.w, fx.w, acc[m].x))));
            acc[m].y = fma(fw1.x, fx.x, fma(fw1.y, fx.y, fma(fw1.z, fx.z, fma(fw1.w, fx.w, acc[m].y))));
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < kSg53GemvM; ++m) {
        const float2 res(simd_sum(acc[m].x), simd_sum(acc[m].y));
        if (lane == 0u) {
            const ulong base = ulong(m) * p.rows + row0;
            if (row0 + 2u <= p.rows) {
                if (p.out_mode == SG53_OUT_F16) {
                    y_f16[base + 0u] = sg53_clamp_f16(res.x);
                    y_f16[base + 1u] = sg53_clamp_f16(res.y);
                } else if (p.out_mode == SG53_OUT_F32) {
                    *reinterpret_cast<device float2*>(y_f32 + base) = res;
                } else {
                    device float2* dst = reinterpret_cast<device float2*>(y_f32 + base);
                    *dst = *dst + res;
                }
            } else {
                if (p.out_mode == SG53_OUT_F16)      y_f16[base] = sg53_clamp_f16(res.x);
                else if (p.out_mode == SG53_OUT_F32) y_f32[base] = res.x;
                else                                 y_f32[base] += res.x;
            }
        }
    }
}

// 4-Way Split-K FP32 GEMV for `hc_fn [24, 16384]` (`96` TGs saturate all 40 M5 Max cores -> ~2x BW!)
// Grid: `(rows, 4, 1)`, `256` threads/TG. Writes `y_part[m * rows * 4 + row * 4 + split]`.
kernel void sg53_f32_gemv_splitk4_spec(
    device const float*           w_f32  [[buffer(0)]], // [rows, cols]
    device const float*           x_f32  [[buffer(1)]], // [M, cols]
    device float*                 y_part [[buffer(2)]], // [M, rows, 4]
    constant Sg53GemvBatchParams& p      [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],         // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],      // 0..7
    uint  lane [[thread_index_in_simdgroup]]            // 0..31
) {
    threadgroup float tg_red[8][8];
    const uint row = tgid.x;
    const uint sp  = tgid.y; // 0..3
    if (row >= p.rows || sp >= 4u) return;

    const uint c4_tot   = p.cols >> 2u;
    const uint c4_chunk = c4_tot >> 2u;
    const uint c4_beg   = sp * c4_chunk;
    const uint c4_end   = (sp == 3u) ? c4_tot : (c4_beg + c4_chunk);
    device const float4* w4 = reinterpret_cast<device const float4*>(w_f32 + ulong(row) * p.cols);

    float acc[8];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < kSg53GemvM; ++m) acc[m] = 0.0f;

    for (uint c4 = c4_beg + tid; c4 < c4_end; c4 += 256u) {
        const float4 wv = w4[c4];
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < kSg53GemvM; ++m) {
            const float4 xv = reinterpret_cast<device const float4*>(x_f32 + ulong(m) * p.cols)[c4];
            acc[m] = fma(wv.x, xv.x, fma(wv.y, xv.y, fma(wv.z, xv.z, fma(wv.w, xv.w, acc[m]))));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < kSg53GemvM; ++m) {
        float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < kSg53GemvM; ++m) {
            float v   = (lane < 8u) ? tg_red[m][lane] : 0.0f;
            float tot = simd_sum(v);
            if (lane == 0u) {
                y_part[(ulong(m) * p.rows + row) * 4ul + sp] = tot;
            }
        }
    }
}

// Zero-spill `2` rows/SG (`8` rows/TG) Q4 GEMV (`M = 1..8` compile-time unrolled, `constant` x-cache path!)
kernel void sg53_q4_gemv_batch(
    device const uint2*           w_u2   [[buffer(0)]],
    device const half*            scales [[buffer(1)]],
    device const half*            biases [[buffer(2)]],
    constant half*                x_in   [[buffer(3)]], // [M, cols]
    device half*                  y_f16  [[buffer(4)]], // [M, rows]
    device float*                 y_f32  [[buffer(5)]], // [M, rows]
    constant Sg53GemvBatchParams& p      [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],       // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 4u + sgid) * 2u;
    if (row0 >= p.rows) return;

    #define SG53_Q4_CASE(MF) { \
        float2 res[MF]; \
        sg53_q4_gemv2_rows_fixed<MF>(w_u2, scales, biases, x_in, row0, p.rows, p.cols, lane, res); \
        if (lane == 0u) sg53_store_gemv2<MF>(y_f16, y_f32, row0, p.rows, p.out_mode, res); \
        break; \
    }
    switch (p.M) {
        case 1u: SG53_Q4_CASE(1u)
        case 2u: SG53_Q4_CASE(2u)
        case 3u: SG53_Q4_CASE(3u)
        case 4u: SG53_Q4_CASE(4u)
        case 5u: SG53_Q4_CASE(5u)
        case 6u: SG53_Q4_CASE(6u)
        case 7u: SG53_Q4_CASE(7u)
        case 8u: SG53_Q4_CASE(8u)
        default: break;
    }
    #undef SG53_Q4_CASE
}

template <uint M_FIX>
inline void sg53_f16_gemv2_fixed(
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
    const uint c4_cnt = cols >> 2u;
    const uint r0 = min(row0 + 0u, rows - 1u);
    const uint r1 = min(row0 + 1u, rows - 1u);

    device const half4* w0 = reinterpret_cast<device const half4*>(w_f16 + ulong(r0) * cols);
    device const half4* w1 = reinterpret_cast<device const half4*>(w_f16 + ulong(r1) * cols);

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint c4 = lane; c4 < c4_cnt; c4 += 32u) {
        const float4 fw0 = float4(w0[c4]);
        const float4 fw1 = float4(w1[c4]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            const float4 fx = float4(reinterpret_cast<device const half4*>(x_in + ulong(m) * cols)[c4]);
            acc[m].x = fma(fw0.x, fx.x, fma(fw0.y, fx.y, fma(fw0.z, fx.z, fma(fw0.w, fx.w, acc[m].x))));
            acc[m].y = fma(fw1.x, fx.x, fma(fw1.y, fx.y, fma(fw1.z, fx.z, fma(fw1.w, fx.w, acc[m].y))));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        acc[m] = float2(simd_sum(acc[m].x), simd_sum(acc[m].y));
    }
    if (lane == 0u) sg53_store_gemv2<M_FIX>(y_f16, y_f32, row0, rows, out_mode, acc);
}

kernel void sg53_f16_gemv_batch(
    device const half*            w_f16 [[buffer(0)]], // [rows, cols]
    device const half*            x_in  [[buffer(1)]], // [M, cols]
    device half*                  y_f16 [[buffer(2)]], // [M, rows]
    device float*                 y_f32 [[buffer(3)]], // [M, rows]
    constant Sg53GemvBatchParams& p     [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],      // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 4u + sgid) * 2u;
    if (row0 >= p.rows) return;

    #define SG53_F16_CASE(MF) { sg53_f16_gemv2_fixed<MF>(w_f16, x_in, y_f16, y_f32, row0, p.rows, p.cols, p.out_mode, lane); break; }
    switch (p.M) {
        case 1u: SG53_F16_CASE(1u)
        case 2u: SG53_F16_CASE(2u)
        case 3u: SG53_F16_CASE(3u)
        case 4u: SG53_F16_CASE(4u)
        case 5u: SG53_F16_CASE(5u)
        case 6u: SG53_F16_CASE(6u)
        case 7u: SG53_F16_CASE(7u)
        case 8u: SG53_F16_CASE(8u)
        default: break;
    }
    #undef SG53_F16_CASE
}

// High-Occupancy Wide-K FP32 Matrix-Vector / Tile-8 Multiply (`1` row/TG of `256` threads = `8` SGs):
// Compile-time `mb = 1..8` unrolled (zero `acc[8]` spill) for `hc_fn [24, 16384]` and `indexer_compress_gate [128, 4096]`.
// Grid: `(rows, ceil(M / 8), 1)`, `256` threads/TG
template <uint MB_FIX>
inline void sg53_f32_gemv_fixed(
    device const float* w_f32,
    device const float* x_f32,
    device float* y_f32,
    threadgroup float (&tg_red)[8][8],
    uint row,
    uint m0,
    uint rows,
    uint cols,
    uint out_mode,
    uint tid,
    uint sgid,
    uint lane
) {
    const uint c4_cnt = cols >> 2u;
    device const float4* w4 = reinterpret_cast<device const float4*>(w_f32 + ulong(row) * cols);

    float acc[MB_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < MB_FIX; ++m) acc[m] = 0.0f;

    for (uint c4 = tid; c4 < c4_cnt; c4 += 256u) {
        const float4 wv = w4[c4];
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < MB_FIX; ++m) {
            const float4 xv = reinterpret_cast<device const float4*>(x_f32 + ulong(m0 + m) * cols)[c4];
            acc[m] = fma(wv.x, xv.x, fma(wv.y, xv.y, fma(wv.z, xv.z, fma(wv.w, xv.w, acc[m]))));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < MB_FIX; ++m) {
        float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < MB_FIX; ++m) {
            float v   = (lane < 8u) ? tg_red[m][lane] : 0.0f;
            float tot = simd_sum(v);
            if (lane == 0u) {
                ulong idx = ulong(m0 + m) * rows + row;
                if (out_mode == SG53_OUT_RES_F32) y_f32[idx] += tot;
                else                              y_f32[idx]  = tot;
            }
        }
    }
}

kernel void sg53_f32_gemv_batch(
    device const float*           w_f32 [[buffer(0)]], // [rows, cols]
    device const float*           x_f32 [[buffer(1)]], // [M, cols]
    device float*                 y_f32 [[buffer(2)]], // [M, rows]
    constant Sg53GemvBatchParams& p     [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],        // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],     // 0..7
    uint  lane [[thread_index_in_simdgroup]]           // 0..31
) {
    threadgroup float tg_red[8][8];
    const uint row = tgid.x;
    const uint m0  = tgid.y * 8u;
    if (row >= p.rows || m0 >= p.M) return;

    const uint mb = min(8u, p.M - m0);
    #define SG53_F32_CASE(MB) { sg53_f32_gemv_fixed<MB>(w_f32, x_f32, y_f32, tg_red, row, m0, p.rows, p.cols, p.out_mode, tid, sgid, lane); break; }
    switch (mb) {
        case 1u: SG53_F32_CASE(1u)
        case 2u: SG53_F32_CASE(2u)
        case 3u: SG53_F32_CASE(3u)
        case 4u: SG53_F32_CASE(4u)
        case 5u: SG53_F32_CASE(5u)
        case 6u: SG53_F32_CASE(6u)
        case 7u: SG53_F32_CASE(7u)
        case 8u: SG53_F32_CASE(8u)
        default: break;
    }
    #undef SG53_F32_CASE
}

/* ============================================================================
 * 4. Batched Prefill Metal 4 `NAX` (`mpp::tensor_ops::matmul2d`) Q4 & FP16 GEMM
 * ============================================================================ */

struct Sg53NaxGemmParams {
    uint M_valid;
    uint M_padded;
    uint N_logical;
    uint N_padded;  // Multiple of 64 for Q4
    uint K;         // Multiple of 64
    uint out_mode;  // SG53_OUT_*
    uint pad0;
    uint pad1;
};

#if defined(__HAVE_TENSOR__)
// ---- Q4_A (gs=64) Prefill NAX: `uint2` 8B loads + Dual-Rate FP16 unpack + `BK=128` (dodges K=64 cliff!) ----
template <int BM, int BK, int SG>
inline void sg53_q4_gemm_nax_impl(
    device const uint2*         w_u2,
    device const half*          scales,
    device const half*          biases,
    device const half*          x_in,
    device half*                y_f16,
    device float*               y_f32,
    constant Sg53NaxGemmParams& p,
    threadgroup half            (&W_tile)[64][BK],
    uint2                       tgid,
    uint                        tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    device half* x_mut = const_cast<device half*>(x_in);
    tensor<device half, dextents<int, 2>, tensor_inline> tX(x_mut, dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup half, extents<int, BK, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, BK, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<BK, BM>(0, int(m0));
    auto cT  = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sX0)>, sg53_strip_as<decltype(tW)>, float>();

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
            sg53_unpack_u32_nibbles_h4(qv.x, n0h, n1h);
            sg53_unpack_u32_nibbles_h4(qv.y, n2h, n3h);

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

    if (p.out_mode == SG53_OUT_F32 && (p.N_logical % 64u) == 0u && (m0 + uint(BM)) <= p.M_valid) {
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
            if (n_glob < p.N_logical) {
                if (p.out_mode == SG53_OUT_F16) {
                    if (m_glob < p.M_padded) {
                        y_f16[ulong(m_glob) * p.N_logical + n_glob] =
                            (m_glob < p.M_valid) ? sg53_clamp_f16(cT[i]) : half(0.0h);
                    }
                } else if (m_glob < p.M_valid) {
                    ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                    if (p.out_mode == SG53_OUT_F32) y_f32[out_idx] = cT[i];
                    else                            y_f32[out_idx] += cT[i];
                }
            }
        }
    }
}

// ---- Direct-Device `static_slice` `K=dynamic_extent` FP16 NAX (`~58 TF`, 0 TG mem, 0 barriers!) ----
template <int BM, int BN, int SG>
inline void sg53_f16_gemm_nax_direct_impl(
    device const half*          w_f16,
    device const half*          x_in,
    device half*                y_f16,
    device float*               y_f32,
    constant Sg53NaxGemmParams& p,
    uint2                       tgid
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

    if (p.out_mode == SG53_OUT_F32 && (p.N_logical % uint(BN)) == 0u && (m0 + uint(BM)) <= p.M_valid) {
        tensor<device float, dextents<int, 2>, tensor_inline> tY(y_f32, dextents<int, 2>(int(p.N_logical), int(p.M_padded)));
        auto sY = tY.template slice<BN, BM>(int(n0), int(m0));
        op.run(sX, sW, sY);
        return;
    }

    auto cT = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sX)>, sg53_strip_as<decltype(sW)>, float>();
    op.run(sX, sW, cT);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (n_glob < p.N_logical) {
                if (p.out_mode == SG53_OUT_F16) {
                    if (m_glob < p.M_padded) {
                        y_f16[ulong(m_glob) * p.N_logical + n_glob] =
                            (m_glob < p.M_valid) ? sg53_clamp_f16(cT[i]) : half(0.0h);
                    }
                } else if (m_glob < p.M_valid) {
                    ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                    if (p.out_mode == SG53_OUT_F32) y_f32[out_idx] = cT[i];
                    else                            y_f32[out_idx] += cT[i];
                }
            }
        }
    }
}
#endif

// `BM=16, BK=128, SG=2` (`64` threads/TG -> 2x occupancy!)
kernel void sg53_q4_gemm_nax_m16(
    device const uint2*         w_u2   [[buffer(0)]],
    device const half*          scales [[buffer(1)]],
    device const half*          biases [[buffer(2)]],
    device const half*          x_in   [[buffer(3)]],
    device half*                y_f16  [[buffer(4)]],
    device float*               y_f32  [[buffer(5)]],
    constant Sg53NaxGemmParams& p      [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sg53_q4_gemm_nax_impl<16, 128, 2>(w_u2, scales, biases, x_in, y_f16, y_f32, p, W_tile, tgid, tid);
#else
    (void)w_u2; (void)scales; (void)biases; (void)x_in; (void)y_f16; (void)y_f32; (void)p; (void)W_tile; (void)tgid; (void)tid;
#endif
}

// `BM=64, BK=128, SG=4` (`128` threads/TG)
kernel void sg53_q4_gemm_nax_m64(
    device const uint2*         w_u2   [[buffer(0)]],
    device const half*          scales [[buffer(1)]],
    device const half*          biases [[buffer(2)]],
    device const half*          x_in   [[buffer(3)]],
    device half*                y_f16  [[buffer(4)]],
    device float*               y_f32  [[buffer(5)]],
    constant Sg53NaxGemmParams& p      [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sg53_q4_gemm_nax_impl<64, 128, 4>(w_u2, scales, biases, x_in, y_f16, y_f32, p, W_tile, tgid, tid);
#else
    (void)w_u2; (void)scales; (void)biases; (void)x_in; (void)y_f16; (void)y_f32; (void)p; (void)W_tile; (void)tgid; (void)tid;
#endif
}

// Synthetic `K=64` fallback (`BM=16, BK=64, SG=4`, `128` threads/TG)
kernel void sg53_q4_gemm_nax_bk64_m16(
    device const uint2*         w_u2   [[buffer(0)]],
    device const half*          scales [[buffer(1)]],
    device const half*          biases [[buffer(2)]],
    device const half*          x_in   [[buffer(3)]],
    device half*                y_f16  [[buffer(4)]],
    device float*               y_f32  [[buffer(5)]],
    constant Sg53NaxGemmParams& p      [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sg53_q4_gemm_nax_impl<16, 64, 4>(w_u2, scales, biases, x_in, y_f16, y_f32, p, W_tile, tgid, tid);
#else
    (void)w_u2; (void)scales; (void)biases; (void)x_in; (void)y_f16; (void)y_f32; (void)p; (void)W_tile; (void)tgid; (void)tid;
#endif
}

// Direct-device FP16 NAX `BM=16, BN=64, SG=2` (`64` threads/TG, 0 TG mem)
kernel void sg53_f16_gemm_nax_m16(
    device const half*          w_f16 [[buffer(0)]],
    device const half*          x_in  [[buffer(1)]],
    device half*                y_f16 [[buffer(2)]],
    device float*               y_f32 [[buffer(3)]],
    constant Sg53NaxGemmParams& p     [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sg53_f16_gemm_nax_direct_impl<16, 64, 2>(w_f16, x_in, y_f16, y_f32, p, tgid);
#else
    (void)w_f16; (void)x_in; (void)y_f16; (void)y_f32; (void)p; (void)tgid;
#endif
}

// Direct-device FP16 NAX `BM=64, BN=64, SG=4` (`128` threads/TG, 0 TG mem)
kernel void sg53_f16_gemm_nax_m64(
    device const half*          w_f16 [[buffer(0)]],
    device const half*          x_in  [[buffer(1)]],
    device half*                y_f16 [[buffer(2)]],
    device float*               y_f32 [[buffer(3)]],
    constant Sg53NaxGemmParams& p     [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sg53_f16_gemm_nax_direct_impl<64, 64, 4>(w_f16, x_in, y_f16, y_f32, p, tgid);
#else
    (void)w_f16; (void)x_in; (void)y_f16; (void)y_f32; (void)p; (void)tgid;
#endif
}

/* ============================================================================
 * 4b. Opt-In Prefill (`M >= 32`) Metal 4.1 `INT8×INT8→INT32` NAX `Q4_A` GEMM
 *     (`70–90 TOPS`, `1.4x–1.63x` vs FP16 NAX, `cos >= 0.9995`, 0 extra weight RAM!)
 * ============================================================================ */

struct Sg53QuantI8Params {
    uint M_valid;
    uint M_padded;
    uint K;       // Multiple of 64
    uint pad0;
};

// Dynamic per-row symmetric INT8 activation quantization + exact per-group-64 `x_sums` (`[M_padded, K/64]`)
// Grid: `(M_padded, 1, 1)`, `256` threads/TG (`8` SGs)
kernel void sg53_quant_f16_to_i8_row(
    device const half*        x_in   [[buffer(0)]], // [M_padded, K]
    device int8_t*            x_i8   [[buffer(1)]], // [M_padded, K]
    device half*              x_rsc  [[buffer(2)]], // [M_padded]
    device half*              x_sums [[buffer(3)]], // [M_padded, K/64]
    constant Sg53QuantI8Params& p    [[buffer(4)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],      // 0..255
    uint sgid [[simdgroup_index_in_threadgroup]],   // 0..7
    uint lane [[thread_index_in_simdgroup]]         // 0..31
) {
    if (m >= p.M_padded) return;
    const uint grp_cnt    = p.K >> 6u;
    const uint grp_stride = (grp_cnt + 63u) & ~63u;
    const uint k4_cnt     = p.K >> 2u;

    if (m >= p.M_valid) {
        if (tid == 0u) x_rsc[m] = half(0.0h);
        for (uint g = tid; g < grp_stride; g += 256u) {
            x_sums[ulong(m) * grp_stride + g] = half(0.0h);
        }
        device char4* dst4 = reinterpret_cast<device char4*>(x_i8 + ulong(m) * p.K);
        for (uint i = tid; i < k4_cnt; i += 256u) dst4[i] = char4(0);
        return;
    }

    device const half4* src4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.K);
    float amax = 0.0f;
    for (uint i = tid; i < k4_cnt; i += 256u) {
        half4 v = abs(src4[i]);
        amax = max(amax, max(max(float(v.x), float(v.y)), max(float(v.z), float(v.w))));
    }
    amax = simd_max(amax);
    threadgroup float tg_max[8];
    if (lane == 0u) tg_max[sgid] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float gmax = simd_max((lane < 8u) ? tg_max[lane] : 0.0f);
        if (lane == 0u) tg_max[0] = gmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float row_amax = tg_max[0];
    const float rscale   = (row_amax > 1e-12f) ? (row_amax / 127.0f) : 0.0f;
    const float inv_rsc  = (row_amax > 1e-12f) ? (127.0f / row_amax) : 0.0f;
    if (tid == 0u) x_rsc[m] = half(rscale);

    device const half2* src2 = reinterpret_cast<device const half2*>(x_in + ulong(m) * p.K);
    device char2*       dst2 = reinterpret_cast<device char2*>(x_i8 + ulong(m) * p.K);
    for (uint g = sgid; g < grp_cnt; g += 8u) {
        const half2 v  = src2[g * 32u + lane];
        const float f0 = float(v.x), f1 = float(v.y);
        const float gs = simd_sum(f0 + f1);
        if (lane == 0u) x_sums[ulong(m) * grp_stride + g] = sg53_clamp_f16(gs);
        dst2[g * 32u + lane] = char2(int(rint(clamp(f0 * inv_rsc, -127.0f, 127.0f))),
                                     int(rint(clamp(f1 * inv_rsc, -127.0f, 127.0f))));
    }
    for (uint g = grp_cnt + tid; g < grp_stride; g += 256u) {
        x_sums[ulong(m) * grp_stride + g] = half(0.0h);
    }
}

#if defined(__HAVE_TENSOR__)
union Sg53Q4NaxI8Smem {
    int8_t w_i8[64][128]; // 8 KiB
    half   b_ef[64][64];  // 8 KiB (reused for fused exact `b_eff = biases + 8*scales` epilogue!)
};

template <int BM>
inline void sg53_q4_gemm_nax_i8_impl(
    device const uint2*         w_u2,
    device const half*          scales,
    device const half*          biases,
    device const int8_t*        x_i8,
    device const half*          x_rsc,
    device const half*          x_sums,
    device half*                y_f16,
    device float*               y_f32,
    constant Sg53NaxGemmParams& p,
    threadgroup Sg53Q4NaxI8Smem& smem,
    threadgroup float*          tg_wsc,
    uint2                       tgid,
    uint                        tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    const uint grp_per_row = p.K >> 6u;
    const uint u2_per_row  = p.K >> 4u;
    const uint kt_cnt      = p.K >> 7u;

    if (tid < 64u) {
        const ulong rg = ulong(n0 + tid);
        float smax = 0.0f;
        if ((grp_per_row & 3u) == 0u) {
            device const half4* sp = reinterpret_cast<device const half4*>(scales + rg * grp_per_row);
            half4 mx(0.0h);
            for (uint g4 = 0u; g4 < (grp_per_row >> 2u); ++g4) mx = max(mx, abs(sp[g4]));
            smax = max(max(float(mx.x), float(mx.y)), max(float(mx.z), float(mx.w)));
        } else {
            for (uint g = 0u; g < grp_per_row; ++g) smax = max(smax, abs(float(scales[rg * grp_per_row + g])));
        }
        tg_wsc[tid] = smax * (8.0f / 127.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tX(
        const_cast<device int8_t*>(x_i8), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup int8_t, extents<int, 128, 64>, tensor_inline> tW(
        &smem.w_i8[0][0], extents<int, 128, 64>());
    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto sX0 = tX.template slice<128, BM>(0, int(m0));
    auto cI  = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sX0)>, sg53_strip_as<decltype(tW)>, int32_t>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cI.get_capacity(); ++i) {
        if (cI.is_valid_element(i)) cI[i] = 0;
    }

    const uint r_grp = tid >> 3u; // 0..15
    const uint sub   = tid & 7u;  // 0..7
    float inv_w[4];
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < 4u; ++q) {
        const float wsc = tg_wsc[q * 16u + r_grp];
        inv_w[q] = (wsc > 1e-12f) ? (1.0f / wsc) : 0.0f;
    }

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        const uint col_u2 = kt * 8u + sub;
        const uint grp    = col_u2 >> 2u;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 4u; ++q) {
            const uint  r_loc = q * 16u + r_grp;
            const ulong rg    = ulong(n0 + r_loc);
            const uint2 qv    = w_u2[rg * u2_per_row + col_u2];
            const short m_q4  = short(rint(float(scales[rg * grp_per_row + grp]) * inv_w[q] * 8.0f));
            const uint  ux = qv.x, uy = qv.y;
            short4 s0(short(ux & 0xFu) - 8, short((ux >> 4u) & 0xFu) - 8, short((ux >> 8u) & 0xFu) - 8, short((ux >> 12u) & 0xFu) - 8);
            short4 s1(short((ux >> 16u) & 0xFu) - 8, short((ux >> 20u) & 0xFu) - 8, short((ux >> 24u) & 0xFu) - 8, short((ux >> 28u) & 0xFu) - 8);
            short4 s2(short(uy & 0xFu) - 8, short((uy >> 4u) & 0xFu) - 8, short((uy >> 8u) & 0xFu) - 8, short((uy >> 12u) & 0xFu) - 8);
            short4 s3(short((uy >> 16u) & 0xFu) - 8, short((uy >> 20u) & 0xFu) - 8, short((uy >> 24u) & 0xFu) - 8, short((uy >> 28u) & 0xFu) - 8);
            threadgroup char4* dst = reinterpret_cast<threadgroup char4*>(&smem.w_i8[r_loc][sub * 16u]);
            dst[0] = char4((s0 * m_q4 + short4(4)) >> 3);
            dst[1] = char4((s1 * m_q4 + short4(4)) >> 3);
            dst[2] = char4((s2 * m_q4 + short4(4)) >> 3);
            dst[3] = char4((s3 * m_q4 + short4(4)) >> 3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX.template slice<128, BM>(int(kt * 128u), int(m0));
        op.run(sX, tW, cI);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write scaled INT32 dot product (`cI` dies before bias epilogue -> zero dual-cT register spill!)
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cI.get_capacity(); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx = cI.get_multidimensional_index(i);
            const uint nl = uint(idx[0]);
            const uint mg = m0 + uint(idx[1]);
            const uint ng = n0 + nl;
            if (ng < p.N_logical) {
                const float val = float(cI[i]) * float(x_rsc[mg]) * tg_wsc[nl];
                if (p.out_mode == SG53_OUT_F16) {
                    if (mg < p.M_padded) {
                        y_f16[ulong(mg) * p.N_logical + ng] = (mg < p.M_valid) ? sg53_clamp_f16(val) : half(0.0h);
                    }
                } else if (mg < p.M_valid) {
                    const ulong oidx = ulong(mg) * p.N_logical + ng;
                    if (p.out_mode == SG53_OUT_F32) y_f32[oidx]  = val;
                    else                            y_f32[oidx] += val;
                }
            }
        }
    }

    // Fused exact `b_eff = biases + 8*scales` bias term (`X_sums[M, G_pad] @ B_eff[N, G_pad]^T`, 64x smaller!)
    const uint grp_stride = (grp_per_row + 63u) & ~63u;
    tensor<device half, dextents<int, 2>, tensor_inline> tXs(
        const_cast<device half*>(x_sums), dextents<int, 2>(int(grp_stride), int(p.M_padded)));
    tensor<threadgroup half, extents<int, 64, 64>, tensor_inline> tB(
        &smem.b_ef[0][0], extents<int, 64, 64>());
    constexpr auto b_desc = matmul2d_descriptor(
        BM, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<b_desc, execution_simdgroups<4>> op_b;

    auto sXs0 = tXs.template slice<64, BM>(0, int(m0));
    auto cB   = op_b.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sXs0)>, sg53_strip_as<decltype(tB)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cB.get_capacity(); ++i) {
        if (cB.is_valid_element(i)) cB[i] = 0.0f;
    }

    const uint g_tiles = (grp_per_row + 63u) >> 6u;
    const uint br = tid >> 1u;        // 0..63
    const uint bh = (tid & 1u) * 32u; // 0 or 32
    const ulong rg_b = ulong(n0 + br);
    for (uint gt = 0u; gt < g_tiles; ++gt) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 32u; j += 4u) {
            const uint gg = gt * 64u + bh + j;
            threadgroup half4* db = reinterpret_cast<threadgroup half4*>(&smem.b_ef[br][bh + j]);
            if (gg + 4u <= grp_per_row) {
                half4 sc = *reinterpret_cast<device const half4*>(scales + rg_b * grp_per_row + gg);
                half4 bi = *reinterpret_cast<device const half4*>(biases + rg_b * grp_per_row + gg);
                *db = fma(sc, half4(8.0h), bi);
            } else {
                half4 tmp(0.0h);
                for (uint c = 0u; c < 4u; ++c) {
                    if (gg + c < grp_per_row) {
                        tmp[c] = fma(scales[rg_b * grp_per_row + gg + c], half(8.0h), biases[rg_b * grp_per_row + gg + c]);
                    }
                }
                *db = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sXs = tXs.template slice<64, BM>(int(gt * 64u), int(m0));
        op_b.run(sXs, tB, cB);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cB.get_capacity(); ++i) {
        if (cB.is_valid_element(i)) {
            auto idx = cB.get_multidimensional_index(i);
            const uint mg = m0 + uint(idx[1]);
            const uint ng = n0 + uint(idx[0]);
            if (mg < p.M_valid && ng < p.N_logical) {
                const ulong oidx = ulong(mg) * p.N_logical + ng;
                if (p.out_mode == SG53_OUT_F16) {
                    y_f16[oidx] = sg53_clamp_f16(float(y_f16[oidx]) + cB[i]);
                } else {
                    y_f32[oidx] += cB[i];
                }
            }
        }
    }
}
#endif

kernel void sg53_q4_gemm_nax_i8_m64(
    device const uint2*         w_u2   [[buffer(0)]],
    device const half*          scales [[buffer(1)]],
    device const half*          biases [[buffer(2)]],
    device const int8_t*        x_i8   [[buffer(3)]],
    device const half*          x_rsc  [[buffer(4)]],
    device const half*          x_sums [[buffer(5)]],
    device half*                y_f16  [[buffer(6)]],
    device float*               y_f32  [[buffer(7)]],
    constant Sg53NaxGemmParams& p      [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup Sg53Q4NaxI8Smem smem;
    threadgroup float tg_wsc[64];
    sg53_q4_gemm_nax_i8_impl<64>(w_u2, scales, biases, x_i8, x_rsc, x_sums, y_f16, y_f32, p, smem, tg_wsc, tgid, tid);
#else
    (void)w_u2; (void)scales; (void)biases; (void)x_i8; (void)x_rsc; (void)x_sums; (void)y_f16; (void)y_f32; (void)p; (void)tgid; (void)tid;
#endif
}

kernel void sg53_q4_gemm_nax_i8_m128(
    device const uint2*         w_u2   [[buffer(0)]],
    device const half*          scales [[buffer(1)]],
    device const half*          biases [[buffer(2)]],
    device const int8_t*        x_i8   [[buffer(3)]],
    device const half*          x_rsc  [[buffer(4)]],
    device const half*          x_sums [[buffer(5)]],
    device half*                y_f16  [[buffer(6)]],
    device float*               y_f32  [[buffer(7)]],
    constant Sg53NaxGemmParams& p      [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup Sg53Q4NaxI8Smem smem;
    threadgroup float tg_wsc[64];
    sg53_q4_gemm_nax_i8_impl<128>(w_u2, scales, biases, x_i8, x_rsc, x_sums, y_f16, y_f32, p, smem, tg_wsc, tgid, tid);
#else
    (void)w_u2; (void)scales; (void)biases; (void)x_i8; (void)x_rsc; (void)x_sums; (void)y_f16; (void)y_f32; (void)p; (void)tgid; (void)tid;
#endif
}

/* ============================================================================
 * 5. Direct `T5LayerNorm` (`NO +1.0f!`), 4-Stream `mHC` (`Sinkhorn-Knopp`),
 *    Single-Stream Residual & `Glm5NextTextHyperHead` Kernels
 * ============================================================================ */

struct Sg53RmsNormParams {
    uint  M_valid;
    uint  M_padded;
    uint  dim;
    uint  src_stride;
    uint  src_col_off;
    uint  dst_f16_stride;
    uint  dst_f16_col_off;
    float eps;
    uint  write_f16;
    uint  write_f32;
    uint  pad0;
    uint  pad1;
};

// Grid: `(1, M_padded, 1)`, `256` threads/TG
kernel void sg53_rmsnorm_strided(
    device const float*         src_f32 [[buffer(0)]],
    device const float*         norm_w  [[buffer(1)]], // [dim] (NO +1.0f!)
    device half*                dst_f16 [[buffer(2)]],
    device float*               dst_f32 [[buffer(3)]],
    constant Sg53RmsNormParams& p       [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],        // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],     // 0..7
    uint  lane [[thread_index_in_simdgroup]]           // 0..31
) {
    const uint m = tgid.y;
    if (m >= p.M_padded) return;

    const uint d4_cnt = p.dim >> 2u;
    if (m >= p.M_valid) {
        if (p.write_f16 != 0u) {
            device half4* dh = reinterpret_cast<device half4*>(
                dst_f16 + ulong(m) * p.dst_f16_stride + p.dst_f16_col_off);
            for (uint i = tid; i < d4_cnt; i += 256u) {
                dh[i] = half4(0.0h);
            }
        }
        return;
    }

    threadgroup float tg_sq[8];
    device const float4* s4 = reinterpret_cast<device const float4*>(
        src_f32 + ulong(m) * p.src_stride + p.src_col_off);
    device const float4* w4 = reinterpret_cast<device const float4*>(norm_w);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 v = s4[i];
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }

    float sg_sum = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float v   = (lane < 8u) ? tg_sq[lane] : 0.0f;
        float tot = simd_sum(v);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.dim) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    device half4* dh = (p.write_f16 != 0u)
        ? reinterpret_cast<device half4*>(dst_f16 + ulong(m) * p.dst_f16_stride + p.dst_f16_col_off)
        : nullptr;
    device float4* df = (p.write_f32 != 0u)
        ? reinterpret_cast<device float4*>(dst_f32 + ulong(m) * p.dim)
        : nullptr;

    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 y = (s4[i] * inv_rms) * w4[i];
        if (dh) dh[i] = sg53_clamp_f16x4(y);
        if (df) df[i] = y;
    }
}

struct Sg53MhcNormParams {
    uint  M_valid;
    uint  hc_dim; // hc_mult * D = 16384
    float eps;    // 1e-5f
    uint  pad0;
};

// Unweighted, ungrouped 16384-dim RMSNorm: `z = y * rsqrt(mean(y^2) + 1e-5f)`
// Grid: `(M_valid, 1, 1)`, `256` threads/TG
kernel void sg53_mhc_unweighted_norm(
    device const float*         h_in_f32 [[buffer(0)]], // [M_valid, hc_dim]
    device float*               z_out    [[buffer(1)]], // [M_valid, hc_dim]
    constant Sg53MhcNormParams& p        [[buffer(2)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],          // 0..255
    uint sgid [[simdgroup_index_in_threadgroup]],       // 0..7
    uint lane [[thread_index_in_simdgroup]]             // 0..31
) {
    if (m >= p.M_valid) return;
    threadgroup float tg_sq[8];

    const uint d4_cnt = p.hc_dim >> 2u;
    device const float4* src4 = reinterpret_cast<device const float4*>(h_in_f32 + ulong(m) * p.hc_dim);
    device float4*       dst4 = reinterpret_cast<device float4*>(z_out + ulong(m) * p.hc_dim);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 v = src4[i];
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }

    float sg_sum = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.hc_dim) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    for (uint i = tid; i < d4_cnt; i += 256u) {
        dst4[i] = src4[i] * inv_rms;
    }
}

struct Sg53MhcSinkhornParams {
    uint  M_valid;
    uint  hc_mult;        // 4
    uint  sinkhorn_iters; // 20
    float hc_eps;         // 1e-6f
};

// Exact `pre`, `post`, and 20-Iteration Doubly-Stochastic Sinkhorn-Knopp on `4x4` `comb`
// Grid: `((M_valid + 31) / 32, 1, 1)`, `32` threads/TG (`1` token per lane)
kernel void sg53_mhc_sinkhorn(
    device const float*             mix_in   [[buffer(0)]], // [M_valid, 24]
    device const float*             base_in  [[buffer(1)]], // [24]
    device const float*             scale_in [[buffer(2)]], // [3]
    device float*                   pre_out  [[buffer(3)]], // [M_valid, 4]
    device float*                   post_out [[buffer(4)]], // [M_valid, 4]
    device float*                   comb_out [[buffer(5)]], // [M_valid, 16]
    constant Sg53MhcSinkhornParams& p        [[buffer(6)]],
    uint m [[thread_position_in_grid]]
) {
    if (m >= p.M_valid) return;

    const uint  H       = p.hc_mult; // 4
    const uint  mix_dim = (2u + H) * H;
    const float sc_pre  = scale_in[0];
    const float sc_post = scale_in[1];
    const float sc_comb = scale_in[2];
    const float eps     = p.hc_eps;

    device const float* mix = mix_in + ulong(m) * mix_dim;
    device float*       pre = pre_out + ulong(m) * H;
    device float*       pst = post_out + ulong(m) * H;
    device float*       cmb = comb_out + ulong(m) * (H * H);

    for (uint s = 0u; s < H; ++s) {
        pre[s] = sg53_sigmoid(fma(mix[s], sc_pre, base_in[s])) + eps; // +hc_eps on pre!
        pst[s] = 2.0f * sg53_sigmoid(fma(mix[H + s], sc_post, base_in[H + s]));
    }

    float R[4][4];
    const uint comb_base = 2u * H;

    // Iteration 0: numerically stabilized row-wise softmax + hc_eps, then col norm
    for (uint r = 0u; r < 4u; ++r) {
        float v_row[4];
        float v_max = -INFINITY;
        for (uint c = 0u; c < 4u; ++c) {
            uint idx = comb_base + 4u * r + c;
            float v  = fma(mix[idx], sc_comb, base_in[idx]);
            v_row[c] = v;
            v_max    = max(v_max, v);
        }
        float r_sum = 0.0f;
        for (uint c = 0u; c < 4u; ++c) {
            float ev = exp(v_row[c] - v_max);
            R[r][c]  = ev;
            r_sum   += ev;
        }
        float inv_r = 1.0f / (r_sum + eps);
        for (uint c = 0u; c < 4u; ++c) {
            R[r][c] = fma(R[r][c], inv_r, eps); // iter 0: softmax + 1e-6f!
        }
    }

    for (uint c = 0u; c < 4u; ++c) {
        float c_sum = (R[0][c] + R[1][c]) + (R[2][c] + R[3][c]);
        float inv_c = 1.0f / (c_sum + eps);
        for (uint r = 0u; r < 4u; ++r) {
            R[r][c] *= inv_c;
        }
    }

    // Remaining `sinkhorn_iters - 1` (`19`) row + col sweeps
    for (uint iter = 1u; iter < p.sinkhorn_iters; ++iter) {
        for (uint r = 0u; r < 4u; ++r) {
            float r_sum = (R[r][0] + R[r][1]) + (R[r][2] + R[r][3]);
            float inv_r = 1.0f / (r_sum + eps);
            for (uint c = 0u; c < 4u; ++c) {
                R[r][c] *= inv_r;
            }
        }
        for (uint c = 0u; c < 4u; ++c) {
            float c_sum = (R[0][c] + R[1][c]) + (R[2][c] + R[3][c]);
            float inv_c = 1.0f / (c_sum + eps);
            for (uint r = 0u; r < 4u; ++r) {
                R[r][c] *= inv_c;
            }
        }
    }

    for (uint r = 0u; r < 4u; ++r) {
        for (uint c = 0u; c < 4u; ++c) {
            cmb[4u * r + c] = R[r][c];
        }
    }
}

struct Sg53MhcCollapseParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    uint  hc_mult; // 4
    float eps;     // 1e-5f
    uint  write_f32;
    uint  write_unit_f16;
    uint  pad1;
};

// Fused Collapse (`x_col[d] = sum_{s=0..3} pre[s] * H_in[s, d]`) + Sublayer Pre-RMSNorm (`NO +1.0f`)
// Grid: `(1, M_padded, 1)`, `256` threads/TG
kernel void sg53_mhc_collapse_rmsnorm(
    device const float*           h_in_f32 [[buffer(0)]], // [M_valid, 4 * D]
    device const float*           pre_in   [[buffer(1)]], // [M_valid, 4]
    device const float*           norm_w   [[buffer(2)]], // [D]
    device half*                  out_f16  [[buffer(3)]], // [M_padded, D]
    device float*                 out_f32  [[buffer(4)]], // [M_valid, D]
    device half*                  unit_f16 [[buffer(5)]], // optional [16, D] unit-RMS for 16c ANE probe!
    constant Sg53MhcCollapseParams& p      [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],           // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],        // 0..7
    uint  lane [[thread_index_in_simdgroup]]              // 0..31
) {
    const uint m      = tgid.y;
    const uint d4_cnt = p.D >> 2u;
    if (m >= p.M_padded) return;

    if (m >= p.M_valid) {
        device half4* dh = reinterpret_cast<device half4*>(out_f16 + ulong(m) * p.D);
        for (uint i = tid; i < d4_cnt; i += 256u) {
            dh[i] = half4(0.0h);
        }
        return;
    }

    threadgroup float tg_sq[8];
    const float p0 = pre_in[ulong(m) * 4ul + 0u];
    const float p1 = pre_in[ulong(m) * 4ul + 1u];
    const float p2 = pre_in[ulong(m) * 4ul + 2u];
    const float p3 = pre_in[ulong(m) * 4ul + 3u];

    device const float4* h0 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 0ul) * p.D);
    device const float4* h1 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 1ul) * p.D);
    device const float4* h2 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 2ul) * p.D);
    device const float4* h3 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 3ul) * p.D);
    device const float4* w4 = reinterpret_cast<device const float4*>(norm_w);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 xc = fma(float4(p0), h0[i], fma(float4(p1), h1[i], fma(float4(p2), h2[i], float4(p3) * h3[i])));
        sq_acc = fma(xc.x, xc.x, fma(xc.y, xc.y, fma(xc.z, xc.z, fma(xc.w, xc.w, sq_acc))));
    }

    float sg_sum = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    device half4*  dh = reinterpret_cast<device half4*>(out_f16 + ulong(m) * p.D);
    device float4* df = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + ulong(m) * p.D) : nullptr;
    device half4*  du = (p.write_unit_f16 != 0u && m < 16u)
                            ? reinterpret_cast<device half4*>(unit_f16 + ulong(m) * p.D)
                            : nullptr;

    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 xc   = fma(float4(p0), h0[i], fma(float4(p1), h1[i], fma(float4(p2), h2[i], float4(p3) * h3[i])));
        float4 unit = xc * inv_rms;
        float4 y    = unit * w4[i];
        dh[i]       = sg53_clamp_f16x4(y);
        if (df) df[i] = y;
        if (du) du[i] = sg53_clamp_f16x4(unit);
    }
}

inline void sg53_rht64_warp_f32(thread float& v0, thread float& v1, uint lane);

struct Sg53MhcFusedCollapseParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    uint  hc_mult;          // 4
    uint  sinkhorn_iters;   // 20
    float hc_eps;           // 1e-6f
    float norm_eps;         // 1e-5f
    uint  write_f32;
    uint  write_unit_f16;
    uint  split_k;          // 4 for split-K `hc_fn`, 1 for direct `mix_in`
    uint  apply_hc_inv_rms; // 1 => scale `mix` by `inv_rms(h_in_f32)` (eliminates 90 `mhc_norm` dispatches/tok!)
    uint  write_x_rot_f16;  // 1 => fuse exact RHT-64 into `out_rot_f16` (eliminates 42 `rht64` dispatches/tok!)
};

// Fused Unweighted-HC-InvRMS + Split-K Reduce + 20-Iter Sinkhorn-Knopp + Collapse + Pre-RMSNorm + Optional RHT-64!
// Grid: `(1, M_padded, 1)`, `256` threads/TG
kernel void sg53_mhc_sinkhorn_collapse_rmsnorm(
    device const float*                h_in_f32    [[buffer(0)]], // [M_valid, 4 * D]
    device const float*                mix_in      [[buffer(1)]], // [M_valid, 24 * split_k]
    device const float*                base_in     [[buffer(2)]], // [24]
    device const float*                scale_in    [[buffer(3)]], // [3]
    device const float*                norm_w      [[buffer(4)]], // [D]
    device float*                      post_out    [[buffer(5)]], // [M_valid, 4]
    device float*                      comb_out    [[buffer(6)]], // [M_valid, 16]
    device half*                       out_f16     [[buffer(7)]], // [M_padded, D]
    device float*                      out_f32     [[buffer(8)]], // [M_valid, D]
    device half*                       unit_f16    [[buffer(9)]], // optional [16, D]
    device half*                       out_rot_f16 [[buffer(10)]],// optional [M_valid, D] (RHT-64 of out_f16!)
    constant Sg53MhcFusedCollapseParams& p         [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],                   // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],                // 0..7
    uint  lane [[thread_index_in_simdgroup]]                      // 0..31
) {
    const uint m      = tgid.y;
    const uint d4_cnt = p.D >> 2u;
    if (m >= p.M_padded) return;

    if (m >= p.M_valid) {
        device half4* dh = reinterpret_cast<device half4*>(out_f16 + ulong(m) * p.D);
        for (uint i = tid; i < d4_cnt; i += 256u) dh[i] = half4(0.0h);
        return;
    }

    threadgroup float tg_mix[24];
    threadgroup float tg_pre[4];
    threadgroup float tg_sq[8];

    device const float4* h0 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 0ul) * p.D);
    device const float4* h1 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 1ul) * p.D);
    device const float4* h2 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 2ul) * p.D);
    device const float4* h3 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 3ul) * p.D);

    float inv_rms_hc = 1.0f;
    if (p.apply_hc_inv_rms != 0u) {
        const uint hc_d4 = p.D; // (4 * D) / 4 = D
        device const float4* h_all = reinterpret_cast<device const float4*>(h_in_f32 + ulong(m) * 4ul * p.D);
        float hc_sq = 0.0f;
        for (uint i = tid; i < hc_d4; i += 256u) {
            float4 v = h_all[i];
            hc_sq = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, hc_sq))));
        }
        float sg_hc = simd_sum(hc_sq);
        if (lane == 0u) tg_sq[sgid] = sg_hc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float tot_hc = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot_hc / float(4u * p.D) + p.norm_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv_rms_hc = tg_sq[0];
    }

    if (tid < 24u) {
        float mv;
        if (p.split_k == 4u) {
            const ulong b = (ulong(m) * 24ul + tid) * 4ul;
            mv = (mix_in[b + 0u] + mix_in[b + 1u]) + (mix_in[b + 2u] + mix_in[b + 3u]);
        } else {
            mv = mix_in[ulong(m) * 24ul + tid];
        }
        tg_mix[tid] = mv * inv_rms_hc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        const float sc_pre  = scale_in[0];
        const float sc_post = scale_in[1];
        const float sc_comb = scale_in[2];
        const float eps     = p.hc_eps;

        device float* pst = post_out + ulong(m) * 4ul;
        device float* cmb = comb_out + ulong(m) * 16ul;

        for (uint s = 0u; s < 4u; ++s) {
            tg_pre[s] = sg53_sigmoid(fma(tg_mix[s], sc_pre, base_in[s])) + eps;
            pst[s]    = 2.0f * sg53_sigmoid(fma(tg_mix[4u + s], sc_post, base_in[4u + s]));
        }

        float R[4][4];
        for (uint r = 0u; r < 4u; ++r) {
            float v_row[4];
            float v_max = -INFINITY;
            for (uint c = 0u; c < 4u; ++c) {
                uint idx = 8u + 4u * r + c;
                float v  = fma(tg_mix[idx], sc_comb, base_in[idx]);
                v_row[c] = v;
                v_max    = max(v_max, v);
            }
            float r_sum = 0.0f;
            for (uint c = 0u; c < 4u; ++c) {
                float ev = exp(v_row[c] - v_max);
                R[r][c]  = ev;
                r_sum   += ev;
            }
            float inv_r = 1.0f / (r_sum + eps);
            for (uint c = 0u; c < 4u; ++c) R[r][c] = fma(R[r][c], inv_r, eps);
        }
        for (uint c = 0u; c < 4u; ++c) {
            float inv_c = 1.0f / ((R[0][c] + R[1][c]) + (R[2][c] + R[3][c]) + eps);
            for (uint r = 0u; r < 4u; ++r) R[r][c] *= inv_c;
        }
        for (uint iter = 1u; iter < p.sinkhorn_iters; ++iter) {
            for (uint r = 0u; r < 4u; ++r) {
                float inv_r = 1.0f / ((R[r][0] + R[r][1]) + (R[r][2] + R[r][3]) + eps);
                for (uint c = 0u; c < 4u; ++c) R[r][c] *= inv_r;
            }
            for (uint c = 0u; c < 4u; ++c) {
                float inv_c = 1.0f / ((R[0][c] + R[1][c]) + (R[2][c] + R[3][c]) + eps);
                for (uint r = 0u; r < 4u; ++r) R[r][c] *= inv_c;
            }
        }
        for (uint r = 0u; r < 4u; ++r) {
            for (uint c = 0u; c < 4u; ++c) cmb[4u * r + c] = R[r][c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float p0 = tg_pre[0], p1 = tg_pre[1], p2 = tg_pre[2], p3 = tg_pre[3];
    device const float4* w4 = reinterpret_cast<device const float4*>(norm_w);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 xc = fma(float4(p0), h0[i], fma(float4(p1), h1[i], fma(float4(p2), h2[i], float4(p3) * h3[i])));
        sq_acc = fma(xc.x, xc.x, fma(xc.y, xc.y, fma(xc.z, xc.z, fma(xc.w, xc.w, sq_acc))));
    }
    float sg_sum = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    if (p.write_x_rot_f16 != 0u && (p.D & 63u) == 0u) {
        const uint n_blk = p.D >> 6u;
        device const float2* h0_2 = reinterpret_cast<device const float2*>(h_in_f32 + (ulong(m) * 4ul + 0ul) * p.D);
        device const float2* h1_2 = reinterpret_cast<device const float2*>(h_in_f32 + (ulong(m) * 4ul + 1ul) * p.D);
        device const float2* h2_2 = reinterpret_cast<device const float2*>(h_in_f32 + (ulong(m) * 4ul + 2ul) * p.D);
        device const float2* h3_2 = reinterpret_cast<device const float2*>(h_in_f32 + (ulong(m) * 4ul + 3ul) * p.D);
        device const float2* w2   = reinterpret_cast<device const float2*>(norm_w);
        device half2*  dh2  = reinterpret_cast<device half2*>(out_f16 + ulong(m) * p.D);
        device half2*  dr2  = reinterpret_cast<device half2*>(out_rot_f16 + ulong(m) * p.D);
        device float2* df2  = (p.write_f32 != 0u) ? reinterpret_cast<device float2*>(out_f32 + ulong(m) * p.D) : nullptr;
        device half2*  du2  = (p.write_unit_f16 != 0u && m < 16u)
                                  ? reinterpret_cast<device half2*>(unit_f16 + ulong(m) * p.D)
                                  : nullptr;
        for (uint blk = sgid; blk < n_blk; blk += 8u) {
            const uint idx  = blk * 32u + lane;
            float2 xc   = fma(float2(p0), h0_2[idx], fma(float2(p1), h1_2[idx], fma(float2(p2), h2_2[idx], float2(p3) * h3_2[idx])));
            float2 unit = xc * inv_rms;
            float2 y    = unit * w2[idx];
            half2  yh   = half2(sg53_clamp_f16(y.x), sg53_clamp_f16(y.y));
            dh2[idx]    = yh;
            if (df2) df2[idx] = y;
            if (du2) du2[idx] = half2(sg53_clamp_f16(unit.x), sg53_clamp_f16(unit.y));
            float rv0 = float(yh.x);
            float rv1 = float(yh.y);
            sg53_rht64_warp_f32(rv0, rv1, lane);
            dr2[idx] = half2(sg53_clamp_f16(rv0), sg53_clamp_f16(rv1));
        }
    } else {
        device half4*  dh = reinterpret_cast<device half4*>(out_f16 + ulong(m) * p.D);
        device float4* df = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + ulong(m) * p.D) : nullptr;
        device half4*  du = (p.write_unit_f16 != 0u && m < 16u)
                                ? reinterpret_cast<device half4*>(unit_f16 + ulong(m) * p.D)
                                : nullptr;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            float4 xc   = fma(float4(p0), h0[i], fma(float4(p1), h1[i], fma(float4(p2), h2[i], float4(p3) * h3[i])));
            float4 unit = xc * inv_rms;
            float4 y    = unit * w4[i];
            dh[i]       = sg53_clamp_f16x4(y);
            if (df) df[i] = y;
            if (du) du[i] = sg53_clamp_f16x4(unit);
        }
    }
}

struct Sg53MhcExpandParams {
    uint M_valid;
    uint D;
    uint hc_mult; // 4
    uint pad0;
};

// In-Place Expand + Doubly-Stochastic Residual (`comb^T @ H_in` — NOTE TRANSPOSE `comb[r, s]`!):
// `H_out[s, d] = post[s] * y_sub[d] + sum_{r=0..3} comb[r, s] * H_in[r, d]`
// Grid: `((D/4 + 255) / 256, M_valid, 1)`, `256` threads/TG
kernel void sg53_mhc_expand(
    device float*                 h_inout [[buffer(0)]], // [M_valid, 4 * D]
    device const float*           y_sub   [[buffer(1)]], // [M_valid, D]
    device const float*           post_in [[buffer(2)]], // [M_valid, 4]
    device const float*           comb_in [[buffer(3)]], // [M_valid, 16]
    constant Sg53MhcExpandParams& p       [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]           // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    device float4* s0_p = reinterpret_cast<device float4*>(h_inout + (ulong(m) * 4ul + 0ul) * p.D);
    device float4* s1_p = reinterpret_cast<device float4*>(h_inout + (ulong(m) * 4ul + 1ul) * p.D);
    device float4* s2_p = reinterpret_cast<device float4*>(h_inout + (ulong(m) * 4ul + 2ul) * p.D);
    device float4* s3_p = reinterpret_cast<device float4*>(h_inout + (ulong(m) * 4ul + 3ul) * p.D);

    // Load all 4 streams into registers BEFORE writing back in-place!
    const float4 h0 = s0_p[idx4];
    const float4 h1 = s1_p[idx4];
    const float4 h2 = s2_p[idx4];
    const float4 h3 = s3_p[idx4];
    const float4 yv = reinterpret_cast<device const float4*>(y_sub + ulong(m) * p.D)[idx4];

    device const float* pst = post_in + ulong(m) * 4ul;
    device const float* cmb = comb_in + ulong(m) * 16ul;

    // `comb[r, s] = cmb[4 * r + s]`
    s0_p[idx4] = fma(float4(pst[0]), yv,
                 fma(float4(cmb[0]), h0, fma(float4(cmb[4]), h1, fma(float4(cmb[8]),  h2, float4(cmb[12]) * h3))));
    s1_p[idx4] = fma(float4(pst[1]), yv,
                 fma(float4(cmb[1]), h0, fma(float4(cmb[5]), h1, fma(float4(cmb[9]),  h2, float4(cmb[13]) * h3))));
    s2_p[idx4] = fma(float4(pst[2]), yv,
                 fma(float4(cmb[2]), h0, fma(float4(cmb[6]), h1, fma(float4(cmb[10]), h2, float4(cmb[14]) * h3))));
    s3_p[idx4] = fma(float4(pst[3]), yv,
                 fma(float4(cmb[3]), h0, fma(float4(cmb[7]), h1, fma(float4(cmb[11]), h2, float4(cmb[15]) * h3))));
}

struct Sg53ResAddParams {
    uint M_valid;
    uint D;
};

// Single-stream residual add for MTP Layer 45 (`has_mhc == 0`)
kernel void sg53_single_residual_add(
    device float*              x_inout [[buffer(0)]], // [M_valid, D]
    device const float*        y_sub   [[buffer(1)]], // [M_valid, D]
    constant Sg53ResAddParams& p       [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    const ulong off4 = ulong(m) * d4_cnt + idx4;
    reinterpret_cast<device float4*>(x_inout)[off4] += reinterpret_cast<device const float4*>(y_sub)[off4];
}

struct Sg53HyperHeadParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    float eps; // 1e-5f
    uint  write_f16;
    uint  write_f32;
    uint  pad0;
    uint  pad1;
};

// Final `Glm5NextTextHyperHead` after Layer 44:
// `h_mean[d] = 0.25f * (H[0, d] + H[1, d] + H[2, d] + H[3, d])` -> `rms_norm_D(h_mean, final_norm_w, 1e-5f)`
// Grid: `(1, M_padded, 1)`, `256` threads/TG
kernel void sg53_hyper_head_final_norm(
    device const float*          h_in_f32 [[buffer(0)]], // [M_valid, 4 * D]
    device const float*          norm_w   [[buffer(1)]], // [D]
    device half*                 out_f16  [[buffer(2)]], // [M_padded, D] (for lm_head)
    device float*                out_f32  [[buffer(3)]], // [M_valid, D]  (H_main for Native MTP!)
    constant Sg53HyperHeadParams& p       [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],          // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],       // 0..7
    uint  lane [[thread_index_in_simdgroup]]             // 0..31
) {
    const uint m      = tgid.y;
    const uint d4_cnt = p.D >> 2u;
    if (m >= p.M_padded) return;

    if (m >= p.M_valid) {
        if (p.write_f16 != 0u) {
            device half4* dh = reinterpret_cast<device half4*>(out_f16 + ulong(m) * p.D);
            for (uint i = tid; i < d4_cnt; i += 256u) {
                dh[i] = half4(0.0h);
            }
        }
        return;
    }

    threadgroup float tg_sq[8];
    device const float4* h0 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 0ul) * p.D);
    device const float4* h1 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 1ul) * p.D);
    device const float4* h2 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 2ul) * p.D);
    device const float4* h3 = reinterpret_cast<device const float4*>(h_in_f32 + (ulong(m) * 4ul + 3ul) * p.D);
    device const float4* w4 = reinterpret_cast<device const float4*>(norm_w);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 hm = ((h0[i] + h1[i]) + (h2[i] + h3[i])) * 0.25f;
        sq_acc = fma(hm.x, hm.x, fma(hm.y, hm.y, fma(hm.z, hm.z, fma(hm.w, hm.w, sq_acc))));
    }

    float sg_sum = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    device half4*  dh = (p.write_f16 != 0u) ? reinterpret_cast<device half4*>(out_f16 + ulong(m) * p.D) : nullptr;
    device float4* df = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + ulong(m) * p.D) : nullptr;

    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 hm = ((h0[i] + h1[i]) + (h2[i] + h3[i])) * 0.25f;
        float4 y  = (hm * inv_rms) * w4[i];
        if (dh) dh[i] = sg53_clamp_f16x4(y);
        if (df) df[i] = y;
    }
}

/* ============================================================================
 * 6. 34-Layer `KDA` (`Key-Driven Linear Attention`, Per-Channel Bounded Decay)
 * ============================================================================ */

struct Sg53KdaFbgSplitParams {
    uint M_valid;
    uint M_padded;
    uint nh; // 64
    uint hd; // 128
};

// Splits `fbg [M_valid, 2*hd + nh]` into `f_a_f16 [M_padded, hd]`, `b_f32 [M_valid, nh]`, `g_a_f16 [M_padded, hd]`
// Grid: `(1, M_padded, 1)`, `256` threads/TG
kernel void sg53_kda_fbg_split(
    device const float*            fbg_in  [[buffer(0)]], // [M_valid, 2*hd + nh]
    device half*                   f_a_out [[buffer(1)]], // [M_padded, hd]
    device float*                  b_out   [[buffer(2)]], // [M_valid, nh]
    device half*                   g_a_out [[buffer(3)]], // [M_padded, hd]
    constant Sg53KdaFbgSplitParams& p      [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]            // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_padded) return;

    if (m >= p.M_valid) {
        for (uint d = tid; d < p.hd; d += 256u) {
            f_a_out[ulong(m) * p.hd + d] = half(0.0h);
            g_a_out[ulong(m) * p.hd + d] = half(0.0h);
        }
        return;
    }

    const uint fbg_dim = 2u * p.hd + p.nh;
    device const float* row = fbg_in + ulong(m) * fbg_dim;
    for (uint i = tid; i < fbg_dim; i += 256u) {
        float v = row[i];
        if (i < p.hd) {
            f_a_out[ulong(m) * p.hd + i] = sg53_clamp_f16(v);
        } else if (i < p.hd + p.nh) {
            b_out[ulong(m) * p.nh + (i - p.hd)] = v;
        } else {
            g_a_out[ulong(m) * p.hd + (i - p.hd - p.nh)] = sg53_clamp_f16(v);
        }
    }
}

struct Sg53KdaConvParams {
    uint M;
    uint channels; // 24576
    uint save_snap_mask;
    uint snap_layer_idx;
    uint num_kda_layers;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void sg53_kda_conv1d_silu(
    device const half*         qkv_in     [[buffer(0)]], // [M, channels]
    device const float4*       conv_w     [[buffer(1)]], // [channels] (`[t-3, t-2, t-1, t]`)
    device half*               conv_state [[buffer(2)]], // [3, channels] for this KDA layer
    device half*               qkv_out    [[buffer(3)]], // [M, channels]
    device half*               conv_snaps [[buffer(4)]], // optional [4, num_kda, 3, channels]
    constant Sg53KdaConvParams& p         [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.channels) return;

    const uint   C = p.channels;
    const float4 w = conv_w[gid];

    half h0 = conv_state[0u * C + gid];
    half h1 = conv_state[1u * C + gid];
    half h2 = conv_state[2u * C + gid];

    for (uint t = 0u; t < p.M; ++t) {
        const ulong idx = ulong(t) * C + gid;
        const half  h3  = qkv_in[idx];
        const float raw = fma(w.x, float(h0), fma(w.y, float(h1), fma(w.z, float(h2), w.w * float(h3))));
        qkv_out[idx]    = sg53_clamp_f16(sg53_silu(raw));

        h0 = h1;
        h1 = h2;
        h2 = h3;

        if ((p.save_snap_mask & (1u << t)) != 0u) {
            const ulong layer_stride = 3ul * C;
            const ulong slot_stride  = ulong(p.num_kda_layers) * layer_stride;
            device half* snap = conv_snaps + ulong(t) * slot_stride + ulong(p.snap_layer_idx) * layer_stride;
            snap[0u * C + gid] = h0;
            snap[1u * C + gid] = h1;
            snap[2u * C + gid] = h2;
        }
    }

    conv_state[0u * C + gid] = h0;
    conv_state[1u * C + gid] = h1;
    conv_state[2u * C + gid] = h2;
}

struct Sg53KdaParams {
    uint  M_valid;
    uint  M_padded;
    uint  nh;               // 64
    uint  hd;               // 128
    float l2_eps;           // 1e-6f
    float norm_eps;         // 1e-5f
    float gate_lower_bound; // -5.0f (`decay = exp(-5.0 * sigmoid(exp(A_log)*(a + dt_bias)))`)
    uint  save_snap_mask;
    uint  snap_layer_idx;
    uint  num_kda_layers;
    uint  pad0;
    uint  pad1;
};

// Exact KDA Recurrent Update (`M = 1 .. 8`, 100% bit-identical per-token math between K=0 & MTP!)
// State layout per KDA layer: `S[nh, hd, hd]` where `state_h[dk * hd + dv]` stores `S[h, dv, dk]`!
// Grid: `(nh, 1, 1)`, `512` threads/TG (`dv = tid & 127`, `qtr = tid >> 7` in 0..3 when `hd == 128`)
kernel void sg53_kda_recurrent_gated_norm(
    device const half*     qkv_conv  [[buffer(0)]], // [M_valid, 3 * nh * hd]
    device const float*    a_in      [[buffer(1)]], // [M_valid, nh * hd] (per-channel forget!)
    device const float*    b_in      [[buffer(2)]], // [M_valid, nh]
    device const half*     gate_in   [[buffer(3)]], // [M_valid, nh * hd]
    device const float*    A_log     [[buffer(4)]], // [nh]
    device const float*    dt_bias   [[buffer(5)]], // [nh * hd]
    device const float*    o_norm_w  [[buffer(6)]], // [hd] (NO +1.0f!)
    device float*          rec_state [[buffer(7)]], // [nh, hd, hd]
    device half*           y_out     [[buffer(8)]], // [M_padded, nh * hd]
    device float*          rec_snaps [[buffer(9)]], // optional [4, num_kda, nh, hd, hd]
    constant Sg53KdaParams& p        [[buffer(10)]],
    uint h    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],      // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],   // 0..15
    uint lane [[thread_index_in_simdgroup]]         // 0..31
) {
    if (h >= p.nh) return;

    const uint qkv_dim  = p.nh * p.hd;
    const uint conv_dim = 3u * qkv_dim;
    const uint hd_sq    = p.hd * p.hd;

    // General small-hd fallback for miniature synthetic configs (`hd < 128`)
    if (p.hd != 128u) {
        if (tid >= p.hd) return;
        const uint dv = tid;
        threadgroup float q_sh[128];
        threadgroup float k_sh[128];
        threadgroup float dec_sh[128];
        threadgroup float o_sh[128];
        threadgroup float tg_sq[4];
        threadgroup float tg_sc[4];

        device float* state_h = rec_state + ulong(h) * hd_sq;
        const float exp_alog  = exp(A_log[h]);
        const float nw        = o_norm_w[dv];
        const float inv_sqrt  = rsqrt(float(p.hd));
        const uint  num_sgs   = (p.hd + 31u) >> 5u;

        for (uint t = 0u; t < p.M_valid; ++t) {
            device const half* q_raw = qkv_conv + ulong(t) * conv_dim + h * p.hd;
            device const half* k_raw = qkv_conv + ulong(t) * conv_dim + qkv_dim + h * p.hd;
            device const half* v_raw = qkv_conv + ulong(t) * conv_dim + 2u * qkv_dim + h * p.hd;

            float qv = float(q_raw[dv]);
            float kv = float(k_raw[dv]);
            float q_s = simd_sum(qv * qv);
            float k_s = simd_sum(kv * kv);
            if (lane == 0u && sgid < 4u) {
                tg_sq[sgid] = q_s;
                tg_sc[sgid] = k_s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                float qt = simd_sum((lane < num_sgs) ? tg_sq[lane] : 0.0f);
                float kt = simd_sum((lane < num_sgs) ? tg_sc[lane] : 0.0f);
                if (lane == 0u) {
                    tg_sq[0] = rsqrt(qt + p.l2_eps) * inv_sqrt;
                    tg_sc[0] = rsqrt(kt + p.l2_eps);
                    tg_sc[1] = sg53_sigmoid(b_in[ulong(t) * p.nh + h]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            q_sh[dv] = qv * tg_sq[0];
            k_sh[dv] = kv * tg_sc[0];
            float av = a_in[ulong(t) * qkv_dim + h * p.hd + dv] + dt_bias[h * p.hd + dv];
            dec_sh[dv] = exp(p.gate_lower_bound * sg53_sigmoid(exp_alog * av));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float kv_mem = 0.0f;
            for (uint dk = 0u; dk < p.hd; ++dk) {
                float sv = state_h[dk * p.hd + dv] * dec_sh[dk];
                state_h[dk * p.hd + dv] = sv;
                kv_mem = fma(sv, k_sh[dk], kv_mem);
            }
            float delta = tg_sc[1] * (float(v_raw[dv]) - kv_mem);
            float ov    = 0.0f;
            for (uint dk = 0u; dk < p.hd; ++dk) {
                float sv = fma(k_sh[dk], delta, state_h[dk * p.hd + dv]);
                state_h[dk * p.hd + dv] = sv;
                ov = fma(sv, q_sh[dk], ov);
            }
            if ((p.save_snap_mask & (1u << t)) != 0u) {
                const ulong layer_str = ulong(p.nh) * hd_sq;
                const ulong slot_str  = ulong(p.num_kda_layers) * layer_str;
                device float* snap_h  = rec_snaps + ulong(t) * slot_str + ulong(p.snap_layer_idx) * layer_str + ulong(h) * hd_sq;
                for (uint dk = 0u; dk < p.hd; ++dk) {
                    snap_h[dk * p.hd + dv] = state_h[dk * p.hd + dv];
                }
            }
            o_sh[dv] = ov;
            float os = simd_sum(ov * ov);
            if (lane == 0u && sgid < 4u) tg_sq[sgid] = os;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                float ot = simd_sum((lane < num_sgs) ? tg_sq[lane] : 0.0f);
                if (lane == 0u) tg_sq[0] = rsqrt(ot / float(p.hd) + p.norm_eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float g_val = float(gate_in[ulong(t) * qkv_dim + h * p.hd + dv]);
            y_out[ulong(t) * qkv_dim + h * p.hd + dv] = sg53_clamp_f16(o_sh[dv] * tg_sq[0] * nw * sg53_sigmoid(g_val));
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint t = p.M_valid; t < p.M_padded; ++t) {
            y_out[ulong(t) * qkv_dim + h * p.hd + dv] = half(0.0h);
        }
        return;
    }

    // Canonical `hd == 128` 512-thread register-resident fast path (`s_reg[32]` per thread!)
    const uint dv      = tid & 127u;
    const uint qtr     = tid >> 7u;  // 0..3
    const uint dk_base = qtr * 32u;

    threadgroup float q_hat[128];
    threadgroup float k_hat[128];
    threadgroup float decay_k[128];
    threadgroup float o_buf[128];
    threadgroup float tg_kv_red[4][128];
    threadgroup float tg_out_red[4][128];
    threadgroup float tg_qk_sq[8];
    threadgroup float tg_o_sq[4];
    threadgroup float tg_scalars[4];

    device float* state_h  = rec_state + ulong(h) * 16384ul;
    const float exp_a_log  = exp(A_log[h]);
    const float norm_w_dv  = o_norm_w[dv];

    float s_reg[32];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 32u; ++i) {
        s_reg[i] = state_h[(dk_base + i) * 128u + dv];
    }

    constexpr float kInvSqrt128 = 0.08838834764831845f;

    for (uint t = 0u; t < p.M_valid; ++t) {
        device const half* qkv_t  = qkv_conv + ulong(t) * conv_dim;
        device const half* q_raw  = qkv_t + h * 128u;
        device const half* k_raw  = qkv_t + qkv_dim + h * 128u;
        device const half* v_raw  = qkv_t + 2u * qkv_dim + h * 128u;
        device const half* g_raw  = gate_in + ulong(t) * qkv_dim + h * 128u;

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
        } else if (sgid < 12u) {
            uint  dk    = tid - 256u;
            float a_val = a_in[ulong(t) * qkv_dim + h * 128u + dk] + dt_bias[h * 128u + dk];
            decay_k[dk] = exp(p.gate_lower_bound * sg53_sigmoid(exp_a_log * a_val));
        }
        if (tid == 384u) {
            tg_scalars[0] = sg53_sigmoid(b_in[ulong(t) * p.nh + h]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float q_tot = simd_sum((lane < 4u) ? tg_qk_sq[lane] : 0.0f);
            float k_tot = simd_sum((lane < 4u) ? tg_qk_sq[4u + lane] : 0.0f);
            if (lane == 0u) {
                tg_scalars[1] = rsqrt(q_tot + p.l2_eps) * kInvSqrt128;
                tg_scalars[2] = rsqrt(k_tot + p.l2_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 128u) {
            q_hat[tid] *= tg_scalars[1];
        } else if (tid < 256u) {
            k_hat[tid - 128u] *= tg_scalars[2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float beta = tg_scalars[0];

        // Per-channel decay multiplies `dk` KEY axis!
        float kv_part = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 32u; ++i) {
            float sv = s_reg[i] * decay_k[dk_base + i];
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
            const ulong layer_stride = ulong(p.nh) * 16384ul;
            const ulong slot_stride  = ulong(p.num_kda_layers) * layer_stride;
            device float* snap_h = rec_snaps + ulong(t) * slot_stride +
                                   ulong(p.snap_layer_idx) * layer_stride + ulong(h) * 16384ul;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 32u; ++i) {
                snap_h[(dk_base + i) * 128u + dv] = s_reg[i];
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
                tg_scalars[3] = rsqrt(o_tot * (1.0f / 128.0f) + p.norm_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 128u) {
            float normed = o_buf[dv] * tg_scalars[3] * norm_w_dv;
            float g_val  = sg53_sigmoid(float(g_raw[dv]));
            y_out[ulong(t) * qkv_dim + h * 128u + dv] = sg53_clamp_f16(normed * g_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 128u) {
        for (uint t = p.M_valid; t < p.M_padded; ++t) {
            y_out[ulong(t) * qkv_dim + h * 128u + dv] = half(0.0h);
        }
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 32u; ++i) {
        state_h[(dk_base + i) * 128u + dv] = s_reg[i];
    }
}

// Fast 256-TG Zero-Barrier Batched Prefill KDA (`M > 8`, `hd == 128`)
kernel void sg53_kda_prep_batch(
    device const half*     qkv_conv   [[buffer(0)]], // [M_valid, 3 * nh * 128]
    device const float*    a_in       [[buffer(1)]], // [M_valid, nh * 128]
    device const float*    b_in       [[buffer(2)]], // [M_valid, nh]
    device const float*    A_log      [[buffer(3)]], // [nh]
    device const float*    dt_bias    [[buffer(4)]], // [nh * 128]
    device float*          qk_hat_out [[buffer(5)]], // [M_valid, 2 * nh * 128]
    device float*          decay_out  [[buffer(6)]], // [M_valid, nh * 128]
    device float*          beta_out   [[buffer(7)]], // [M_valid, nh]
    constant Sg53KdaParams& p         [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],     // (0..3*nh, 0..M_valid-1)
    uint  tid  [[thread_index_in_threadgroup]],      // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]],   // 0..3
    uint  lane [[thread_index_in_simdgroup]]         // 0..31
) {
    const uint item = tgid.x;
    const uint t    = tgid.y;
    if (item > 3u * p.nh || t >= p.M_valid) return;

    const uint qkv_dim  = p.nh * 128u;
    const uint conv_dim = 3u * qkv_dim;

    if (item == 3u * p.nh) {
        for (uint h = tid; h < p.nh; h += 128u) {
            beta_out[ulong(t) * p.nh + h] = sg53_sigmoid(b_in[ulong(t) * p.nh + h]);
        }
        return;
    }

    if (item >= 2u * p.nh) {
        const uint  h        = item - 2u * p.nh;
        const float exp_alog = exp(A_log[h]);
        const float av       = a_in[ulong(t) * qkv_dim + h * 128u + tid] + dt_bias[h * 128u + tid];
        decay_out[ulong(t) * qkv_dim + h * 128u + tid] = exp(p.gate_lower_bound * sg53_sigmoid(exp_alog * av));
        return;
    }

    threadgroup float tg_sq[4];
    const bool  is_q = (item < p.nh);
    const uint  h    = is_q ? item : (item - p.nh);
    const float v    = float(qkv_conv[ulong(t) * conv_dim + (is_q ? 0ul : ulong(qkv_dim)) + h * 128u + tid]);

    float sg_sq = simd_sum(v * v);
    if (lane == 0u) tg_sq[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) {
            constexpr float kInvSqrt128 = 0.08838834764831845f;
            float inv = rsqrt(tot + p.l2_eps);
            tg_sq[0]  = is_q ? (inv * kInvSqrt128) : inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    qk_hat_out[ulong(t) * (2u * qkv_dim) + (is_q ? 0ul : ulong(qkv_dim)) + h * 128u + tid] = v * tg_sq[0];
}

kernel void sg53_kda_rec_batch(
    device const half*     qkv_conv  [[buffer(0)]],
    device const float4*   qk_hat    [[buffer(1)]],
    device const float4*   decay_in  [[buffer(2)]],
    device const float*    beta_in   [[buffer(3)]],
    device float*          rec_state [[buffer(4)]],
    device float*          raw_out   [[buffer(5)]],
    constant Sg53KdaParams& p        [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],    // (0..3, 0..nh-1)
    uint  sgid [[simdgroup_index_in_threadgroup]],  // 0..3
    uint  lane [[thread_index_in_simdgroup]]        // 0..31
) {
    const uint col_blk = tgid.x;
    const uint h       = tgid.y;
    if (col_blk >= 4u || h >= p.nh) return;

    const uint qkv_dim  = p.nh * 128u;
    const uint conv_dim = 3u * qkv_dim;
    const uint k4_str   = qkv_dim >> 2u;
    const uint qk4_str  = (2u * qkv_dim) >> 2u;

    const uint dv0 = col_blk * 32u + sgid * 8u;
    const uint dk0 = lane * 4u;

    device float* state_h = rec_state + ulong(h) * 16384ul;
    float4 s_col[8];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        uint dv = dv0 + c;
        s_col[c] = float4(state_h[(dk0 + 0u) * 128u + dv],
                          state_h[(dk0 + 1u) * 128u + dv],
                          state_h[(dk0 + 2u) * 128u + dv],
                          state_h[(dk0 + 3u) * 128u + dv]);
    }

    for (uint t = 0u; t < p.M_valid; ++t) {
        const float4 q4   = qk_hat[ulong(t) * qk4_str + h * 32u + lane];
        const float4 k4   = qk_hat[ulong(t) * qk4_str + k4_str + h * 32u + lane];
        const float4 dec4 = decay_in[ulong(t) * k4_str + h * 32u + lane];
        const float  beta = beta_in[ulong(t) * p.nh + h];
        device const half* v_ptr = qkv_conv + ulong(t) * conv_dim + 2ul * qkv_dim + h * 128u + dv0;
        device float*      o_ptr = raw_out + ulong(t) * qkv_dim + h * 128u + dv0;

        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; ++c) {
            float4 sc    = s_col[c] * dec4;
            float kv_mem = simd_sum(dot(sc, k4));
            float delta  = beta * (float(v_ptr[c]) - kv_mem);
            sc           = fma(k4, float4(delta), sc);
            s_col[c]     = sc;
            float ov     = simd_sum(dot(sc, q4));
            if (lane == 0u) o_ptr[c] = ov;
        }
    }

    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        uint dv = dv0 + c;
        state_h[(dk0 + 0u) * 128u + dv] = s_col[c].x;
        state_h[(dk0 + 1u) * 128u + dv] = s_col[c].y;
        state_h[(dk0 + 2u) * 128u + dv] = s_col[c].z;
        state_h[(dk0 + 3u) * 128u + dv] = s_col[c].w;
    }
}

kernel void sg53_kda_post_batch(
    device const float*    raw_out  [[buffer(0)]],
    device const half*     gate_in  [[buffer(1)]],
    device const float*    o_norm_w [[buffer(2)]],
    device half*           y_out    [[buffer(3)]],
    constant Sg53KdaParams& p       [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],   // (0..nh-1, 0..M_padded-1)
    uint  tid  [[thread_index_in_threadgroup]],    // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint  lane [[thread_index_in_simdgroup]]       // 0..31
) {
    const uint h = tgid.x;
    const uint t = tgid.y;
    if (h >= p.nh || t >= p.M_padded || tid >= p.hd) return;

    const uint  qkv_dim = p.nh * p.hd;
    const ulong idx     = ulong(t) * qkv_dim + h * p.hd + tid;
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
            tg_sq[0] = rsqrt(tot / float(p.hd) + p.norm_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float normed = ov * tg_sq[0] * o_norm_w[tid];
    const float g_val  = sg53_sigmoid(float(gate_in[idx]));
    y_out[idx]         = sg53_clamp_f16(normed * g_val);
}

/* ---- Chunk-32 WY Metal 4.1 NAX KDA Prefill (`M > 8`, `hd == 128`, Per-Channel Key Decay!) ----
 * Exact chunk-32 WY formulation where EVERY `exp(...)` argument is `<= 0` (values in `(0, 1]`, ZERO overflow!):
 *   G[r, dk] = sum_{u=0..r} g[u, dk] <= 0
 *   M[r, j]  = beta[r] * sum_{dk} k[r, dk] * exp(G[r, dk] - G[j, dk]) * k[j, dk]   (r > j)
 *   P[r, j]  = sum_{dk} q[r, dk] * exp(G[r, dk] - G[j, dk]) * k[j, dk]             (r >= j)
 *   T        = (I + M)^{-1}  (unit lower-triangular FP32 forward substitution)
 *   W        = T @ (beta * k * exp(G)),  U = T @ (beta * v)
 *   Delta    = U - W @ S_prev,  O = Q_sc @ S_prev + P @ Delta
 *   S_next   = diag(exp(G[31])) @ S_prev + K_end^T @ Delta
 */
struct Sg53KdaWyParams {
    uint  M_valid;
    uint  M_padded;          // Multiple of 32 (`num_chunks * 32`)
    uint  num_chunks;        // M_padded / 32
    uint  nh;                // 64
    float l2_eps;            // 1e-6f
    float gate_lower_bound;  // -5.0f
    uint  pad0;
    uint  pad1;
};

// Kernel 1: `(num_chunks, nh, 1)`, `128` threads/TG (`4` SGs, `28.2 KiB` TG mem <= 32 KiB)
kernel void sg53_kda_wy_prep_chunk32(
    device const half*        qkv_conv   [[buffer(0)]], // [M_valid, 3 * nh * 128]
    device const float*       a_in       [[buffer(1)]], // [M_valid, nh * 128]
    device const float*       b_in       [[buffer(2)]], // [M_valid, nh]
    device const float*       A_log      [[buffer(3)]], // [nh]
    device const float*       dt_bias    [[buffer(4)]], // [nh * 128]
    device half*              q_sc_out   [[buffer(5)]], // [nh, M_padded, 128]
    device half*              k_end_out  [[buffer(6)]], // [nh, M_padded, 128]
    device half*              w_out      [[buffer(7)]], // [nh, M_padded, 128]
    device half*              u_out      [[buffer(8)]], // [nh, M_padded, 128]
    device half*              p_out      [[buffer(9)]], // [nh, num_chunks, 32, 32]
    device float*             g_last_out [[buffer(10)]],// [nh, num_chunks, 128]
    constant Sg53KdaWyParams& p          [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],        // (c, h)
    uint  tid  [[thread_index_in_threadgroup]],         // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]],      // 0..3
    uint  lane [[thread_index_in_simdgroup]]            // 0..31
) {
    const uint c = tgid.x;
    const uint h = tgid.y;
    if (c >= p.num_chunks || h >= p.nh) return;

#if defined(__HAVE_TENSOR__)
    union Sg53KdaWyGUnion {
        float G[32][128]; // 16 KiB
        half  Th[32][32]; //  2 KiB (reused after G is dead!)
    };
    threadgroup Sg53KdaWyGUnion u_g;
    threadgroup half  K_tg[32][128]; //  8 KiB (k_hat -> K_sc -> V_sc)
    threadgroup float T_tg[32][32];  //  4 KiB (M -> T = (I + M)^-1)
    threadgroup float beta_tg[32];   // 128 B

    const uint  t_base   = c * 32u;
    const uint  qkv_dim  = p.nh * 128u;
    const uint  conv_dim = 3u * qkv_dim;
    const float exp_alog = exp(A_log[h]);
    constexpr float kInvSqrt128 = 0.08838834764831845f;

    device const float4* dt4_h = reinterpret_cast<device const float4*>(dt_bias + ulong(h) * 128ul);
    const float4 dt_v = dt4_h[lane];

    // 1. Load & L2-normalize `k_hat` into `K_tg`, compute per-channel log-decay `g` into `u_g.G`, compute `beta_tg`
    for (uint r = sgid; r < 32u; r += 4u) {
        const uint t = t_base + r;
        if (t < p.M_valid) {
            device const half4* k_raw4 = reinterpret_cast<device const half4*>(
                qkv_conv + ulong(t) * conv_dim + ulong(qkv_dim) + ulong(h) * 128ul);
            const float4 kv    = float4(k_raw4[lane]);
            const float  ksq   = simd_sum(dot(kv, kv));
            const float  inv_k = rsqrt(ksq + p.l2_eps);
            reinterpret_cast<threadgroup half4*>(&K_tg[r][0])[lane] = sg53_clamp_f16x4(kv * inv_k);

            device const float4* a4 = reinterpret_cast<device const float4*>(
                a_in + ulong(t) * qkv_dim + ulong(h) * 128ul);
            const float4 av = a4[lane] + dt_v;
            const float4 gv = float4(p.gate_lower_bound) / (float4(1.0f) + exp(-float4(exp_alog) * av));
            reinterpret_cast<threadgroup float4*>(&u_g.G[r][0])[lane] = gv;

            if (lane == 0u) {
                beta_tg[r] = sg53_sigmoid(b_in[ulong(t) * p.nh + h]);
            }
        } else {
            reinterpret_cast<threadgroup half4*>(&K_tg[r][0])[lane]   = half4(0.0h);
            reinterpret_cast<threadgroup float4*>(&u_g.G[r][0])[lane] = float4(0.0f);
            if (lane == 0u) {
                beta_tg[r] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Cumulative per-channel prefix sum `G[r, dk]` along `r = 0..31` (1 thread per `dk = tid`)
    {
        float run = 0.0f;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 32u; ++r) {
            run += u_g.G[r][tid];
            u_g.G[r][tid] = run;
        }
        g_last_out[(ulong(h) * p.num_chunks + c) * 128ul + tid] = exp(run);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Fused pass: write `q_sc`, `k_end`, lower-tri `P[r, j]`, strictly-lower `M[r, j]` into `T_tg`
    const float4 G_31  = reinterpret_cast<threadgroup float4*>(&u_g.G[31][0])[lane];
    const ulong  h_off = ulong(h) * p.M_padded * 128ul;
    device half* p_blk = p_out + (ulong(h) * p.num_chunks + c) * 1024ul;

    for (uint r = sgid; r < 32u; r += 4u) {
        const uint   t    = t_base + r;
        const float4 G_r  = reinterpret_cast<threadgroup float4*>(&u_g.G[r][0])[lane];
        const float4 k_r  = float4(reinterpret_cast<threadgroup half4*>(&K_tg[r][0])[lane]);
        float4 q_r        = float4(0.0f);
        if (t < p.M_valid) {
            device const half4* q_raw4 = reinterpret_cast<device const half4*>(
                qkv_conv + ulong(t) * conv_dim + ulong(h) * 128ul);
            const float4 qv    = float4(q_raw4[lane]);
            const float  qsq   = simd_sum(dot(qv, qv));
            const float  inv_q = rsqrt(qsq + p.l2_eps) * kInvSqrt128;
            q_r = qv * inv_q;
        }

        reinterpret_cast<device half4*>(q_sc_out  + h_off + ulong(t) * 128ul)[lane] = sg53_clamp_f16x4(q_r * exp(G_r));
        reinterpret_cast<device half4*>(k_end_out + h_off + ulong(t) * 128ul)[lane] = sg53_clamp_f16x4(k_r * exp(G_31 - G_r));

        const float beta_r = beta_tg[r];
        for (uint j = 0u; j <= r; ++j) {
            const float4 G_j = reinterpret_cast<threadgroup float4*>(&u_g.G[j][0])[lane];
            const float4 k_j = float4(reinterpret_cast<threadgroup half4*>(&K_tg[j][0])[lane]);
            const float4 e4  = exp(G_r - G_j) * k_j;
            const float  prj = simd_sum(dot(q_r, e4));
            if (lane == 0u) {
                p_blk[r * 32u + j] = sg53_clamp_f16(prj);
            }
            if (j < r) {
                const float mrj = beta_r * simd_sum(dot(k_r, e4));
                if (lane == 0u) {
                    T_tg[r][j] = mrj;
                }
            }
        }
        if (lane > r) {
            p_blk[r * 32u + lane] = half(0.0h);
        }
        if (lane >= r) {
            T_tg[r][lane] = (lane == r) ? 1.0f : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Overwrite `K_tg` with `K_sc = beta * k_hat * exp(G)` while `u_g.G` is still live, then invert `T = (I + M)^-1`
    for (uint r = sgid; r < 32u; r += 4u) {
        const float4 G_r = reinterpret_cast<threadgroup float4*>(&u_g.G[r][0])[lane];
        const float4 k_r = float4(reinterpret_cast<threadgroup half4*>(&K_tg[r][0])[lane]);
        reinterpret_cast<threadgroup half4*>(&K_tg[r][0])[lane] = sg53_clamp_f16x4(k_r * (float4(beta_tg[r]) * exp(G_r)));
    }
    if (sgid == 0u) {
        for (uint r = 1u; r < 32u; ++r) {
            float acc = (lane < r) ? -T_tg[r][lane] : 0.0f;
            for (uint j = lane + 1u; j < r; ++j) {
                acc = fma(-T_tg[r][j], T_tg[j][lane], acc);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < r) {
                T_tg[r][lane] = acc;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. `u_g.G` is now dead -> cast `T_tg` (FP32) into `u_g.Th` (FP16)
    for (uint r = sgid; r < 32u; r += 4u) {
        u_g.Th[r][lane] = sg53_clamp_f16(T_tg[r][lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 6. NAX `32 x 64 x 32` (`execution_simdgroups<4>`) for `W = T_h @ K_sc` and `U = T_h @ V_sc`
    tensor<threadgroup half, extents<int, 32, 32>,  tensor_inline> tTh(&u_g.Th[0][0], extents<int, 32, 32>());
    tensor<threadgroup half, extents<int, 128, 32>, tensor_inline> tKV(&K_tg[0][0],   extents<int, 128, 32>());

    constexpr auto desc_32x64 = matmul2d_descriptor(
        32, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_32x64, execution_simdgroups<4>> op_32x64;
    auto sKV0 = tKV.slice<64, 32>(0, 0);
    auto sKV1 = tKV.slice<64, 32>(64, 0);
    auto c64  = op_32x64.template get_destination_cooperative_tensor<sg53_strip_as<decltype(tTh)>, sg53_strip_as<decltype(sKV0)>, float>();

    device half* w_dst = w_out + h_off + ulong(t_base) * 128ul;
    op_32x64.run(tTh, sKV0, c64);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < c64.get_capacity(); ++i) {
        if (c64.is_valid_element(i)) {
            auto idx = c64.get_multidimensional_index(i);
            w_dst[ulong(uint(idx[1])) * 128ul + uint(idx[0])] = sg53_clamp_f16(c64[i]);
        }
    }
    op_32x64.run(tTh, sKV1, c64);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < c64.get_capacity(); ++i) {
        if (c64.is_valid_element(i)) {
            auto idx = c64.get_multidimensional_index(i);
            w_dst[ulong(uint(idx[1])) * 128ul + 64u + uint(idx[0])] = sg53_clamp_f16(c64[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load `V_sc = beta * v` into `K_tg`
    for (uint r = sgid; r < 32u; r += 4u) {
        const uint t = t_base + r;
        if (t < p.M_valid) {
            device const half4* v4 = reinterpret_cast<device const half4*>(
                qkv_conv + ulong(t) * conv_dim + 2ul * qkv_dim + ulong(h) * 128ul);
            reinterpret_cast<threadgroup half4*>(&K_tg[r][0])[lane] = sg53_clamp_f16x4(float4(v4[lane]) * beta_tg[r]);
        } else {
            reinterpret_cast<threadgroup half4*>(&K_tg[r][0])[lane] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half* u_dst = u_out + h_off + ulong(t_base) * 128ul;
    op_32x64.run(tTh, sKV0, c64);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < c64.get_capacity(); ++i) {
        if (c64.is_valid_element(i)) {
            auto idx = c64.get_multidimensional_index(i);
            u_dst[ulong(uint(idx[1])) * 128ul + uint(idx[0])] = sg53_clamp_f16(c64[i]);
        }
    }
    op_32x64.run(tTh, sKV1, c64);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < c64.get_capacity(); ++i) {
        if (c64.is_valid_element(i)) {
            auto idx = c64.get_multidimensional_index(i);
            u_dst[ulong(uint(idx[1])) * 128ul + 64u + uint(idx[0])] = sg53_clamp_f16(c64[i]);
        }
    }
#else
    (void)qkv_conv; (void)a_in; (void)b_in; (void)A_log; (void)dt_bias;
    (void)q_sc_out; (void)k_end_out; (void)w_out; (void)u_out; (void)p_out; (void)g_last_out;
    (void)p; (void)tid; (void)sgid; (void)lane;
#endif
}

// Kernel 2: `(4, nh, 1)` = `256` TGs, `64` th/TG (`sg2`, `14.5 KiB` TG mem <= 16 KiB -> 2x occupancy, 0 spill!)
kernel void sg53_kda_wy_recur_chunk32(
    device const half*        q_sc_in   [[buffer(0)]], // [nh, M_padded, 128]
    device const half*        k_end_in  [[buffer(1)]], // [nh, M_padded, 128]
    device const half*        w_in      [[buffer(2)]], // [nh, M_padded, 128]
    device const half*        u_in      [[buffer(3)]], // [nh, M_padded, 128]
    device const half*        p_in      [[buffer(4)]], // [nh, num_chunks, 32, 32]
    device const float*       g_last_in [[buffer(5)]], // [nh, num_chunks, 128]
    device float*             rec_state [[buffer(6)]], // [nh, 128, 128]
    device float*             raw_out   [[buffer(7)]], // [M_padded, nh * 128]
    constant Sg53KdaWyParams& p         [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],       // (0..3, 0..nh-1)
    uint  tid  [[thread_index_in_threadgroup]]         // 0..63
) {
    const uint col_blk = tgid.x;
    const uint h       = tgid.y;
    if (col_blk >= 4u || h >= p.nh) return;

#if defined(__HAVE_TENSOR__)
    const uint dv0   = col_blk * 32u;
    const uint v_dim = p.nh * 128u;

    threadgroup half  S_tg[128][32]; // 8 KiB
    threadgroup float O_tg[32][32];  // 4 KiB
    threadgroup half  D_tg[32][32];  // 2 KiB
    threadgroup float g_tg[128];     // 512 B -> Total = 14.5 KiB <= 16 KiB!

    device float* state_h = rec_state + ulong(h) * 16384ul;
    tensor<device float, dextents<int, 2>, tensor_inline> tRec(state_h, dextents<int, 2>(128, 128));
    auto sRec0 = tRec.slice<32, 64>(int(dv0), 0);
    auto sRec1 = tRec.slice<32, 64>(int(dv0), 64);

    const ulong h_off = ulong(h) * p.M_padded * 128ul;
    tensor<device half, dextents<int, 2>, tensor_inline> tQ(const_cast<device half*>(q_sc_in  + h_off), dextents<int, 2>(128, int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tK(const_cast<device half*>(k_end_in + h_off), dextents<int, 2>(128, int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tW(const_cast<device half*>(w_in     + h_off), dextents<int, 2>(128, int(p.M_padded)));
    device const half*  u_h = u_in      + h_off;
    device const half*  p_h = p_in      + ulong(h) * p.num_chunks * 1024ul;
    device const float* g_h = g_last_in + ulong(h) * p.num_chunks * 128ul;
    tensor<device float, dextents<int, 2>, tensor_inline> tRaw(raw_out, dextents<int, 2>(int(v_dim), int(p.M_padded)));

    tensor<threadgroup half,  extents<int, 32, 128>, tensor_inline> tSh(&S_tg[0][0], extents<int, 32, 128>());
    tensor<threadgroup float, extents<int, 32, 32>,  tensor_inline> tO (&O_tg[0][0], extents<int, 32, 32>());
    tensor<threadgroup half,  extents<int, 32, 32>,  tensor_inline> tD (&D_tg[0][0], extents<int, 32, 32>());

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
    auto cS0 = op_upd.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sK0_init)>, sg53_strip_as<decltype(tD)>, float>();
    auto cS1 = op_upd.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sK0_init)>, sg53_strip_as<decltype(tD)>, float>();
    cS0.load(sRec0);
    cS1.load(sRec1);

    auto sW_init = tW.slice<128, 32>(0, 0);
    auto cWS = op_ws.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sW_init)>, sg53_strip_as<decltype(tSh)>, float>();

    for (uint c = 0u; c < p.num_chunks; ++c) {
        const uint t_base = c * 32u;

        // 1. Cast pure-FP32 `cS0/cS1` -> `S_tg[128][32]` (`half`) + load `g_tg[0..127]`
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cS0.get_capacity(); ++i) {
            if (cS0.is_valid_element(i)) {
                auto idx      = cS0.get_multidimensional_index(i);
                const uint dk = uint(idx[1]);
                const uint dv = uint(idx[0]);
                S_tg[dk][dv]       = sg53_clamp_f16(cS0[i]);
                S_tg[dk + 64u][dv] = sg53_clamp_f16(cS1[i]);
            }
        }
        g_tg[tid]       = g_h[ulong(c) * 128ul + tid];
        g_tg[tid + 64u] = g_h[ulong(c) * 128ul + tid + 64u];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Direct-device `Delta = U - W @ S_h` (`BK=128` NAX) -> `D_tg[32][32]`
        auto sW = tW.slice<128, 32>(0, int(t_base));
        op_ws.run(sW, tSh, cWS);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cWS.get_capacity(); ++i) {
            if (cWS.is_valid_element(i)) {
                auto idx     = cWS.get_multidimensional_index(i);
                const uint r = uint(idx[1]);
                const uint d = uint(idx[0]);
                D_tg[r][d]   = sg53_clamp_f16(float(u_h[ulong(t_base + r) * 128ul + dv0 + d]) - cWS[i]);
            }
        }

        // 3. Direct-device `O_inter = Q_sc @ S_h` (`BK=128` NAX, `S_tg` still live!)
        auto sQ = tQ.slice<128, 32>(0, int(t_base));
        op_ws.run(sQ, tSh, cWS);
        cWS.store(tO);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Direct-device `O = O_inter + P @ D_tg` -> coalesced `cO.store(sRaw)`!
        tensor<device half, extents<int, 32, 32>, tensor_inline> tP(const_cast<device half*>(p_h + ulong(c) * 1024ul), extents<int, 32, 32>());
        auto cO = op_32_acc.template get_destination_cooperative_tensor<sg53_strip_as<decltype(tP)>, sg53_strip_as<decltype(tD)>, float>();
        cO.load(tO);
        op_32_acc.run(tP, tD, cO);
        cO.store(tRaw.slice<32, 32>(int(h * 128u + dv0), int(t_base)));

        // 5. Per-channel `dk` decay `S = diag(g_last) @ S + K_end^T @ D_tg` (pure FP32!)
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cS0.get_capacity(); ++i) {
            if (cS0.is_valid_element(i)) {
                auto idx      = cS0.get_multidimensional_index(i);
                const uint dk = uint(idx[1]);
                cS0[i] *= g_tg[dk];
                cS1[i] *= g_tg[dk + 64u];
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
    (void)q_sc_in; (void)k_end_in; (void)w_in; (void)u_in; (void)p_in; (void)g_last_in;
    (void)rec_state; (void)raw_out; (void)p; (void)tid;
#endif
}

/* ============================================================================
 * 7. 12-Layer `DSA` Sparse NoPE `MLA` + `Glm5NextIndexer` (`11` Main + `1` MTP)
 * ============================================================================ */

struct Sg53IndexerParams {
    uint  M_valid;
    uint  idx_n_heads;       // 32
    uint  idx_head_dim;      // 128
    uint  index_kpool;       // 4
    uint  select_k_pools;    // 512 (`2048 / 4`)
    uint  seq_pos_start;
    uint  first_pool;
    uint  num_new_pools;
    float layernorm_eps;     // 1e-6f
    uint  max_pools_stride;
    uint  max_sel_words;     // `ceil(max_pools_stride / 32)` in `b_idx_sel_bits`
    uint  pad0;
};

// Full `nn.LayerNorm(128, eps = 1e-6f)` WITH Mean Subtraction & Bias on `wk_raw`,
// paired with `k_gate` into `cur_kg_out [M_valid, 2, idx_head_dim]`!
// Grid: `(M_valid, 1, 1)`, `128` threads/TG
kernel void sg53_indexer_k_layernorm(
    device const float*        wk_raw_in  [[buffer(0)]], // [M_valid, idx_head_dim]
    device const float*        k_gate_in  [[buffer(1)]], // [M_valid, idx_head_dim]
    device const float*        k_norm_w   [[buffer(2)]], // [idx_head_dim]
    device const float*        k_norm_b   [[buffer(3)]], // [idx_head_dim]
    device float*              cur_kg_out [[buffer(4)]], // [M_valid, 2, idx_head_dim] (0=raw_k, 1=k_gate)
    constant Sg53IndexerParams& p         [[buffer(5)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],           // 0..127
    uint sgid [[simdgroup_index_in_threadgroup]],        // 0..3
    uint lane [[thread_index_in_simdgroup]]              // 0..31
) {
    if (m >= p.M_valid || tid >= p.idx_head_dim) return;

    threadgroup float tg_red[4];
    threadgroup float tg_stat[2]; // [0]=mu, [1]=inv_std (array avoids -Wsometimes-uninitialized and tg_red[0] race!)
    const uint num_sgs = (p.idx_head_dim + 31u) >> 5u;
    const float xv = wk_raw_in[ulong(m) * p.idx_head_dim + tid];

    // 1. Mean `mu`
    float sg_s = simd_sum(xv);
    if (lane == 0u) tg_red[sgid] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float tot = simd_sum((lane < num_sgs) ? tg_red[lane] : 0.0f);
        if (lane == 0u) tg_stat[0] = tot / float(p.idx_head_dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mu = tg_stat[0];

    // 2. Variance `var`
    const float diff = xv - mu;
    float sg_v = simd_sum(diff * diff);
    if (lane == 0u) tg_red[sgid] = sg_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float tot = simd_sum((lane < num_sgs) ? tg_red[lane] : 0.0f);
        if (lane == 0u) tg_stat[1] = rsqrt(tot / float(p.idx_head_dim) + p.layernorm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float raw_k  = fma(diff * tg_stat[1], k_norm_w[tid], k_norm_b[tid]);
    const float k_gate = k_gate_in[ulong(m) * p.idx_head_dim + tid];

    device float* dst = cur_kg_out + ulong(m) * (2ul * p.idx_head_dim);
    dst[tid]                  = raw_k;
    dst[p.idx_head_dim + tid] = k_gate;
}

// Per-Channel 4-Way Softmax Pool Compression (`L[j, d] = k_gate[j, d] + ape[j, d]`, `P = softmax_{j=0..3}(L)`):
// Reads tokens `< seq_pos_start` from 16-slot `tail_ring` and `>= seq_pos_start` from `cur_kg_in`!
// Grid: `(num_new_pools, 1, 1)`, `128` threads/TG
kernel void sg53_indexer_update_pools(
    device const float*        tail_ring    [[buffer(0)]], // [16, 2, idx_head_dim]
    device const float*        cur_kg_in    [[buffer(1)]], // [M_valid, 2, idx_head_dim]
    device const float*        compress_ape [[buffer(2)]], // [4, idx_head_dim]
    device half*               pool_k_cache [[buffer(3)]], // [max_pools_alloc, idx_head_dim]
    constant Sg53IndexerParams& p           [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]              // 0..127
) {
    if (tgid >= p.num_new_pools || tid >= p.idx_head_dim) return;

    const uint pool_idx = p.first_pool + tgid;
    const uint base_pos = pool_idx * p.index_kpool;

    float rk[4];
    float lg[4];
    float max_l = -INFINITY;

    for (uint j = 0u; j < 4u; ++j) {
        const uint t_pos = base_pos + j;
        device const float* src = (t_pos < p.seq_pos_start)
            ? (tail_ring + ulong(t_pos & 15u) * (2ul * p.idx_head_dim))
            : (cur_kg_in + ulong(t_pos - p.seq_pos_start) * (2ul * p.idx_head_dim));
        rk[j] = src[tid];
        lg[j] = src[p.idx_head_dim + tid] + compress_ape[ulong(j) * p.idx_head_dim + tid];
        max_l = max(max_l, lg[j]);
    }

    float e0 = exp(lg[0] - max_l);
    float e1 = exp(lg[1] - max_l);
    float e2 = exp(lg[2] - max_l);
    float e3 = exp(lg[3] - max_l);
    float inv_s = 1.0f / ((e0 + e1) + (e2 + e3));

    float pooled = ((e0 * rk[0] + e1 * rk[1]) + (e2 * rk[2] + e3 * rk[3])) * inv_s;
    pool_k_cache[ulong(pool_idx) * p.idx_head_dim + tid] = sg53_clamp_f16(pooled);
}

// Commits the last `min(M_valid, 16)` `(raw_k, k_gate)` entries into the 16-slot `tail_ring` (`pos & 15`):
// Preserves prior uncommitted slots when `M_valid <= 5` (`spec_decode_step` rollback-safe!).
// Grid: `(min(M_valid, 16), 1, 1)`, `128` threads/TG
kernel void sg53_indexer_commit_tail(
    device const float*        cur_kg_in [[buffer(0)]], // [M_valid, 2, idx_head_dim]
    device float*              tail_ring [[buffer(1)]], // [16, 2, idx_head_dim]
    constant Sg53IndexerParams& p        [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]           // 0..127
) {
    const uint keep = min(p.M_valid, 16u);
    if (tgid >= keep || tid >= p.idx_head_dim) return;

    const uint m    = (p.M_valid - keep) + tgid;
    const uint slot = (p.seq_pos_start + m) & 15u;

    device const float* src = cur_kg_in + ulong(m) * (2ul * p.idx_head_dim);
    device float*       dst = tail_ring + ulong(slot) * (2ul * p.idx_head_dim);
    dst[tid]                  = src[tid];
    dst[p.idx_head_dim + tid] = src[p.idx_head_dim + tid];
}

// Scales `w_head_raw [M_valid, idx_n_heads]` by `1.0f / sqrt(float(idx_n_heads))`
kernel void sg53_indexer_scale_weights(
    device float*              w_head    [[buffer(0)]], // [M_valid, idx_n_heads]
    constant Sg53IndexerParams& p        [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint tot = p.M_valid * p.idx_n_heads;
    if (gid < tot) {
        w_head[gid] *= rsqrt(float(p.idx_n_heads));
    }
}

// Scores complete pools `b = 0 .. num_complete - 1`:
// `score[m, b] = sum_{h=0..31} w_head[m, h] * max(0.0f, dot_128(q_idx[m, h], pool_k[b]) / sqrt(128))`
// Grid: `(ceil(max_complete / 8), M_valid, 1)`, `256` threads/TG (`8` SGs = `8` pools/TG)
kernel void sg53_indexer_score_pools(
    device const float*        q_idx_in     [[buffer(0)]], // [M_valid, idx_n_heads, idx_head_dim]
    device const float*        w_head_in    [[buffer(1)]], // [M_valid, idx_n_heads] (pre-scaled by 1/sqrt(32))
    device const half*         pool_k_cache [[buffer(2)]], // [max_pools_alloc, idx_head_dim]
    device float*              scores_out   [[buffer(3)]], // [M_valid, max_pools_stride]
    constant Sg53IndexerParams& p           [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],         // 0..7
    uint  lane [[thread_index_in_simdgroup]]               // 0..31
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;

    const uint q_pos        = p.seq_pos_start + m;
    const uint num_complete = (q_pos + 1u) / p.index_kpool;
    if (num_complete <= p.select_k_pools) return;

    const uint b = tgid.x * 8u + sgid;
    if (b >= num_complete) return;

    const uint   d4_cnt = p.idx_head_dim >> 2u; // 32 for idx_head_dim=128
    const float  inv_hd = rsqrt(float(p.idx_head_dim));
    device const half4*  pk4 = reinterpret_cast<device const half4*>(pool_k_cache + ulong(b) * p.idx_head_dim);
    device const float*  wh  = w_head_in + ulong(m) * p.idx_n_heads;

    if (d4_cnt == 32u) {
        const float4 bk = float4(pk4[lane]);
        float total_sc  = 0.0f;
        for (uint h = 0u; h < p.idx_n_heads; ++h) {
            const float4 qh = reinterpret_cast<device const float4*>(
                q_idx_in + (ulong(m) * p.idx_n_heads + h) * 128ul)[lane];
            float d = simd_sum(dot(qh, bk)) * inv_hd;
            total_sc = fma(wh[h], max(0.0f, d), total_sc);
        }
        if (lane == 0u) {
            scores_out[ulong(m) * p.max_pools_stride + b] = total_sc;
        }
    } else {
        float total_sc = 0.0f;
        for (uint h = 0u; h < p.idx_n_heads; ++h) {
            device const float4* q4 = reinterpret_cast<device const float4*>(
                q_idx_in + (ulong(m) * p.idx_n_heads + h) * p.idx_head_dim);
            float acc = 0.0f;
            for (uint c4 = lane; c4 < d4_cnt; c4 += 32u) {
                acc += dot(q4[c4], float4(pk4[c4]));
            }
            float d = simd_sum(acc) * inv_hd;
            total_sc = fma(wh[h], max(0.0f, d), total_sc);
        }
        if (lane == 0u) {
            scores_out[ulong(m) * p.max_pools_stride + b] = total_sc;
        }
    }
}

// Tiled Long-Context Prefill Pool Scorer (`M_TILE = 8` queries x `B_TILE = 64` pools per TG, `256` threads = `8` SGs):
// Stages `pool_k` tiles in threadgroup memory so `pool_k_cache` DRAM traffic drops by `8x` at 16K..256K prefill!
// Grid: `((max_q_complete + 63) / 64, (M_valid + 7) / 8, 1)`, `256` threads/TG
kernel void sg53_indexer_score_pools_tiled(
    device const float4*       q_idx_in     [[buffer(0)]], // [M_valid, 32, 32]
    device const float*        w_head_in    [[buffer(1)]], // [M_valid, 32]
    device const half4*        pool_k_cache [[buffer(2)]], // [max_pools_alloc, 32]
    device float*              scores_out   [[buffer(3)]], // [M_valid, max_pools_stride]
    constant Sg53IndexerParams& p           [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],         // 0..7 (1 query per SG!)
    uint  lane [[thread_index_in_simdgroup]]               // 0..31
) {
    const uint m_tg = tgid.y * 8u;
    if (m_tg >= p.M_valid) return;

    const uint m_last          = min(m_tg + 7u, p.M_valid - 1u);
    const uint max_tg_complete = ((p.seq_pos_start + m_last) + 1u) / p.index_kpool;
    const uint b_tg            = tgid.x * 64u;
    if (max_tg_complete <= p.select_k_pools || b_tg >= max_tg_complete) return;

    const uint m = m_tg + sgid;
    uint q_comp  = 0u;
    float4 q_reg[32];
    float  w_reg[32];

    if (m < p.M_valid) {
        uint nc = ((p.seq_pos_start + m) + 1u) / p.index_kpool;
        q_comp  = (nc > p.select_k_pools) ? nc : 0u;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 32u; ++h) {
            q_reg[h] = q_idx_in[(ulong(m) * 32ul + h) * 32ul + lane];
            w_reg[h] = w_head_in[ulong(m) * 32ul + h];
        }
    } else {
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 32u; ++h) {
            q_reg[h] = float4(0.0f);
            w_reg[h] = 0.0f;
        }
    }

    threadgroup half4 tg_bk[8][32];
    const float inv_hd = rsqrt(128.0f);

    for (uint s = 0u; s < 8u; ++s) {
        const uint b_step = b_tg + s * 8u;
        if (b_step >= max_tg_complete) break;

        const uint b_load = b_step + sgid;
        tg_bk[sgid][lane] = (b_load < max_tg_complete)
            ? pool_k_cache[ulong(b_load) * 32ul + lane]
            : half4(0.0h);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint bi = 0u; bi < 8u; ++bi) {
            const uint b = b_step + bi;
            if (b >= max_tg_complete) break;
            if (b < q_comp) {
                const float4 bk = float4(tg_bk[bi][lane]);
                float tot_sc    = 0.0f;
                #pragma clang loop unroll(full)
                for (uint h = 0u; h < 32u; ++h) {
                    float d  = simd_sum(dot(q_reg[h], bk)) * inv_hd;
                    tot_sc   = fma(w_reg[h], max(0.0f, d), tot_sc);
                }
                if (lane == 0u) {
                    scores_out[ulong(m) * p.max_pools_stride + b] = tot_sc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Transposes `q_idx_in [M_valid, 32, 128]` (FP32) -> `q_idx_hm_f16 [32, M_padded, 128]` (FP16, zero-padded!)
// Grid: `(idx_n_heads, M_padded, 1)`, `128` threads/TG (`p.pad0` = `M_padded`)
kernel void sg53_indexer_q_to_hm_f16(
    device const float*         q_idx_in     [[buffer(0)]], // [M_valid, idx_n_heads, 128]
    device half*                q_idx_hm_f16 [[buffer(1)]], // [idx_n_heads, M_padded, 128]
    constant Sg53IndexerParams& p            [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],            // (h, m)
    uint  tid  [[thread_index_in_threadgroup]]              // 0..127
) {
    const uint h = tgid.x;
    const uint m = tgid.y;
    const uint M_pad = p.pad0;
    if (h >= p.idx_n_heads || m >= M_pad || tid >= p.idx_head_dim) return;

    const ulong dst_idx = (ulong(h) * M_pad + m) * p.idx_head_dim + tid;
    if (m < p.M_valid) {
        const ulong src_idx = (ulong(m) * p.idx_n_heads + h) * p.idx_head_dim + tid;
        q_idx_hm_f16[dst_idx] = sg53_clamp_f16(q_idx_in[src_idx]);
    } else {
        q_idx_hm_f16[dst_idx] = half(0.0h);
    }
}

// Metal 4.1 Direct-Device NAX Indexer Pool Scorer (`64x64x128 sg4`, ~48 TF!):
// `score[m, b] = (sum_{h=0..31} w_head[m, h] * max(0.0f, dot_128(q_idx[h, m], pool_k[b]))) * rsqrt(128)`
// Grid: `((max_q_complete + 63) / 64, M_padded / 64, 1)`, `128` threads/TG (`4` SGs; `p.pad0` = `M_padded`, `max_pools_stride % 64 == 0`)
kernel void sg53_indexer_score_pools_nax(
    device const half*          q_idx_hm_f16 [[buffer(0)]], // [32, M_padded, 128]
    device const float*         w_head_in    [[buffer(1)]], // [M_valid, 32]
    device const half*          pool_k_cache [[buffer(2)]], // [max_pools_stride, 128]
    device float*               scores_out   [[buffer(3)]], // [M_valid, max_pools_stride]
    constant Sg53IndexerParams& p            [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]              // 0..127
) {
    const uint b_tg  = tgid.x * 64u;
    const uint m_tg  = tgid.y * 64u;
    const uint M_pad = p.pad0;
    if (m_tg >= p.M_valid) return;

    const uint m_tg_last       = min(m_tg + 63u, p.M_valid - 1u);
    const uint max_tg_complete = (p.seq_pos_start + m_tg_last + 1u) / p.index_kpool;
    if (max_tg_complete <= p.select_k_pools || b_tg >= max_tg_complete) return;

#if defined(__HAVE_TENSOR__)
    // Stage `w_head[m_tg..m_tg+63, 0..31]` (`8 KiB`) cooperatively
    threadgroup float tg_wh[64][32];
    const uint r_wh = tid >> 5u; // 0..3
    const uint h_wh = tid & 31u; // 0..31
    #pragma clang loop unroll(full)
    for (uint s = 0u; s < 16u; ++s) {
        const uint r = s * 4u + r_wh;
        const uint m = m_tg + r;
        tg_wh[r][h_wh] = (m < p.M_valid) ? w_head_in[ulong(m) * 32ul + h_wh] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<device half, dextents<int, 2>, tensor_inline> tB(
        const_cast<device half*>(pool_k_cache), dextents<int, 2>(128, int(p.max_pools_stride)));
    auto sB = tB.slice<128, 64>(0, int(b_tg));

    constexpr auto desc = matmul2d_descriptor(
        64, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    const ulong head_stride = ulong(M_pad) * 128ul;
    tensor<device half, dextents<int, 2>, tensor_inline> tQ0(
        const_cast<device half*>(q_idx_hm_f16), dextents<int, 2>(128, int(M_pad)));
    auto sQ0 = tQ0.slice<128, 64>(0, int(m_tg));

    auto cSum = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sQ0)>, sg53_strip_as<decltype(sB)>, float>();
    auto cH   = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sQ0)>, sg53_strip_as<decltype(sB)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cSum.get_capacity(); ++i) {
        if (cSum.is_valid_element(i)) cSum[i] = 0.0f;
    }

    for (uint h = 0u; h < 32u; ++h) {
        tensor<device half, dextents<int, 2>, tensor_inline> tQh(
            const_cast<device half*>(q_idx_hm_f16) + ulong(h) * head_stride,
            dextents<int, 2>(128, int(M_pad)));
        auto sQh = tQh.slice<128, 64>(0, int(m_tg));
        op.run(sQh, sB, cH);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cSum.get_capacity(); ++i) {
            if (cSum.is_valid_element(i)) {
                auto idx = cSum.get_multidimensional_index(i);
                cSum[i] = fma(tg_wh[uint(idx[1])][h], max(0.0f, cH[i]), cSum[i]);
            }
        }
    }

    const float inv_sqrt_hd = rsqrt(128.0f);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cSum.get_capacity(); ++i) {
        if (cSum.is_valid_element(i)) {
            auto idx     = cSum.get_multidimensional_index(i);
            const uint m = m_tg + uint(idx[1]);
            const uint b = b_tg + uint(idx[0]);
            if (m < p.M_valid) {
                const uint nc = (p.seq_pos_start + m + 1u) / p.index_kpool;
                if (nc > p.select_k_pools && b < nc) {
                    scores_out[ulong(m) * ulong(p.max_pools_stride) + ulong(b)] = cSum[i] * inv_sqrt_hd;
                }
            }
        }
    }
#else
    (void)q_idx_hm_f16; (void)w_head_in; (void)pool_k_cache; (void)scores_out; (void)p; (void)tid;
#endif
}

// Metal 4.1 Decode / Spec-Verify (`M = 1..8`) Direct-Device NAX Indexer Pool Scorer (`32x64x128 sg4`):
// Uses ALL 4 SGs (100% active even at M=1!), zero register spilling, 100.0000% bit-exact between M=1 and M=5!
// Grid: `((max_q_complete + 63) / 64, M_valid, 1)`, `128` threads/TG (`idx_n_heads == 32`, `idx_head_dim == 128`, `max_pools_stride % 64 == 0`)
kernel void sg53_indexer_score_pools_decode_nax(
    device const float4*        q_idx_in     [[buffer(0)]], // [M_valid, 32, 32] (float4 = 4 floats, 128 floats/head)
    device const float*         w_head_in    [[buffer(1)]], // [M_valid, 32]
    device const half*          pool_k_cache [[buffer(2)]], // [max_pools_stride, 128]
    device float*               scores_out   [[buffer(3)]], // [M_valid, max_pools_stride]
    constant Sg53IndexerParams& p            [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],             // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]],          // 0..3
    uint  lane [[thread_index_in_simdgroup]]                // 0..31
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;

    const uint nc   = (p.seq_pos_start + m + 1u) / p.index_kpool;
    const uint b_tg = tgid.x * 64u;
    if (nc <= p.select_k_pools || b_tg >= nc) return;

#if defined(__HAVE_TENSOR__)
    threadgroup half  tg_q[32][128];   // 8 KiB: Q_idx[m, 0..31, 0..127] clamped to FP16
    threadgroup float tg_wh[32];       // 128 B: w_head[m, 0..31]
    threadgroup float tg_sc[64][33];   // 8.4 KiB: [pool 0..63][head 0..31] (stride 33 = zero bank conflicts!)

    const ulong q_tok_off = ulong(m) * 32ul;
    #pragma clang loop unroll(full)
    for (uint s = 0u; s < 8u; ++s) {
        const uint   h  = s * 4u + sgid;
        const float4 qv = q_idx_in[(q_tok_off + h) * 32ul + lane];
        reinterpret_cast<threadgroup half4*>(&tg_q[h][0])[lane] = half4(
            sg53_clamp_f16(qv.x), sg53_clamp_f16(qv.y),
            sg53_clamp_f16(qv.z), sg53_clamp_f16(qv.w));
    }
    if (tid < 32u) {
        tg_wh[tid] = w_head_in[q_tok_off + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup half, extents<int, 128, 32>, tensor_inline> tQ(&tg_q[0][0], extents<int, 128, 32>());
    tensor<device half, dextents<int, 2>, tensor_inline> tB(
        const_cast<device half*>(pool_k_cache), dextents<int, 2>(128, int(p.max_pools_stride)));
    auto sB = tB.slice<128, 64>(0, int(b_tg));

    constexpr auto desc = matmul2d_descriptor(
        32, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto cH = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(tQ)>, sg53_strip_as<decltype(sB)>, float>();
    op.run(tQ, sB, cH);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cH.get_capacity(); ++i) {
        if (cH.is_valid_element(i)) {
            auto idx     = cH.get_multidimensional_index(i);
            const uint b = uint(idx[0]); // 0..63
            const uint h = uint(idx[1]); // 0..31
            tg_sc[b][h]  = tg_wh[h] * max(0.0f, cH[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_sqrt_hd = rsqrt(128.0f);
    const ulong out_base    = ulong(m) * ulong(p.max_pools_stride) + ulong(b_tg);
    #pragma clang loop unroll(full)
    for (uint s = 0u; s < 16u; ++s) {
        const uint p_loc = s * 4u + sgid;
        const float tot  = simd_sum(tg_sc[p_loc][lane]) * inv_sqrt_hd;
        if (lane == 0u && (b_tg + p_loc) < nc) {
            scores_out[out_base + p_loc] = tot;
        }
    }
#else
    (void)q_idx_in; (void)w_head_in; (void)pool_k_cache; (void)scores_out; (void)p;
    (void)tid; (void)sgid; (void)lane;
#endif
}

// 4-Pass 8-Bit Signed IEEE-754 Radix Select + Device Bitset Chronological Pool Selector (`0 .. 1M` tokens!)
// Grid: `(M_valid, 1, 1)`, `1024` threads/TG (`32` SGs)
kernel void sg53_indexer_topk_pools(
    device const float*        scores_in     [[buffer(0)]], // [M_valid, max_pools_stride]
    device atomic_uint*        sel_bits_buf  [[buffer(1)]], // [M_valid, max_sel_words]
    device uint*               sel_pools_out [[buffer(2)]], // [M_valid, select_k_pools]
    constant Sg53IndexerParams& p            [[buffer(3)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],              // 0..1023
    uint sgid [[simdgroup_index_in_threadgroup]],           // 0..31
    uint lane [[thread_index_in_simdgroup]]                 // 0..31
) {
    if (m >= p.M_valid) return;

    const uint q_pos        = p.seq_pos_start + m;
    const uint raw_complete = (q_pos + 1u) / p.index_kpool;
    const uint num_complete = min(raw_complete, p.max_sel_words * 32u);
    if (num_complete <= p.select_k_pools) return;

    threadgroup atomic_uint hist[256];
    threadgroup uint        tg_state[4];
    threadgroup uint        tg_sg_cnt[32];

    device const float* row_scores = scores_in + ulong(m) * p.max_pools_stride;
    device atomic_uint* sel_bits   = sel_bits_buf + ulong(m) * p.max_sel_words;
    device uint*        row_out    = sel_pools_out + ulong(m) * p.select_k_pools;

    uint prefix_val  = 0u;
    uint prefix_mask = 0u;
    uint rem_k       = p.select_k_pools;

    // 1. 4-pass 8-bit MSB->LSB signed IEEE-754 radix select
    for (uint pass = 0u; pass < 4u; ++pass) {
        const uint shift = 24u - pass * 8u;
        if (tid < 256u) {
            atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = tid; b < num_complete; b += 1024u) {
            uint u = sg53_f32_to_sortable_u32(row_scores[b]);
            if ((u & prefix_mask) == prefix_val) {
                uint bkt = (u >> shift) & 0xFFu;
                atomic_fetch_add_explicit(&hist[bkt], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint acc    = 0u;
            uint chosen = 0u;
            for (int bkt = 255; bkt >= 0; --bkt) {
                uint c = atomic_load_explicit(&hist[uint(bkt)], memory_order_relaxed);
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
    const uint num_words = min(p.max_sel_words, (num_complete + 31u) >> 5u);

    // Fast TG-coherent bitset for `<= 131,072` tokens (`num_words <= 1024`, `4 KiB` TG mem, 0 device atomic races!),
    // with device-fenced fallback for `131K .. 1M` tokens!
    threadgroup atomic_uint tg_sel[1024];
    const bool use_tg_sel = (num_words <= 1024u);

    if (use_tg_sel) {
        if (tid < num_words) atomic_store_explicit(&tg_sel[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (uint w = tid; w < num_words; w += 1024u) {
            atomic_store_explicit(&sel_bits[w], 0u, memory_order_relaxed);
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_threadgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    // 2. Mark all pools strictly `> thresh_u`
    for (uint b = tid; b < num_complete; b += 1024u) {
        uint u = sg53_f32_to_sortable_u32(row_scores[b]);
        if (u > thresh_u) {
            if (use_tg_sel) atomic_fetch_or_explicit(&tg_sel[b >> 5u],   1u << (b & 31u), memory_order_relaxed);
            else            atomic_fetch_or_explicit(&sel_bits[b >> 5u], 1u << (b & 31u), memory_order_relaxed);
        }
    }
    if (use_tg_sel) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_threadgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    // 3. Mark first `rem_k` ties (`u == thresh_u`) in deterministic lowest-`b`-first order
    uint ties_taken = 0u;
    for (uint base = 0u; base < num_complete && ties_taken < rem_k; base += 1024u) {
        const uint b      = base + tid;
        const bool is_tie = (b < num_complete) && (sg53_f32_to_sortable_u32(row_scores[b]) == thresh_u);
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
        if (is_tie && rank < rem_k) {
            if (use_tg_sel) atomic_fetch_or_explicit(&tg_sel[b >> 5u],   1u << (b & 31u), memory_order_relaxed);
            else            atomic_fetch_or_explicit(&sel_bits[b >> 5u], 1u << (b & 31u), memory_order_relaxed);
        }
        ties_taken += tg_state[3];
        if (use_tg_sel) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_threadgroup);
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
    }

    // 4. Emit `select_k_pools` indices in strictly ascending chronological order
    uint written = 0u;
    for (uint w_base = 0u; w_base < num_words; w_base += 1024u) {
        const uint w_idx = w_base + tid;
        uint       word  = 0u;
        if (w_idx < num_words) {
            word = use_tg_sel ? atomic_load_explicit(&tg_sel[w_idx],   memory_order_relaxed)
                              : atomic_load_explicit(&sel_bits[w_idx], memory_order_relaxed);
        }
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
            if (out_pos < p.select_k_pools) {
                row_out[out_pos] = w_idx * 32u + bit;
            }
            ++out_pos;
            word &= (word - 1u);
        }
        written += tg_state[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct Sg53MlaBatchedGemmParams {
    uint M_valid;
    uint M_padded;
    uint nh; // 64
    uint N;  // 512 for W_UK, 256 for W_UV
    uint K;  // 256 for W_UK, 512 for W_UV
    uint pad0;
    uint pad1;
    uint pad2;
};

// Zero-spill Batched Per-Head FP16 GEMV (`M <= 8`, `2` rows/SG = `8` rows/TG):
// Reads `W[h]` (`16 MiB` total) ONCE across `m = 0 .. M_valid - 1`!
// Grid: `((N + 7) / 8, nh, 1)`, `128` threads/TG (`4` SGs, `2` rows/SG)
template <uint M_FIX>
inline void sg53_mla_f16_gemv2_fixed(
    device const half* w_f16,
    device const half* x_in,
    device half* y_out,
    constant Sg53MlaBatchedGemmParams& p,
    uint row0,
    uint h,
    uint lane
) {
    const uint c4_cnt = p.K >> 2u;
    const uint r0 = min(row0 + 0u, p.N - 1u);
    const uint r1 = min(row0 + 1u, p.N - 1u);

    device const half*  wh = w_f16 + ulong(h) * p.N * p.K;
    device const half4* w0 = reinterpret_cast<device const half4*>(wh + ulong(r0) * p.K);
    device const half4* w1 = reinterpret_cast<device const half4*>(wh + ulong(r1) * p.K);

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint c4 = lane; c4 < c4_cnt; c4 += 32u) {
        const float4 fw0 = float4(w0[c4]);
        const float4 fw1 = float4(w1[c4]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xh = reinterpret_cast<device const half4*>(
                x_in + (ulong(m) * p.nh + h) * p.K);
            const float4 fx = float4(xh[c4]);
            acc[m].x = fma(fw0.x, fx.x, fma(fw0.y, fx.y, fma(fw0.z, fx.z, fma(fw0.w, fx.w, acc[m].x))));
            acc[m].y = fma(fw1.x, fx.x, fma(fw1.y, fx.y, fma(fw1.z, fx.z, fma(fw1.w, fx.w, acc[m].y))));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        float2 s(simd_sum(acc[m].x), simd_sum(acc[m].y));
        if (lane == 0u) {
            const ulong base = (ulong(m) * p.nh + h) * p.N + row0;
            y_out[base + 0u] = sg53_clamp_f16(s.x);
            if (row0 + 1u < p.N) y_out[base + 1u] = sg53_clamp_f16(s.y);
        }
    }
    if (lane == 0u && p.M_padded > M_FIX) {
        for (uint m = M_FIX; m < p.M_padded; ++m) {
            const ulong base = (ulong(m) * p.nh + h) * p.N + row0;
            y_out[base + 0u] = half(0.0h);
            if (row0 + 1u < p.N) y_out[base + 1u] = half(0.0h);
        }
    }
}

kernel void sg53_mla_batched_f16_gemv(
    device const half*              w_f16 [[buffer(0)]], // [nh, N, K]
    device const half*              x_in  [[buffer(1)]], // [M_valid, nh, K]
    device half*                    y_out [[buffer(2)]], // [M_padded, nh, N]
    constant Sg53MlaBatchedGemmParams& p  [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],       // 0..3
    uint  lane [[thread_index_in_simdgroup]]             // 0..31
) {
    const uint row0 = (tgid.x * 4u + sgid) * 2u;
    const uint h    = tgid.y;
    if (row0 >= p.N || h >= p.nh) return;

    #define SG53_MLA_CASE(MF) { sg53_mla_f16_gemv2_fixed<MF>(w_f16, x_in, y_out, p, row0, h, lane); break; }
    switch (p.M_valid) {
        case 1u: SG53_MLA_CASE(1u)
        case 2u: SG53_MLA_CASE(2u)
        case 3u: SG53_MLA_CASE(3u)
        case 4u: SG53_MLA_CASE(4u)
        case 5u: SG53_MLA_CASE(5u)
        case 6u: SG53_MLA_CASE(6u)
        case 7u: SG53_MLA_CASE(7u)
        case 8u: SG53_MLA_CASE(8u)
        default: break;
    }
    #undef SG53_MLA_CASE
}

// Compile-time M-specialized (`M = 1..5`) Per-Head FP16 MLA GEMV (eliminates `case 6..8` register overhead!)
kernel void sg53_mla_batched_f16_gemv_spec(
    device const half*                 w_f16 [[buffer(0)]], // [nh, N, K]
    device const half*                 x_in  [[buffer(1)]], // [M_valid, nh, K]
    device half*                       y_out [[buffer(2)]], // [M_padded, nh, N]
    constant Sg53MlaBatchedGemmParams& p     [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid.x * 4u + sgid) * 2u;
    const uint h    = tgid.y;
    if (row0 >= p.N || h >= p.nh) return;

    const uint c4_cnt = p.K >> 2u;
    const uint r0 = min(row0 + 0u, p.N - 1u);
    const uint r1 = min(row0 + 1u, p.N - 1u);
    device const half*  wh = w_f16 + ulong(h) * p.N * p.K;
    device const half4* w0 = reinterpret_cast<device const half4*>(wh + ulong(r0) * p.K);
    device const half4* w1 = reinterpret_cast<device const half4*>(wh + ulong(r1) * p.K);

    float2 acc[8];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < kSg53GemvM; ++m) acc[m] = float2(0.0f);

    for (uint c4 = lane; c4 < c4_cnt; c4 += 32u) {
        const float4 fw0 = float4(w0[c4]);
        const float4 fw1 = float4(w1[c4]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < kSg53GemvM; ++m) {
            device const half4* xh = reinterpret_cast<device const half4*>(x_in + (ulong(m) * p.nh + h) * p.K);
            const float4 fx = float4(xh[c4]);
            acc[m].x = fma(fw0.x, fx.x, fma(fw0.y, fx.y, fma(fw0.z, fx.z, fma(fw0.w, fx.w, acc[m].x))));
            acc[m].y = fma(fw1.x, fx.x, fma(fw1.y, fx.y, fma(fw1.z, fx.z, fma(fw1.w, fx.w, acc[m].y))));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < kSg53GemvM; ++m) {
        float2 s(simd_sum(acc[m].x), simd_sum(acc[m].y));
        if (lane == 0u) {
            const ulong base = (ulong(m) * p.nh + h) * p.N + row0;
            y_out[base + 0u] = sg53_clamp_f16(s.x);
            if (row0 + 1u < p.N) y_out[base + 1u] = sg53_clamp_f16(s.y);
        }
    }
    if (lane == 0u && p.M_padded > kSg53GemvM) {
        for (uint m = kSg53GemvM; m < p.M_padded; ++m) {
            const ulong base = (ulong(m) * p.nh + h) * p.N + row0;
            y_out[base + 0u] = half(0.0h);
            if (row0 + 1u < p.N) y_out[base + 1u] = half(0.0h);
        }
    }
}

#if defined(__HAVE_TENSOR__)
template <int M_TILE>
inline void sg53_mla_batched_f16_gemm_nax_impl(
    device const half*                 w_f16, // [nh, N, K]
    device const half*                 x_in,  // [M_valid, nh, K]
    device half*                       y_out, // [M_padded, nh, N]
    constant Sg53MlaBatchedGemmParams& p,
    threadgroup half                   (&X_tile)[M_TILE][64],
    threadgroup half                   (&W_tile)[64][64],
    uint3                              tgid,
    uint                               tid
) {
    const uint n0 = tgid.x * 64u;
    const uint m0 = tgid.y * uint(M_TILE);
    const uint h  = tgid.z;
    if (n0 >= p.N || m0 >= p.M_padded || h >= p.nh) return;

    tensor<threadgroup half, extents<int, 64, M_TILE>, tensor_inline> tX(&X_tile[0][0], extents<int, 64, M_TILE>());
    tensor<threadgroup half, extents<int, 64, 64>,     tensor_inline> tW(&W_tile[0][0], extents<int, 64, 64>());

    constexpr auto desc = matmul2d_descriptor(
        M_TILE, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto cT = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(tX)>, sg53_strip_as<decltype(tW)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint k_tiles = p.K >> 6u;
    const uint r_grp   = tid >> 3u; // 0..15
    const uint sub     = tid & 7u;  // 0..7
    device const half* wh = w_f16 + ulong(h) * p.N * p.K;

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        // Cooperatively stage `X_tile[M_TILE][64]` for head `h`
        constexpr uint XM_STEPS = uint(M_TILE) / 16u;
        #pragma clang loop unroll(full)
        for (uint ms = 0u; ms < XM_STEPS; ++ms) {
            const uint m_loc  = ms * 16u + r_grp;
            const uint m_glob = m0 + m_loc;
            threadgroup half4* dx = reinterpret_cast<threadgroup half4*>(&X_tile[m_loc][sub * 8u]);
            if (m_glob < p.M_valid) {
                device const half4* sx = reinterpret_cast<device const half4*>(
                    x_in + (ulong(m_glob) * p.nh + h) * p.K + kt * 64u + sub * 8u);
                dx[0] = sx[0];
                dx[1] = sx[1];
            } else {
                dx[0] = half4(0.0h);
                dx[1] = half4(0.0h);
            }
        }

        // Cooperatively stage `W_tile[64][64]` for head `h`
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 4u; ++q) {
            const uint r_loc  = q * 16u + r_grp;
            const uint r_glob = n0 + r_loc;
            threadgroup half4* dw = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 8u]);
            if (r_glob < p.N) {
                device const half4* sw = reinterpret_cast<device const half4*>(
                    wh + ulong(r_glob) * p.K + kt * 64u + sub * 8u);
                dw[0] = sw[0];
                dw[1] = sw[1];
            } else {
                dw[0] = half4(0.0h);
                dw[1] = half4(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_padded && n_glob < p.N) {
                y_out[(ulong(m_glob) * p.nh + h) * p.N + n_glob] =
                    (m_glob < p.M_valid) ? sg53_clamp_f16(cT[i]) : half(0.0h);
            }
        }
    }
}
#endif

kernel void sg53_mla_batched_f16_gemm_nax_m16(
    device const half*                 w_f16 [[buffer(0)]],
    device const half*                 x_in  [[buffer(1)]],
    device half*                       y_out [[buffer(2)]],
    constant Sg53MlaBatchedGemmParams& p     [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half X_tile[16][64];
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sg53_mla_batched_f16_gemm_nax_impl<16>(w_f16, x_in, y_out, p, X_tile, W_tile, tgid, tid);
#else
    (void)w_f16; (void)x_in; (void)y_out; (void)p; (void)X_tile; (void)W_tile; (void)tgid; (void)tid;
#endif
}

kernel void sg53_mla_batched_f16_gemm_nax_m64(
    device const half*                 w_f16 [[buffer(0)]],
    device const half*                 x_in  [[buffer(1)]],
    device half*                       y_out [[buffer(2)]],
    constant Sg53MlaBatchedGemmParams& p     [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half X_tile[64][64];
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sg53_mla_batched_f16_gemm_nax_impl<64>(w_f16, x_in, y_out, p, X_tile, W_tile, tgid, tid);
#else
    (void)w_f16; (void)x_in; (void)y_out; (void)p; (void)X_tile; (void)W_tile; (void)tgid; (void)tid;
#endif
}

// Transposes `q_in [M_valid, nh, K]` -> head-major `q_hm_out [nh, M_padded, K]` (zero-padded for `m >= M_valid`)
// Grid: `(nh, M_padded, 1)`, `64` threads/TG (`K` multiple of 4; `64 * half4 = 256` halfs in 1 shot!)
kernel void sg53_mla_q_to_hm_f16(
    device const half*                 q_in     [[buffer(0)]], // [M_valid, nh, K]
    device half*                       q_hm_out [[buffer(1)]], // [nh, M_padded, K]
    constant Sg53MlaBatchedGemmParams& p        [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],               // (h, m)
    uint  tid  [[thread_index_in_threadgroup]]                 // 0..63
) {
    const uint h = tgid.x;
    const uint m = tgid.y;
    if (h >= p.nh || m >= p.M_padded) return;

    const uint k4_cnt = p.K >> 2u;
    device half4* dst4 = reinterpret_cast<device half4*>(
        q_hm_out + (ulong(h) * p.M_padded + m) * p.K);

    if (m < p.M_valid) {
        device const half4* src4 = reinterpret_cast<device const half4*>(
            q_in + (ulong(m) * p.nh + h) * p.K);
        for (uint c4 = tid; c4 < k4_cnt; c4 += 64u) {
            dst4[c4] = src4[c4];
        }
    } else {
        for (uint c4 = tid; c4 < k4_cnt; c4 += 64u) {
            dst4[c4] = half4(0.0h);
        }
    }
}

#if defined(__HAVE_TENSOR__)
// Direct-Device `static_slice` `K=dynamic_extent` Head-Major Batched Per-Head FP16 NAX (`W_UK` & `W_UV`!)
// ZERO TG memory, ZERO barriers!
//   - `p.pad0 == 1` (`W_UK`): writes head-major `[nh, M_padded, N]` directly for NAX Flash-MLA!
//   - `p.pad0 == 0` (`W_UV`): writes token-major `[M_padded, nh, N]` directly for `o_proj`!
template <int BM, int SG>
inline void sg53_mla_batched_f16_nax_hm_impl(
    device const half*                 w_f16, // [nh, N, K]
    device const half*                 x_hm,  // [nh, M_padded, K]
    device half*                       y_out, // [nh, M_padded, N] or [M_padded, nh, N]
    constant Sg53MlaBatchedGemmParams& p,
    uint3                              tgid
) {
    const uint n0 = tgid.x * 64u;
    const uint m0 = tgid.y * uint(BM);
    const uint h  = tgid.z;
    if (n0 >= p.N || m0 >= p.M_padded || h >= p.nh) return;

    device half* x_h = const_cast<device half*>(x_hm)  + ulong(h) * p.M_padded * p.K;
    device half* w_h = const_cast<device half*>(w_f16) + ulong(h) * p.N * p.K;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(x_h, dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tW(w_h, dextents<int, 2>(int(p.K), int(p.N)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX = tX.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW = tW.template slice<dynamic_extent, 64>(0, int(n0));
    auto cT = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sX)>, sg53_strip_as<decltype(sW)>, float>();
    op.run(sX, sW, cT);

    const bool out_hm = (p.pad0 != 0u);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_padded && n_glob < p.N) {
                half val = (m_glob < p.M_valid) ? sg53_clamp_f16(cT[i]) : half(0.0h);
                if (out_hm) {
                    y_out[(ulong(h) * p.M_padded + m_glob) * p.N + n_glob] = val;
                } else {
                    y_out[(ulong(m_glob) * p.nh + h) * p.N + n_glob] = val;
                }
            }
        }
    }
}
#endif

// Grid: `((N + 63)/64, M_padded / 16, nh)`, `64` threads/TG (`2` SGs, 0 TG mem)
kernel void sg53_mla_batched_f16_nax_hm_m16(
    device const half*                 w_f16 [[buffer(0)]],
    device const half*                 x_hm  [[buffer(1)]],
    device half*                       y_out [[buffer(2)]],
    constant Sg53MlaBatchedGemmParams& p     [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sg53_mla_batched_f16_nax_hm_impl<16, 2>(w_f16, x_hm, y_out, p, tgid);
#else
    (void)w_f16; (void)x_hm; (void)y_out; (void)p; (void)tgid;
#endif
}

// Grid: `((N + 63)/64, M_padded / 64, nh)`, `128` threads/TG (`4` SGs, 0 TG mem)
kernel void sg53_mla_batched_f16_nax_hm_m64(
    device const half*                 w_f16 [[buffer(0)]],
    device const half*                 x_hm  [[buffer(1)]],
    device half*                       y_out [[buffer(2)]],
    constant Sg53MlaBatchedGemmParams& p     [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sg53_mla_batched_f16_nax_hm_impl<64, 4>(w_f16, x_hm, y_out, p, tgid);
#else
    (void)w_f16; (void)x_hm; (void)y_out; (void)p; (void)tgid;
#endif
}

inline uint sg53_kv_phys_pos(device const uint* block_loc, uint logical_pos, uint paged) {
    if (paged == 0u) return logical_pos;
    const uint loc = block_loc[logical_pos >> 2u];
    return ((loc & STRATA_GLM53_KV_LOC_IDX_MASK) << 2u) | (logical_pos & 3u);
}

struct Sg53MlaKvWriteParams {
    uint  M_valid;
    uint  kvr;           // 512
    uint  src_stride;    // qr + kvr = 2048
    uint  src_col_off;   // qr = 1536
    uint  seq_pos_start;
    uint  paged;
    float eps;
    uint  pad0;
};

// RMSNorm on `qkv_a_f32[m, qr .. qr + kvr - 1]` written directly to `c_kv_cache[phys_pos, 0 .. kvr - 1]`!
// Grid: `(1, M_valid, 1)`, `256` threads/TG
kernel void sg53_mla_kv_rmsnorm_write(
    device const float*            src_f32    [[buffer(0)]],
    device const float*            norm_w     [[buffer(1)]],
    device half*                   c_kv_cache [[buffer(2)]],
    device const uint*             block_loc  [[buffer(3)]],
    constant Sg53MlaKvWriteParams& p          [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;

    threadgroup float tg_sq[8];
    const uint d4_cnt = p.kvr >> 2u;
    device const float4* s4 = reinterpret_cast<device const float4*>(
        src_f32 + ulong(m) * p.src_stride + p.src_col_off);
    device const float4* w4 = reinterpret_cast<device const float4*>(norm_w);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 v = s4[i];
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }

    float sg_sum = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float v   = (lane < 8u) ? tg_sq[lane] : 0.0f;
        float tot = simd_sum(v);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.kvr) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms  = tg_sq[0];
    const uint  phys_pos = sg53_kv_phys_pos(block_loc, p.seq_pos_start + m, p.paged);
    device half4* dh     = reinterpret_cast<device half4*>(c_kv_cache + ulong(phys_pos) * p.kvr);
    for (uint i = tid; i < d4_cnt; i += 256u) {
        dh[i] = sg53_clamp_f16x4((s4[i] * inv_rms) * w4[i]);
    }
}

struct Sg53MlaDecodeParams {
    uint  M_valid;
    uint  nh;             // 64
    uint  kv_lora_rank;   // 512 (or 64..512 in synthetic tests)
    uint  q_per_tg;       // 4 (4 heads sharing single c_kv [512] load!)
    uint  seq_pos_start;
    uint  max_seq_alloc;
    uint  max_splits;     // 16
    uint  index_kpool;    // 4
    uint  select_k_pools; // 512
    float attn_scale;     // 1.0f / sqrt(qk_nope_head_dim) = 0.0625f
    uint  m_base;
    uint  paged;
    uint  m_end;          // 0 => M_valid; >0 => exclusive end for paged group
    uint  M_padded;       // Multiple of 16/64 for NAX Flash-MLA
    uint  total_kv;       // seq_pos_start + M_valid
    uint  pad0;
};

inline void sg53_mla_query_split_plan(
    uint q_pos,
    uint index_kpool,
    uint select_k_pools,
    uint max_splits,
    thread bool& is_sparse,
    thread uint& n_tok,
    thread uint& my_splits,
    thread uint& my_kv_per_split
) {
    const uint num_complete = (q_pos + 1u) / index_kpool;
    is_sparse = (num_complete > select_k_pools);
    if (!is_sparse) {
        n_tok = q_pos + 1u;
    } else {
        const uint tail = (q_pos + 1u) - num_complete * index_kpool;
        n_tok = select_k_pools * index_kpool + tail;
    }
    uint target     = clamp((n_tok + 63u) >> 6u, 1u, max_splits);
    my_kv_per_split = ((n_tok + target - 1u) / target + 7u) & ~7u;
    my_splits       = max(1u, (n_tok + my_kv_per_split - 1u) / my_kv_per_split);
}

// Absorbed Latent Multi-Query (`64:1`) Split-KV Attention over Shared `c_kv [max_seq_alloc, 512]`!
// Supports BOTH Fast Dense Causal (`q_pos < 2051`) and `Glm5NextIndexer` Sparse Pools (`q_pos >= 2051`).
// Grid: `(launch_splits, ceil(nh / q_per_tg), M_valid)`, `256` threads/TG (`8` SGs)
kernel void sg53_mla_latent_splitkv(
    device const half*         q_latent    [[buffer(0)]], // [M_padded, nh, kv_lora_rank]
    device const half*         c_kv_cache  [[buffer(1)]], // [kv_stride_tokens, kv_lora_rank]
    device const uint*         sel_pools   [[buffer(2)]], // [M_valid, select_k_pools]
    device float*              part_m      [[buffer(3)]], // [M_valid, max_splits, nh]
    device float*              part_l      [[buffer(4)]], // [M_valid, max_splits, nh]
    device float*              part_o      [[buffer(5)]], // [M_valid, max_splits, nh, kv_lora_rank]
    constant Sg53MlaDecodeParams& p        [[buffer(6)]],
    device const uint*         block_loc   [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],           // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],        // 0..7
    uint  lane [[thread_index_in_simdgroup]]              // 0..31
) {
    const uint split_idx = tgid.x;
    const uint q_grp     = tgid.y;
    const uint m         = p.m_base + tgid.z;
    const uint h_base    = q_grp * p.q_per_tg;
    if (m >= p.M_valid || h_base >= p.nh) return;

    const uint g_cnt = min(p.q_per_tg, p.nh - h_base);
    const uint q_pos = p.seq_pos_start + m;

    bool is_sparse = false;
    uint n_tok = 0u, my_splits = 1u, my_kv_per_split = 8u;
    sg53_mla_query_split_plan(q_pos, p.index_kpool, p.select_k_pools, p.max_splits,
                              is_sparse, n_tok, my_splits, my_kv_per_split);
    if (split_idx >= my_splits) return;

    const uint kv_beg       = min(n_tok, split_idx * my_kv_per_split);
    const uint kv_end       = min(n_tok, kv_beg + my_kv_per_split);
    const uint num_complete = (q_pos + 1u) / p.index_kpool;
    const uint pool_tokens  = p.select_k_pools * p.index_kpool;

    // Supports `kv_lora_rank <= 512` (`dims_per_lane <= 16`) and `q_per_tg <= 4`
    threadgroup float tg_q[4][512];
    threadgroup float tg_m[4][8];
    threadgroup float tg_l[4][8];
    threadgroup float tg_o[8][512];

    for (uint d = tid; d < p.kv_lora_rank; d += 256u) {
        for (uint g = 0u; g < g_cnt; ++g) {
            tg_q[g][d] = float(q_latent[(ulong(m) * p.nh + (h_base + g)) * p.kv_lora_rank + d]) * p.attn_scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint dims_per_lane = p.kv_lora_rank >> 5u; // 16 for kv_lora_rank=512
    const uint d_base        = lane * dims_per_lane;

    float q_reg[4][16];
    float m_sg[4];
    float l_sg[4];
    float o_sg[4][16];

    for (uint g = 0u; g < g_cnt; ++g) {
        m_sg[g] = -INFINITY;
        l_sg[g] = 0.0f;
        for (uint j = 0u; j < dims_per_lane; ++j) {
            q_reg[g][j] = tg_q[g][d_base + j];
            o_sg[g][j]  = 0.0f;
        }
    }

    device const uint* row_sel = sel_pools + ulong(m) * p.select_k_pools;

    for (uint idx = kv_beg + sgid; idx < kv_end; idx += 8u) {
        uint t_pos = idx;
        if (is_sparse) {
            if (idx < pool_tokens) {
                uint p_slot = idx / p.index_kpool;
                uint p_off  = idx % p.index_kpool;
                t_pos = row_sel[p_slot] * p.index_kpool + p_off;
            } else {
                t_pos = num_complete * p.index_kpool + (idx - pool_tokens);
            }
        }

        // Load shared `c_kv[phys_pos]` ONCE for all `g_cnt` query heads!
        float c_val[16];
        const uint phys_pos = sg53_kv_phys_pos(block_loc, t_pos, p.paged);
        device const half* ck_ptr = c_kv_cache + ulong(phys_pos) * p.kv_lora_rank + d_base;
        if ((dims_per_lane & 3u) == 0u) {
            device const half4* ck4 = reinterpret_cast<device const half4*>(ck_ptr);
            for (uint j4 = 0u; j4 < (dims_per_lane >> 2u); ++j4) {
                const float4 fv = float4(ck4[j4]);
                c_val[j4 * 4u + 0u] = fv.x;
                c_val[j4 * 4u + 1u] = fv.y;
                c_val[j4 * 4u + 2u] = fv.z;
                c_val[j4 * 4u + 3u] = fv.w;
            }
        } else {
            for (uint j = 0u; j < dims_per_lane; ++j) {
                c_val[j] = float(ck_ptr[j]);
            }
        }

        for (uint g = 0u; g < g_cnt; ++g) {
            float dot_part = 0.0f;
            for (uint j = 0u; j < dims_per_lane; ++j) {
                dot_part = fma(q_reg[g][j], c_val[j], dot_part);
            }
            float sc    = simd_sum(dot_part);
            float m_new = max(m_sg[g], sc);
            float alpha = (m_sg[g] == -INFINITY) ? 0.0f : exp(m_sg[g] - m_new);
            float p_val = (m_new == -INFINITY) ? 0.0f : exp(sc - m_new);
            l_sg[g]     = fma(l_sg[g], alpha, p_val);
            m_sg[g]     = m_new;
            for (uint j = 0u; j < dims_per_lane; ++j) {
                o_sg[g][j] = fma(o_sg[g][j], alpha, p_val * c_val[j]);
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

    // Combine the 8 SIMD-group partial online-softmax accumulators per head `g`
    for (uint g = 0u; g < g_cnt; ++g) {
        float m_max = -INFINITY;
        for (uint s = 0u; s < 8u; ++s) {
            m_max = max(m_max, tg_m[g][s]);
        }
        float w_sg = (m_max == -INFINITY || tg_m[g][sgid] == -INFINITY) ? 0.0f : exp(tg_m[g][sgid] - m_max);
        for (uint j = 0u; j < dims_per_lane; ++j) {
            tg_o[sgid][d_base + j] = o_sg[g][j] * w_sg;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint h = h_base + g;
        const ulong meta_idx = (ulong(m - p.m_base) * p.max_splits + split_idx) * p.nh + h;
        if (tid == 0u) {
            float l_tot = 0.0f;
            for (uint s = 0u; s < 8u; ++s) {
                float ws = (m_max == -INFINITY || tg_m[g][s] == -INFINITY) ? 0.0f : exp(tg_m[g][s] - m_max);
                l_tot = fma(tg_l[g][s], ws, l_tot);
            }
            part_m[meta_idx] = m_max;
            part_l[meta_idx] = l_tot;
        }

        device float* po = part_o + meta_idx * p.kv_lora_rank;
        for (uint d = tid; d < p.kv_lora_rank; d += 256u) {
            float o_tot = (tg_o[0][d] + tg_o[1][d]) + (tg_o[2][d] + tg_o[3][d]) +
                          (tg_o[4][d] + tg_o[5][d]) + (tg_o[6][d] + tg_o[7][d]);
            po[d] = o_tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Metal 4.1 NAX Split-KV Decode MLA (`kv_lora_rank == 512`, `(nh % 16) == 0`, `4` SGs = `128` dims/SG, `KV_TILE = 16`!):
// Loads `16` query heads x `512` dims ONCE into `qCT`, burst-loads `16 x 512` `C_kv` via coalesced `uint4` ONCE per tile
// for BOTH `QK` (`16x16x128`) & `PV` (`16x64x16` x2), writes `part_m/l/o` 100% compatible with `sg53_mla_latent_reduce`!
// 100.0000% bit-exact between `M=1` and `M=5` (each token `m` runs independent `tgid.z`)!
// Grid: `(launch_splits, nh / 16, mcnt)`, `128` threads/TG (`4` SGs)
kernel void sg53_mla_decode_splitkv_nax(
    device const uint4*           q_latent   [[buffer(0)]], // [M_padded, nh, 64] (each uint4 = 8 halfs -> 512 halfs/head)
    device const uint4*           c_kv_cache [[buffer(1)]], // [kv_stride_tokens, 64]
    device const uint*            sel_pools  [[buffer(2)]], // [M_valid, select_k_pools]
    device float*                 part_m     [[buffer(3)]], // [mcnt, max_splits, nh]
    device float*                 part_l     [[buffer(4)]], // [mcnt, max_splits, nh]
    device float*                 part_o     [[buffer(5)]], // [mcnt, max_splits, nh, 512]
    constant Sg53MlaDecodeParams& p          [[buffer(6)]],
    device const uint*            block_loc  [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],          // 0..3 (Split-D: 128 dims/SG)
    uint  lane [[thread_index_in_simdgroup]]                // 0..31
) {
    const uint split_idx = tgid.x;
    const uint h0        = tgid.y * 16u;
    const uint m         = p.m_base + tgid.z;
    if (m >= p.M_valid || h0 >= p.nh) return;

    const uint q_pos = p.seq_pos_start + m;
    bool is_sparse = false;
    uint n_tok = 0u, my_splits = 1u, my_kv_per_split = 8u;
    sg53_mla_query_split_plan(q_pos, p.index_kpool, p.select_k_pools, p.max_splits,
                              is_sparse, n_tok, my_splits, my_kv_per_split);
    if (split_idx >= my_splits) return;

#if defined(__HAVE_TENSOR__)
    const uint kv_beg       = min(n_tok, split_idx * my_kv_per_split);
    const uint kv_end       = min(n_tok, kv_beg + my_kv_per_split);
    const uint num_complete = (q_pos + 1u) / p.index_kpool;
    const uint pool_tokens  = p.select_k_pools * p.index_kpool;

    const ulong sg_c8  = ulong(sgid) * 16ul; // 16 uint4 = 128 halfs
    const uint  r_pair = lane >> 4u;         // 0..1
    const uint  c8     = lane & 15u;         // 0..15

    threadgroup half  C_tg[4][16][128]; // 16 KiB (`4 KiB` sgid-private, loaded ONCE for QK & PV!)
    threadgroup float S_tg[4][16][16];  //  4 KiB
    threadgroup half  P_tg[16][16];     // 512 B
    threadgroup float M_tg[16];
    threadgroup float L_tg[16];
    threadgroup float A_tg[16];

    if (sgid == 0u && lane < 16u) {
        M_tg[lane] = -INFINITY;
        L_tg[lane] = 0.0f;
    }

    // 1. Coalesced 16B `uint4` load of `16` query heads `[h0..h0+15, sgid*128..+127]`, pin in NAX `qCT`!
    const ulong q_tok_off = ulong(m) * ulong(p.nh);
    #pragma clang loop unroll(full)
    for (uint rp = 0u; rp < 8u; ++rp) {
        const uint r = rp * 2u + r_pair;
        reinterpret_cast<threadgroup uint4*>(&C_tg[sgid][r][0])[c8] =
            q_latent[(q_tok_off + ulong(h0 + r)) * 64ul + sg_c8 + c8];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup half, extents<int, 128, 16>, tensor_inline> tC_sg(&C_tg[sgid][0][0], extents<int, 128, 16>());
    tensor<threadgroup half, extents<int, 16, 16>,  tensor_inline> tP(&P_tg[0][0], extents<int, 16, 16>());

    constexpr auto desc_qk = matmul2d_descriptor(
        16, 16, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;

    constexpr auto desc_pv = matmul2d_descriptor(
        16, 64, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    auto qCT = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(tC_sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto sC0  = tC_sg.slice<64, 16>(0, 0);
    auto sC1  = tC_sg.slice<64, 16>(64, 0);
    auto oCT0 = op_pv.get_destination_cooperative_tensor<sg53_strip_as<decltype(tP)>, sg53_strip_as<decltype(sC0)>, float>();
    auto oCT1 = op_pv.get_destination_cooperative_tensor<sg53_strip_as<decltype(tP)>, sg53_strip_as<decltype(sC1)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            oCT0[i] = 0.0f;
            oCT1[i] = 0.0f;
        }
    }

    device const uint* row_sel = sel_pools + ulong(m) * ulong(p.select_k_pools);

    // 2. Main Split-KV Tile Loop (`KV_TILE = 16`, `C_kv` burst-loaded ONCE for BOTH `op_qk` and `op_pv`!)
    for (uint kv = kv_beg; kv < kv_end; kv += 16u) {
        const uint n_valid = min(16u, kv_end - kv);

        uint my_ppos = 0u;
        if (lane < n_valid) {
            const uint idx = kv + lane;
            uint t_pos = idx;
            if (is_sparse) {
                if (idx < pool_tokens) {
                    const uint p_slot = idx / p.index_kpool;
                    const uint p_off  = idx % p.index_kpool;
                    t_pos = row_sel[p_slot] * p.index_kpool + p_off;
                } else {
                    t_pos = num_complete * p.index_kpool + (idx - pool_tokens);
                }
            }
            my_ppos = sg53_kv_phys_pos(block_loc, t_pos, p.paged);
        }

        #pragma clang loop unroll(full)
        for (uint rp = 0u; rp < 8u; ++rp) {
            const uint r  = rp * 2u + r_pair;
            const uint pp = simd_shuffle(my_ppos, r);
            reinterpret_cast<threadgroup uint4*>(&C_tg[sgid][r][0])[c8] = (r < n_valid)
                ? c_kv_cache[ulong(pp) * 64ul + sg_c8 + c8]
                : uint4(0u);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        auto sCT = op_qk.get_destination_cooperative_tensor<sg53_strip_as<decltype(qCT)>, sg53_strip_as<decltype(tC_sg)>, float>();
        op_qk.run(qCT, tC_sg, sCT);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < sCT.get_capacity(); ++i) {
            if (sCT.is_valid_element(i)) {
                auto idx = sCT.get_multidimensional_index(i);
                S_tg[sgid][uint(idx[1])][uint(idx[0])] = sCT[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #1: S_tg ready + prev P_tg/A_tg consumed!

        const bool valid_col = (lane < n_valid);
        const uint r_start   = sgid * 4u;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 4u; ++r) {
            const uint  row    = r_start + r;
            const float s_raw  = valid_col
                ? ((S_tg[0][row][lane] + S_tg[1][row][lane]) + (S_tg[2][row][lane] + S_tg[3][row][lane]))
                : 0.0f;
            const float s      = valid_col ? (s_raw * p.attn_scale) : -INFINITY;
            const float rmax   = simd_max(s);
            const float m_prev = M_tg[row];
            const float m_next = max(m_prev, rmax);
            float alpha = 0.0f;
            float p_val = 0.0f;
            if (m_next > -INFINITY) {
                alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_next);
                p_val = valid_col ? exp(s - m_next) : 0.0f;
            }
            const float p_sum = simd_sum(p_val);
            if (lane < 16u) P_tg[row][lane] = half(p_val);
            if (lane == 0u) {
                M_tg[row] = m_next;
                L_tg[row] = fma(L_tg[row], alpha, p_sum);
                A_tg[row] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #2: P_tg/A_tg ready + S_tg consumed!

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
            if (oCT0.is_valid_element(i)) {
                auto idx = oCT0.get_multidimensional_index(i);
                float a  = A_tg[uint(idx[1])];
                oCT0[i] *= a;
                oCT1[i] *= a;
            }
        }

        // Reuse `C_tg[sgid]` STILL LIVE in TG memory — ZERO second KV read!
        op_pv.run(tP, sC0, oCT0);
        op_pv.run(tP, sC1, oCT1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Write Split-KV partial accumulators `part_m`, `part_l`, `part_o` (100% compatible with `sg53_mla_latent_reduce`!)
    const ulong meta_base = (ulong(m - p.m_base) * ulong(p.max_splits) + ulong(split_idx)) * ulong(p.nh) + ulong(h0);
    if (sgid == 0u && lane < 16u) {
        part_m[meta_base + lane] = M_tg[lane];
        part_l[meta_base + lane] = L_tg[lane];
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            auto idx = oCT0.get_multidimensional_index(i);
            const uint r = uint(idx[1]); // 0..15 (head offset)
            const uint c = uint(idx[0]); // 0..63 (dim offset)
            device float* po = part_o + (meta_base + ulong(r)) * 512ul + ulong(sgid) * 128ul;
            po[c]       = oCT0[i];
            po[64u + c] = oCT1[i];
        }
    }
#else
    (void)q_latent; (void)c_kv_cache; (void)sel_pools; (void)part_m; (void)part_l; (void)part_o;
    (void)p; (void)block_loc; (void)sgid; (void)lane;
#endif
}

// Reduces `my_splits` partial latent accumulators into `c_out [M_valid, nh, kv_lora_rank]` FP16
// Grid: `(nh, M_valid, 1)`, `256` threads/TG
kernel void sg53_mla_latent_reduce(
    device const float*        part_m [[buffer(0)]], // [M_valid, max_splits, nh]
    device const float*        part_l [[buffer(1)]], // [M_valid, max_splits, nh]
    device const float*        part_o [[buffer(2)]], // [M_valid, max_splits, nh, kv_lora_rank]
    device half*               c_out  [[buffer(3)]], // [M_valid, nh, kv_lora_rank]
    constant Sg53MlaDecodeParams& p   [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]       // 0..256
) {
    const uint h = tgid.x;
    const uint m = p.m_base + tgid.y;
    if (h >= p.nh || m >= p.M_valid) return;

    const uint q_pos = p.seq_pos_start + m;
    bool is_sparse = false;
    uint n_tok = 0u, my_splits = 1u, my_kv_per_split = 8u;
    sg53_mla_query_split_plan(q_pos, p.index_kpool, p.select_k_pools, p.max_splits,
                              is_sparse, n_tok, my_splits, my_kv_per_split);

    threadgroup float tg_w[65];

    if (tid == 0u) {
        float m_max = -INFINITY;
        for (uint s = 0u; s < my_splits; ++s) {
            float ms = part_m[(ulong(m - p.m_base) * p.max_splits + s) * p.nh + h];
            m_max = max(m_max, ms);
        }
        float l_sum = 0.0f;
        for (uint s = 0u; s < my_splits; ++s) {
            ulong idx = (ulong(m - p.m_base) * p.max_splits + s) * p.nh + h;
            float ms  = part_m[idx];
            float ws  = (m_max == -INFINITY || ms == -INFINITY) ? 0.0f : exp(ms - m_max);
            tg_w[s]   = ws;
            l_sum     = fma(part_l[idx], ws, l_sum);
        }
        tg_w[64] = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_l = tg_w[64];
    device half* dst  = c_out + (ulong(m) * p.nh + h) * p.kv_lora_rank;

    for (uint d = tid; d < p.kv_lora_rank; d += 256u) {
        float acc = 0.0f;
        for (uint s = 0u; s < my_splits; ++s) {
            ulong idx = (ulong(m - p.m_base) * p.max_splits + s) * p.nh + h;
            acc = fma(part_o[idx * p.kv_lora_rank + d], tg_w[s], acc);
        }
        dst[d] = sg53_clamp_f16(acc * inv_l);
    }
}

/* ---- Zero-`part_o` Metal 4.1 NAX Flash-MLA Prefill (`kv_lora_rank == 512`, `4` SGs = `128` dims/SG!) ----
 *   - `sg53_mla_prefill_dense_nax`:  Unpaged dense causal prefill (`KV_TILE = 32`, direct-device `tC`, `9.2 KiB` TG mem!)
 *   - `sg53_mla_prefill_sparse_nax`: Sparse top-512 &/or Paged (`block_loc`) prefill (`KV_TILE = 16` = 4 blocks,
 *     16-query SIMD-merge block-union, loads `C_kv` ONCE for BOTH `QK` & `PV`, `20.7 KiB` TG mem!)
 */

// Grid: `(M_padded / 16, nh, 1)`, `128` threads/TG (`4` SGs, `max_seq_alloc % 32 == 0`)
kernel void sg53_mla_prefill_dense_nax(
    device const half*            q_hm       [[buffer(0)]], // [nh, M_padded, 512]
    device const half*            c_kv_cache [[buffer(1)]], // [max_seq_alloc, 512]
    device half*                  c_out_hm   [[buffer(2)]], // [nh, M_padded, 512]
    constant Sg53MlaDecodeParams& p          [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],          // 0..3 (Split-D: 128 dims/SG)
    uint  lane [[thread_index_in_simdgroup]]                // 0..31
) {
    const uint m0 = tgid.x * 16u;
    const uint h  = tgid.y;
    if (m0 >= p.M_padded || h >= p.nh) return;

    const ulong h_off = ulong(h) * p.M_padded * 512ul;
    if (m0 >= p.M_valid) {
        const uint pad_rows = min(16u, p.M_padded - m0);
        for (uint r = 0u; r < pad_rows; ++r) {
            reinterpret_cast<device half4*>(c_out_hm + h_off + ulong(m0 + r) * 512ul + sgid * 128u)[lane] = half4(0.0h);
        }
        return;
    }

#if defined(__HAVE_TENSOR__)
    const uint max_q_pos = p.seq_pos_start + min(m0 + 15u, p.M_valid - 1u);
    const uint kv_limit  = min(p.total_kv, max_q_pos + 1u);

    threadgroup float S_tg[4][16][32]; // 8 KiB
    threadgroup half  P_tg[16][32];    // 1 KiB
    threadgroup float M_tg[16];
    threadgroup float L_tg[16];
    threadgroup float A_tg[16];

    if (sgid == 0u && lane < 16u) {
        M_tg[lane] = -INFINITY;
        L_tg[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half* q_head = const_cast<device half*>(q_hm) + h_off;
    tensor<device half, dextents<int, 2>, tensor_inline> tQ(q_head, dextents<int, 2>(512, int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tC(const_cast<device half*>(c_kv_cache), dextents<int, 2>(512, int(p.max_seq_alloc)));
    tensor<threadgroup half, extents<int, 32, 16>, tensor_inline> tP(&P_tg[0][0], extents<int, 32, 16>());

    constexpr auto desc_qk = matmul2d_descriptor(
        16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;

    constexpr auto desc_pv = matmul2d_descriptor(
        16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    const int d_off = int(sgid * 128u);
    auto sQ  = tQ.slice<128, 16>(d_off, int(m0));
    auto qCT = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(sQ);

    auto sV0_init = tC.slice<64, 32>(d_off, 0);
    auto oCT0 = op_pv.get_destination_cooperative_tensor<sg53_strip_as<decltype(tP)>, sg53_strip_as<decltype(sV0_init)>, float>();
    auto oCT1 = op_pv.get_destination_cooperative_tensor<sg53_strip_as<decltype(tP)>, sg53_strip_as<decltype(sV0_init)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            oCT0[i] = 0.0f;
            oCT1[i] = 0.0f;
        }
    }

    for (uint kv_pos = 0u; kv_pos < kv_limit; kv_pos += 32u) {
        auto sK  = tC.slice<128, 32>(d_off, int(kv_pos));
        auto sCT = op_qk.get_destination_cooperative_tensor<sg53_strip_as<decltype(qCT)>, sg53_strip_as<decltype(sK)>, float>();
        op_qk.run(qCT, sK, sCT);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < sCT.get_capacity(); ++i) {
            if (sCT.is_valid_element(i)) {
                auto idx = sCT.get_multidimensional_index(i);
                S_tg[sgid][uint(idx[1])][uint(idx[0])] = sCT[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #1: S_tg ready + prev P_tg/A_tg consumed!

        const uint kv_col_pos = kv_pos + lane;
        const uint r_start    = sgid * 4u;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 4u; ++r) {
            const uint  row    = r_start + r;
            const uint  m_glob = m0 + row;
            const uint  q_pos  = p.seq_pos_start + m_glob;
            const bool  valid  = (m_glob < p.M_valid) && (kv_col_pos <= q_pos) && (kv_col_pos < p.total_kv);
            const float s_raw  = (S_tg[0][row][lane] + S_tg[1][row][lane]) + (S_tg[2][row][lane] + S_tg[3][row][lane]);
            const float s      = valid ? (s_raw * p.attn_scale) : -INFINITY;

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
        threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #2: P_tg/A_tg ready + S_tg consumed!

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
            if (oCT0.is_valid_element(i)) {
                auto idx = oCT0.get_multidimensional_index(i);
                float a  = A_tg[uint(idx[1])];
                oCT0[i] *= a;
                oCT1[i] *= a;
            }
        }

        auto sV0 = tC.slice<64, 32>(d_off,      int(kv_pos));
        auto sV1 = tC.slice<64, 32>(d_off + 64, int(kv_pos));
        op_pv.run(tP, sV0, oCT0);
        op_pv.run(tP, sV1, oCT1);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            auto idx    = oCT0.get_multidimensional_index(i);
            uint r      = uint(idx[1]);
            uint c      = uint(idx[0]);
            uint m_glob = m0 + r;
            if (m_glob < p.M_valid) {
                float inv_l  = (L_tg[r] > 0.0f) ? (1.0f / L_tg[r]) : 0.0f;
                ulong o_base = h_off + ulong(m_glob) * 512ul + ulong(sgid) * 128ul;
                c_out_hm[o_base + c]       = sg53_clamp_f16(oCT0[i] * inv_l);
                c_out_hm[o_base + 64u + c] = sg53_clamp_f16(oCT1[i] * inv_l);
            } else if (m_glob < p.M_padded) {
                ulong o_base = h_off + ulong(m_glob) * 512ul + ulong(sgid) * 128ul;
                c_out_hm[o_base + c]       = half(0.0h);
                c_out_hm[o_base + 64u + c] = half(0.0h);
            }
        }
    }
#else
    (void)q_hm; (void)c_kv_cache; (void)c_out_hm; (void)p; (void)sgid; (void)lane;
#endif
}

// Grid: `(num_q_tiles_in_group, nh, 1)`, `128` threads/TG (`4` SGs, `kv_lora_rank == 512`, `index_kpool == 4`)
kernel void sg53_mla_prefill_sparse_nax(
    device const uint4*           q_hm       [[buffer(0)]], // [nh, M_padded, 64] (each uint4 = 8 halfs)
    device const uint4*           c_kv_cache [[buffer(1)]], // [max_seq_alloc, 64]
    device const uint*            sel_pools  [[buffer(2)]], // [M_valid, select_k_pools]
    device half*                  c_out_hm   [[buffer(3)]], // [nh, M_padded, 512]
    constant Sg53MlaDecodeParams& p          [[buffer(4)]],
    device const uint*            block_loc  [[buffer(5)]], // [max_blocks] (read iff p.paged)
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],          // 0..3 (Split-D: 128 dims/SG)
    uint  lane [[thread_index_in_simdgroup]]                // 0..31
) {
    const uint m0 = p.m_base + tgid.x * 16u;
    const uint h  = tgid.y;
    if (m0 >= p.M_padded || h >= p.nh) return;

    const uint  m_end  = (p.m_end > 0u) ? min(p.m_end, p.M_valid) : p.M_valid;
    const ulong h_off  = ulong(h) * p.M_padded * 512ul;
    const ulong sg_c8  = ulong(sgid) * 16ul;
    const uint  r_pair = lane >> 4u; // 0..1
    const uint  c8     = lane & 15u; // 0..15 (`16 * 8 = 128` halfs)

    if (m0 >= m_end) {
        if (m0 >= p.M_valid) {
            const uint pad_rows = min(16u, p.M_padded - m0);
            for (uint r = r_pair; r < pad_rows; r += 2u) {
                device uint4* out8 = reinterpret_cast<device uint4*>(c_out_hm + h_off + ulong(m0 + r) * 512ul);
                out8[sg_c8 + c8] = uint4(0u);
            }
        }
        return;
    }

#if defined(__HAVE_TENSOR__)
    const uint q_cnt = min(16u, m_end - m0);

    threadgroup half  C_tg[4][16][128]; // 16 KiB (`4 KiB` sgid-private, loaded ONCE for QK & PV!)
    threadgroup float S_tg[4][16][16];  //  4 KiB
    threadgroup half  P_tg[16][16];     // 512 B
    threadgroup float M_tg[16];
    threadgroup float L_tg[16];
    threadgroup float A_tg[16];

    if (sgid == 0u && lane < 16u) {
        M_tg[lane] = -INFINITY;
        L_tg[lane] = 0.0f;
    }

    // 1. Coalesced 16B `uint4` load of `q_cnt <= 16` queries (`d = sgid*128..+127`), pin in NAX `qCT`!
    const ulong q_head_off = ulong(h) * ulong(p.M_padded);
    #pragma clang loop unroll(full)
    for (uint rp = 0u; rp < 8u; ++rp) {
        const uint r = rp * 2u + r_pair;
        reinterpret_cast<threadgroup uint4*>(&C_tg[sgid][r][0])[c8] = (r < q_cnt)
            ? q_hm[(q_head_off + ulong(m0 + r)) * 64ul + sg_c8 + c8]
            : uint4(0u);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup half, extents<int, 128, 16>, tensor_inline> tC_sg(&C_tg[sgid][0][0], extents<int, 128, 16>());
    tensor<threadgroup half, extents<int, 16, 16>,  tensor_inline> tP(&P_tg[0][0], extents<int, 16, 16>());

    constexpr auto desc_qk = matmul2d_descriptor(
        16, 16, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;

    constexpr auto desc_pv = matmul2d_descriptor(
        16, 64, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    auto qCT = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(tC_sg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto sC0  = tC_sg.slice<64, 16>(0, 0);
    auto sC1  = tC_sg.slice<64, 16>(64, 0);
    auto oCT0 = op_pv.get_destination_cooperative_tensor<sg53_strip_as<decltype(tP)>, sg53_strip_as<decltype(sC0)>, float>();
    auto oCT1 = op_pv.get_destination_cooperative_tensor<sg53_strip_as<decltype(tP)>, sg53_strip_as<decltype(sC1)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            oCT0[i] = 0.0f;
            oCT1[i] = 0.0f;
        }
    }

    // 2. Initialize 16-way ascending sorted block-union merge cursors (lanes 0..15 track queries m0..m0+q_cnt-1!)
    bool is_sparse_q = false;
    bool has_tail_q  = false;
    uint ptr_q       = 0u;
    uint end_ptr_q   = 0u;
    uint tail_blk_q  = 0u;
    uint cur_b       = 0xFFFFFFFFu;
    device const uint* m_sel_q = sel_pools;

    if (lane < q_cnt) {
        const uint m_q   = m0 + lane;
        const uint q_pos = p.seq_pos_start + m_q;
        const uint nc    = (q_pos + 1u) >> 2u;
        is_sparse_q      = (nc > p.select_k_pools);
        has_tail_q       = (((q_pos + 1u) & 3u) != 0u);
        tail_blk_q       = nc;
        m_sel_q          = sel_pools + ulong(m_q) * ulong(p.select_k_pools);
        end_ptr_q        = is_sparse_q ? p.select_k_pools : nc;
        if (ptr_q < end_ptr_q) {
            cur_b = is_sparse_q ? m_sel_q[0] : 0u;
        } else if (has_tail_q) {
            cur_b      = tail_blk_q;
            has_tail_q = false;
        }
    }

    const uint lane_bi  = lane >> 2u; // 0..3 for lane < 16
    const uint lane_sub = lane & 3u;

    // 3. Main Block-Union Loop (`4` union blocks = `16` KV tokens per iter, `C_kv` loaded ONCE for both QK & PV!)
    for (;;) {
        uint u_ppos[4];
        uint my_col_blk = 0u;
        uint my_col_msk = 0u;
        uint n_blks     = 0u;

        #pragma clang loop unroll(full)
        for (uint bi = 0u; bi < 4u; ++bi) {
            const uint min_b = simd_min(cur_b);
            if (min_b == 0xFFFFFFFFu) {
                u_ppos[bi] = 0u;
                continue;
            }
            const uint msk = simd_or((lane < 16u && cur_b == min_b) ? (1u << lane) : 0u);
            if (lane < 16u && lane_bi == bi) {
                my_col_blk = min_b;
                my_col_msk = msk;
            }
            const uint phys_blk = (p.paged != 0u) ? (block_loc[min_b] & STRATA_GLM53_KV_LOC_IDX_MASK) : min_b;
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

        // Burst-load shared `C_kv` (`16` rows x `128` halfs per SG) ONCE for both `op_qk` and `op_pv`!
        #pragma clang loop unroll(full)
        for (uint bi = 0u; bi < 4u; ++bi) {
            const uint r0 = bi * 4u;
            if (bi < n_blks) {
                const ulong pp = ulong(u_ppos[bi]);
                reinterpret_cast<threadgroup uint4*>(&C_tg[sgid][r0 + r_pair][0])[c8] =
                    c_kv_cache[(pp + ulong(r_pair)) * 64ul + sg_c8 + c8];
                reinterpret_cast<threadgroup uint4*>(&C_tg[sgid][r0 + 2u + r_pair][0])[c8] =
                    c_kv_cache[(pp + ulong(2u + r_pair)) * 64ul + sg_c8 + c8];
            } else {
                reinterpret_cast<threadgroup uint4*>(&C_tg[sgid][r0 + r_pair][0])[c8]      = uint4(0u);
                reinterpret_cast<threadgroup uint4*>(&C_tg[sgid][r0 + 2u + r_pair][0])[c8] = uint4(0u);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        auto sCT = op_qk.get_destination_cooperative_tensor<sg53_strip_as<decltype(qCT)>, sg53_strip_as<decltype(tC_sg)>, float>();
        op_qk.run(qCT, tC_sg, sCT);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < sCT.get_capacity(); ++i) {
            if (sCT.is_valid_element(i)) {
                auto idx = sCT.get_multidimensional_index(i);
                S_tg[sgid][uint(idx[1])][uint(idx[0])] = sCT[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #1: S_tg ready + prev P_tg/A_tg consumed!

        const bool in_blks = (lane < 16u) && (lane_bi < n_blks);
        const uint col_tok = (my_col_blk << 2u) | lane_sub;
        const uint r_start = sgid * 4u;
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 4u; ++r) {
            const uint  row    = r_start + r;
            const uint  q_pos  = p.seq_pos_start + m0 + row;
            const bool  valid  = (row < q_cnt) && in_blks && (((my_col_msk >> row) & 1u) != 0u) && (col_tok <= q_pos);
            const float s_raw  = (lane < 16u)
                ? ((S_tg[0][row][lane] + S_tg[1][row][lane]) + (S_tg[2][row][lane] + S_tg[3][row][lane]))
                : 0.0f;
            const float s      = valid ? (s_raw * p.attn_scale) : -INFINITY;
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
            if (lane < 16u) P_tg[row][lane] = half(p_val);
            if (lane == 0u) {
                M_tg[row] = m_next;
                L_tg[row] = fma(L_tg[row], alpha, p_sum);
                A_tg[row] = alpha;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // Barrier #2: P_tg/A_tg ready + S_tg consumed!

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
            if (oCT0.is_valid_element(i)) {
                auto idx = oCT0.get_multidimensional_index(i);
                float a  = A_tg[uint(idx[1])];
                oCT0[i] *= a;
                oCT1[i] *= a;
            }
        }

        // Reuse `C_tg[sgid]` STILL LIVE in TG memory — ZERO second KV read!
        op_pv.run(tP, sC0, oCT0);
        op_pv.run(tP, sC1, oCT1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            auto idx    = oCT0.get_multidimensional_index(i);
            uint r      = uint(idx[1]);
            uint c      = uint(idx[0]);
            uint m_glob = m0 + r;
            if (r < q_cnt) {
                float inv_l  = (L_tg[r] > 0.0f) ? (1.0f / L_tg[r]) : 0.0f;
                ulong o_base = h_off + ulong(m_glob) * 512ul + ulong(sgid) * 128ul;
                c_out_hm[o_base + c]       = sg53_clamp_f16(oCT0[i] * inv_l);
                c_out_hm[o_base + 64u + c] = sg53_clamp_f16(oCT1[i] * inv_l);
            } else if (m_glob >= p.M_valid && m_glob < p.M_padded) {
                ulong o_base = h_off + ulong(m_glob) * 512ul + ulong(sgid) * 128ul;
                c_out_hm[o_base + c]       = half(0.0h);
                c_out_hm[o_base + 64u + c] = half(0.0h);
            }
        }
    }
#else
    (void)q_hm; (void)c_kv_cache; (void)sel_pools; (void)c_out_hm; (void)p; (void)block_loc; (void)sgid; (void)lane;
#endif
}

/* ============================================================================
 * 8. `LimitedSwiGLU(10.0f)`, `288`-Expert `noaux_tc` Sigmoid Router &
 *    Indirect Multi-Buffer (`b_hot_experts` + `b_ring_dma`) MoE Kernels
 * ============================================================================ */

struct Sg53SwigluParams {
    uint  M_valid;
    uint  M_padded;
    uint  d_ff;  // 12288 for dense, 2048 for shared_experts
    float limit; // 10.0f (`min(gate, 10) & clamp(up, -10, 10)`)
};

// Splits concatenated `gate_up_f32 [M_valid, 2 * d_ff]` -> `LimitedSwiGLU` -> `mid_f16 [M_padded, d_ff]`
kernel void sg53_limited_swiglu_split(
    device const float*        gate_up_f32 [[buffer(0)]], // [M_valid, 2 * d_ff]
    device half*               mid_f16     [[buffer(1)]], // [M_padded, d_ff]
    constant Sg53SwigluParams& p           [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]            // 0..255
) {
    const uint m      = tgid.y;
    const uint d4_cnt = p.d_ff >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (m >= p.M_padded || idx4 >= d4_cnt) return;

    if (m >= p.M_valid) {
        reinterpret_cast<device half4*>(mid_f16 + ulong(m) * p.d_ff)[idx4] = half4(0.0h);
        return;
    }

    device const float4* row_g = reinterpret_cast<device const float4*>(gate_up_f32 + ulong(m) * (2ul * p.d_ff));
    device const float4* row_u = reinterpret_cast<device const float4*>(gate_up_f32 + ulong(m) * (2ul * p.d_ff) + p.d_ff);

    float4 prod = sg53_limited_swiglu4(row_g[idx4], row_u[idx4], p.limit);
    reinterpret_cast<device half4*>(mid_f16 + ulong(m) * p.d_ff)[idx4] = sg53_clamp_f16x4(prod);
}

struct Sg53MoeRouterParams {
    uint  M_valid;
    uint  D;                     // 4096
    uint  num_experts;           // 288
    uint  top_k;                 // 8
    uint  norm_topk_prob;        // 1
    float routed_scaling_factor; // 2.5f
    uint  n_group;               // 1
    uint  topk_group;            // 1
};

// Exact `GLM-5.3-Flash` `noaux_tc` Sigmoid Router (`logits = dot_FP32(x, W_gate)`):
//   1. `scores[e] = sigmoid(logits[e])`
//   2. `choice[e] = scores[e] + e_score_bias[e]` (supports `n_group > 1` top-2 group masking too!)
//   3. Pick `top_k = 8` indices maximizing `choice[e]`
//   4. Gather UNBIASED `scores[idx[i]]`, normalize (`sum + 1e-20f`), multiply by `2.5f`!
// Grid: `(M_valid, 1, 1)`, `288` -> `512` threads/TG (`16` SGs; supports `num_experts <= 512`)
kernel void sg53_moe_router_topk(
    device const float*           x_in_f32     [[buffer(0)]], // [M_valid, D]
    device const half*            router_gate  [[buffer(1)]], // [num_experts, D] FP16
    device const float*           e_score_bias [[buffer(2)]], // [num_experts] FP32
    device uint*                  topk_idx_out [[buffer(3)]], // [M_valid, top_k]
    device float*                 topk_w_out   [[buffer(4)]], // [M_valid, top_k]
    constant Sg53MoeRouterParams& p            [[buffer(5)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],                // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],             // 0..15
    uint lane [[thread_index_in_simdgroup]]                   // 0..31
) {
    if (m >= p.M_valid) return;

    threadgroup float tg_scores[512]; // unbiased sigmoid(logits[e])
    threadgroup float tg_choice[512]; // scores[e] + bias[e]
    threadgroup float tg_val[16];
    threadgroup uint  tg_idx[16];
    threadgroup uint  sel_ids[16];

    const uint d4_cnt = p.D >> 2u;
    device const float4* xm4 = reinterpret_cast<device const float4*>(x_in_f32 + ulong(m) * p.D);

    // 1. Cooperative SIMD-group dot product (`16` experts per sweep across `16` SGs -> `18` sweeps for `288` experts!)
    for (uint e_base = 0u; e_base < p.num_experts; e_base += 16u) {
        const uint e = e_base + sgid;
        if (e < p.num_experts) {
            device const half4* we4 = reinterpret_cast<device const half4*>(router_gate + ulong(e) * p.D);
            float acc = 0.0f;
            for (uint c4 = lane; c4 < d4_cnt; c4 += 32u) {
                float4 xv = xm4[c4];
                float4 wv = float4(we4[c4]);
                acc = fma(xv.x, wv.x, fma(xv.y, wv.y, fma(xv.z, wv.z, fma(xv.w, wv.w, acc))));
            }
            float logit = simd_sum(acc);
            if (lane == 0u) {
                float sc     = sg53_sigmoid(logit);
                tg_scores[e] = sc;
                tg_choice[e] = sc + e_score_bias[e];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Optional group mask (`n_group > 1 && topk_group < n_group`)
    if (p.n_group > 1u && p.topk_group < p.n_group && p.n_group <= 32u) {
        if (sgid == 0u) {
            const uint exp_per_grp = p.num_experts / p.n_group;
            float g_score = -INFINITY;
            if (lane < p.n_group) {
                float m1 = -INFINITY, m2 = -INFINITY;
                for (uint j = 0u; j < exp_per_grp; ++j) {
                    float v = tg_choice[lane * exp_per_grp + j];
                    if (v > m1) { m2 = m1; m1 = v; }
                    else if (v > m2) { m2 = v; }
                }
                g_score = m1 + m2;
            }
            uint g_active_mask = 0u;
            for (uint gk = 0u; gk < p.topk_group; ++gk) {
                float bv = ((g_active_mask & (1u << lane)) == 0u && lane < p.n_group) ? g_score : -INFINITY;
                uint  bi = lane;
                for (ushort off = 16u; off > 0u; off >>= 1u) {
                    float ov = simd_shuffle_xor(bv, off);
                    uint  oi = simd_shuffle_xor(bi, off);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                g_active_mask |= (1u << bi);
            }
            if (lane == 0u) tg_idx[0] = g_active_mask;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint g_mask = tg_idx[0];
        const uint exp_per_grp = p.num_experts / p.n_group;
        if (tid < p.num_experts) {
            uint g = tid / exp_per_grp;
            if ((g_mask & (1u << g)) == 0u) {
                tg_choice[tid] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 2. Deterministic Top-K (`K = 8`) selection over `tg_choice[0 .. num_experts - 1]`
    const uint K = min(p.top_k, 16u);
    for (uint k = 0u; k < K; ++k) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;
        if (tid < p.num_experts) {
            bool already = false;
            for (uint prev = 0u; prev < k; ++prev) {
                if (sel_ids[prev] == tid) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                best_v = tg_choice[tid];
                best_i = tid;
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
            float v   = (lane < 16u) ? tg_val[lane] : -INFINITY;
            uint  idx = (lane < 16u) ? tg_idx[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 8u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(v, off);
                uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) {
                    v   = ov;
                    idx = oi;
                }
            }
            if (lane == 0u) {
                sel_ids[k] = (idx < p.num_experts) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Gather UNBIASED `scores[sel_ids[k]]`, normalize (`sum + 1e-20f`), scale by `2.5f`!
    if (tid == 0u) {
        float raw_w[16];
        float sum_w = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float rw = tg_scores[sel_ids[k]];
            raw_w[k] = rw;
            sum_w   += rw;
        }
        float scale = p.routed_scaling_factor;
        if (p.norm_topk_prob != 0u) {
            scale = p.routed_scaling_factor / (sum_w + 1e-20f);
        }
        for (uint k = 0u; k < K; ++k) {
            topk_idx_out[ulong(m) * p.top_k + k] = sel_ids[k];
            topk_w_out[ulong(m) * p.top_k + k]   = raw_w[k] * scale;
        }
    }
}

// Parallel 40-Core Batched Router Logits (`1` expert/SG = `4` experts/TG -> `72` TGs for 288 experts, reads `router_gate` ONCE across M=1..8!)
// 100.0000% bit-exact to `sg53_moe_router_topk` (0 ULP, 0 bit diffs)!
// Grid: `((num_experts + 3) / 4, (M_valid + 7) / 8, 1)`, `128` threads/TG (`4` SGs)
template <uint MB_FIX>
inline void sg53_moe_router_logits_fixed(
    constant float*    x_in_f32,
    device const half* router_gate,
    device float*      logits_out,
    uint e,
    uint m0,
    uint D,
    uint num_experts,
    uint lane
) {
    const uint d4_cnt = D >> 2u;
    device const half4* we4 = reinterpret_cast<device const half4*>(router_gate + ulong(e) * ulong(D));

    float acc[MB_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < MB_FIX; ++m) acc[m] = 0.0f;

    for (uint c4 = lane; c4 < d4_cnt; c4 += 32u) {
        const float4 wv = float4(we4[c4]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < MB_FIX; ++m) {
            const float4 xv = reinterpret_cast<constant float4*>(x_in_f32 + ulong(m0 + m) * ulong(D))[c4];
            acc[m] = fma(xv.x, wv.x, fma(xv.y, wv.y, fma(xv.z, wv.z, fma(xv.w, wv.w, acc[m]))));
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < MB_FIX; ++m) {
        const float logit = simd_sum(acc[m]);
        if (lane == 0u) {
            logits_out[ulong(m0 + m) * ulong(num_experts) + ulong(e)] = logit;
        }
    }
}

kernel void sg53_moe_router_logits(
    constant float*               x_in_f32    [[buffer(0)]], // [M_valid, D]
    device const half*            router_gate [[buffer(1)]], // [num_experts, D] FP16
    device float*                 logits_out  [[buffer(2)]], // [M_valid, num_experts] FP32
    constant Sg53MoeRouterParams& p           [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],           // 0..3
    uint  lane [[thread_index_in_simdgroup]]                 // 0..31
) {
    const uint e  = tgid.x * 4u + sgid;
    const uint m0 = tgid.y * 8u;
    if (e >= p.num_experts || m0 >= p.M_valid) return;

    const uint mb = min(8u, p.M_valid - m0);
    #define SG53_RL_CASE(MB) { sg53_moe_router_logits_fixed<MB>(x_in_f32, router_gate, logits_out, e, m0, p.D, p.num_experts, lane); break; }
    switch (mb) {
        case 1u: SG53_RL_CASE(1u)
        case 2u: SG53_RL_CASE(2u)
        case 3u: SG53_RL_CASE(3u)
        case 4u: SG53_RL_CASE(4u)
        case 5u: SG53_RL_CASE(5u)
        case 6u: SG53_RL_CASE(6u)
        case 7u: SG53_RL_CASE(7u)
        case 8u: SG53_RL_CASE(8u)
        default: break;
    }
    #undef SG53_RL_CASE
}

// Fast `noaux_tc` Sigmoid + Bias + Top-K Selection over Precomputed `router_logits [M_valid, num_experts]` (`< 2 us`!)
// 100.0000% bit-exact to `sg53_moe_router_topk`!
// Grid: `(M_valid, 1, 1)`, `512` threads/TG (`16` SGs; supports `num_experts <= 512`)
kernel void sg53_moe_router_select(
    device const float*           router_logits [[buffer(0)]], // [M_valid, num_experts] FP32
    device const float*           e_score_bias  [[buffer(1)]], // [num_experts] FP32
    device uint*                  topk_idx_out  [[buffer(2)]], // [M_valid, top_k]
    device float*                 topk_w_out    [[buffer(3)]], // [M_valid, top_k]
    constant Sg53MoeRouterParams& p             [[buffer(4)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],                 // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],              // 0..15
    uint lane [[thread_index_in_simdgroup]]                    // 0..31
) {
    if (m >= p.M_valid) return;

    threadgroup float tg_scores[512];
    threadgroup float tg_choice[512];
    threadgroup float tg_val[16];
    threadgroup uint  tg_idx[16];
    threadgroup uint  sel_ids[16];

    if (tid < p.num_experts) {
        const float logit = router_logits[ulong(m) * ulong(p.num_experts) + ulong(tid)];
        const float sc    = sg53_sigmoid(logit);
        tg_scores[tid]    = sc;
        tg_choice[tid]    = sc + e_score_bias[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (p.n_group > 1u && p.topk_group < p.n_group && p.n_group <= 32u) {
        if (sgid == 0u) {
            const uint exp_per_grp = p.num_experts / p.n_group;
            float g_score = -INFINITY;
            if (lane < p.n_group) {
                float m1 = -INFINITY, m2 = -INFINITY;
                for (uint j = 0u; j < exp_per_grp; ++j) {
                    float v = tg_choice[lane * exp_per_grp + j];
                    if (v > m1) { m2 = m1; m1 = v; }
                    else if (v > m2) { m2 = v; }
                }
                g_score = m1 + m2;
            }
            uint g_active_mask = 0u;
            for (uint gk = 0u; gk < p.topk_group; ++gk) {
                float bv = ((g_active_mask & (1u << lane)) == 0u && lane < p.n_group) ? g_score : -INFINITY;
                uint  bi = lane;
                for (ushort off = 16u; off > 0u; off >>= 1u) {
                    float ov = simd_shuffle_xor(bv, off);
                    uint  oi = simd_shuffle_xor(bi, off);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                g_active_mask |= (1u << bi);
            }
            if (lane == 0u) tg_idx[0] = g_active_mask;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint g_mask = tg_idx[0];
        const uint exp_per_grp = p.num_experts / p.n_group;
        if (tid < p.num_experts) {
            uint g = tid / exp_per_grp;
            if ((g_mask & (1u << g)) == 0u) {
                tg_choice[tid] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint K = min(p.top_k, 16u);
    for (uint k = 0u; k < K; ++k) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;
        if (tid < p.num_experts) {
            bool already = false;
            for (uint prev = 0u; prev < k; ++prev) {
                if (sel_ids[prev] == tid) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                best_v = tg_choice[tid];
                best_i = tid;
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
            float v   = (lane < 16u) ? tg_val[lane] : -INFINITY;
            uint  idx = (lane < 16u) ? tg_idx[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 8u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(v, off);
                uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) {
                    v   = ov;
                    idx = oi;
                }
            }
            if (lane == 0u) {
                sel_ids[k] = (idx < p.num_experts) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float raw_w[16];
        float sum_w = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float rw = tg_scores[sel_ids[k]];
            raw_w[k] = rw;
            sum_w   += rw;
        }
        float scale = p.routed_scaling_factor;
        if (p.norm_topk_prob != 0u) {
            scale = p.routed_scaling_factor / (sum_w + 1e-20f);
        }
        for (uint k = 0u; k < K; ++k) {
            topk_idx_out[ulong(m) * p.top_k + k] = sel_ids[k];
            topk_w_out[ulong(m) * p.top_k + k]   = raw_w[k] * scale;
        }
    }
}

struct Sg53MoeIndirectParams {
    ulong expert_record_bytes; // 14,155,776 B (`54 * 256 KiB` — 64-bit ulong for >4 GiB `b_hot_experts`!)
    ulong expert_proj_bytes;   // 4,718,592 B
    ulong expert_weight_bytes; // 4,194,304 B
    ulong expert_scales_bytes; // 262,144 B
    uint  M_valid;
    uint  top_k;               // 8
    uint  D;                   // 4096
    uint  d_moe;               // 2048
    float swiglu_limit;        // 10.0f
    uint  num_tiles;
    uint  pad0;
    uint  pad1;
};

struct Sg53MoeWorkTile {
    uint source_buf; // STRATA_GLM53_EXPERT_SRC_HOT_UMA (0) or STRATA_GLM53_EXPERT_SRC_RING_DMA (1)
    uint slot_idx;   // Physical slot index inside `b_hot_experts` or `b_ring_dma`
    uint pair_start; // Offset inside `sorted_pairs`
    uint pair_count; // 1..8
};

// Zero-spill Indirect Multi-Buffer (`b_hot_experts` + `b_ring_dma`) Gather `gate_proj + up_proj + LimitedSwiGLU`!
// `1` row/SG (`4` rows/TG, `128` th/TG), single `x_in` pass — 100.0000% bit-identical to `sg53_moe_indirect_grouped_gu`!
// Grid: `((d_moe + 3) / 4, top_k, M_valid)`, `128` threads/TG (`4` SGs, `1` row/SG)
kernel void sg53_moe_indirect_gather_gu(
    device const uchar*                     hot_base  [[buffer(0)]], // UMA LFU Hot-Expert Cache
    device const uchar*                     ring_base [[buffer(1)]], // Ring_DMA Staging Buffer
    device const StrataGlm53ExpertDispatch* disp_list [[buffer(2)]], // [M_valid * top_k]
    device const half*                      x_in      [[buffer(3)]], // [M_valid, D]
    device half*                            mid_out   [[buffer(4)]], // [M_valid * top_k, d_moe]
    constant Sg53MoeIndirectParams&         p         [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],                   // 0..3
    uint  lane [[thread_index_in_simdgroup]]                         // 0..31
) {
    const uint row = tgid.x * 4u + sgid;
    const uint k   = tgid.y;
    const uint m   = tgid.z;
    if (row >= p.d_moe || k >= p.top_k || m >= p.M_valid) return;

    const ulong pair_idx = ulong(m) * p.top_k + k;
    const StrataGlm53ExpertDispatch disp = disp_list[pair_idx];
    if (disp.gate_weight == 0.0f) {
        if (lane == 0u) mid_out[pair_idx * p.d_moe + row] = half(0.0h);
        return;
    }

    // CRITICAL 64-bit `ulong` byte addressing (`b_hot_experts` > 4 GiB!)
    device const uchar* rec_base = ((disp.source_buf == STRATA_GLM53_EXPERT_SRC_HOT_UMA) ? hot_base : ring_base) +
                                   ulong(disp.slot_idx) * p.expert_record_bytes;

    device const uint2* gw = reinterpret_cast<device const uint2*>(rec_base + 0ul);
    device const half*  gs = reinterpret_cast<device const half*>(rec_base + p.expert_weight_bytes);
    device const half*  gb = reinterpret_cast<device const half*>(rec_base + p.expert_weight_bytes + p.expert_scales_bytes);

    device const uchar* up_base = rec_base + p.expert_proj_bytes;
    device const uint2* uw = reinterpret_cast<device const uint2*>(up_base + 0ul);
    device const half*  us = reinterpret_cast<device const half*>(up_base + p.expert_weight_bytes);
    device const half*  ub = reinterpret_cast<device const half*>(up_base + p.expert_weight_bytes + p.expert_scales_bytes);

    const uint u2_per_row  = p.D >> 4u;
    const uint grp_per_row = p.D >> 6u;
    device const half* xm  = x_in + ulong(m) * p.D;

    float ga = 0.0f;
    float ua = 0.0f;
    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 qg      = gw[ulong(row) * u2_per_row + u2_idx];
        const uint2 qu      = uw[ulong(row) * u2_per_row + u2_idx];
        const float gsc     = float(gs[ulong(row) * grp_per_row + grp_idx]);
        const float gbi     = float(gb[ulong(row) * grp_per_row + grp_idx]);
        const float usc     = float(us[ulong(row) * grp_per_row + grp_idx]);
        const float ubi     = float(ub[ulong(row) * grp_per_row + grp_idx]);

        half4 g0, g1, g2, g3, u0, u1, u2, u3;
        sg53_unpack_u32_nibbles_h4(qg.x, g0, g1); sg53_unpack_u32_nibbles_h4(qg.y, g2, g3);
        sg53_unpack_u32_nibbles_h4(qu.x, u0, u1); sg53_unpack_u32_nibbles_h4(qu.y, u2, u3);

        device const half4* xp = reinterpret_cast<device const half4*>(xm + u2_idx * 16u);
        const half4  x0  = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        const float4 xs4 = (float4(x0) + float4(x1)) + (float4(x2) + float4(x3));
        const float  xs  = (xs4.x + xs4.y) + (xs4.z + xs4.w);

        ga = fma(gsc, sg53_dot_h4x4(g0, g1, g2, g3, x0, x1, x2, x3), fma(gbi, xs, ga));
        ua = fma(usc, sg53_dot_h4x4(u0, u1, u2, u3, x0, x1, x2, x3), fma(ubi, xs, ua));
    }

    const float gv = simd_sum(ga);
    const float uv = simd_sum(ua);
    if (lane == 0u) {
        mid_out[pair_idx * p.d_moe + row] = sg53_clamp_f16(sg53_limited_swiglu(gv, uv, p.swiglu_limit));
    }
}

// Zero-spill Indirect Multi-Buffer Gather `down_proj` (`2` rows/SG = `8` rows/TG)
// 100.0000% bit-identical to `sg53_moe_indirect_grouped_dn`!
// Grid: `((D + 7) / 8, top_k, M_valid)`, `128` threads/TG (`4` SGs, `2` rows/SG)
kernel void sg53_moe_indirect_gather_dn(
    device const uchar*                     hot_base  [[buffer(0)]],
    device const uchar*                     ring_base [[buffer(1)]],
    device const StrataGlm53ExpertDispatch* disp_list [[buffer(2)]], // [M_valid * top_k]
    constant half*                          mid_in    [[buffer(3)]], // [M_valid * top_k, d_moe]
    device float*                           down_part [[buffer(4)]], // [M_valid * top_k, D]
    constant Sg53MoeIndirectParams&         p         [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],                   // 0..3
    uint  lane [[thread_index_in_simdgroup]]                         // 0..31
) {
    const uint row0 = (tgid.x * 4u + sgid) * 2u;
    const uint k    = tgid.y;
    const uint m    = tgid.z;
    if (row0 >= p.D || k >= p.top_k || m >= p.M_valid) return;

    const ulong pair_idx = ulong(m) * p.top_k + k;
    const StrataGlm53ExpertDispatch disp = disp_list[pair_idx];
    if (disp.gate_weight == 0.0f) {
        if (lane == 0u) {
            if (row0 + 2u <= p.D) *reinterpret_cast<device float2*>(down_part + pair_idx * p.D + row0) = float2(0.0f);
            else                  down_part[pair_idx * p.D + row0] = 0.0f;
        }
        return;
    }

    device const uchar* dn_base = ((disp.source_buf == STRATA_GLM53_EXPERT_SRC_HOT_UMA) ? hot_base : ring_base) +
                                  ulong(disp.slot_idx) * p.expert_record_bytes +
                                  2ul * p.expert_proj_bytes;

    device const uint2* dw = reinterpret_cast<device const uint2*>(dn_base + 0ul);
    device const half*  ds = reinterpret_cast<device const half*>(dn_base + p.expert_weight_bytes);
    device const half*  db = reinterpret_cast<device const half*>(dn_base + p.expert_weight_bytes + p.expert_scales_bytes);

    float2 res[1];
    sg53_q4_gemv2_rows_fixed<1u>(dw, ds, db, mid_in + pair_idx * p.d_moe, row0, p.D, p.d_moe, lane, res);

    if (lane == 0u) {
        if (row0 + 2u <= p.D) *reinterpret_cast<device float2*>(down_part + pair_idx * p.D + row0) = res[0];
        else                  down_part[pair_idx * p.D + row0] = res[0].x;
    }
}

// Zero-spill Expert-Grouped Register-Batched (`B = 1..8` compile-time unrolled!) Indirect `gate_proj + up_proj + LimitedSwiGLU`!
// Loads each expert's `9.0 MiB` Q4 gate+up weights ONCE across up to `8` tokens, 100.0000% bit-identical to `gather_gu`!
// Grid: `((d_moe + 3) / 4, num_tiles, 1)`, `128` threads/TG (`4` SGs, `1` row/SG)
template <uint B_FIX>
inline void sg53_moe_grouped_gu_fixed(
    device const uchar*             hot_base,
    device const uchar*             ring_base,
    device const half*              x_in,
    device const uint*              sorted_pairs,
    device half*                    mid_out,
    Sg53MoeWorkTile                 tile,
    constant Sg53MoeIndirectParams& p,
    uint row,
    uint lane
) {
    uint tok_m[B_FIX];
    uint pair_id[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        uint pid   = sorted_pairs[tile.pair_start + b];
        pair_id[b] = pid;
        tok_m[b]   = pid / p.top_k;
    }

    device const uchar* rec_base = ((tile.source_buf == STRATA_GLM53_EXPERT_SRC_HOT_UMA) ? hot_base : ring_base) +
                                   ulong(tile.slot_idx) * p.expert_record_bytes;

    device const uint2* gw = reinterpret_cast<device const uint2*>(rec_base + 0ul);
    device const half*  gs = reinterpret_cast<device const half*>(rec_base + p.expert_weight_bytes);
    device const half*  gb = reinterpret_cast<device const half*>(rec_base + p.expert_weight_bytes + p.expert_scales_bytes);

    device const uchar* up_base = rec_base + p.expert_proj_bytes;
    device const uint2* uw = reinterpret_cast<device const uint2*>(up_base + 0ul);
    device const half*  us = reinterpret_cast<device const half*>(up_base + p.expert_weight_bytes);
    device const half*  ub = reinterpret_cast<device const half*>(up_base + p.expert_weight_bytes + p.expert_scales_bytes);

    const uint u2_per_row  = p.D >> 4u;
    const uint grp_per_row = p.D >> 6u;

    float g_acc[B_FIX];
    float u_acc[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        g_acc[b] = 0.0f;
        u_acc[b] = 0.0f;
    }

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 qg      = gw[ulong(row) * u2_per_row + u2_idx];
        const uint2 qu      = uw[ulong(row) * u2_per_row + u2_idx];
        const float gsc     = float(gs[ulong(row) * grp_per_row + grp_idx]);
        const float gbi     = float(gb[ulong(row) * grp_per_row + grp_idx]);
        const float usc     = float(us[ulong(row) * grp_per_row + grp_idx]);
        const float ubi     = float(ub[ulong(row) * grp_per_row + grp_idx]);

        half4 g0, g1, g2, g3, u0, u1, u2, u3;
        sg53_unpack_u32_nibbles_h4(qg.x, g0, g1); sg53_unpack_u32_nibbles_h4(qg.y, g2, g3);
        sg53_unpack_u32_nibbles_h4(qu.x, u0, u1); sg53_unpack_u32_nibbles_h4(qu.y, u2, u3);

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) {
            device const half4* xp = reinterpret_cast<device const half4*>(
                x_in + ulong(tok_m[b]) * p.D + u2_idx * 16u);
            const half4  x0  = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float4 xs4 = (float4(x0) + float4(x1)) + (float4(x2) + float4(x3));
            const float  xs  = (xs4.x + xs4.y) + (xs4.z + xs4.w);

            g_acc[b] = fma(gsc, sg53_dot_h4x4(g0, g1, g2, g3, x0, x1, x2, x3), fma(gbi, xs, g_acc[b]));
            u_acc[b] = fma(usc, sg53_dot_h4x4(u0, u1, u2, u3, x0, x1, x2, x3), fma(ubi, xs, u_acc[b]));
        }
    }

    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        const float gv = simd_sum(g_acc[b]);
        const float uv = simd_sum(u_acc[b]);
        if (lane == 0u) {
            mid_out[ulong(pair_id[b]) * p.d_moe + row] = sg53_clamp_f16(sg53_limited_swiglu(gv, uv, p.swiglu_limit));
        }
    }
}

kernel void sg53_moe_indirect_grouped_gu(
    device const uchar*             hot_base     [[buffer(0)]],
    device const uchar*             ring_base    [[buffer(1)]],
    device const half*              x_in         [[buffer(2)]], // [M_padded, D]
    device const uint*              sorted_pairs [[buffer(3)]], // [<= M_valid * top_k]
    device const Sg53MoeWorkTile*   work_tiles   [[buffer(4)]], // [num_tiles]
    device half*                    mid_out      [[buffer(5)]], // [M_valid * top_k, d_moe]
    constant Sg53MoeIndirectParams& p            [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],              // 0..3
    uint  lane [[thread_index_in_simdgroup]]                    // 0..31
) {
    const uint tile_idx = tgid.y;
    const uint row      = tgid.x * 4u + sgid;
    if (tile_idx >= p.num_tiles || row >= p.d_moe) return;

    const Sg53MoeWorkTile tile = work_tiles[tile_idx];
    #define SG53_GGU_CASE(BF) { sg53_moe_grouped_gu_fixed<BF>(hot_base, ring_base, x_in, sorted_pairs, mid_out, tile, p, row, lane); break; }
    switch (tile.pair_count) {
        case 1u: SG53_GGU_CASE(1u)
        case 2u: SG53_GGU_CASE(2u)
        case 3u: SG53_GGU_CASE(3u)
        case 4u: SG53_GGU_CASE(4u)
        case 5u: SG53_GGU_CASE(5u)
        case 6u: SG53_GGU_CASE(6u)
        case 7u: SG53_GGU_CASE(7u)
        case 8u: SG53_GGU_CASE(8u)
        default: break;
    }
    #undef SG53_GGU_CASE
}

// Zero-spill Expert-Grouped Register-Batched (`B = 1..8` compile-time unrolled!) Indirect `down_proj` (`2` rows/SG = `8` rows/TG)
// 100.0000% bit-identical to `sg53_moe_indirect_gather_dn`!
// Grid: `((D + 7) / 8, num_tiles, 1)`, `128` threads/TG (`4` SGs, `2` rows/SG)
template <uint B_FIX>
inline void sg53_moe_grouped_dn_fixed(
    device const uchar*             hot_base,
    device const uchar*             ring_base,
    device const half*              mid_in,
    device const uint*              sorted_pairs,
    device float*                   down_part,
    Sg53MoeWorkTile                 tile,
    constant Sg53MoeIndirectParams& p,
    uint row0,
    uint lane
) {
    uint pair_id[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        pair_id[b] = sorted_pairs[tile.pair_start + b];
    }

    device const uchar* dn_base = ((tile.source_buf == STRATA_GLM53_EXPERT_SRC_HOT_UMA) ? hot_base : ring_base) +
                                  ulong(tile.slot_idx) * p.expert_record_bytes +
                                  2ul * p.expert_proj_bytes;

    device const uint2* dw = reinterpret_cast<device const uint2*>(dn_base + 0ul);
    device const half*  ds = reinterpret_cast<device const half*>(dn_base + p.expert_weight_bytes);
    device const half*  db = reinterpret_cast<device const half*>(dn_base + p.expert_weight_bytes + p.expert_scales_bytes);

    const uint u2_per_row  = p.d_moe >> 4u;
    const uint grp_per_row = p.d_moe >> 6u;
    const uint r0          = min(row0 + 0u, p.D - 1u);
    const uint r1          = min(row0 + 1u, p.D - 1u);

    float2 acc[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) acc[b] = float2(0.0f);

    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 q0      = dw[ulong(r0) * u2_per_row + u2_idx];
        const uint2 q1      = dw[ulong(r1) * u2_per_row + u2_idx];
        const float2 sc     = float2(float(ds[ulong(r0) * grp_per_row + grp_idx]),
                                     float(ds[ulong(r1) * grp_per_row + grp_idx]));
        const float2 bi     = float2(float(db[ulong(r0) * grp_per_row + grp_idx]),
                                     float(db[ulong(r1) * grp_per_row + grp_idx]));

        half4 n00, n01, n02, n03, n10, n11, n12, n13;
        sg53_unpack_u32_nibbles_h4(q0.x, n00, n01); sg53_unpack_u32_nibbles_h4(q0.y, n02, n03);
        sg53_unpack_u32_nibbles_h4(q1.x, n10, n11); sg53_unpack_u32_nibbles_h4(q1.y, n12, n13);

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) {
            device const half4* xp = reinterpret_cast<device const half4*>(
                mid_in + ulong(pair_id[b]) * p.d_moe + u2_idx * 16u);
            const half4  x0  = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            const float4 xs4 = (float4(x0) + float4(x1)) + (float4(x2) + float4(x3));
            const float  xs  = (xs4.x + xs4.y) + (xs4.z + xs4.w);

            acc[b].x = fma(sc.x, sg53_dot_h4x4(n00, n01, n02, n03, x0, x1, x2, x3), fma(bi.x, xs, acc[b].x));
            acc[b].y = fma(sc.y, sg53_dot_h4x4(n10, n11, n12, n13, x0, x1, x2, x3), fma(bi.y, xs, acc[b].y));
        }
    }

    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        float2 s(simd_sum(acc[b].x), simd_sum(acc[b].y));
        if (lane == 0u) {
            if (row0 + 2u <= p.D) *reinterpret_cast<device float2*>(down_part + ulong(pair_id[b]) * p.D + row0) = s;
            else                  down_part[ulong(pair_id[b]) * p.D + row0] = s.x;
        }
    }
}

kernel void sg53_moe_indirect_grouped_dn(
    device const uchar*             hot_base     [[buffer(0)]],
    device const uchar*             ring_base    [[buffer(1)]],
    device const half*              mid_in       [[buffer(2)]], // [M_valid * top_k, d_moe]
    device const uint*              sorted_pairs [[buffer(3)]],
    device const Sg53MoeWorkTile*   work_tiles   [[buffer(4)]], // [num_tiles]
    device float*                   down_part    [[buffer(5)]], // [M_valid * top_k, D]
    constant Sg53MoeIndirectParams& p            [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],              // 0..3
    uint  lane [[thread_index_in_simdgroup]]                    // 0..31
) {
    const uint tile_idx = tgid.y;
    const uint row0     = (tgid.x * 4u + sgid) * 2u;
    if (tile_idx >= p.num_tiles || row0 >= p.D) return;

    const Sg53MoeWorkTile tile = work_tiles[tile_idx];
    #define SG53_GDN_CASE(BF) { sg53_moe_grouped_dn_fixed<BF>(hot_base, ring_base, mid_in, sorted_pairs, down_part, tile, p, row0, lane); break; }
    switch (tile.pair_count) {
        case 1u: SG53_GDN_CASE(1u)
        case 2u: SG53_GDN_CASE(2u)
        case 3u: SG53_GDN_CASE(3u)
        case 4u: SG53_GDN_CASE(4u)
        case 5u: SG53_GDN_CASE(5u)
        case 6u: SG53_GDN_CASE(6u)
        case 7u: SG53_GDN_CASE(7u)
        case 8u: SG53_GDN_CASE(8u)
        default: break;
    }
    #undef SG53_GDN_CASE
}

struct Sg53MoeCombineParams {
    uint M_valid;
    uint top_k;
    uint D;
    uint pad0;
};

// Combines `y_mlp[m] = shared_out[m] + sum_{k=0..top_k-1} disp[m, k].gate_weight * down_part[m, k]`
// (Un-gated `1.0x` `shared_experts` + deterministic `k=0..7` routed sum!)
kernel void sg53_moe_combine(
    device const float*                     down_part  [[buffer(0)]], // [M_valid * top_k, D]
    device const StrataGlm53ExpertDispatch* disp_list  [[buffer(1)]], // [M_valid * top_k]
    device const float*                     shared_out [[buffer(2)]], // [M_valid, D]
    device float*                           mlp_out    [[buffer(3)]], // [M_valid, D]
    constant Sg53MoeCombineParams&          p          [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]                        // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    float4 y_moe = reinterpret_cast<device const float4*>(shared_out)[ulong(m) * d4_cnt + idx4];
    for (uint k = 0u; k < p.top_k; ++k) {
        const ulong pair_idx = ulong(m) * p.top_k + k;
        const float wk       = disp_list[pair_idx].gate_weight;
        if (wk != 0.0f) {
            const float4 dk4 = reinterpret_cast<device const float4*>(down_part)[pair_idx * d4_cnt + idx4];
            y_moe = fma(float4(wk), dk4, y_moe);
        }
    }
    reinterpret_cast<device float4*>(mlp_out)[ulong(m) * d4_cnt + idx4] = y_moe;
}

/* ============================================================================
 * 8b. v2 100% RAM-Resident `RHT-64 + E8S-18-V4` MoE & Single-CB UMA MTP Suite
 * ============================================================================ */

// Exact warp-shuffle block-64 Randomized Hadamard Transform (`0` TG mem, `0` barriers, `< 1 us`!):
//   `y = (1/8) * H_64 * diag(sign) * x` (bit-exact to host `sg53_rht64_f32`!)
// Each lane `0..31` holds `(i0, i1) = (2 * lane, 2 * lane + 1)`.
inline void sg53_rht64_warp_f32(thread float& v0, thread float& v1, uint lane) {
    constexpr ulong kMask = STRATA_GLM53_RHT64_SIGN_MASK;
    const uint i0 = lane * 2u;
    if (((kMask >> i0) & 1ul) != 0ul)        v0 = -v0;
    if (((kMask >> (i0 + 1u)) & 1ul) != 0ul) v1 = -v1;

    // Stage 0 (len = 1): in-lane butterfly
    const float a0 = v0 + v1;
    const float b0 = v0 - v1;
    v0 = a0;
    v1 = b0;

    // Stages 1..5 (len = 2, 4, 8, 16, 32 -> simd_shuffle_xor 1, 2, 4, 8, 16)
    #pragma clang loop unroll(full)
    for (ushort sh = 1u; sh <= 16u; sh <<= 1u) {
        const float o0 = simd_shuffle_xor(v0, sh);
        const float o1 = simd_shuffle_xor(v1, sh);
        const bool  hi = (lane & uint(sh)) != 0u;
        v0 = hi ? (o0 - v0) : (v0 + o0);
        v1 = hi ? (o1 - v1) : (v1 + o1);
    }

    v0 *= 0.125f;
    v1 *= 0.125f;
}

struct Sg53Rht64Params {
    uint M_valid;
    uint M_padded;
    uint K; // Multiple of 64
    uint pad0;
};

// Out-of-place OR in-place (`x_in == x_out`) FP16 `RHT-64` along `K` (`0` TG mem, `0` barriers!)
// Grid: `(((K / 64) + 3) / 4, M_padded, 1)`, `128` threads/TG (`4` SGs = `4` blocks of 64/TG)
kernel void sg53_rht64_f16(
    device const half*        x_in  [[buffer(0)]], // [M_valid, K]
    device half*              x_out [[buffer(1)]], // [M_padded, K]
    constant Sg53Rht64Params& p     [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint  lane [[thread_index_in_simdgroup]]       // 0..31
) {
    const uint blk = tgid.x * 4u + sgid;
    const uint m   = tgid.y;
    if (blk >= (p.K >> 6u) || m >= p.M_padded) return;

    device half2* dst2 = reinterpret_cast<device half2*>(x_out + ulong(m) * p.K + blk * 64u);
    if (m >= p.M_valid) {
        dst2[lane] = half2(0.0h);
        return;
    }

    const half2 h2 = reinterpret_cast<device const half2*>(x_in + ulong(m) * p.K + blk * 64u)[lane];
    float v0 = float(h2.x);
    float v1 = float(h2.y);
    sg53_rht64_warp_f32(v0, v1, lane);
    dst2[lane] = half2(sg53_clamp_f16(v0), sg53_clamp_f16(v1));
}

// Standalone fused `LimitedSwiGLU(gate, up) + RHT-64` (`0` TG mem, `0` barriers!)
// Grid: `(((d_ff / 64) + 3) / 4, M_padded, 1)`, `128` threads/TG (`4` SGs)
kernel void sg53_limited_swiglu_rht64(
    device const float*        gate_up_f32 [[buffer(0)]], // [M_valid, 2 * d_ff]
    device half*               mid_rot_f16 [[buffer(1)]], // [M_padded, d_ff]
    constant Sg53SwigluParams& p           [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint blk = tgid.x * 4u + sgid;
    const uint m   = tgid.y;
    if (blk >= (p.d_ff >> 6u) || m >= p.M_padded) return;

    device half2* dst2 = reinterpret_cast<device half2*>(mid_rot_f16 + ulong(m) * p.d_ff + blk * 64u);
    if (m >= p.M_valid) {
        dst2[lane] = half2(0.0h);
        return;
    }

    const ulong row_base = ulong(m) * (2ul * p.d_ff) + blk * 64u;
    const float2 g2 = reinterpret_cast<device const float2*>(gate_up_f32 + row_base)[lane];
    const float2 u2 = reinterpret_cast<device const float2*>(gate_up_f32 + row_base + p.d_ff)[lane];
    float v0 = sg53_limited_swiglu(g2.x, u2.x, p.limit);
    float v1 = sg53_limited_swiglu(g2.y, u2.y, p.limit);
    sg53_rht64_warp_f32(v0, v1, lane);
    dst2[lane] = half2(sg53_clamp_f16(v0), sg53_clamp_f16(v1));
}

// ---- GPU-Side Single-CB MoE Binning, Gather & Direct `b_topk_w` Combine ----

struct Sg53MoeV2WorkTile {
    uint expert_id;
    uint pair_start;
    uint pair_count;
    uint pad0;
};

struct Sg53MoeBinParams {
    uint M_valid;
    uint top_k;             // 8
    uint num_experts;       // 288 (<= 512)
    uint d_moe;             // 2048
    uint D;                 // 4096
    uint max_tiles_per_bin; // 1024
    uint pad0;
    uint pad1;
};

// GPU-side binning for `M = 1..8` spec/decode (`1..8` pairs/tile) — eliminates ALL mid-step CPU syncs!
// Writes `2` `MTLDispatchThreadgroupsIndirectArguments` (`gu` at `+0`, `dn` at `+12`) into `indirect_args`!
// Grid: `(1, 1, 1)`, `512` threads/TG (`16` SGs)
kernel void sg53_moe_bin_tokens_v2(
    device const uint*        topk_idx      [[buffer(0)]], // [M_valid * top_k]
    device uint*              sorted_pairs  [[buffer(1)]], // [M_valid * top_k]
    device Sg53MoeV2WorkTile* work_tiles    [[buffer(2)]], // [max_tiles_per_bin]
    device uint*              indirect_args [[buffer(3)]], // [8] (`gu` 3x u32, `dn` 3x u32, `num_tiles`, pad)
    constant Sg53MoeBinParams& p            [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],             // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],          // 0..15
    uint lane [[thread_index_in_simdgroup]]                // 0..31
) {
    threadgroup atomic_uint tg_cnt[512];
    threadgroup uint        tg_off[512];
    threadgroup uint        tg_toff[512];
    threadgroup atomic_uint tg_cur[512];
    threadgroup uint        sg_c[16];
    threadgroup uint        sg_t[16];
    threadgroup uint        tg_tot_tiles;

    atomic_store_explicit(&tg_cnt[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pairs = p.M_valid * p.top_k;
    for (uint i = tid; i < total_pairs; i += 512u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) {
            atomic_fetch_add_explicit(&tg_cnt[e], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c  = (tid < p.num_experts) ? atomic_load_explicit(&tg_cnt[tid], memory_order_relaxed) : 0u;
    const uint nt = (c + 7u) >> 3u;

    const uint c_pre = simd_prefix_exclusive_sum(c);
    const uint c_tot = simd_sum(c);
    const uint t_pre = simd_prefix_exclusive_sum(nt);
    const uint t_tot = simd_sum(nt);
    if (lane == 0u) {
        sg_c[sgid] = c_tot;
        sg_t[sgid] = t_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const uint cv   = (lane < 16u) ? sg_c[lane] : 0u;
        const uint tv   = (lane < 16u) ? sg_t[lane] : 0u;
        const uint c_sp = simd_prefix_exclusive_sum(cv);
        const uint t_sp = simd_prefix_exclusive_sum(tv);
        const uint t_gt = simd_sum(tv);
        if (lane < 16u) {
            sg_c[lane] = c_sp;
            sg_t[lane] = t_sp;
        }
        if (lane == 0u) {
            tg_tot_tiles = min(t_gt, p.max_tiles_per_bin);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint my_off  = sg_c[sgid] + c_pre;
    const uint my_toff = sg_t[sgid] + t_pre;
    tg_off[tid]  = my_off;
    tg_toff[tid] = my_toff;
    atomic_store_explicit(&tg_cur[tid], my_off, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < total_pairs; i += 512u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) {
            const uint pos = atomic_fetch_add_explicit(&tg_cur[e], 1u, memory_order_relaxed);
            sorted_pairs[pos] = i;
        }
    }

    if (tid < p.num_experts && nt > 0u) {
        for (uint k = 0u; k < nt; ++k) {
            const uint t_idx = my_toff + k;
            if (t_idx < p.max_tiles_per_bin) {
                Sg53MoeV2WorkTile tile;
                tile.expert_id  = tid;
                tile.pair_start = my_off + k * 8u;
                tile.pair_count = min(8u, c - k * 8u);
                tile.pad0       = 0u;
                work_tiles[t_idx] = tile;
            }
        }
    }

    if (tid == 0u) {
        const uint num_tiles = tg_tot_tiles;
        // Indirect args for `q4_uma_grouped_gu` (`4` rows/TG, offset 0)
        indirect_args[0]  = (p.d_moe + 3u) >> 2u;
        indirect_args[1]  = num_tiles;
        indirect_args[2]  = 1u;
        // Indirect args for `q4_uma_grouped_dn` (`8` rows/TG, offset 12)
        indirect_args[3]  = (p.D + 7u) >> 3u;
        indirect_args[4]  = num_tiles;
        indirect_args[5]  = 1u;
        indirect_args[6]  = num_tiles;
        indirect_args[7]  = 0u;
        // Indirect args for `e8s18_grouped_gu` (`8` rows/TG = `2` rows/SG, offset 32)
        indirect_args[8]  = (p.d_moe + 7u) >> 3u;
        indirect_args[9]  = num_tiles;
        indirect_args[10] = 1u;
        // Indirect args for `e8s18_grouped_dn` (`16` rows/TG = `4` rows/SG, offset 44)
        indirect_args[11] = (p.D + 15u) >> 4u;
        indirect_args[12] = num_tiles;
        indirect_args[13] = 1u;
    }
}

// GPU-side Tri-Bin (`BM = 64 / 32 / 16`) NAX MoE binning for Prefill (`M > 8`)!
// Writes `6` `MTLDispatchThreadgroupsIndirectArguments` (`gu_m64/m32/m16` + `dn_m64/m32/m16`)!
// Grid: `(1, 1, 1)`, `512` threads/TG (`16` SGs)
kernel void sg53_moe_bin_tokens_nax(
    device const uint*        topk_idx      [[buffer(0)]], // [M_valid * top_k]
    device uint*              sorted_pairs  [[buffer(1)]], // [M_valid * top_k]
    device Sg53MoeV2WorkTile* work_tiles    [[buffer(2)]], // [3 * max_tiles_per_bin] (`M64`, `M32`, `M16`)
    device uint*              indirect_args [[buffer(3)]], // [24] (`6` x 3 u32 + counts)
    constant Sg53MoeBinParams& p            [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],             // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],          // 0..15
    uint lane [[thread_index_in_simdgroup]]                // 0..31
) {
    threadgroup atomic_uint tg_cnt[512];
    threadgroup atomic_uint tg_cur[512];
    threadgroup uint        sg_c[16];
    threadgroup uint        sg_64[16];
    threadgroup uint        sg_32[16];
    threadgroup uint        sg_16[16];
    threadgroup uint        tg_tots[4];

    atomic_store_explicit(&tg_cnt[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pairs = p.M_valid * p.top_k;
    for (uint i = tid; i < total_pairs; i += 512u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) {
            atomic_fetch_add_explicit(&tg_cnt[e], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c   = (tid < p.num_experts) ? atomic_load_explicit(&tg_cnt[tid], memory_order_relaxed) : 0u;
    const uint q64 = c >> 6u;
    const uint rem = c & 63u;
    const uint n64 = q64 + ((rem >= 33u) ? 1u : 0u);
    const uint n32 = (rem >= 17u && rem <= 32u) ? 1u : 0u;
    const uint n16 = (rem >= 1u  && rem <= 16u) ? 1u : 0u;

    const uint c_pre   = simd_prefix_exclusive_sum(c);
    const uint n64_pre = simd_prefix_exclusive_sum(n64);
    const uint n32_pre = simd_prefix_exclusive_sum(n32);
    const uint n16_pre = simd_prefix_exclusive_sum(n16);
    const uint c_tot   = simd_sum(c);
    const uint n64_tot = simd_sum(n64);
    const uint n32_tot = simd_sum(n32);
    const uint n16_tot = simd_sum(n16);

    if (lane == 0u) {
        sg_c[sgid]  = c_tot;
        sg_64[sgid] = n64_tot;
        sg_32[sgid] = n32_tot;
        sg_16[sgid] = n16_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const uint vc  = (lane < 16u) ? sg_c[lane]  : 0u;
        const uint v64 = (lane < 16u) ? sg_64[lane] : 0u;
        const uint v32 = (lane < 16u) ? sg_32[lane] : 0u;
        const uint v16 = (lane < 16u) ? sg_16[lane] : 0u;

        const uint pc  = simd_prefix_exclusive_sum(vc);
        const uint p64 = simd_prefix_exclusive_sum(v64);
        const uint p32 = simd_prefix_exclusive_sum(v32);
        const uint p16 = simd_prefix_exclusive_sum(v16);

        const uint t64 = simd_sum(v64);
        const uint t32 = simd_sum(v32);
        const uint t16 = simd_sum(v16);

        if (lane < 16u) {
            sg_c[lane]  = pc;
            sg_64[lane] = p64;
            sg_32[lane] = p32;
            sg_16[lane] = p16;
        }
        if (lane == 0u) {
            tg_tots[0] = min(t64, p.max_tiles_per_bin);
            tg_tots[1] = min(t32, p.max_tiles_per_bin);
            tg_tots[2] = min(t16, p.max_tiles_per_bin);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint my_off = sg_c[sgid] + c_pre;
    atomic_store_explicit(&tg_cur[tid], my_off, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < total_pairs; i += 512u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) {
            const uint pos = atomic_fetch_add_explicit(&tg_cur[e], 1u, memory_order_relaxed);
            sorted_pairs[pos] = i;
        }
    }

    if (tid < p.num_experts && c > 0u) {
        const uint o64 = sg_64[sgid] + n64_pre;
        const uint o32 = sg_32[sgid] + n32_pre;
        const uint o16 = sg_16[sgid] + n16_pre;
        uint cur_p = my_off;
        for (uint k = 0u; k < n64; ++k) {
            const uint take = min(64u, my_off + c - cur_p);
            if (o64 + k < p.max_tiles_per_bin) {
                work_tiles[o64 + k] = { tid, cur_p, take, 0u };
            }
            cur_p += take;
        }
        if (n32 != 0u && o32 < p.max_tiles_per_bin) {
            work_tiles[p.max_tiles_per_bin + o32] = { tid, cur_p, my_off + c - cur_p, 0u };
            cur_p = my_off + c;
        }
        if (n16 != 0u && o16 < p.max_tiles_per_bin) {
            work_tiles[2u * p.max_tiles_per_bin + o16] = { tid, cur_p, my_off + c - cur_p, 0u };
        }
    }

    if (tid == 0u) {
        const uint gx_gu = (p.d_moe + 63u) >> 6u;
        const uint gx_dn = (p.D + 63u) >> 6u;
        const uint t64   = tg_tots[0];
        const uint t32   = tg_tots[1];
        const uint t16   = tg_tots[2];

        // gu M64 (+0), M32 (+12), M16 (+24)
        indirect_args[0]  = gx_gu; indirect_args[1]  = t64; indirect_args[2]  = 1u;
        indirect_args[3]  = gx_gu; indirect_args[4]  = t32; indirect_args[5]  = 1u;
        indirect_args[6]  = gx_gu; indirect_args[7]  = t16; indirect_args[8]  = 1u;
        // dn M64 (+36), M32 (+48), M16 (+60)
        indirect_args[9]  = gx_dn; indirect_args[10] = t64; indirect_args[11] = 1u;
        indirect_args[12] = gx_dn; indirect_args[13] = t32; indirect_args[14] = 1u;
        indirect_args[15] = gx_dn; indirect_args[16] = t16; indirect_args[17] = 1u;
        indirect_args[18] = t64;   indirect_args[19] = t32; indirect_args[20] = t16;
        indirect_args[21] = 0u;
    }
}

// Gathers `x_in [M_valid, D]` into expert-sorted `x_sorted [total_pairs + 64, D]` AND zeroes `+64` pad tail
// of BOTH `x_sorted` and `mid_sorted` so NAX `BM=64` slices never read uninitialized memory!
// Grid: `(total_pairs + 64, 1, 1)`, `256` threads/TG
kernel void sg53_moe_gather_x_f16(
    device const half*        x_in         [[buffer(0)]], // [M_valid, D]
    device const uint*        sorted_pairs [[buffer(1)]], // [total_pairs]
    device half*              x_sorted     [[buffer(2)]], // [total_pairs + 64, D]
    device half*              mid_sorted   [[buffer(3)]], // [total_pairs + 64, d_moe]
    constant Sg53MoeBinParams& p           [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]              // 0..255
) {
    const uint total_pairs = p.M_valid * p.top_k;
    if (row >= total_pairs + 64u) return;

    const uint d4_cnt = p.D >> 2u;
    device half4* dst4 = reinterpret_cast<device half4*>(x_sorted + ulong(row) * p.D);
    if (row < total_pairs) {
        const uint m = sorted_pairs[row] / p.top_k;
        device const half4* src4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.D);
        for (uint c4 = tid; c4 < d4_cnt; c4 += 256u) {
            dst4[c4] = src4[c4];
        }
    } else {
        for (uint c4 = tid; c4 < d4_cnt; c4 += 256u) {
            dst4[c4] = half4(0.0h);
        }
        const uint m4_cnt = p.d_moe >> 2u;
        device half4* mdst4 = reinterpret_cast<device half4*>(mid_sorted + ulong(row) * p.d_moe);
        for (uint c4 = tid; c4 < m4_cnt; c4 += 256u) {
            mdst4[c4] = half4(0.0h);
        }
    }
}

// Direct `b_topk_w` MoE combine (`y_mlp[m] = shared_out[m] + sum_{k=0..top_k-1} topk_w[m, k] * down_part[m, k]`)
kernel void sg53_moe_combine_v2(
    device const float*            down_part  [[buffer(0)]], // [M_valid * top_k, D]
    device const float*            topk_w     [[buffer(1)]], // [M_valid * top_k]
    device const float*            shared_out [[buffer(2)]], // [M_valid, D]
    device float*                  mlp_out    [[buffer(3)]], // [M_valid, D]
    constant Sg53MoeCombineParams& p          [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]               // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    float4 y_moe = reinterpret_cast<device const float4*>(shared_out)[ulong(m) * d4_cnt + idx4];
    for (uint k = 0u; k < p.top_k; ++k) {
        const ulong pair_idx = ulong(m) * p.top_k + k;
        const float wk       = topk_w[pair_idx];
        if (wk != 0.0f) {
            const float4 dk4 = reinterpret_cast<device const float4*>(down_part)[pair_idx * d4_cnt + idx4];
            y_moe = fma(float4(wk), dk4, y_moe);
        }
    }
    reinterpret_cast<device float4*>(mlp_out)[ulong(m) * d4_cnt + idx4] = y_moe;
}

// Fused v2 MoE Combine + `mHC` Expand (`-42` serial barriers/token & eliminates `b_block_y_f32` roundtrip!)
// Grid: `((D/4 + 255) / 256, M_valid, 1)`, `256` threads/TG
kernel void sg53_moe_combine_mhc_expand(
    device float*                  h_inout    [[buffer(0)]], // [M_valid, 4 * D]
    device const float*            down_part  [[buffer(1)]], // [M_valid * top_k, D]
    device const float*            topk_w     [[buffer(2)]], // [M_valid * top_k]
    device const float*            shared_out [[buffer(3)]], // [M_valid, D]
    device const float*            post_in    [[buffer(4)]], // [M_valid, 4]
    device const float*            comb_in    [[buffer(5)]], // [M_valid, 16]
    constant Sg53MoeCombineParams& p          [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]               // 0..255
) {
    const uint m = tgid.y;
    if (m >= p.M_valid) return;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (idx4 >= d4_cnt) return;

    float4 yv = reinterpret_cast<device const float4*>(shared_out)[ulong(m) * d4_cnt + idx4];
    for (uint k = 0u; k < p.top_k; ++k) {
        const ulong pair_idx = ulong(m) * p.top_k + k;
        const float wk       = topk_w[pair_idx];
        if (wk != 0.0f) {
            const float4 dk4 = reinterpret_cast<device const float4*>(down_part)[pair_idx * d4_cnt + idx4];
            yv = fma(float4(wk), dk4, yv);
        }
    }

    device float4* s0_p = reinterpret_cast<device float4*>(h_inout + (ulong(m) * 4ul + 0ul) * p.D);
    device float4* s1_p = reinterpret_cast<device float4*>(h_inout + (ulong(m) * 4ul + 1ul) * p.D);
    device float4* s2_p = reinterpret_cast<device float4*>(h_inout + (ulong(m) * 4ul + 2ul) * p.D);
    device float4* s3_p = reinterpret_cast<device float4*>(h_inout + (ulong(m) * 4ul + 3ul) * p.D);
    const float4 h0 = s0_p[idx4], h1 = s1_p[idx4], h2 = s2_p[idx4], h3 = s3_p[idx4];

    device const float* pst = post_in + ulong(m) * 4ul;
    device const float* cmb = comb_in + ulong(m) * 16ul;

    s0_p[idx4] = fma(float4(pst[0]), yv,
                 fma(float4(cmb[0]), h0, fma(float4(cmb[4]), h1, fma(float4(cmb[8]),  h2, float4(cmb[12]) * h3))));
    s1_p[idx4] = fma(float4(pst[1]), yv,
                 fma(float4(cmb[1]), h0, fma(float4(cmb[5]), h1, fma(float4(cmb[9]),  h2, float4(cmb[13]) * h3))));
    s2_p[idx4] = fma(float4(pst[2]), yv,
                 fma(float4(cmb[2]), h0, fma(float4(cmb[6]), h1, fma(float4(cmb[10]), h2, float4(cmb[14]) * h3))));
    s3_p[idx4] = fma(float4(pst[3]), yv,
                 fma(float4(cmb[3]), h0, fma(float4(cmb[7]), h1, fma(float4(cmb[11]), h2, float4(cmb[15]) * h3))));
}

// ---- `E8S-18-V4` Decode GEMV (`constant` 64 KiB LUT + 2x LUT prefetch + dual-rate FP16 inner FMA, 0 bit diffs M=1==M=5!) ----

struct Sg53MoeE8Params {
    ulong layer_byte_off;   // Byte offset of this MoE layer inside `part0` or `part1`
    ulong e8_record_bytes;  // 7,471,104 B (or synthetic size)
    ulong e8_proj_bytes;    // 2,490,368 B
    ulong e8_lo16_bytes;    // 2,097,152 B
    ulong e8_hi2_bytes;     //   262,144 B
    uint  M_valid;
    uint  top_k;            // 8
    uint  D;                // 4096 (multiple of 64)
    uint  d_moe;            // 2048 (multiple of 64)
    float swiglu_limit;     // 10.0f
    uint  total_pairs_pad;  // M_valid * top_k + 64
    uint  tile_base;        // Tile start offset inside `work_tiles` (`0`, `max_tiles`, `2*max_tiles`)
    uint  pad0;
    uint  pad1;
};

inline void sg53_e8s18_decode_entry(
    StrataGlm53E8Half8Entry e,
    uint                    h,
    uint                    v4,
    thread half4&           out_lo,
    thread half4&           out_hi
) {
    const bool  sw = (v4 & 1u) != 0u;
    const bool  zw = (v4 & 2u) != 0u;
    const half4 a  = sw ? e.hi : e.lo;
    const half4 b  = sw ? e.lo : e.hi;
    const half4 l  = zw ? a.zwxy : a;
    const half4 r  = zw ? b.zwxy : b;
    out_lo = as_type<half4>(as_type<ushort4>(l) ^ kSg53H8SgnLo[h]);
    out_hi = as_type<half4>(as_type<ushort4>(r) ^ kSg53H8SgnHi[h]);
}

inline void sg53_e8s18_decode_h4_c(
    constant StrataGlm53E8Half8Entry* lut,
    ushort                            lo16,
    uint                              v4,
    thread half4&                     out_lo,
    thread half4&                     out_hi
) {
    sg53_e8s18_decode_entry(lut[uint(lo16) >> 4u], uint(lo16) & 15u, v4, out_lo, out_hi);
}

// Canonical 64-weight `E8S-18-V4` dot product (single `half4` inner accum -> FP32 outer):
// Exact bit-for-bit identical FMA order shared by `gather` (`M=1`) AND 2-weight fused `grouped` (`M=2..8`) -> `0` bit diffs!
inline void sg53_e8s18_dot64_2r_c(
    constant StrataGlm53E8Half8Entry* lut,
    uint4                             q0,
    ushort                            h0,
    uint4                             q1,
    ushort                            h1,
    constant half4*                   xpg,
    thread float&                     d0,
    thread float&                     d1
) {
    half4 a(0.0h), b(0.0h);
    half4 wl, wh;
    #define SG53_E8_2R(LO0, LO1, SH, X0, X1) { \
        const half4 xv0 = xpg[X0], xv1 = xpg[X1]; \
        const StrataGlm53E8Half8Entry e0 = lut[uint(LO0) >> 4u]; \
        const StrataGlm53E8Half8Entry e1 = lut[uint(LO1) >> 4u]; \
        sg53_e8s18_decode_entry(e0, uint(LO0) & 15u, (uint(h0) >> (SH)) & 3u, wl, wh); a = fma(wl, xv0, fma(wh, xv1, a)); \
        sg53_e8s18_decode_entry(e1, uint(LO1) & 15u, (uint(h1) >> (SH)) & 3u, wl, wh); b = fma(wl, xv0, fma(wh, xv1, b)); \
    }
    SG53_E8_2R(ushort(q0.x),        ushort(q1.x),         0u,  0,  1)
    SG53_E8_2R(ushort(q0.x >> 16u), ushort(q1.x >> 16u),  2u,  2,  3)
    SG53_E8_2R(ushort(q0.y),        ushort(q1.y),         4u,  4,  5)
    SG53_E8_2R(ushort(q0.y >> 16u), ushort(q1.y >> 16u),  6u,  6,  7)
    SG53_E8_2R(ushort(q0.z),        ushort(q1.z),         8u,  8,  9)
    SG53_E8_2R(ushort(q0.z >> 16u), ushort(q1.z >> 16u), 10u, 10, 11)
    SG53_E8_2R(ushort(q0.w),        ushort(q1.w),        12u, 12, 13)
    SG53_E8_2R(ushort(q0.w >> 16u), ushort(q1.w >> 16u), 14u, 14, 15)
    #undef SG53_E8_2R
    const float4 fa = float4(a);
    const float4 fb = float4(b);
    d0 = (fa.x + fa.y) + (fa.z + fa.w);
    d1 = (fb.x + fb.y) + (fb.z + fb.w);
}

// Decode `M=1` `2` rows/SG (`8` rows/TG = `2048` TGs across `top_k=8`!) Sequential `gate` + `up` + `LimitedSwiGLU`!
// Uses `constant` 64 KiB LUT + 2-row fused `xpg` reuse (`+20%` random-LUT BW, 0 bit diffs vs `grouped`!).
// Grid: `((d_moe + 7) / 8, top_k, M_valid)`, `128` threads/TG (`4` SGs, `2` rows/SG)
kernel void sg53_moe_e8s18_gather_gu(
    device const uchar*               part_base [[buffer(0)]], // `part0` or `part1`
    constant StrataGlm53E8Half8Entry* lut       [[buffer(1)]], // 64 KiB private `E8S-18-V4` codebook (constant cache!)
    device const uint*                topk_idx  [[buffer(2)]], // [M_valid * top_k]
    constant half*                    x_rot     [[buffer(3)]], // [M_valid, D] (RHT-64 rotated!)
    device half*                      mid_out   [[buffer(4)]], // [M_valid * top_k, d_moe]
    constant Sg53MoeE8Params&         p         [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],             // 0..3
    uint  lane [[thread_index_in_simdgroup]]                   // 0..31
) {
    const uint row0 = (tgid.x * 4u + sgid) * 2u;
    const uint k    = tgid.y;
    const uint m    = tgid.z;
    if (row0 >= p.d_moe || k >= p.top_k || m >= p.M_valid) return;

    const ulong pair_idx = ulong(m) * p.top_k + k;
    const uint  eid      = topk_idx[pair_idx];
    const uint  grp      = p.D >> 6u;
    const ulong sc_off   = p.e8_lo16_bytes + p.e8_hi2_bytes;
    const uint  r0       = min(row0 + 0u, p.d_moe - 1u);
    const uint  r1       = min(row0 + 1u, p.d_moe - 1u);

    device const uchar* rec_base = part_base + p.layer_byte_off + ulong(eid) * p.e8_record_bytes;
    constant half* xm            = x_rot + ulong(m) * p.D;

    float2 ga(0.0f);
    {
        device const uint4*  lo = reinterpret_cast<device const uint4*> (rec_base);
        device const ushort* hi = reinterpret_cast<device const ushort*>(rec_base + p.e8_lo16_bytes);
        device const uchar*  sc = rec_base + sc_off;
        for (uint g = lane; g < grp; g += 32u) {
            constant half4* xpg = reinterpret_cast<constant half4*>(xm + g * 64u);
            const ulong g0 = ulong(r0) * grp + g;
            const ulong g1 = ulong(r1) * grp + g;
            float d0, d1;
            sg53_e8s18_dot64_2r_c(lut, lo[g0], hi[g0], lo[g1], hi[g1], xpg, d0, d1);
            ga.x = fma(float(sg53_e4m4_decode_h(sc[g0])), d0, ga.x);
            ga.y = fma(float(sg53_e4m4_decode_h(sc[g1])), d1, ga.y);
        }
    }

    float2 ua(0.0f);
    {
        device const uchar*  ub = rec_base + p.e8_proj_bytes;
        device const uint4*  lo = reinterpret_cast<device const uint4*> (ub);
        device const ushort* hi = reinterpret_cast<device const ushort*>(ub + p.e8_lo16_bytes);
        device const uchar*  sc = ub + sc_off;
        for (uint g = lane; g < grp; g += 32u) {
            constant half4* xpg = reinterpret_cast<constant half4*>(xm + g * 64u);
            const ulong g0 = ulong(r0) * grp + g;
            const ulong g1 = ulong(r1) * grp + g;
            float d0, d1;
            sg53_e8s18_dot64_2r_c(lut, lo[g0], hi[g0], lo[g1], hi[g1], xpg, d0, d1);
            ua.x = fma(float(sg53_e4m4_decode_h(sc[g0])), d0, ua.x);
            ua.y = fma(float(sg53_e4m4_decode_h(sc[g1])), d1, ua.y);
        }
    }

    const float2 gv(simd_sum(ga.x), simd_sum(ga.y));
    const float2 uv(simd_sum(ua.x), simd_sum(ua.y));
    if (lane == 0u) {
        device half* dst = mid_out + pair_idx * p.d_moe + row0;
        const half h0 = sg53_clamp_f16(sg53_limited_swiglu(gv.x, uv.x, p.swiglu_limit));
        if (row0 + 2u <= p.d_moe) {
            const half h1 = sg53_clamp_f16(sg53_limited_swiglu(gv.y, uv.y, p.swiglu_limit));
            *reinterpret_cast<device half2*>(dst) = half2(h0, h1);
        } else {
            dst[0] = h0;
        }
    }
}

// High-occupancy `256`-thread/TG (`8` SGs, `2` rows/SG = `16` rows/TG) `E8S-18-V4` `gather_gu` (`409+ GB/s` random LUT!)
// Grid: `((d_moe + 15) / 16, top_k, M_valid)`, `256` threads/TG
kernel void sg53_moe_e8s18_gather_gu_256(
    device const uchar*               part_base [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut       [[buffer(1)]],
    device const uint*                topk_idx  [[buffer(2)]],
    constant half*                    x_rot     [[buffer(3)]],
    device half*                      mid_out   [[buffer(4)]],
    constant Sg53MoeE8Params&         p         [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],             // 0..7
    uint  lane [[thread_index_in_simdgroup]]                   // 0..31
) {
    const uint row0 = (tgid.x * 8u + sgid) * 2u;
    const uint k    = tgid.y;
    const uint m    = tgid.z;
    if (row0 >= p.d_moe || k >= p.top_k || m >= p.M_valid) return;

    const ulong pair_idx = ulong(m) * p.top_k + k;
    const uint  eid      = topk_idx[pair_idx];
    const uint  grp      = p.D >> 6u;
    const ulong sc_off   = p.e8_lo16_bytes + p.e8_hi2_bytes;
    const uint  r0       = min(row0 + 0u, p.d_moe - 1u);
    const uint  r1       = min(row0 + 1u, p.d_moe - 1u);

    device const uchar* rec_base = part_base + p.layer_byte_off + ulong(eid) * p.e8_record_bytes;
    constant half* xm            = x_rot + ulong(m) * p.D;

    float2 ga(0.0f);
    {
        device const uint4*  lo = reinterpret_cast<device const uint4*> (rec_base);
        device const ushort* hi = reinterpret_cast<device const ushort*>(rec_base + p.e8_lo16_bytes);
        device const uchar*  sc = rec_base + sc_off;
        for (uint g = lane; g < grp; g += 32u) {
            constant half4* xpg = reinterpret_cast<constant half4*>(xm + g * 64u);
            const ulong g0 = ulong(r0) * grp + g;
            const ulong g1 = ulong(r1) * grp + g;
            float d0, d1;
            sg53_e8s18_dot64_2r_c(lut, lo[g0], hi[g0], lo[g1], hi[g1], xpg, d0, d1);
            ga.x = fma(float(sg53_e4m4_decode_h(sc[g0])), d0, ga.x);
            ga.y = fma(float(sg53_e4m4_decode_h(sc[g1])), d1, ga.y);
        }
    }

    float2 ua(0.0f);
    {
        device const uchar*  ub = rec_base + p.e8_proj_bytes;
        device const uint4*  lo = reinterpret_cast<device const uint4*> (ub);
        device const ushort* hi = reinterpret_cast<device const ushort*>(ub + p.e8_lo16_bytes);
        device const uchar*  sc = ub + sc_off;
        for (uint g = lane; g < grp; g += 32u) {
            constant half4* xpg = reinterpret_cast<constant half4*>(xm + g * 64u);
            const ulong g0 = ulong(r0) * grp + g;
            const ulong g1 = ulong(r1) * grp + g;
            float d0, d1;
            sg53_e8s18_dot64_2r_c(lut, lo[g0], hi[g0], lo[g1], hi[g1], xpg, d0, d1);
            ua.x = fma(float(sg53_e4m4_decode_h(sc[g0])), d0, ua.x);
            ua.y = fma(float(sg53_e4m4_decode_h(sc[g1])), d1, ua.y);
        }
    }

    const float2 gv(simd_sum(ga.x), simd_sum(ga.y));
    const float2 uv(simd_sum(ua.x), simd_sum(ua.y));
    if (lane == 0u) {
        device half* dst = mid_out + pair_idx * p.d_moe + row0;
        const half h0 = sg53_clamp_f16(sg53_limited_swiglu(gv.x, uv.x, p.swiglu_limit));
        if (row0 + 2u <= p.d_moe) {
            const half h1 = sg53_clamp_f16(sg53_limited_swiglu(gv.y, uv.y, p.swiglu_limit));
            *reinterpret_cast<device half2*>(dst) = half2(h0, h1);
        } else {
            dst[0] = h0;
        }
    }
}

// Decode `M=1` `4` rows/SG (`16` rows/TG = `2048` TGs across `top_k=8`!) `E8S-18-V4` `down_proj`!
// Grid: `((D + 15) / 16, top_k, M_valid)`, `128` threads/TG (`4` SGs, `4` rows/SG)
kernel void sg53_moe_e8s18_gather_dn(
    device const uchar*               part_base [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut       [[buffer(1)]],
    device const uint*                topk_idx  [[buffer(2)]], // [M_valid * top_k]
    constant half*                    mid_rot   [[buffer(3)]], // [M_valid * top_k, d_moe] (RHT-64 rotated!)
    device float*                     down_part [[buffer(4)]], // [M_valid * top_k, D]
    constant Sg53MoeE8Params&         p         [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid.x * 4u + sgid) * 4u;
    const uint k    = tgid.y;
    const uint m    = tgid.z;
    if (row0 >= p.D || k >= p.top_k || m >= p.M_valid) return;

    const ulong pair_idx = ulong(m) * p.top_k + k;
    const uint  eid      = topk_idx[pair_idx];
    const uint  grp      = p.d_moe >> 6u;
    const ulong sc_off   = p.e8_lo16_bytes + p.e8_hi2_bytes;

    device const uchar*  d_base = part_base + p.layer_byte_off + ulong(eid) * p.e8_record_bytes + 2ul * p.e8_proj_bytes;
    device const uint4*  lo     = reinterpret_cast<device const uint4*> (d_base);
    device const ushort* hi     = reinterpret_cast<device const ushort*>(d_base + p.e8_lo16_bytes);
    device const uchar*  sc     = d_base + sc_off;

    const uint r0 = min(row0 + 0u, p.D - 1u);
    const uint r1 = min(row0 + 1u, p.D - 1u);
    const uint r2 = min(row0 + 2u, p.D - 1u);
    const uint r3 = min(row0 + 3u, p.D - 1u);

    float4 acc(0.0f);
    constant half* mp = mid_rot + pair_idx * p.d_moe;

    for (uint g = lane; g < grp; g += 32u) {
        constant half4* xpg = reinterpret_cast<constant half4*>(mp + g * 64u);
        const ulong g0 = ulong(r0) * grp + g;
        const ulong g1 = ulong(r1) * grp + g;
        const ulong g2 = ulong(r2) * grp + g;
        const ulong g3 = ulong(r3) * grp + g;
        float d0, d1, d2, d3;
        sg53_e8s18_dot64_2r_c(lut, lo[g0], hi[g0], lo[g1], hi[g1], xpg, d0, d1);
        sg53_e8s18_dot64_2r_c(lut, lo[g2], hi[g2], lo[g3], hi[g3], xpg, d2, d3);
        acc.x = fma(float(sg53_e4m4_decode_h(sc[g0])), d0, acc.x);
        acc.y = fma(float(sg53_e4m4_decode_h(sc[g1])), d1, acc.y);
        acc.z = fma(float(sg53_e4m4_decode_h(sc[g2])), d2, acc.z);
        acc.w = fma(float(sg53_e4m4_decode_h(sc[g3])), d3, acc.w);
    }

    const float4 res(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
    if (lane == 0u) {
        device float* dst = down_part + pair_idx * p.D + row0;
        if (row0 + 4u <= p.D) {
            *reinterpret_cast<device float4*>(dst) = res;
        } else {
            for (uint ri = 0u; ri < 4u && (row0 + ri) < p.D; ++ri) dst[ri] = res[ri];
        }
    }
}

// High-occupancy `256`-thread/TG (`8` SGs, `2` rows/SG = `16` rows/TG) `E8S-18-V4` `gather_dn` (`414+ GB/s` random LUT!)
// Grid: `((D + 15) / 16, top_k, M_valid)`, `256` threads/TG
kernel void sg53_moe_e8s18_gather_dn_256(
    device const uchar*               part_base [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut       [[buffer(1)]],
    device const uint*                topk_idx  [[buffer(2)]],
    constant half*                    mid_rot   [[buffer(3)]],
    device float*                     down_part [[buffer(4)]],
    constant Sg53MoeE8Params&         p         [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],             // 0..7
    uint  lane [[thread_index_in_simdgroup]]                   // 0..31
) {
    const uint row0 = (tgid.x * 8u + sgid) * 2u;
    const uint k    = tgid.y;
    const uint m    = tgid.z;
    if (row0 >= p.D || k >= p.top_k || m >= p.M_valid) return;

    const ulong pair_idx = ulong(m) * p.top_k + k;
    const uint  eid      = topk_idx[pair_idx];
    const uint  grp      = p.d_moe >> 6u;
    const ulong sc_off   = p.e8_lo16_bytes + p.e8_hi2_bytes;

    device const uchar*  d_base = part_base + p.layer_byte_off + ulong(eid) * p.e8_record_bytes + 2ul * p.e8_proj_bytes;
    device const uint4*  lo     = reinterpret_cast<device const uint4*> (d_base);
    device const ushort* hi     = reinterpret_cast<device const ushort*>(d_base + p.e8_lo16_bytes);
    device const uchar*  sc     = d_base + sc_off;

    const uint r0 = min(row0 + 0u, p.D - 1u);
    const uint r1 = min(row0 + 1u, p.D - 1u);

    float2 acc(0.0f);
    constant half* mp = mid_rot + pair_idx * p.d_moe;

    for (uint g = lane; g < grp; g += 32u) {
        constant half4* xpg = reinterpret_cast<constant half4*>(mp + g * 64u);
        const ulong g0 = ulong(r0) * grp + g;
        const ulong g1 = ulong(r1) * grp + g;
        float d0, d1;
        sg53_e8s18_dot64_2r_c(lut, lo[g0], hi[g0], lo[g1], hi[g1], xpg, d0, d1);
        acc.x = fma(float(sg53_e4m4_decode_h(sc[g0])), d0, acc.x);
        acc.y = fma(float(sg53_e4m4_decode_h(sc[g1])), d1, acc.y);
    }

    const float2 res(simd_sum(acc.x), simd_sum(acc.y));
    if (lane == 0u) {
        device float* dst = down_part + pair_idx * p.D + row0;
        if (row0 + 2u <= p.D) {
            *reinterpret_cast<device float2*>(dst) = res;
        } else {
            dst[0] = res.x;
        }
    }
}

// 2-weight fused `E8S-18-V4` grouped dot product (`gate+up` or `2` rows of `dn` sharing EVERY `x` L1D load!)
// Uses single `half4` inner accum in 100.0000% bit-exact FMA order to `sg53_e8s18_dot64_2r_c` -> `0` bit diffs `M=1 == M=5`!
template <uint NB, uint B_OFF, uint B_TOT>
inline void sg53_e8s18_dot64_2w_chunk(
    device const StrataGlm53E8Half8Entry* lut,
    uint4                                 qA,
    ushort                                hA,
    float                                 sA,
    uint4                                 qB,
    ushort                                hB,
    float                                 sB,
    device const half*                    x_base,
    thread const uint                     (&row_id)[B_TOT],
    uint                                  row_stride,
    uint                                  g_off,
    thread float                          (&accA)[B_TOT],
    thread float                          (&accB)[B_TOT]
) {
    half4 a[NB], b_acc[NB];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < NB; ++b) { a[b] = half4(0.0h); b_acc[b] = half4(0.0h); }

    #define SG53_E8_2W_STEP(LOA, LOB, SH, S8) { \
        half4 wAl, wAh, wBl, wBh; \
        const StrataGlm53E8Half8Entry eA = lut[uint(LOA) >> 4u]; \
        const StrataGlm53E8Half8Entry eB = lut[uint(LOB) >> 4u]; \
        sg53_e8s18_decode_entry(eA, uint(LOA) & 15u, (uint(hA) >> (SH)) & 3u, wAl, wAh); \
        sg53_e8s18_decode_entry(eB, uint(LOB) & 15u, (uint(hB) >> (SH)) & 3u, wBl, wBh); \
        for (uint b = 0u; b < NB; ++b) { \
            const uint4 xv = *reinterpret_cast<device const uint4*>(x_base + ulong(row_id[B_OFF + b]) * ulong(row_stride) + ulong(g_off) + (S8) * 8u); \
            const half4 x0 = as_type<half4>(xv.xy), x1 = as_type<half4>(xv.zw); \
            a[b]     = fma(wAl, x0, fma(wAh, x1, a[b])); \
            b_acc[b] = fma(wBl, x0, fma(wBh, x1, b_acc[b])); \
        } \
    }
    SG53_E8_2W_STEP(ushort(qA.x),        ushort(qB.x),         0u, 0u)
    SG53_E8_2W_STEP(ushort(qA.x >> 16u), ushort(qB.x >> 16u),  2u, 1u)
    SG53_E8_2W_STEP(ushort(qA.y),        ushort(qB.y),         4u, 2u)
    SG53_E8_2W_STEP(ushort(qA.y >> 16u), ushort(qB.y >> 16u),  6u, 3u)
    SG53_E8_2W_STEP(ushort(qA.z),        ushort(qB.z),         8u, 4u)
    SG53_E8_2W_STEP(ushort(qA.z >> 16u), ushort(qB.z >> 16u), 10u, 5u)
    SG53_E8_2W_STEP(ushort(qA.w),        ushort(qB.w),        12u, 6u)
    SG53_E8_2W_STEP(ushort(qA.w >> 16u), ushort(qB.w >> 16u), 14u, 7u)
    #undef SG53_E8_2W_STEP

    #pragma clang loop unroll(full)
    for (uint b = 0u; b < NB; ++b) {
        const float4 fa = float4(a[b]);
        const float4 fb = float4(b_acc[b]);
        accA[B_OFF + b] = fma(sA, (fa.x + fa.y) + (fa.z + fa.w), accA[B_OFF + b]);
        accB[B_OFF + b] = fma(sB, (fb.x + fb.y) + (fb.z + fb.w), accB[B_OFF + b]);
    }
}

template <uint B_FIX, bool SPLIT = (B_FIX > 5u)>
struct Sg53E8Dot2WAll {
    static inline void run(
        device const StrataGlm53E8Half8Entry* lut,
        uint4 qA, ushort hA, float sA, uint4 qB, ushort hB, float sB,
        device const half* x_base, thread const uint (&row_id)[B_FIX],
        uint row_stride, uint g_off, thread float (&accA)[B_FIX], thread float (&accB)[B_FIX]
    ) {
        sg53_e8s18_dot64_2w_chunk<B_FIX, 0u, B_FIX>(lut, qA, hA, sA, qB, hB, sB, x_base, row_id, row_stride, g_off, accA, accB);
    }
};

template <uint B_FIX>
struct Sg53E8Dot2WAll<B_FIX, true> {
    static inline void run(
        device const StrataGlm53E8Half8Entry* lut,
        uint4 qA, ushort hA, float sA, uint4 qB, ushort hB, float sB,
        device const half* x_base, thread const uint (&row_id)[B_FIX],
        uint row_stride, uint g_off, thread float (&accA)[B_FIX], thread float (&accB)[B_FIX]
    ) {
        sg53_e8s18_dot64_2w_chunk<4u, 0u, B_FIX>(lut, qA, hA, sA, qB, hB, sB, x_base, row_id, row_stride, g_off, accA, accB);
        sg53_e8s18_dot64_2w_chunk<B_FIX - 4u, 4u, B_FIX>(lut, qA, hA, sA, qB, hB, sB, x_base, row_id, row_stride, g_off, accA, accB);
    }
};

inline void sg53_e8s18_dot64_2r_d(
    device const StrataGlm53E8Half8Entry* lut,
    uint4                                 q0,
    ushort                                h0,
    uint4                                 q1,
    ushort                                h1,
    device const half4*                   xpg,
    thread float&                         d0,
    thread float&                         d1
) {
    half4 a(0.0h), b(0.0h);
    half4 wl, wh;
    #define SG53_E8_2RD(LO0, LO1, SH, X0, X1) { \
        const half4 xv0 = xpg[X0], xv1 = xpg[X1]; \
        const StrataGlm53E8Half8Entry e0 = lut[uint(LO0) >> 4u]; \
        const StrataGlm53E8Half8Entry e1 = lut[uint(LO1) >> 4u]; \
        sg53_e8s18_decode_entry(e0, uint(LO0) & 15u, (uint(h0) >> (SH)) & 3u, wl, wh); a = fma(wl, xv0, fma(wh, xv1, a)); \
        sg53_e8s18_decode_entry(e1, uint(LO1) & 15u, (uint(h1) >> (SH)) & 3u, wl, wh); b = fma(wl, xv0, fma(wh, xv1, b)); \
    }
    SG53_E8_2RD(ushort(q0.x),        ushort(q1.x),         0u,  0,  1)
    SG53_E8_2RD(ushort(q0.x >> 16u), ushort(q1.x >> 16u),  2u,  2,  3)
    SG53_E8_2RD(ushort(q0.y),        ushort(q1.y),         4u,  4,  5)
    SG53_E8_2RD(ushort(q0.y >> 16u), ushort(q1.y >> 16u),  6u,  6,  7)
    SG53_E8_2RD(ushort(q0.z),        ushort(q1.z),         8u,  8,  9)
    SG53_E8_2RD(ushort(q0.z >> 16u), ushort(q1.z >> 16u), 10u, 10, 11)
    SG53_E8_2RD(ushort(q0.w),        ushort(q1.w),        12u, 12, 13)
    SG53_E8_2RD(ushort(q0.w >> 16u), ushort(q1.w >> 16u), 14u, 14, 15)
    #undef SG53_E8_2RD
    const float4 fa = float4(a);
    const float4 fb = float4(b);
    d0 = (fa.x + fa.y) + (fa.z + fa.w);
    d1 = (fb.x + fb.y) + (fb.z + fb.w);
}

// Grouped (`B = 1..8`) `E8S-18-V4` `gate_proj + up_proj + LimitedSwiGLU` (`2` rows/SG = `8` rows/TG) — 100.0000% bit-exact to `gather_gu`!
// Indirect Grid: `((d_moe + 7) / 8, num_tiles, 1)` (offset 32), `128` threads/TG (`4` SGs, `2` rows/SG)
inline void sg53_e8s18_grouped_gu_b1(
    device const uchar*                   part_base,
    device const StrataGlm53E8Half8Entry* lut,
    device const half*                    x_rot,
    device const uint*                    sorted_pairs,
    device half*                          mid_out,
    Sg53MoeV2WorkTile                     tile,
    constant Sg53MoeE8Params&             p,
    uint row0,
    uint lane
) {
    const uint  pid      = sorted_pairs[tile.pair_start];
    const uint  tok_m    = pid / p.top_k;
    const uint  grp      = p.D >> 6u;
    const ulong sc_off   = p.e8_lo16_bytes + p.e8_hi2_bytes;
    const uint  r0       = min(row0 + 0u, p.d_moe - 1u);
    const uint  r1       = min(row0 + 1u, p.d_moe - 1u);
    device const uchar* rec_base = part_base + p.layer_byte_off + ulong(tile.expert_id) * p.e8_record_bytes;
    device const half*  xm       = x_rot + ulong(tok_m) * p.D;

    float2 ga(0.0f);
    {
        device const uint4*  lo = reinterpret_cast<device const uint4*> (rec_base);
        device const ushort* hi = reinterpret_cast<device const ushort*>(rec_base + p.e8_lo16_bytes);
        device const uchar*  sc = rec_base + sc_off;
        for (uint g = lane; g < grp; g += 32u) {
            device const half4* xpg = reinterpret_cast<device const half4*>(xm + g * 64u);
            const ulong g0 = ulong(r0) * grp + g;
            const ulong g1 = ulong(r1) * grp + g;
            float d0, d1;
            sg53_e8s18_dot64_2r_d(lut, lo[g0], hi[g0], lo[g1], hi[g1], xpg, d0, d1);
            ga.x = fma(float(sg53_e4m4_decode_h(sc[g0])), d0, ga.x);
            ga.y = fma(float(sg53_e4m4_decode_h(sc[g1])), d1, ga.y);
        }
    }

    float2 ua(0.0f);
    {
        device const uchar*  ub = rec_base + p.e8_proj_bytes;
        device const uint4*  lo = reinterpret_cast<device const uint4*> (ub);
        device const ushort* hi = reinterpret_cast<device const ushort*>(ub + p.e8_lo16_bytes);
        device const uchar*  sc = ub + sc_off;
        for (uint g = lane; g < grp; g += 32u) {
            device const half4* xpg = reinterpret_cast<device const half4*>(xm + g * 64u);
            const ulong g0 = ulong(r0) * grp + g;
            const ulong g1 = ulong(r1) * grp + g;
            float d0, d1;
            sg53_e8s18_dot64_2r_d(lut, lo[g0], hi[g0], lo[g1], hi[g1], xpg, d0, d1);
            ua.x = fma(float(sg53_e4m4_decode_h(sc[g0])), d0, ua.x);
            ua.y = fma(float(sg53_e4m4_decode_h(sc[g1])), d1, ua.y);
        }
    }

    const float2 gv(simd_sum(ga.x), simd_sum(ga.y));
    const float2 uv(simd_sum(ua.x), simd_sum(ua.y));
    if (lane == 0u) {
        device half* dst = mid_out + ulong(pid) * p.d_moe + row0;
        const half h0 = sg53_clamp_f16(sg53_limited_swiglu(gv.x, uv.x, p.swiglu_limit));
        if (row0 + 2u <= p.d_moe) {
            const half h1 = sg53_clamp_f16(sg53_limited_swiglu(gv.y, uv.y, p.swiglu_limit));
            *reinterpret_cast<device half2*>(dst) = half2(h0, h1);
        } else {
            dst[0] = h0;
        }
    }
}

template <uint B_FIX>
inline void sg53_e8s18_grouped_gu_fixed(
    device const uchar*                   part_base,
    device const StrataGlm53E8Half8Entry* lut,
    device const half*                    x_rot,
    device const uint*                    sorted_pairs,
    device half*                          mid_out,
    Sg53MoeV2WorkTile                     tile,
    constant Sg53MoeE8Params&             p,
    uint row0,
    uint lane
) {
    uint tok_m[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        tok_m[b] = sorted_pairs[tile.pair_start + b] / p.top_k;
    }

    device const uchar* rec_base = part_base + p.layer_byte_off + ulong(tile.expert_id) * p.e8_record_bytes;
    device const uchar* ub       = rec_base + p.e8_proj_bytes;
    const uint  grp              = p.D >> 6u;
    const ulong sc_off           = p.e8_lo16_bytes + p.e8_hi2_bytes;

    device const uint4*  g_lo = reinterpret_cast<device const uint4*> (rec_base);
    device const ushort* g_hi = reinterpret_cast<device const ushort*>(rec_base + p.e8_lo16_bytes);
    device const uchar*  g_sc = rec_base + sc_off;
    device const uint4*  u_lo = reinterpret_cast<device const uint4*> (ub);
    device const ushort* u_hi = reinterpret_cast<device const ushort*>(ub + p.e8_lo16_bytes);
    device const uchar*  u_sc = ub + sc_off;

    for (uint ri = 0u; ri < 2u; ++ri) {
        const uint row = row0 + ri;
        if (row >= p.d_moe) break;

        float ga[B_FIX], ua[B_FIX];
        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) { ga[b] = 0.0f; ua[b] = 0.0f; }

        for (uint g = lane; g < grp; g += 32u) {
            const ulong gi = ulong(row) * grp + g;
            Sg53E8Dot2WAll<B_FIX>::run(
                lut,
                g_lo[gi], g_hi[gi], float(sg53_e4m4_decode_h(g_sc[gi])),
                u_lo[gi], u_hi[gi], float(sg53_e4m4_decode_h(u_sc[gi])),
                x_rot, tok_m, p.D, g * 64u, ga, ua);
        }

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) {
            const float gv = simd_sum(ga[b]);
            const float uv = simd_sum(ua[b]);
            if (lane == 0u) {
                const uint pid = sorted_pairs[tile.pair_start + b];
                mid_out[ulong(pid) * p.d_moe + row] = sg53_clamp_f16(sg53_limited_swiglu(gv, uv, p.swiglu_limit));
            }
        }
    }
}

kernel void sg53_moe_e8s18_grouped_gu(
    device const uchar*                   part_base    [[buffer(0)]],
    device const StrataGlm53E8Half8Entry* lut          [[buffer(1)]],
    device const half*                    x_rot        [[buffer(2)]], // [M_valid, D]
    device const uint*                    sorted_pairs [[buffer(3)]],
    device const Sg53MoeV2WorkTile*       work_tiles   [[buffer(4)]],
    device half*                          mid_out      [[buffer(5)]], // [M_valid * top_k, d_moe]
    constant Sg53MoeE8Params&             p            [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],                    // 0..3
    uint  lane [[thread_index_in_simdgroup]]                          // 0..31
) {
    const uint row0 = (tgid.x * 4u + sgid) * 2u;
    if (row0 >= p.d_moe) return;
    const Sg53MoeV2WorkTile tile = work_tiles[p.tile_base + tgid.y];
    #define SG53_E8GU_CASE(BF) { sg53_e8s18_grouped_gu_fixed<BF>(part_base, lut, x_rot, sorted_pairs, mid_out, tile, p, row0, lane); break; }
    switch (tile.pair_count) {
        case 1u: sg53_e8s18_grouped_gu_b1(part_base, lut, x_rot, sorted_pairs, mid_out, tile, p, row0, lane); break;
        case 2u: SG53_E8GU_CASE(2u)
        case 3u: SG53_E8GU_CASE(3u)
        case 4u: SG53_E8GU_CASE(4u)
        case 5u: SG53_E8GU_CASE(5u)
        case 6u: SG53_E8GU_CASE(6u)
        case 7u: SG53_E8GU_CASE(7u)
        case 8u: SG53_E8GU_CASE(8u)
        default: break;
    }
    #undef SG53_E8GU_CASE
}

// Grouped (`B = 1..8`) `E8S-18-V4` `down_proj` (`4` rows/SG = `16` rows/TG) — 100.0000% bit-exact to `gather_dn`!
// Indirect Grid: `((D + 15) / 16, num_tiles, 1)` (offset 44), `128` threads/TG (`4` SGs, `4` rows/SG)
inline void sg53_e8s18_grouped_dn_b1(
    device const uchar*                   part_base,
    device const StrataGlm53E8Half8Entry* lut,
    device const half*                    mid_rot,
    device const uint*                    sorted_pairs,
    device float*                         down_part,
    Sg53MoeV2WorkTile                     tile,
    constant Sg53MoeE8Params&             p,
    uint row0,
    uint lane
) {
    const uint  pid    = sorted_pairs[tile.pair_start];
    const uint  grp    = p.d_moe >> 6u;
    const ulong sc_off = p.e8_lo16_bytes + p.e8_hi2_bytes;

    device const uchar*  d_base = part_base + p.layer_byte_off + ulong(tile.expert_id) * p.e8_record_bytes + 2ul * p.e8_proj_bytes;
    device const uint4*  lo     = reinterpret_cast<device const uint4*> (d_base);
    device const ushort* hi     = reinterpret_cast<device const ushort*>(d_base + p.e8_lo16_bytes);
    device const uchar*  sc     = d_base + sc_off;

    const uint r0 = min(row0 + 0u, p.D - 1u);
    const uint r1 = min(row0 + 1u, p.D - 1u);
    const uint r2 = min(row0 + 2u, p.D - 1u);
    const uint r3 = min(row0 + 3u, p.D - 1u);

    float4 acc(0.0f);
    device const half* mp = mid_rot + ulong(pid) * p.d_moe;

    for (uint g = lane; g < grp; g += 32u) {
        device const half4* xpg = reinterpret_cast<device const half4*>(mp + g * 64u);
        const ulong g0 = ulong(r0) * grp + g;
        const ulong g1 = ulong(r1) * grp + g;
        const ulong g2 = ulong(r2) * grp + g;
        const ulong g3 = ulong(r3) * grp + g;
        float d0, d1, d2, d3;
        sg53_e8s18_dot64_2r_d(lut, lo[g0], hi[g0], lo[g1], hi[g1], xpg, d0, d1);
        sg53_e8s18_dot64_2r_d(lut, lo[g2], hi[g2], lo[g3], hi[g3], xpg, d2, d3);
        acc.x = fma(float(sg53_e4m4_decode_h(sc[g0])), d0, acc.x);
        acc.y = fma(float(sg53_e4m4_decode_h(sc[g1])), d1, acc.y);
        acc.z = fma(float(sg53_e4m4_decode_h(sc[g2])), d2, acc.z);
        acc.w = fma(float(sg53_e4m4_decode_h(sc[g3])), d3, acc.w);
    }

    const float4 res(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
    if (lane == 0u) {
        device float* dst = down_part + ulong(pid) * p.D + row0;
        if (row0 + 4u <= p.D) {
            *reinterpret_cast<device float4*>(dst) = res;
        } else {
            for (uint ri = 0u; ri < 4u && (row0 + ri) < p.D; ++ri) dst[ri] = res[ri];
        }
    }
}

template <uint B_FIX>
inline void sg53_e8s18_grouped_dn_fixed(
    device const uchar*                   part_base,
    device const StrataGlm53E8Half8Entry* lut,
    device const half*                    mid_rot,
    device const uint*                    sorted_pairs,
    device float*                         down_part,
    Sg53MoeV2WorkTile                     tile,
    constant Sg53MoeE8Params&             p,
    uint row0,
    uint lane
) {
    uint pair_id[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        pair_id[b] = sorted_pairs[tile.pair_start + b];
    }

    device const uchar*  d_base = part_base + p.layer_byte_off + ulong(tile.expert_id) * p.e8_record_bytes + 2ul * p.e8_proj_bytes;
    const uint  grp             = p.d_moe >> 6u;
    const ulong sc_off          = p.e8_lo16_bytes + p.e8_hi2_bytes;

    device const uint4*  lo = reinterpret_cast<device const uint4*> (d_base);
    device const ushort* hi = reinterpret_cast<device const ushort*>(d_base + p.e8_lo16_bytes);
    device const uchar*  sc = d_base + sc_off;

    for (uint rp = 0u; rp < 2u; ++rp) {
        const uint r0 = row0 + rp * 2u;
        if (r0 >= p.D) break;
        const uint r1 = min(r0 + 1u, p.D - 1u);

        float a0[B_FIX], a1[B_FIX];
        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) { a0[b] = 0.0f; a1[b] = 0.0f; }

        for (uint g = lane; g < grp; g += 32u) {
            const ulong i0 = ulong(r0) * grp + g;
            const ulong i1 = ulong(r1) * grp + g;
            Sg53E8Dot2WAll<B_FIX>::run(
                lut,
                lo[i0], hi[i0], float(sg53_e4m4_decode_h(sc[i0])),
                lo[i1], hi[i1], float(sg53_e4m4_decode_h(sc[i1])),
                mid_rot, pair_id, p.d_moe, g * 64u, a0, a1);
        }

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) {
            const float v0 = simd_sum(a0[b]);
            const float v1 = simd_sum(a1[b]);
            if (lane == 0u) {
                down_part[ulong(pair_id[b]) * p.D + r0] = v0;
                if (r0 + 1u < p.D) down_part[ulong(pair_id[b]) * p.D + r1] = v1;
            }
        }
    }
}

kernel void sg53_moe_e8s18_grouped_dn(
    device const uchar*                   part_base    [[buffer(0)]],
    device const StrataGlm53E8Half8Entry* lut          [[buffer(1)]],
    device const half*                    mid_rot      [[buffer(2)]], // [M_valid * top_k, d_moe]
    device const uint*                    sorted_pairs [[buffer(3)]],
    device const Sg53MoeV2WorkTile*       work_tiles   [[buffer(4)]],
    device float*                         down_part    [[buffer(5)]], // [M_valid * top_k, D]
    constant Sg53MoeE8Params&             p            [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid.x * 4u + sgid) * 4u;
    if (row0 >= p.D) return;
    const Sg53MoeV2WorkTile tile = work_tiles[p.tile_base + tgid.y];
    #define SG53_E8DN_CASE(BF) { sg53_e8s18_grouped_dn_fixed<BF>(part_base, lut, mid_rot, sorted_pairs, down_part, tile, p, row0, lane); break; }
    switch (tile.pair_count) {
        case 1u: sg53_e8s18_grouped_dn_b1(part_base, lut, mid_rot, sorted_pairs, down_part, tile, p, row0, lane); break;
        case 2u: SG53_E8DN_CASE(2u)
        case 3u: SG53_E8DN_CASE(3u)
        case 4u: SG53_E8DN_CASE(4u)
        case 5u: SG53_E8DN_CASE(5u)
        case 6u: SG53_E8DN_CASE(6u)
        case 7u: SG53_E8DN_CASE(7u)
        case 8u: SG53_E8DN_CASE(8u)
        default: break;
    }
    #undef SG53_E8DN_CASE
}

// ---- `E8S-18-V4` Prefill Metal 4.1 NAX MoE (`BM = 16 / 32 / 64`, `BN = 64`, `BK = 128`, `constant` LUT!) ----
// `gu` fuses `LimitedSwiGLU + RHT-64` epilogue directly in dead `W_tile` (`16 KiB` union -> ZERO extra TG mem or kernel!)

#if defined(__HAVE_TENSOR__)
// High-throughput `BK=128` E8S-18-V4 tile unpacker (`ushort2` = 16 weights/thread + `constant` 64 KiB LUT!)
template <int SG>
inline void sg53_e8s18_unpack_tile_bk128(
    device const ushort2*             proj_lo2,
    device const ushort*              proj_hi2,
    device const uchar*               proj_sc,
    constant StrataGlm53E8Half8Entry* lut,
    uint n0,
    uint kt,
    uint K,
    threadgroup half (&W_tile)[64][128],
    uint tid
) {
    const uint u2_per_row  = K >> 4u;
    const uint grp_per_row = K >> 6u;
    const uint r_grp       = tid >> 3u;     // 0..15 (SG=4) or 0..7 (SG=2)
    const uint c_u2        = tid & 7u;      // 0..7 (8 x 16 = 128 cols!)
    const uint g_loc       = c_u2 >> 2u;    // 0 or 1
    const uint sh0         = (c_u2 & 3u) * 4u;
    constexpr uint r_step  = uint(SG) * 4u; // 16 (SG=4) or 8 (SG=2)
    constexpr uint q_iters = 64u / r_step;  // 4 (SG=4) or 8 (SG=2)

    const uint col_u2  = kt * 8u + c_u2;
    const uint col_grp = kt * 2u + g_loc;

    #pragma clang loop unroll(full)
    for (uint q = 0u; q < q_iters; ++q) {
        const uint    r_loc  = q * r_step + r_grp;
        const ulong   r_glob = ulong(n0 + r_loc);
        const ushort2 lv     = proj_lo2[r_glob * u2_per_row + col_u2];
        const ushort  hi2_w  = proj_hi2[r_glob * grp_per_row + col_grp];
        const half    sc_h   = sg53_e4m4_decode_h(proj_sc[r_glob * grp_per_row + col_grp]);

        half4 wlo, whi;
        threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][c_u2 * 16u]);
        sg53_e8s18_decode_h4_c(lut, lv.x, (uint(hi2_w) >>  sh0      ) & 3u, wlo, whi);
        dst[0] = wlo * sc_h;
        dst[1] = whi * sc_h;
        sg53_e8s18_decode_h4_c(lut, lv.y, (uint(hi2_w) >> (sh0 + 2u)) & 3u, wlo, whi);
        dst[2] = wlo * sc_h;
        dst[3] = whi * sc_h;
    }
}

template <int BM>
union Sg53E8GuSmem {
    half  w[64][128];   // 16 KiB
    float mid[BM][64];  // <= 16 KiB (exact 16 KiB at BM=64!)
};

// Single-`cT` sequential Pass 1 (`gate` -> `mid_sorted` FP16 staging) + Pass 2 (`up` -> `LimitedSwiGLU + RHT-64` -> `mid_sorted`)!
template <int BM, int SG>
inline void sg53_moe_e8s18_nax_gu_impl(
    device const uchar*               part_base,
    constant StrataGlm53E8Half8Entry* lut,
    device const half*                x_sorted,
    device const Sg53MoeV2WorkTile*   work_tiles,
    device half*                      mid_sorted,
    constant Sg53MoeE8Params&         p,
    threadgroup Sg53E8GuSmem<BM>&     smem,
    uint2 tgid,
    uint  tid,
    uint  sgid,
    uint  lane
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.d_moe) return;

    const Sg53MoeV2WorkTile tile = work_tiles[p.tile_base + tgid.y];
    if (tile.pair_count == 0u) return;

    device const uchar* rec_base = part_base + p.layer_byte_off + ulong(tile.expert_id) * p.e8_record_bytes;
    const ulong sc_off           = p.e8_lo16_bytes + p.e8_hi2_bytes;
    const uint  kt_cnt           = p.D >> 7u;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(
        const_cast<device half*>(x_sorted), dextents<int, 2>(int(p.D), int(p.total_pairs_pad)));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&smem.w[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<128, BM>(0, int(tile.pair_start));
    auto cT  = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sX0)>, sg53_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    {
        device const ushort2* g_lo = reinterpret_cast<device const ushort2*>(rec_base);
        device const ushort*  g_hi = reinterpret_cast<device const ushort*> (rec_base + p.e8_lo16_bytes);
        device const uchar*   g_sc = rec_base + sc_off;
        for (uint kt = 0u; kt < kt_cnt; ++kt) {
            sg53_e8s18_unpack_tile_bk128<SG>(g_lo, g_hi, g_sc, lut, n0, kt, p.D, smem.w, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto sX = tX.template slice<128, BM>(int(kt * 128u), int(tile.pair_start));
            op.run(sX, tW, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                auto idx = cT.get_multidimensional_index(i);
                const uint r = uint(idx[1]);
                const uint c = uint(idx[0]);
                if (r < tile.pair_count) {
                    mid_sorted[ulong(tile.pair_start + r) * p.d_moe + n0 + c] =
                        sg53_clamp_f16(clamp(cT[i], -24.0f, p.swiglu_limit));
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    {
        device const uchar*   u_base = rec_base + p.e8_proj_bytes;
        device const ushort2* u_lo   = reinterpret_cast<device const ushort2*>(u_base);
        device const ushort*  u_hi   = reinterpret_cast<device const ushort*> (u_base + p.e8_lo16_bytes);
        device const uchar*   u_sc   = u_base + sc_off;
        for (uint kt = 0u; kt < kt_cnt; ++kt) {
            sg53_e8s18_unpack_tile_bk128<SG>(u_lo, u_hi, u_sc, lut, n0, kt, p.D, smem.w, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto sX = tX.template slice<128, BM>(int(kt * 128u), int(tile.pair_start));
            op.run(sX, tW, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            const uint r = uint(idx[1]);
            const uint c = uint(idx[0]);
            if (r < tile.pair_count) {
                const float g_val = float(mid_sorted[ulong(tile.pair_start + r) * p.d_moe + n0 + c]);
                smem.mid[r][c] = sg53_limited_swiglu(g_val, cT[i], p.swiglu_limit);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = sgid; r < tile.pair_count; r += uint(SG)) {
        float v0 = smem.mid[r][2u * lane];
        float v1 = smem.mid[r][2u * lane + 1u];
        sg53_rht64_warp_f32(v0, v1, lane);
        device half2* dst2 = reinterpret_cast<device half2*>(
            mid_sorted + ulong(tile.pair_start + r) * p.d_moe + n0);
        dst2[lane] = half2(sg53_clamp_f16(v0), sg53_clamp_f16(v1));
    }
}

template <int BM, int SG>
inline void sg53_moe_e8s18_nax_dn_impl(
    device const uchar*               part_base,
    constant StrataGlm53E8Half8Entry* lut,
    device const half*                mid_sorted,
    device const uint*                sorted_pairs,
    device const Sg53MoeV2WorkTile*   work_tiles,
    device float*                     down_part,
    constant Sg53MoeE8Params&         p,
    threadgroup half                  (&W_tile)[64][128],
    uint2 tgid,
    uint  tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.D) return;

    const Sg53MoeV2WorkTile tile = work_tiles[p.tile_base + tgid.y];
    if (tile.pair_count == 0u) return;

    device const uchar* d_base = part_base + p.layer_byte_off + ulong(tile.expert_id) * p.e8_record_bytes + 2ul * p.e8_proj_bytes;
    const ulong sc_off         = p.e8_lo16_bytes + p.e8_hi2_bytes;

    device const ushort2* d_lo = reinterpret_cast<device const ushort2*>(d_base);
    device const ushort*  d_hi = reinterpret_cast<device const ushort*> (d_base + p.e8_lo16_bytes);
    device const uchar*   d_sc = d_base + sc_off;

    tensor<device half, dextents<int, 2>, tensor_inline> tM(
        const_cast<device half*>(mid_sorted), dextents<int, 2>(int(p.d_moe), int(p.total_pairs_pad)));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sM0 = tM.template slice<128, BM>(0, int(tile.pair_start));
    auto cT  = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sM0)>, sg53_strip_as<decltype(tW)>, float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint kt_cnt = p.d_moe >> 7u;
    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        sg53_e8s18_unpack_tile_bk128<SG>(d_lo, d_hi, d_sc, lut, n0, kt, p.d_moe, W_tile, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sM = tM.template slice<128, BM>(int(kt * 128u), int(tile.pair_start));
        op.run(sM, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx     = cT.get_multidimensional_index(i);
            const uint r = uint(idx[1]);
            const uint c = uint(idx[0]);
            if (r < tile.pair_count && (n0 + c) < p.D) {
                const uint pid = sorted_pairs[tile.pair_start + r];
                down_part[ulong(pid) * p.D + n0 + c] = cT[i];
            }
        }
    }
}
#endif

kernel void sg53_moe_e8s18_nax_gu_m64(
    device const uchar*               part_base  [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut        [[buffer(1)]],
    device const half*                x_sorted   [[buffer(2)]],
    device const Sg53MoeV2WorkTile*   work_tiles [[buffer(3)]],
    device half*                      mid_sorted [[buffer(4)]],
    constant Sg53MoeE8Params&         p          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup Sg53E8GuSmem<64> smem;
    sg53_moe_e8s18_nax_gu_impl<64, 4>(part_base, lut, x_sorted, work_tiles, mid_sorted, p, smem, tgid, tid, sgid, lane);
#else
    (void)part_base; (void)lut; (void)x_sorted; (void)work_tiles; (void)mid_sorted; (void)p; (void)tgid; (void)tid; (void)sgid; (void)lane;
#endif
}

kernel void sg53_moe_e8s18_nax_gu_m32(
    device const uchar*               part_base  [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut        [[buffer(1)]],
    device const half*                x_sorted   [[buffer(2)]],
    device const Sg53MoeV2WorkTile*   work_tiles [[buffer(3)]],
    device half*                      mid_sorted [[buffer(4)]],
    constant Sg53MoeE8Params&         p          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup Sg53E8GuSmem<32> smem;
    sg53_moe_e8s18_nax_gu_impl<32, 4>(part_base, lut, x_sorted, work_tiles, mid_sorted, p, smem, tgid, tid, sgid, lane);
#else
    (void)part_base; (void)lut; (void)x_sorted; (void)work_tiles; (void)mid_sorted; (void)p; (void)tgid; (void)tid; (void)sgid; (void)lane;
#endif
}

kernel void sg53_moe_e8s18_nax_gu_m16(
    device const uchar*               part_base  [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut        [[buffer(1)]],
    device const half*                x_sorted   [[buffer(2)]],
    device const Sg53MoeV2WorkTile*   work_tiles [[buffer(3)]],
    device half*                      mid_sorted [[buffer(4)]],
    constant Sg53MoeE8Params&         p          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup Sg53E8GuSmem<16> smem;
    sg53_moe_e8s18_nax_gu_impl<16, 2>(part_base, lut, x_sorted, work_tiles, mid_sorted, p, smem, tgid, tid, sgid, lane);
#else
    (void)part_base; (void)lut; (void)x_sorted; (void)work_tiles; (void)mid_sorted; (void)p; (void)tgid; (void)tid; (void)sgid; (void)lane;
#endif
}

kernel void sg53_moe_e8s18_nax_dn_m64(
    device const uchar*               part_base    [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut          [[buffer(1)]],
    device const half*                mid_sorted   [[buffer(2)]],
    device const uint*                sorted_pairs [[buffer(3)]],
    device const Sg53MoeV2WorkTile*   work_tiles   [[buffer(4)]],
    device float*                     down_part    [[buffer(5)]],
    constant Sg53MoeE8Params&         p            [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup half W_tile[64][128];
    sg53_moe_e8s18_nax_dn_impl<64, 4>(part_base, lut, mid_sorted, sorted_pairs, work_tiles, down_part, p, W_tile, tgid, tid);
#else
    (void)part_base; (void)lut; (void)mid_sorted; (void)sorted_pairs; (void)work_tiles; (void)down_part; (void)p; (void)tgid; (void)tid;
#endif
}

kernel void sg53_moe_e8s18_nax_dn_m32(
    device const uchar*               part_base    [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut          [[buffer(1)]],
    device const half*                mid_sorted   [[buffer(2)]],
    device const uint*                sorted_pairs [[buffer(3)]],
    device const Sg53MoeV2WorkTile*   work_tiles   [[buffer(4)]],
    device float*                     down_part    [[buffer(5)]],
    constant Sg53MoeE8Params&         p            [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup half W_tile[64][128];
    sg53_moe_e8s18_nax_dn_impl<32, 4>(part_base, lut, mid_sorted, sorted_pairs, work_tiles, down_part, p, W_tile, tgid, tid);
#else
    (void)part_base; (void)lut; (void)mid_sorted; (void)sorted_pairs; (void)work_tiles; (void)down_part; (void)p; (void)tgid; (void)tid;
#endif
}

kernel void sg53_moe_e8s18_nax_dn_m16(
    device const uchar*               part_base    [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut          [[buffer(1)]],
    device const half*                mid_sorted   [[buffer(2)]],
    device const uint*                sorted_pairs [[buffer(3)]],
    device const Sg53MoeV2WorkTile*   work_tiles   [[buffer(4)]],
    device float*                     down_part    [[buffer(5)]],
    constant Sg53MoeE8Params&         p            [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup half W_tile[64][128];
    sg53_moe_e8s18_nax_dn_impl<16, 2>(part_base, lut, mid_sorted, sorted_pairs, work_tiles, down_part, p, W_tile, tgid, tid);
#else
    (void)part_base; (void)lut; (void)mid_sorted; (void)sorted_pairs; (void)work_tiles; (void)down_part; (void)p; (void)tgid; (void)tid;
#endif
}

// ---- Opt-In Prefill (`--int8-prefill`) `INT8×INT8→INT32` NAX `E8S-18-V4` MoE (`cos = 0.99993`, 0 bias, 8 KiB TG!) ----

// Gathers `x_in [M_valid, D]` into expert-sorted `x_sorted_i8 [total_pairs+64, D]` + `x_rsc [total_pairs+64]`!
// Grid: `(total_pairs + 64, 1, 1)`, `256` threads/TG
kernel void sg53_moe_gather_x_i8(
    device const half*         x_in         [[buffer(0)]], // [M_valid, D]
    device const uint*         sorted_pairs [[buffer(1)]], // [total_pairs]
    device int8_t*             x_sorted_i8  [[buffer(2)]], // [total_pairs + 64, D] (aliases b_moe_down_part!)
    device half*               x_rsc        [[buffer(3)]], // [total_pairs + 64]
    device half*               mid_sorted   [[buffer(4)]], // [total_pairs + 64, d_moe] (zero +64 tail)
    constant Sg53MoeBinParams& p            [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],             // 0..255
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint total_pairs = p.M_valid * p.top_k;
    if (row >= total_pairs + 64u) return;

    const uint d4_cnt = p.D >> 2u;
    device char4* dst4 = reinterpret_cast<device char4*>(x_sorted_i8 + ulong(row) * p.D);
    if (row >= total_pairs) {
        if (tid == 0u) x_rsc[row] = half(0.0h);
        for (uint c4 = tid; c4 < d4_cnt; c4 += 256u) dst4[c4] = char4(0);
        const uint m4_cnt = p.d_moe >> 2u;
        device half4* mdst4 = reinterpret_cast<device half4*>(mid_sorted + ulong(row) * p.d_moe);
        for (uint c4 = tid; c4 < m4_cnt; c4 += 256u) mdst4[c4] = half4(0.0h);
        return;
    }

    const uint m = sorted_pairs[row] / p.top_k;
    device const half4* src4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.D);
    float amax = 0.0f;
    for (uint c4 = tid; c4 < d4_cnt; c4 += 256u) {
        half4 v = abs(src4[c4]);
        amax = max(amax, max(max(float(v.x), float(v.y)), max(float(v.z), float(v.w))));
    }
    amax = simd_max(amax);
    threadgroup float tg_max[8];
    if (lane == 0u) tg_max[sgid] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float gm = simd_max((lane < 8u) ? tg_max[lane] : 0.0f);
        if (lane == 0u) tg_max[0] = gm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float ra      = tg_max[0];
    const float rscale  = (ra > 1e-12f) ? (ra / 127.0f) : 0.0f;
    const float inv_rsc = (ra > 1e-12f) ? (127.0f / ra) : 0.0f;
    if (tid == 0u) x_rsc[row] = half(rscale);

    for (uint c4 = tid; c4 < d4_cnt; c4 += 256u) {
        const float4 fv = float4(src4[c4]) * inv_rsc;
        dst4[c4] = char4(rint(clamp(fv, float4(-127.0f), float4(127.0f))));
    }
}

#if defined(__HAVE_TENSOR__)
template <int SG>
inline void sg53_e8s18_unpack_tile_i8_bk128(
    device const ushort2*             proj_lo2,
    device const ushort*              proj_hi2,
    device const uchar*               proj_sc,
    constant StrataGlm53E8Half8Entry* lut,
    thread const float                (&inv_w)[64 / (SG * 4)],
    uint n0,
    uint kt,
    uint K,
    threadgroup int8_t                (&W_i8)[64][128],
    uint tid
) {
    const uint u2_per_row  = K >> 4u;
    const uint grp_per_row = K >> 6u;
    const uint r_grp       = tid >> 3u;
    const uint c_u2        = tid & 7u;
    const uint g_loc       = c_u2 >> 2u;
    const uint sh0         = (c_u2 & 3u) * 4u;
    constexpr uint r_step  = uint(SG) * 4u;
    constexpr uint q_iters = 64u / r_step;

    const uint col_u2  = kt * 8u + c_u2;
    const uint col_grp = kt * 2u + g_loc;

    #pragma clang loop unroll(full)
    for (uint q = 0u; q < q_iters; ++q) {
        const uint    r_loc  = q * r_step + r_grp;
        const ulong   r_glob = ulong(n0 + r_loc);
        const ushort2 lv     = proj_lo2[r_glob * u2_per_row + col_u2];
        const ushort  hi2_w  = proj_hi2[r_glob * grp_per_row + col_grp];
        const half    mul    = half(float(sg53_e4m4_decode_h(proj_sc[r_glob * grp_per_row + col_grp])) * inv_w[q]);

        half4 wlo, whi;
        threadgroup char4* dst = reinterpret_cast<threadgroup char4*>(&W_i8[r_loc][c_u2 * 16u]);
        sg53_e8s18_decode_h4_c(lut, lv.x, (uint(hi2_w) >>  sh0      ) & 3u, wlo, whi);
        dst[0] = char4(rint(clamp(wlo * mul, half4(-127.0h), half4(127.0h))));
        dst[1] = char4(rint(clamp(whi * mul, half4(-127.0h), half4(127.0h))));
        sg53_e8s18_decode_h4_c(lut, lv.y, (uint(hi2_w) >> (sh0 + 2u)) & 3u, wlo, whi);
        dst[2] = char4(rint(clamp(wlo * mul, half4(-127.0h), half4(127.0h))));
        dst[3] = char4(rint(clamp(whi * mul, half4(-127.0h), half4(127.0h))));
    }
}

template <int BM>
union Sg53E8GuI8Smem {
    int8_t w[64][128];  // 8 KiB
    float  mid[BM][64]; // <= 16 KiB
};

template <int BM, int SG>
inline void sg53_moe_e8s18_nax_i8_gu_impl(
    device const uchar*               part_base,
    constant StrataGlm53E8Half8Entry* lut,
    device const int8_t*              x_sorted_i8,
    device const half*                x_rsc,
    device const Sg53MoeV2WorkTile*   work_tiles,
    device half*                      mid_sorted,
    constant Sg53MoeE8Params&         p,
    threadgroup Sg53E8GuI8Smem<BM>&   smem,
    threadgroup float*                tg_wsc,
    uint2 tgid,
    uint  tid,
    uint  sgid,
    uint  lane
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.d_moe) return;
    const Sg53MoeV2WorkTile tile = work_tiles[p.tile_base + tgid.y];
    if (tile.pair_count == 0u) return;

    device const uchar* rec_base = part_base + p.layer_byte_off + ulong(tile.expert_id) * p.e8_record_bytes;
    const ulong sc_off           = p.e8_lo16_bytes + p.e8_hi2_bytes;
    const uint  grp_per_row      = p.D >> 6u;
    const uint  kt_cnt           = p.D >> 7u;

    constexpr uint r_step  = uint(SG) * 4u;
    constexpr uint q_iters = 64u / r_step;
    const uint r_grp = tid >> 3u;

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tX(
        const_cast<device int8_t*>(x_sorted_i8), dextents<int, 2>(int(p.D), int(p.total_pairs_pad)));
    tensor<threadgroup int8_t, extents<int, 128, 64>, tensor_inline> tW(&smem.w[0][0], extents<int, 128, 64>());
    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto sX0 = tX.template slice<128, BM>(0, int(tile.pair_start));
    auto cI  = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sX0)>, sg53_strip_as<decltype(tW)>, int32_t>();

    // Pass 1: `gate_proj`
    {
        device const ushort2* g_lo = reinterpret_cast<device const ushort2*>(rec_base);
        device const ushort*  g_hi = reinterpret_cast<device const ushort*> (rec_base + p.e8_lo16_bytes);
        device const uchar*   g_sc = rec_base + sc_off;
        if (tid < 64u) {
            const ulong rg = ulong(n0 + tid);
            uchar mx = 0u;
            for (uint g = 0u; g < grp_per_row; ++g) mx = max(mx, g_sc[rg * grp_per_row + g]);
            tg_wsc[tid] = float(sg53_e4m4_decode_h(mx)) * (4.5f / 127.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_w[q_iters];
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            float ws = tg_wsc[q * r_step + r_grp];
            inv_w[q] = (ws > 1e-12f) ? (1.0f / ws) : 0.0f;
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cI.get_capacity(); ++i) if (cI.is_valid_element(i)) cI[i] = 0;

        for (uint kt = 0u; kt < kt_cnt; ++kt) {
            sg53_e8s18_unpack_tile_i8_bk128<SG>(g_lo, g_hi, g_sc, lut, inv_w, n0, kt, p.D, smem.w, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto sX = tX.template slice<128, BM>(int(kt * 128u), int(tile.pair_start));
            op.run(sX, tW, cI);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cI.get_capacity(); ++i) {
            if (cI.is_valid_element(i)) {
                auto idx = cI.get_multidimensional_index(i);
                const uint r = uint(idx[1]), c = uint(idx[0]);
                if (r < tile.pair_count) {
                    const float gv = float(cI[i]) * float(x_rsc[tile.pair_start + r]) * tg_wsc[c];
                    mid_sorted[ulong(tile.pair_start + r) * p.d_moe + n0 + c] =
                        sg53_clamp_f16(clamp(gv, -24.0f, p.swiglu_limit));
                }
            }
        }
    }

    // Pass 2: `up_proj` + fused `LimitedSwiGLU + RHT-64` -> FP16 `mid_sorted` (exact match to `dn_i8` or `dn`!)
    {
        device const uchar*   u_base = rec_base + p.e8_proj_bytes;
        device const ushort2* u_lo   = reinterpret_cast<device const ushort2*>(u_base);
        device const ushort*  u_hi   = reinterpret_cast<device const ushort*> (u_base + p.e8_lo16_bytes);
        device const uchar*   u_sc   = u_base + sc_off;
        if (tid < 64u) {
            const ulong rg = ulong(n0 + tid);
            uchar mx = 0u;
            for (uint g = 0u; g < grp_per_row; ++g) mx = max(mx, u_sc[rg * grp_per_row + g]);
            tg_wsc[tid] = float(sg53_e4m4_decode_h(mx)) * (4.5f / 127.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_w[q_iters];
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            float ws = tg_wsc[q * r_step + r_grp];
            inv_w[q] = (ws > 1e-12f) ? (1.0f / ws) : 0.0f;
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cI.get_capacity(); ++i) if (cI.is_valid_element(i)) cI[i] = 0;

        for (uint kt = 0u; kt < kt_cnt; ++kt) {
            sg53_e8s18_unpack_tile_i8_bk128<SG>(u_lo, u_hi, u_sc, lut, inv_w, n0, kt, p.D, smem.w, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto sX = tX.template slice<128, BM>(int(kt * 128u), int(tile.pair_start));
            op.run(sX, tW, cI);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cI.get_capacity(); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx = cI.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < tile.pair_count) {
                const float g_val = float(mid_sorted[ulong(tile.pair_start + r) * p.d_moe + n0 + c]);
                const float u_val = float(cI[i]) * float(x_rsc[tile.pair_start + r]) * tg_wsc[c];
                smem.mid[r][c] = sg53_limited_swiglu(g_val, u_val, p.swiglu_limit);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = sgid; r < tile.pair_count; r += uint(SG)) {
        float v0 = smem.mid[r][2u * lane];
        float v1 = smem.mid[r][2u * lane + 1u];
        sg53_rht64_warp_f32(v0, v1, lane);
        device half2* dst2 = reinterpret_cast<device half2*>(
            mid_sorted + ulong(tile.pair_start + r) * p.d_moe + n0);
        dst2[lane] = half2(sg53_clamp_f16(v0), sg53_clamp_f16(v1));
    }
}

template <int BM, int SG>
inline void sg53_moe_e8s18_nax_i8_dn_impl(
    device const uchar*               part_base,
    constant StrataGlm53E8Half8Entry* lut,
    device const int8_t*              mid_sorted_i8,
    device const half*                mid_rsc,
    device const uint*                sorted_pairs,
    device const Sg53MoeV2WorkTile*   work_tiles,
    device float*                     down_part,
    constant Sg53MoeE8Params&         p,
    threadgroup int8_t                (&W_i8)[64][128],
    threadgroup float*                tg_wsc,
    uint2 tgid,
    uint  tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.D) return;
    const Sg53MoeV2WorkTile tile = work_tiles[p.tile_base + tgid.y];
    if (tile.pair_count == 0u) return;

    device const uchar* d_base = part_base + p.layer_byte_off + ulong(tile.expert_id) * p.e8_record_bytes + 2ul * p.e8_proj_bytes;
    const ulong sc_off         = p.e8_lo16_bytes + p.e8_hi2_bytes;
    device const ushort2* d_lo = reinterpret_cast<device const ushort2*>(d_base);
    device const ushort*  d_hi = reinterpret_cast<device const ushort*> (d_base + p.e8_lo16_bytes);
    device const uchar*   d_sc = d_base + sc_off;

    const uint grp_per_row = p.d_moe >> 6u;
    const uint kt_cnt      = p.d_moe >> 7u;
    constexpr uint r_step  = uint(SG) * 4u;
    constexpr uint q_iters = 64u / r_step;
    const uint r_grp = tid >> 3u;

    if (tid < 64u) {
        const ulong rg = ulong(n0 + tid);
        uchar mx = 0u;
        for (uint g = 0u; g < grp_per_row; ++g) mx = max(mx, d_sc[rg * grp_per_row + g]);
        tg_wsc[tid] = float(sg53_e4m4_decode_h(mx)) * (4.5f / 127.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_w[q_iters];
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < q_iters; ++q) {
        float ws = tg_wsc[q * r_step + r_grp];
        inv_w[q] = (ws > 1e-12f) ? (1.0f / ws) : 0.0f;
    }

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tM(
        const_cast<device int8_t*>(mid_sorted_i8), dextents<int, 2>(int(p.d_moe), int(p.total_pairs_pad)));
    tensor<threadgroup int8_t, extents<int, 128, 64>, tensor_inline> tW(&W_i8[0][0], extents<int, 128, 64>());
    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto sM0 = tM.template slice<128, BM>(0, int(tile.pair_start));
    auto cI  = op.template get_destination_cooperative_tensor<sg53_strip_as<decltype(sM0)>, sg53_strip_as<decltype(tW)>, int32_t>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cI.get_capacity(); ++i) if (cI.is_valid_element(i)) cI[i] = 0;

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        sg53_e8s18_unpack_tile_i8_bk128<SG>(d_lo, d_hi, d_sc, lut, inv_w, n0, kt, p.d_moe, W_i8, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sM = tM.template slice<128, BM>(int(kt * 128u), int(tile.pair_start));
        op.run(sM, tW, cI);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cI.get_capacity(); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx     = cI.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < tile.pair_count && (n0 + c) < p.D) {
                const uint pid = sorted_pairs[tile.pair_start + r];
                down_part[ulong(pid) * p.D + n0 + c] =
                    float(cI[i]) * float(mid_rsc[tile.pair_start + r]) * tg_wsc[c];
            }
        }
    }
}
#endif

kernel void sg53_moe_e8s18_nax_i8_gu_m64(
    device const uchar*               part_base   [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut         [[buffer(1)]],
    device const int8_t*              x_sorted_i8 [[buffer(2)]],
    device const half*                x_rsc       [[buffer(3)]],
    device const Sg53MoeV2WorkTile*   work_tiles  [[buffer(4)]],
    device half*                      mid_sorted  [[buffer(5)]],
    constant Sg53MoeE8Params&         p           [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup Sg53E8GuI8Smem<64> smem;
    threadgroup float tg_wsc[64];
    sg53_moe_e8s18_nax_i8_gu_impl<64, 4>(part_base, lut, x_sorted_i8, x_rsc, work_tiles, mid_sorted, p, smem, tg_wsc, tgid, tid, sgid, lane);
#else
    (void)part_base; (void)lut; (void)x_sorted_i8; (void)x_rsc; (void)work_tiles; (void)mid_sorted; (void)p; (void)tgid; (void)tid; (void)sgid; (void)lane;
#endif
}

kernel void sg53_moe_e8s18_nax_i8_gu_m32(
    device const uchar*               part_base   [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut         [[buffer(1)]],
    device const int8_t*              x_sorted_i8 [[buffer(2)]],
    device const half*                x_rsc       [[buffer(3)]],
    device const Sg53MoeV2WorkTile*   work_tiles  [[buffer(4)]],
    device half*                      mid_sorted  [[buffer(5)]],
    constant Sg53MoeE8Params&         p           [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup Sg53E8GuI8Smem<32> smem;
    threadgroup float tg_wsc[64];
    sg53_moe_e8s18_nax_i8_gu_impl<32, 4>(part_base, lut, x_sorted_i8, x_rsc, work_tiles, mid_sorted, p, smem, tg_wsc, tgid, tid, sgid, lane);
#else
    (void)part_base; (void)lut; (void)x_sorted_i8; (void)x_rsc; (void)work_tiles; (void)mid_sorted; (void)p; (void)tgid; (void)tid; (void)sgid; (void)lane;
#endif
}

kernel void sg53_moe_e8s18_nax_i8_gu_m16(
    device const uchar*               part_base   [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut         [[buffer(1)]],
    device const int8_t*              x_sorted_i8 [[buffer(2)]],
    device const half*                x_rsc       [[buffer(3)]],
    device const Sg53MoeV2WorkTile*   work_tiles  [[buffer(4)]],
    device half*                      mid_sorted  [[buffer(5)]],
    constant Sg53MoeE8Params&         p           [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup Sg53E8GuI8Smem<16> smem;
    threadgroup float tg_wsc[64];
    sg53_moe_e8s18_nax_i8_gu_impl<16, 2>(part_base, lut, x_sorted_i8, x_rsc, work_tiles, mid_sorted, p, smem, tg_wsc, tgid, tid, sgid, lane);
#else
    (void)part_base; (void)lut; (void)x_sorted_i8; (void)x_rsc; (void)work_tiles; (void)mid_sorted; (void)p; (void)tgid; (void)tid; (void)sgid; (void)lane;
#endif
}

// Per-row INT8 quantize of `mid_sorted [total_pairs+64, d_moe]` -> `mid_sorted_i8` + `mid_rsc` (0 extra UMA via alias!)
kernel void sg53_moe_quant_mid_i8(
    device const half*         mid_f16 [[buffer(0)]],
    device int8_t*             mid_i8  [[buffer(1)]],
    device half*               mid_rsc [[buffer(2)]],
    constant Sg53MoeBinParams& p       [[buffer(3)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint total_pairs = p.M_valid * p.top_k;
    if (row >= total_pairs + 64u) return;
    const uint m4_cnt = p.d_moe >> 2u;
    device char4* dst4 = reinterpret_cast<device char4*>(mid_i8 + ulong(row) * p.d_moe);
    if (row >= total_pairs) {
        if (tid == 0u) mid_rsc[row] = half(0.0h);
        for (uint c4 = tid; c4 < m4_cnt; c4 += 256u) dst4[c4] = char4(0);
        return;
    }
    device const half4* src4 = reinterpret_cast<device const half4*>(mid_f16 + ulong(row) * p.d_moe);
    float amax = 0.0f;
    for (uint c4 = tid; c4 < m4_cnt; c4 += 256u) {
        half4 v = abs(src4[c4]);
        amax = max(amax, max(max(float(v.x), float(v.y)), max(float(v.z), float(v.w))));
    }
    amax = simd_max(amax);
    threadgroup float tg_max[8];
    if (lane == 0u) tg_max[sgid] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float gm = simd_max((lane < 8u) ? tg_max[lane] : 0.0f);
        if (lane == 0u) tg_max[0] = gm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float ra      = tg_max[0];
    const float rscale  = (ra > 1e-12f) ? (ra / 127.0f) : 0.0f;
    const float inv_rsc = (ra > 1e-12f) ? (127.0f / ra) : 0.0f;
    if (tid == 0u) mid_rsc[row] = half(rscale);
    for (uint c4 = tid; c4 < m4_cnt; c4 += 256u) {
        const float4 fv = float4(src4[c4]) * inv_rsc;
        dst4[c4] = char4(rint(clamp(fv, float4(-127.0f), float4(127.0f))));
    }
}

kernel void sg53_moe_e8s18_nax_i8_dn_m64(
    device const uchar*               part_base     [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut           [[buffer(1)]],
    device const int8_t*              mid_sorted_i8 [[buffer(2)]],
    device const half*                mid_rsc       [[buffer(3)]],
    device const uint*                sorted_pairs  [[buffer(4)]],
    device const Sg53MoeV2WorkTile*   work_tiles    [[buffer(5)]],
    device float*                     down_part     [[buffer(6)]],
    constant Sg53MoeE8Params&         p             [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup int8_t W_i8[64][128];
    threadgroup float tg_wsc[64];
    sg53_moe_e8s18_nax_i8_dn_impl<64, 4>(part_base, lut, mid_sorted_i8, mid_rsc, sorted_pairs, work_tiles, down_part, p, W_i8, tg_wsc, tgid, tid);
#else
    (void)part_base; (void)lut; (void)mid_sorted_i8; (void)mid_rsc; (void)sorted_pairs; (void)work_tiles; (void)down_part; (void)p; (void)tgid; (void)tid;
#endif
}

kernel void sg53_moe_e8s18_nax_i8_dn_m32(
    device const uchar*               part_base     [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut           [[buffer(1)]],
    device const int8_t*              mid_sorted_i8 [[buffer(2)]],
    device const half*                mid_rsc       [[buffer(3)]],
    device const uint*                sorted_pairs  [[buffer(4)]],
    device const Sg53MoeV2WorkTile*   work_tiles    [[buffer(5)]],
    device float*                     down_part     [[buffer(6)]],
    constant Sg53MoeE8Params&         p             [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup int8_t W_i8[64][128];
    threadgroup float tg_wsc[64];
    sg53_moe_e8s18_nax_i8_dn_impl<32, 4>(part_base, lut, mid_sorted_i8, mid_rsc, sorted_pairs, work_tiles, down_part, p, W_i8, tg_wsc, tgid, tid);
#else
    (void)part_base; (void)lut; (void)mid_sorted_i8; (void)mid_rsc; (void)sorted_pairs; (void)work_tiles; (void)down_part; (void)p; (void)tgid; (void)tid;
#endif
}

kernel void sg53_moe_e8s18_nax_i8_dn_m16(
    device const uchar*               part_base     [[buffer(0)]],
    constant StrataGlm53E8Half8Entry* lut           [[buffer(1)]],
    device const int8_t*              mid_sorted_i8 [[buffer(2)]],
    device const half*                mid_rsc       [[buffer(3)]],
    device const uint*                sorted_pairs  [[buffer(4)]],
    device const Sg53MoeV2WorkTile*   work_tiles    [[buffer(5)]],
    device float*                     down_part     [[buffer(6)]],
    constant Sg53MoeE8Params&         p             [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    threadgroup int8_t W_i8[64][128];
    threadgroup float tg_wsc[64];
    sg53_moe_e8s18_nax_i8_dn_impl<16, 2>(part_base, lut, mid_sorted_i8, mid_rsc, sorted_pairs, work_tiles, down_part, p, W_i8, tg_wsc, tgid, tid);
#else
    (void)part_base; (void)lut; (void)mid_sorted_i8; (void)mid_rsc; (void)sorted_pairs; (void)work_tiles; (void)down_part; (void)p; (void)tgid; (void)tid;
#endif
}

// ---- v2 Single-CB UMA `Q4_A` Layer-45 MTP MoE Kernels (`b_mtp_experts`, ZERO CPU sync!) ----

// Direct `topk_idx` Q4_A `gate_proj + up_proj + LimitedSwiGLU` for `M = 1` MTP draft steps (`1` row/SG = `4` rows/TG)
kernel void sg53_moe_q4_uma_gather_gu(
    device const uchar*             mtp_base [[buffer(0)]], // `b_mtp_experts` (`288 * 13.5 MiB`)
    device const uint*              topk_idx [[buffer(1)]], // [M_valid * top_k]
    constant half*                  x_in     [[buffer(2)]], // [M_valid, D]
    device half*                    mid_out  [[buffer(3)]], // [M_valid * top_k, d_moe]
    constant Sg53MoeIndirectParams& p        [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid.x * 4u + sgid;
    const uint k   = tgid.y;
    const uint m   = tgid.z;
    if (row >= p.d_moe || k >= p.top_k || m >= p.M_valid) return;

    const ulong pair_idx = ulong(m) * p.top_k + k;
    const uint  eid      = topk_idx[pair_idx];
    device const uchar* rec_base = mtp_base + ulong(eid) * p.expert_record_bytes;

    device const uint2* gw = reinterpret_cast<device const uint2*>(rec_base);
    device const half*  gs = reinterpret_cast<device const half*>(rec_base + p.expert_weight_bytes);
    device const half*  gb = reinterpret_cast<device const half*>(rec_base + p.expert_weight_bytes + p.expert_scales_bytes);

    device const uchar* up_base = rec_base + p.expert_proj_bytes;
    device const uint2* uw = reinterpret_cast<device const uint2*>(up_base);
    device const half*  us = reinterpret_cast<device const half*>(up_base + p.expert_weight_bytes);
    device const half*  ub = reinterpret_cast<device const half*>(up_base + p.expert_weight_bytes + p.expert_scales_bytes);

    const uint u2_per_row  = p.D >> 4u;
    const uint grp_per_row = p.D >> 6u;
    constant half* xm      = x_in + ulong(m) * p.D;

    float ga = 0.0f;
    float ua = 0.0f;
    for (uint u2_idx = lane; u2_idx < u2_per_row; u2_idx += 32u) {
        const uint  grp_idx = u2_idx >> 2u;
        const uint2 qg      = gw[ulong(row) * u2_per_row + u2_idx];
        const uint2 qu      = uw[ulong(row) * u2_per_row + u2_idx];
        const float gsc     = float(gs[ulong(row) * grp_per_row + grp_idx]);
        const float gbi     = float(gb[ulong(row) * grp_per_row + grp_idx]);
        const float usc     = float(us[ulong(row) * grp_per_row + grp_idx]);
        const float ubi     = float(ub[ulong(row) * grp_per_row + grp_idx]);

        half4 g0, g1, g2, g3, u0, u1, u2, u3;
        sg53_unpack_u32_nibbles_h4(qg.x, g0, g1); sg53_unpack_u32_nibbles_h4(qg.y, g2, g3);
        sg53_unpack_u32_nibbles_h4(qu.x, u0, u1); sg53_unpack_u32_nibbles_h4(qu.y, u2, u3);

        constant half4* xp = reinterpret_cast<constant half4*>(xm + u2_idx * 16u);
        const half4  x0  = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        const float4 xs4 = (float4(x0) + float4(x1)) + (float4(x2) + float4(x3));
        const float  xs  = (xs4.x + xs4.y) + (xs4.z + xs4.w);

        ga = fma(gsc, sg53_dot_h4x4(g0, g1, g2, g3, x0, x1, x2, x3), fma(gbi, xs, ga));
        ua = fma(usc, sg53_dot_h4x4(u0, u1, u2, u3, x0, x1, x2, x3), fma(ubi, xs, ua));
    }

    const float gv = simd_sum(ga);
    const float uv = simd_sum(ua);
    if (lane == 0u) {
        mid_out[pair_idx * p.d_moe + row] = sg53_clamp_f16(sg53_limited_swiglu(gv, uv, p.swiglu_limit));
    }
}

// Direct `topk_idx` Q4_A `down_proj` for `M = 1` MTP draft steps (`2` rows/SG = `8` rows/TG)
kernel void sg53_moe_q4_uma_gather_dn(
    device const uchar*             mtp_base  [[buffer(0)]],
    device const uint*              topk_idx  [[buffer(1)]],
    constant half*                  mid_in    [[buffer(2)]],
    device float*                   down_part [[buffer(3)]],
    constant Sg53MoeIndirectParams& p         [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid.x * 4u + sgid) * 2u;
    const uint k    = tgid.y;
    const uint m    = tgid.z;
    if (row0 >= p.D || k >= p.top_k || m >= p.M_valid) return;

    const ulong pair_idx = ulong(m) * p.top_k + k;
    const uint  eid      = topk_idx[pair_idx];
    device const uchar* dn_base = mtp_base + ulong(eid) * p.expert_record_bytes + 2ul * p.expert_proj_bytes;

    device const uint2* dw = reinterpret_cast<device const uint2*>(dn_base);
    device const half*  ds = reinterpret_cast<device const half*>(dn_base + p.expert_weight_bytes);
    device const half*  db = reinterpret_cast<device const half*>(dn_base + p.expert_weight_bytes + p.expert_scales_bytes);

    float2 res[1];
    sg53_q4_gemv2_rows_fixed<1u>(dw, ds, db, mid_in + pair_idx * p.d_moe, row0, p.D, p.d_moe, lane, res);

    if (lane == 0u) {
        if (row0 + 2u <= p.D) *reinterpret_cast<device float2*>(down_part + pair_idx * p.D + row0) = res[0];
        else                  down_part[pair_idx * p.D + row0] = res[0].x;
    }
}

// GPU-binned indirect `Q4_A` UMA `gate_proj + up_proj + LimitedSwiGLU` (`M = 2..8` MTP sync & prefill!)
kernel void sg53_moe_q4_uma_grouped_gu(
    device const uchar*             mtp_base     [[buffer(0)]],
    device const half*              x_in         [[buffer(1)]],
    device const uint*              sorted_pairs [[buffer(2)]],
    device const Sg53MoeV2WorkTile* work_tiles   [[buffer(3)]],
    device half*                    mid_out      [[buffer(4)]],
    constant Sg53MoeIndirectParams& p            [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid.x * 4u + sgid;
    if (row >= p.d_moe) return;
    const Sg53MoeV2WorkTile v2t = work_tiles[tgid.y];
    Sg53MoeWorkTile tile;
    tile.source_buf = STRATA_GLM53_EXPERT_SRC_HOT_UMA;
    tile.slot_idx   = v2t.expert_id;
    tile.pair_start = v2t.pair_start;
    tile.pair_count = v2t.pair_count;

    #define SG53_Q4UGU_CASE(BF) { sg53_moe_grouped_gu_fixed<BF>(mtp_base, mtp_base, x_in, sorted_pairs, mid_out, tile, p, row, lane); break; }
    switch (tile.pair_count) {
        case 1u: SG53_Q4UGU_CASE(1u)
        case 2u: SG53_Q4UGU_CASE(2u)
        case 3u: SG53_Q4UGU_CASE(3u)
        case 4u: SG53_Q4UGU_CASE(4u)
        case 5u: SG53_Q4UGU_CASE(5u)
        case 6u: SG53_Q4UGU_CASE(6u)
        case 7u: SG53_Q4UGU_CASE(7u)
        case 8u: SG53_Q4UGU_CASE(8u)
        default: break;
    }
    #undef SG53_Q4UGU_CASE
}

// GPU-binned indirect `Q4_A` UMA `down_proj` (`M = 2..8` MTP sync & prefill!)
kernel void sg53_moe_q4_uma_grouped_dn(
    device const uchar*             mtp_base     [[buffer(0)]],
    device const half*              mid_in       [[buffer(1)]],
    device const uint*              sorted_pairs [[buffer(2)]],
    device const Sg53MoeV2WorkTile* work_tiles   [[buffer(3)]],
    device float*                   down_part    [[buffer(4)]],
    constant Sg53MoeIndirectParams& p            [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid.x * 4u + sgid) * 2u;
    if (row0 >= p.D) return;
    const Sg53MoeV2WorkTile v2t = work_tiles[tgid.y];
    Sg53MoeWorkTile tile;
    tile.source_buf = STRATA_GLM53_EXPERT_SRC_HOT_UMA;
    tile.slot_idx   = v2t.expert_id;
    tile.pair_start = v2t.pair_start;
    tile.pair_count = v2t.pair_count;

    #define SG53_Q4UDN_CASE(BF) { sg53_moe_grouped_dn_fixed<BF>(mtp_base, mtp_base, mid_in, sorted_pairs, down_part, tile, p, row0, lane); break; }
    switch (tile.pair_count) {
        case 1u: SG53_Q4UDN_CASE(1u)
        case 2u: SG53_Q4UDN_CASE(2u)
        case 3u: SG53_Q4UDN_CASE(3u)
        case 4u: SG53_Q4UDN_CASE(4u)
        case 5u: SG53_Q4UDN_CASE(5u)
        case 6u: SG53_Q4UDN_CASE(6u)
        case 7u: SG53_Q4UDN_CASE(7u)
        case 8u: SG53_Q4UDN_CASE(8u)
        default: break;
    }
    #undef SG53_Q4UDN_CASE
}

/* ============================================================================
 * 9. Batched Vocab Argmax & Top-K/Top-P Leviathan Sampler (`vocab = 154,880`)
 * ============================================================================ */

struct Sg53SampleParams {
    uint  M;
    uint  vocab_size;
    uint  greedy;
    float inv_temp;
    uint  top_k;
    float top_p;
    uint  rng_lo;
    uint  rng_hi;
};

inline float sg53_rand_u01(uint lo, uint hi, uint salt) {
    ulong z = ((ulong(hi) << 32ul) | ulong(lo)) + ulong(salt) * 0x9E3779B97F4A7C15ul;
    z += 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30ul)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27ul)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31ul);
    uint u24 = uint(z >> 40ul) & 0xFFFFFFu;
    return (float(u24) + 0.5f) * (1.0f / 16777216.0f);
}

kernel void sg53_sample_vocab_batch(
    device const float*        logits        [[buffer(0)]], // [M, vocab_size]
    device uint*               out_tok       [[buffer(1)]], // [M]
    device uint*               out_top_ids   [[buffer(2)]], // optional [M, 64]
    device float*              out_top_probs [[buffer(3)]], // optional [M, 64]
    device uint*               out_keep_k    [[buffer(4)]], // optional [M]
    constant Sg53SampleParams& p             [[buffer(5)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],              // 0..1023
    uint sgid [[simdgroup_index_in_threadgroup]],           // 0..31
    uint lane [[thread_index_in_simdgroup]]                 // 0..31
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
        const float max_v = top_vals[0];
        float probs[64];
        float sum_e = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float ew = exp((top_vals[k] - max_v) * p.inv_temp);
            probs[k] = ew;
            sum_e   += ew;
        }
        const float inv_s = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
        for (uint k = 0u; k < K; ++k) {
            probs[k] *= inv_s;
        }

        uint  keep_k   = K;
        float kept_sum = 1.0f;
        if (p.top_p > 0.0f && p.top_p < 1.0f) {
            float cum = 0.0f;
            keep_k    = 0u;
            for (uint k = 0u; k < K; ++k) {
                cum += probs[k];
                ++keep_k;
                if (cum >= p.top_p) break;
            }
            kept_sum = (cum > 0.0f) ? cum : 1.0f;
        }

        const float inv_kept = 1.0f / kept_sum;
        for (uint k = 0u; k < keep_k; ++k) {
            probs[k] *= inv_kept;
            out_top_ids[ulong(m) * 64ul + k]   = top_ids[k];
            out_top_probs[ulong(m) * 64ul + k] = probs[k];
        }
        out_keep_k[m] = keep_k;

        const float u01 = sg53_rand_u01(p.rng_lo, p.rng_hi, m);
        float cdf       = 0.0f;
        uint  chosen    = top_ids[0];
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += probs[k];
            if (u01 <= cdf || (k + 1u) == keep_k) {
                chosen = top_ids[k];
                break;
            }
        }
        out_tok[m] = chosen;
    }
}

// 2-Stage Parallel Vocab Sampler (`64` TGs x `256` th -> `1` TG x `256` th):
// 100% bit-exact to `sg53_sample_vocab_batch` for both greedy and stochastic top-K/top-P sampling!
kernel void sg53_sample_vocab_stage1(
    device const float*        logits   [[buffer(0)]], // [M, vocab_size]
    device float*              stage_v  [[buffer(1)]], // [M, 64, K_out]
    device uint*               stage_i  [[buffer(2)]], // [M, 64, K_out]
    constant Sg53SampleParams& p        [[buffer(3)]],
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

    if (p.greedy != 0u) {
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
                stage_v[ulong(m) * 64ul + blk] = v;
                stage_i[ulong(m) * 64ul + blk] = idx;
            }
        }
        return;
    }

    const uint K = min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
    const ulong base_out = (ulong(m) * 64ul + blk) * K;

    if (chunk <= 4096u) {
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
                if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
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
                    if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
                }
            }
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(best_v, off);
                uint  oi = simd_shuffle_xor(best_i, off);
                if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
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
                    picked[k] = idx;
                    stage_v[base_out + k] = v;
                    stage_i[base_out + k] = idx;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void sg53_sample_vocab_stage2(
    device const float*        stage_v       [[buffer(0)]], // [M, 64 * K_out]
    device const uint*         stage_i       [[buffer(1)]], // [M, 64 * K_out]
    device uint*               out_tok       [[buffer(2)]], // [M]
    device uint*               out_top_ids   [[buffer(3)]], // optional [M, 64]
    device float*              out_top_probs [[buffer(4)]], // optional [M, 64]
    device uint*               out_keep_k    [[buffer(5)]], // optional [M]
    constant Sg53SampleParams& p             [[buffer(6)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],              // 0..255
    uint sgid [[simdgroup_index_in_threadgroup]],           // 0..7
    uint lane [[thread_index_in_simdgroup]]                 // 0..31
) {
    if (m >= p.M) return;

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];

    if (p.greedy != 0u) {
        float best_v = (tid < 64u) ? stage_v[ulong(m) * 64ul + tid] : -INFINITY;
        uint  best_i = (tid < 64u) ? stage_i[ulong(m) * 64ul + tid] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
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
            if (lane == 0u) out_tok[m] = (idx < p.vocab_size) ? idx : 0u;
        }
        return;
    }

    const uint  K       = min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
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
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
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
                top_ids[k]  = (idx < p.vocab_size) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (best_j < 16u && ri[best_j] == tg_i[0]) {
            rv[best_j] = -INFINITY;
            ri[best_j] = 0xFFFFFFFFu;
        }
    }

    if (tid == 0u) {
        const float max_v = top_vals[0];
        float probs[64];
        float sum_e = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float ew = exp((top_vals[k] - max_v) * p.inv_temp);
            probs[k] = ew;
            sum_e   += ew;
        }
        const float inv_s = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
        for (uint k = 0u; k < K; ++k) probs[k] *= inv_s;

        uint  keep_k   = K;
        float kept_sum = 1.0f;
        if (p.top_p > 0.0f && p.top_p < 1.0f) {
            float cum = 0.0f;
            keep_k    = 0u;
            for (uint k = 0u; k < K; ++k) {
                cum += probs[k];
                ++keep_k;
                if (cum >= p.top_p) break;
            }
            kept_sum = (cum > 0.0f) ? cum : 1.0f;
        }

        const float inv_kept = 1.0f / kept_sum;
        for (uint k = 0u; k < keep_k; ++k) {
            probs[k] *= inv_kept;
            out_top_ids[ulong(m) * 64ul + k]   = top_ids[k];
            out_top_probs[ulong(m) * 64ul + k] = probs[k];
        }
        out_keep_k[m] = keep_k;

        const float u01 = sg53_rand_u01(p.rng_lo, p.rng_hi, m);
        float cdf       = 0.0f;
        uint  chosen    = top_ids[0];
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += probs[k];
            if (u01 <= cdf || (k + 1u) == keep_k) {
                chosen = top_ids[k];
                break;
            }
        }
        out_tok[m] = chosen;
    }
}

/* ============================================================================
 * 9b. Sliding-Window OpenAI Presence/Frequency & HF Repetition Logit Penalties
 * ============================================================================ */

struct Sg53LogitPenaltyParams {
    uint  M;
    uint  vocab_size;
    uint  num_hist_ids;
    uint  extra_count;
    uint  extra_mode;         // 0: linear verify/decode (row m adds extra_toks[1..m]); 1: single draft row adding extra_toks[1..extra_count]
    float presence_penalty;
    float frequency_penalty;
    float repetition_penalty;
    uint  drop_toks[8];       // Exact sliding-window eviction tokens for speculative steps 1..8 (0xFFFFFFFFu = none)
};

inline float sg53_apply_pen_val(float v, uint count, float pres, float freq, float rep) {
    if (rep > 0.0f && rep != 1.0f) {
        v = (v > 0.0f) ? (v / rep) : (v * rep);
    }
    if (pres != 0.0f) v -= pres;
    if (freq != 0.0f) v -= freq * float(count);
    return v;
}

// Sparse O(U + M) sliding-window logit penalty kernel — 100.0000% bit-exact `MTP K>0 == K=0`!
// Grid: `((num_hist_ids + extra_count + 255) / 256, 1, 1)`, `256` threads/TG
kernel void sg53_apply_logit_penalties(
    device float*                   logits     [[buffer(0)]], // [M, vocab_size] (in-place)
    device const uint*              pen_ids    [[buffer(1)]], // [num_hist_ids] unique token IDs in active window
    device const ushort*            pen_counts [[buffer(2)]], // [vocab_size] occurrence counts in active window
    device const uint*              extra_toks [[buffer(3)]], // [1 + extra_count]
    constant Sg53LogitPenaltyParams& p         [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_items = p.num_hist_ids + p.extra_count;
    if (gid >= total_items) return;

    uint t      = 0u;
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

    if (p.extra_mode == 1u) {
        uint c = base_c;
        for (uint j = 1u; j <= p.extra_count; ++j) {
            if (extra_toks[j] == t) ++c;
            if ((j - 1u) < 8u && p.drop_toks[j - 1u] == t && c > 0u) --c;
        }
        if (c > 0u) {
            logits[t] = sg53_apply_pen_val(logits[t], c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
        }
    } else {
        uint c = base_c;
        for (uint m = 0u; m < p.M; ++m) {
            if (m > 0u && m <= p.extra_count) {
                if (extra_toks[m] == t) ++c;
                if ((m - 1u) < 8u && p.drop_toks[m - 1u] == t && c > 0u) --c;
            }
            if (c > 0u) {
                const ulong idx = ulong(m) * p.vocab_size + t;
                logits[idx] = sg53_apply_pen_val(logits[idx], c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
            }
        }
    }
}

/* ============================================================================
 * 9c. FR-Spec 20K SLC Shortlist Draft Sampler (`p_max` + `short_to_full`) & Penalties
 * ============================================================================ */

struct Sg53DraftSampleParams {
    uint  M;
    uint  vocab_size;    // V_short (e.g. 20480) or full V (154880)
    uint  greedy;
    float inv_temp;
    uint  top_k;
    float top_p;
    uint  rng_lo;
    uint  rng_hi;
    uint  use_token_map; // 1 => translate winner & top_ids via short_to_full
    uint  want_pmax;     // 1 => compute exact softmax probability of drafted token into out_pmax[0]
    uint  pad0;
    uint  pad1;
};

// Stage 1 for MTP Draft: finds per-block top-1 (or top-K) AND online `(block_max, block_sumexp)` for exact `p_max`!
// Grid: `(64, M, 1)`, `256` threads/TG
kernel void sg53_sample_draft_stage1(
    device const float*             logits   [[buffer(0)]], // [M, vocab_size]
    device float*                   stage_v  [[buffer(1)]], // [M, 64, K_out]
    device uint*                    stage_i  [[buffer(2)]], // [M, 64, K_out]
    device float2*                  stage_ms [[buffer(3)]], // [M, 64] (block_max, block_sumexp)
    constant Sg53DraftSampleParams& p        [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
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
    threadgroup float tg_s[8];

    if (p.greedy != 0u) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;
        for (uint i = beg + tid; i < end; i += 256u) {
            float v = row[i];
            if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
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
                tg_v[0] = v;
                stage_v[ulong(m) * 64ul + blk] = v;
                stage_i[ulong(m) * 64ul + blk] = idx;
            }
        }
        if (p.want_pmax != 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float bmax = tg_v[0];
            float se = 0.0f;
            if (isfinite(bmax)) {
                for (uint i = beg + tid; i < end; i += 256u) {
                    float d = row[i] - bmax;
                    if (d > -20.0f) se += exp(d);
                }
            }
            float sg_se = simd_sum(se);
            if (lane == 0u) tg_s[sgid] = sg_se;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                float tot_se = simd_sum((lane < 8u) ? tg_s[lane] : 0.0f);
                if (lane == 0u) stage_ms[ulong(m) * 64ul + blk] = float2(bmax, tot_se);
            }
        }
        return;
    }

    const uint K = min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
    const ulong base_out = (ulong(m) * 64ul + blk) * K;
    if (chunk <= 4096u) {
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
                float v = rv[j]; uint idx = ri[j];
                if (v > best_v || (v == best_v && idx < best_i)) { best_v = v; best_i = idx; best_j = j; }
            }
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(best_v, off);
                uint  oi = simd_shuffle_xor(best_i, off);
                if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
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
                    tg_i[0] = idx;
                    stage_v[base_out + k] = v;
                    stage_i[base_out + k] = idx;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (best_j < 16u && ri[best_j] == tg_i[0]) { rv[best_j] = -INFINITY; ri[best_j] = 0xFFFFFFFFu; }
        }
    } else {
        threadgroup uint picked[64];
        for (uint k = 0u; k < K; ++k) {
            float best_v = -INFINITY;
            uint  best_i = 0xFFFFFFFFu;
            for (uint i = beg + tid; i < end; i += 256u) {
                bool skip = false;
                for (uint prev = 0u; prev < k; ++prev) { if (picked[prev] == i) { skip = true; break; } }
                if (!skip) {
                    float v = row[i];
                    if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
                }
            }
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                float ov = simd_shuffle_xor(best_v, off);
                uint  oi = simd_shuffle_xor(best_i, off);
                if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
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
                    picked[k] = idx;
                    stage_v[base_out + k] = v;
                    stage_i[base_out + k] = idx;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Stage 2 for MTP Draft: maps shortlist indices -> full vocab via `short_to_full` AND writes exact `out_pmax`!
kernel void sg53_sample_draft_stage2(
    device const float*             stage_v       [[buffer(0)]],
    device const uint*              stage_i       [[buffer(1)]],
    device const float2*            stage_ms      [[buffer(2)]],
    device const uint*              short_to_full [[buffer(3)]], // [vocab_size] when use_token_map != 0
    device uint*                    out_tok       [[buffer(4)]], // [M]
    device float*                   out_pmax      [[buffer(5)]], // [M] (when want_pmax != 0)
    device uint*                    out_top_ids   [[buffer(6)]], // optional [M, 64]
    device float*                   out_top_probs [[buffer(7)]], // optional [M, 64]
    device uint*                    out_keep_k    [[buffer(8)]], // optional [M]
    constant Sg53DraftSampleParams& p             [[buffer(9)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M) return;
    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];

    if (p.greedy != 0u) {
        float best_v = (tid < 64u) ? stage_v[ulong(m) * 64ul + tid] : -INFINITY;
        uint  best_i = (tid < 64u) ? stage_i[ulong(m) * 64ul + tid] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
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
                const uint raw_id = (idx < p.vocab_size) ? idx : 0u;
                out_tok[m] = (p.use_token_map != 0u) ? short_to_full[raw_id] : raw_id;
                tg_v[0] = v;
            }
        }
        if (p.want_pmax != 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float gmax = tg_v[0];
            float contrib = 0.0f;
            if (tid < 64u && isfinite(gmax)) {
                float2 ms = stage_ms[ulong(m) * 64ul + tid];
                if (isfinite(ms.x) && ms.y > 0.0f) {
                    float d = ms.x - gmax;
                    if (d > -20.0f) contrib = ms.y * exp(d);
                }
            }
            float sg_c = simd_sum(contrib);
            if (lane == 0u) tg_v[sgid] = sg_c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                float tot = simd_sum((lane < 8u) ? tg_v[lane] : 0.0f);
                if (lane == 0u) out_pmax[m] = (tot > 0.0f) ? clamp(1.0f / tot, 0.0f, 1.0f) : 0.0f;
            }
        }
        return;
    }

    const uint  K       = min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
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
            float v = rv[j]; uint idx = ri[j];
            if (v > best_v || (v == best_v && idx < best_i)) { best_v = v; best_i = idx; best_j = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_xor(best_v, off);
            uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
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
                top_ids[k]  = (idx < p.vocab_size) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (best_j < 16u && ri[best_j] == tg_i[0]) { rv[best_j] = -INFINITY; ri[best_j] = 0xFFFFFFFFu; }
    }

    if (tid == 0u) {
        const float max_v = top_vals[0];
        float probs[64];
        float sum_e = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float ew = exp((top_vals[k] - max_v) * p.inv_temp);
            probs[k] = ew;
            sum_e   += ew;
        }
        const float inv_s = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
        for (uint k = 0u; k < K; ++k) probs[k] *= inv_s;

        uint  keep_k   = K;
        float kept_sum = 1.0f;
        if (p.top_p > 0.0f && p.top_p < 1.0f) {
            float cum = 0.0f;
            keep_k    = 0u;
            for (uint k = 0u; k < K; ++k) {
                cum += probs[k];
                ++keep_k;
                if (cum >= p.top_p) break;
            }
            kept_sum = (cum > 0.0f) ? cum : 1.0f;
        }

        const float inv_kept = 1.0f / kept_sum;
        for (uint k = 0u; k < keep_k; ++k) {
            probs[k] *= inv_kept;
            const uint raw_id = top_ids[k];
            const uint mapped = (p.use_token_map != 0u) ? short_to_full[raw_id] : raw_id;
            out_top_ids[ulong(m) * 64ul + k]   = mapped;
            out_top_probs[ulong(m) * 64ul + k] = probs[k];
        }
        out_keep_k[m] = keep_k;

        const float u01    = sg53_rand_u01(p.rng_lo, p.rng_hi, m);
        float       cdf    = 0.0f;
        uint        chosen = out_top_ids[ulong(m) * 64ul + 0u];
        float       p_ch   = probs[0];
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += probs[k];
            if (u01 <= cdf || (k + 1u) == keep_k) {
                chosen = out_top_ids[ulong(m) * 64ul + k];
                p_ch   = probs[k];
                break;
            }
        }
        out_tok[m] = chosen;
        if (p.want_pmax != 0u) out_pmax[m] = clamp(p_ch, 0.0f, 1.0f);
    }
}

// Sparse O(U + M) sliding-window logit penalty kernel for FR-Spec shortlisted draft logits (`logits[V_short]`)
kernel void sg53_apply_logit_penalties_shortlist(
    device float*                    logits        [[buffer(0)]], // [1, V_short] (in-place)
    device const uint*               pen_ids       [[buffer(1)]], // [num_hist_ids] full vocab IDs
    device const ushort*             pen_counts    [[buffer(2)]], // [V_full] occurrence counts
    device const uint*               extra_toks    [[buffer(3)]], // [1 + extra_count] full vocab IDs
    device const uint*               full_to_short [[buffer(4)]], // [V_full] -> shortlist idx or 0xFFFFFFFFu
    constant Sg53LogitPenaltyParams& p             [[buffer(5)]], // p.vocab_size = V_short!
    uint gid [[thread_position_in_grid]]
) {
    const uint total_items = p.num_hist_ids + p.extra_count;
    if (gid >= total_items) return;

    uint t      = 0u;
    uint base_c = 0u;
    if (gid < p.num_hist_ids) {
        t      = pen_ids[gid];
        base_c = uint(pen_counts[t]);
    } else {
        const uint j = 1u + (gid - p.num_hist_ids);
        t = extra_toks[j];
        if (pen_counts[t] != 0u) return;
        for (uint pj = 1u; pj < j; ++pj) {
            if (extra_toks[pj] == t) return;
        }
        base_c = 0u;
    }

    const uint st = full_to_short[t];
    if (st >= p.vocab_size) return;

    uint c = base_c;
    for (uint j = 1u; j <= p.extra_count; ++j) {
        if (extra_toks[j] == t) ++c;
        if ((j - 1u) < 8u && p.drop_toks[j - 1u] == t && c > 0u) --c;
    }
    if (c > 0u) {
        logits[st] = sg53_apply_pen_val(logits[st], c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
    }
}

struct Sg53FrspecPatchParams {
    uint count;
    uint cols;
    uint pad0;
    uint pad1;
};

// Dynamic Prompt+Online Self-Healing GPU Patcher for 20K SLC-Pinned FR-Spec (`< 1 us/row`!):
// Copies `count` rows `(slot_idx, full_row)` from full `lm_head` in `b_spine` into `b_frspec_lm_head`.
// Grid: `(count, 1, 1)`, `256` threads/TG
kernel void sg53_frspec_patch_rows(
    device const uint4*           src_w       [[buffer(0)]],
    device const half*            src_s       [[buffer(1)]],
    device const half*            src_b       [[buffer(2)]],
    device uint4*                 dst_w       [[buffer(3)]],
    device half*                  dst_s       [[buffer(4)]],
    device half*                  dst_b       [[buffer(5)]],
    device const uint2*           patch_pairs [[buffer(6)]], // [count]: .x = slot_idx, .y = full_row
    constant Sg53FrspecPatchParams& p         [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]                // 0..255
) {
    if (tgid >= p.count) return;
    const uint2 pair        = patch_pairs[tgid];
    const uint  slot        = pair.x;
    const uint  full_row    = pair.y;
    const uint  u4_per_row  = p.cols >> 5u;
    const uint  grp_per_row = p.cols >> 6u;

    for (uint i = tid; i < u4_per_row; i += 256u) {
        dst_w[ulong(slot) * u4_per_row + i] = src_w[ulong(full_row) * u4_per_row + i];
    }
    for (uint g = tid; g < grp_per_row; g += 256u) {
        dst_s[ulong(slot) * grp_per_row + g] = src_s[ulong(full_row) * grp_per_row + g];
        dst_b[ulong(slot) * grp_per_row + g] = src_b[ulong(full_row) * grp_per_row + g];
    }
}

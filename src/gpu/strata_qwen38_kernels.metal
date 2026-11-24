#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#if defined(__HAVE_TENSOR__)
#include <metal_tensor>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

// Inlined by strata_qwen38_gpu.mm at runtime from ../../include/strata_qwen38.h:
// @@STRATA_QWEN38_HEADER@@
#ifndef STRATA_QWEN38_H
#include "../../include/strata_qwen38.h"
#endif

using namespace metal;
#if defined(__HAVE_TENSOR__)
using namespace mpp::tensor_ops;
#endif

/* ============================================================================
 * 1. Lossless IEEE-754 4-Bit Affine Unpack & Numerically Safe Helpers
 * ============================================================================ */

inline half sq38_clamp_f16(float x) {
    return half(clamp(x, -65504.0f, 65504.0f));
}

inline half4 sq38_clamp_f16x4(float4 x) {
    return half4(clamp(x, float4(-65504.0f), float4(65504.0f)));
}

inline float sq38_silu(float x) {
    return x / (1.0f + exp(-x));
}

inline float sq38_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float sq38_softplus(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

// Lossless IEEE-754 binary16 trick: `0x6400 | nib` == `1024.0h + half(nib)` for nib in 0..15!
inline void sq38_unpack_u32_nibbles_h4(uint w, thread half4& lo, thread half4& hi) {
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

// Computes exact FP32 dot(n[0..31], x[0..31]) keeping `x` in compact 16-bit `half4` registers!
inline float sq38_dot_u4_xh8(
    uint4 q,
    half4 xh0, half4 xh1, half4 xh2, half4 xh3,
    half4 xh4, half4 xh5, half4 xh6, half4 xh7
) {
    half4 n0, n1, n2, n3, n4, n5, n6, n7;
    sq38_unpack_u32_nibbles_h4(q.x, n0, n1);
    sq38_unpack_u32_nibbles_h4(q.y, n2, n3);
    sq38_unpack_u32_nibbles_h4(q.z, n4, n5);
    sq38_unpack_u32_nibbles_h4(q.w, n6, n7);

    float4 s0 = fma(float4(n0), float4(xh0), float4(n1) * float4(xh1));
    float4 s1 = fma(float4(n2), float4(xh2), float4(n3) * float4(xh3));
    float4 s2 = fma(float4(n4), float4(xh4), float4(n5) * float4(xh5));
    float4 s3 = fma(float4(n6), float4(xh6), float4(n7) * float4(xh7));
    float4 st = (s0 + s1) + (s2 + s3);
    return (st.x + st.y) + (st.z + st.w);
}

/* ============================================================================
 * 2. Fast GPU Zero-Fill, 4-Bit Affine Embedding Lookup & FP32 Residual RMSNorm
 * ============================================================================ */

kernel void sq38_zero_u32(
    device uint4*  buf       [[buffer(0)]],
    constant uint& count_u4  [[buffer(1)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid < count_u4) {
        buf[gid] = uint4(0u);
    }
}

kernel void sq38_copy_u32(
    device const uint4* src      [[buffer(0)]],
    device uint4*       dst      [[buffer(1)]],
    constant uint&      count_u4 [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid < count_u4) {
        dst[gid] = src[gid];
    }
}

struct Sq38EmbedParams {
    uint M;
    uint cols;       // 5120
    uint vocab_size; // 248320
    uint pad;
};

kernel void sq38_embed_lookup_q4(
    device const uint4*   weight_u4 [[buffer(0)]], // [vocab_size, cols / 32]
    device const half*    scales    [[buffer(1)]], // [vocab_size, cols / 64]
    device const half*    biases    [[buffer(2)]], // [vocab_size, cols / 64]
    device const uint*    token_ids [[buffer(3)]], // [M]
    device float*         x_out_f32 [[buffer(4)]], // [M, cols]
    constant Sq38EmbedParams& p     [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]      // 0..159 (cols / 32)
) {
    if (tgid >= p.M) return;
    const uint u4_per_row  = p.cols >> 5u;
    const uint grp_per_row = p.cols >> 6u;
    if (tid >= u4_per_row) return;

    const uint tok = min(token_ids[tgid], p.vocab_size - 1u);
    const uint grp = tid >> 1u;

    const uint4 q  = weight_u4[ulong(tok) * u4_per_row + tid];
    const float sc = float(scales[ulong(tok) * grp_per_row + grp]);
    const float bi = float(biases[ulong(tok) * grp_per_row + grp]);

    half4 n0, n1, n2, n3, n4, n5, n6, n7;
    sq38_unpack_u32_nibbles_h4(q.x, n0, n1);
    sq38_unpack_u32_nibbles_h4(q.y, n2, n3);
    sq38_unpack_u32_nibbles_h4(q.z, n4, n5);
    sq38_unpack_u32_nibbles_h4(q.w, n6, n7);

    device float4* dst = reinterpret_cast<device float4*>(
        x_out_f32 + ulong(tgid) * p.cols + tid * 32u);
    dst[0] = fma(float4(n0), float4(sc), float4(bi));
    dst[1] = fma(float4(n1), float4(sc), float4(bi));
    dst[2] = fma(float4(n2), float4(sc), float4(bi));
    dst[3] = fma(float4(n3), float4(sc), float4(bi));
    dst[4] = fma(float4(n4), float4(sc), float4(bi));
    dst[5] = fma(float4(n5), float4(sc), float4(bi));
    dst[6] = fma(float4(n6), float4(sc), float4(bi));
    dst[7] = fma(float4(n7), float4(sc), float4(bi));
}

struct Sq38RmsNormParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    float eps;
};

// Standard `y = (x * rsqrt(mean(x^2) + eps)) * weight` (NO +1.0; MLX checkpoint already has +1.0).
// Also zero-pads rows `M_valid .. M_padded - 1` for static-tile Metal 4 NAX `matmul2d`.
kernel void sq38_rmsnorm_f32_to_f16(
    device const float*       x_in_f32  [[buffer(0)]], // [M_valid, D]
    device const float*       norm_w    [[buffer(1)]], // [D]
    device half*              x_out_f16 [[buffer(2)]], // [M_padded, D]
    constant Sq38RmsNormParams& p       [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],         // 0..255
    uint sgid [[simdgroup_index_in_threadgroup]],      // 0..7
    uint lane [[thread_index_in_simdgroup]]            // 0..31
) {
    if (tgid >= p.M_padded) return;

    device half* dst_row = x_out_f16 + ulong(tgid) * p.D;
    if (tgid >= p.M_valid) {
        const uint d4_cnt = p.D >> 2u;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            reinterpret_cast<device half4*>(dst_row)[i] = half4(0.0h);
        }
        return;
    }

    threadgroup float tg_sq[8];
    device const float* src_row = x_in_f32 + ulong(tgid) * p.D;
    const uint d4_cnt = p.D >> 2u;

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 v = reinterpret_cast<device const float4*>(src_row)[i];
        sq_acc = fma(v.x, v.x, sq_acc);
        sq_acc = fma(v.y, v.y, sq_acc);
        sq_acc = fma(v.z, v.z, sq_acc);
        sq_acc = fma(v.w, v.w, sq_acc);
    }

    float sg_sum = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = 0.0f;
    if (sgid == 0u) {
        float v = (lane < 8u) ? tg_sq[lane] : 0.0f;
        total_sq = simd_sum(v);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(total_sq / float(p.D) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 v = reinterpret_cast<device const float4*>(src_row)[i];
        float4 w = reinterpret_cast<device const float4*>(norm_w)[i];
        reinterpret_cast<device half4*>(dst_row)[i] = sq38_clamp_f16x4((v * inv_rms) * w);
    }
}

/* ============================================================================
 * 3. Single-Token Decode (`M = 1`) Coalesced SIMD-Group 4-Bit Affine GEMV Suite
 *    - 2 SIMD-groups/TG (64 threads) = 8 output rows/TG, 0 B TG SRAM, 0 barriers!
 *    - 512 B coalesced `uint4` loads across 32 lanes (1,024 weights/step/SG, MLP=4..8)
 * ============================================================================ */

// High-Occupancy 4-Row/SG (`128` threads = `4` SGs = `16` rows/TG, `< 40` regs/thread -> max waves/core!)
inline float4 sq38_gemv4_rows(
    device const uint4* w_u4,   // [rows_padded, cols / 32]
    device const half*  scales, // [rows_padded, cols / 64]
    device const half*  biases, // [rows_padded, cols / 64]
    device const half*  x,      // [cols]
    uint row0,
    uint cols,
    uint lane
) {
    const uint u4_per_row  = cols >> 5u;
    const uint grp_per_row = cols >> 6u;
    const uint k_iters     = cols >> 10u; // Exact since cols % 1024 == 0 for all matrices

    device const uint4* r0_w = w_u4 + ulong(row0 + 0u) * u4_per_row;
    device const uint4* r1_w = w_u4 + ulong(row0 + 1u) * u4_per_row;
    device const uint4* r2_w = w_u4 + ulong(row0 + 2u) * u4_per_row;
    device const uint4* r3_w = w_u4 + ulong(row0 + 3u) * u4_per_row;

    device const half* r0_s = scales + ulong(row0 + 0u) * grp_per_row;
    device const half* r1_s = scales + ulong(row0 + 1u) * grp_per_row;
    device const half* r2_s = scales + ulong(row0 + 2u) * grp_per_row;
    device const half* r3_s = scales + ulong(row0 + 3u) * grp_per_row;

    device const half* r0_b = biases + ulong(row0 + 0u) * grp_per_row;
    device const half* r1_b = biases + ulong(row0 + 1u) * grp_per_row;
    device const half* r2_b = biases + ulong(row0 + 2u) * grp_per_row;
    device const half* r3_b = biases + ulong(row0 + 3u) * grp_per_row;

    float4 acc(0.0f);

    for (uint it = 0u; it < k_iters; ++it) {
        const uint u4_idx  = it * 32u + lane;
        const uint grp_idx = it * 16u + (lane >> 1u);
        device const half4* x_ptr = reinterpret_cast<device const half4*>(x + u4_idx * 32u);

        const half4 xh0 = x_ptr[0];
        const half4 xh1 = x_ptr[1];
        const half4 xh2 = x_ptr[2];
        const half4 xh3 = x_ptr[3];
        const half4 xh4 = x_ptr[4];
        const half4 xh5 = x_ptr[5];
        const half4 xh6 = x_ptr[6];
        const half4 xh7 = x_ptr[7];

        const float4 x_s4  = ((float4(xh0) + float4(xh1)) + (float4(xh2) + float4(xh3))) +
                             ((float4(xh4) + float4(xh5)) + (float4(xh6) + float4(xh7)));
        const float  x_sum = (x_s4.x + x_s4.y) + (x_s4.z + x_s4.w);

        const uint4 q0 = r0_w[u4_idx];
        const uint4 q1 = r1_w[u4_idx];
        const uint4 q2 = r2_w[u4_idx];
        const uint4 q3 = r3_w[u4_idx];

        acc.x = fma(float(r0_s[grp_idx]), sq38_dot_u4_xh8(q0, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                    fma(float(r0_b[grp_idx]), x_sum, acc.x));
        acc.y = fma(float(r1_s[grp_idx]), sq38_dot_u4_xh8(q1, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                    fma(float(r1_b[grp_idx]), x_sum, acc.y));
        acc.z = fma(float(r2_s[grp_idx]), sq38_dot_u4_xh8(q2, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                    fma(float(r2_b[grp_idx]), x_sum, acc.z));
        acc.w = fma(float(r3_s[grp_idx]), sq38_dot_u4_xh8(q3, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                    fma(float(r3_b[grp_idx]), x_sum, acc.w));
    }

    return float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
}

struct Sq38GemvParams {
    uint rows;
    uint cols;
};

// Standard FP32 output GEMV (`128` threads = `4` SGs = `16` rows/TG, e.g. `248,320 x 5,120` LM Head)
kernel void sq38_q4_gemv_f32(
    device const uint4*      w_u4   [[buffer(0)]],
    device const half*       scales [[buffer(1)]],
    device const half*       biases [[buffer(2)]],
    device const half*       x_in   [[buffer(3)]],
    device float*            y_out  [[buffer(4)]],
    constant Sq38GemvParams& p      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 4u + sgid) * 4u;
    if (row0 >= p.rows) return;
    float4 res = sq38_gemv4_rows(w_u4, scales, biases, x_in, row0, p.cols, lane);
    if (lane == 0u) {
        *reinterpret_cast<device float4*>(y_out + row0) = res;
    }
}

// Fused Output-Projection + FP32 Residual Add (`128` threads = `4` SGs = `16` rows/TG)
kernel void sq38_q4_gemv_res_f32(
    device const uint4*      w_u4          [[buffer(0)]],
    device const half*       scales        [[buffer(1)]],
    device const half*       biases        [[buffer(2)]],
    device const half*       x_in          [[buffer(3)]],
    device float*            x_res_inout   [[buffer(4)]],
    constant Sq38GemvParams& p             [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 4u + sgid) * 4u;
    if (row0 >= p.rows) return;
    float4 res = sq38_gemv4_rows(w_u4, scales, biases, x_in, row0, p.cols, lane);
    if (lane == 0u) {
        device float4* dst = reinterpret_cast<device float4*>(x_res_inout + row0);
        *dst = *dst + res;
    }
}

// Fused `gate_proj + up_proj + SiLU` (`[17408, 5120] -> mid_f16 [17408]`): reads `x_norm` ONCE!
kernel void sq38_q4_gemv_fused_gate_up_silu(
    device const uint4*      gate_w [[buffer(0)]],
    device const half*       gate_s [[buffer(1)]],
    device const half*       gate_b [[buffer(2)]],
    device const uint4*      up_w   [[buffer(3)]],
    device const half*       up_s   [[buffer(4)]],
    device const half*       up_b   [[buffer(5)]],
    device const half*       x_in   [[buffer(6)]],
    device half*             mid_out[[buffer(7)]],
    constant Sq38GemvParams& p      [[buffer(8)]], // rows = 17408, cols = 5120
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],  // 0..1
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tgid * 2u + sgid) * 4u;
    if (row0 >= p.rows) return;

    const uint u4_per_row  = p.cols >> 5u;
    const uint grp_per_row = p.cols >> 6u;
    const uint k_iters     = p.cols >> 10u;

    device const uint4* g0_w = gate_w + ulong(row0 + 0u) * u4_per_row;
    device const uint4* g1_w = gate_w + ulong(row0 + 1u) * u4_per_row;
    device const uint4* g2_w = gate_w + ulong(row0 + 2u) * u4_per_row;
    device const uint4* g3_w = gate_w + ulong(row0 + 3u) * u4_per_row;

    device const uint4* u0_w = up_w + ulong(row0 + 0u) * u4_per_row;
    device const uint4* u1_w = up_w + ulong(row0 + 1u) * u4_per_row;
    device const uint4* u2_w = up_w + ulong(row0 + 2u) * u4_per_row;
    device const uint4* u3_w = up_w + ulong(row0 + 3u) * u4_per_row;

    device const half* g0_s = gate_s + ulong(row0 + 0u) * grp_per_row;
    device const half* g1_s = gate_s + ulong(row0 + 1u) * grp_per_row;
    device const half* g2_s = gate_s + ulong(row0 + 2u) * grp_per_row;
    device const half* g3_s = gate_s + ulong(row0 + 3u) * grp_per_row;

    device const half* u0_s = up_s + ulong(row0 + 0u) * grp_per_row;
    device const half* u1_s = up_s + ulong(row0 + 1u) * grp_per_row;
    device const half* u2_s = up_s + ulong(row0 + 2u) * grp_per_row;
    device const half* u3_s = up_s + ulong(row0 + 3u) * grp_per_row;

    device const half* g0_b = gate_b + ulong(row0 + 0u) * grp_per_row;
    device const half* g1_b = gate_b + ulong(row0 + 1u) * grp_per_row;
    device const half* g2_b = gate_b + ulong(row0 + 2u) * grp_per_row;
    device const half* g3_b = gate_b + ulong(row0 + 3u) * grp_per_row;

    device const half* u0_b = up_b + ulong(row0 + 0u) * grp_per_row;
    device const half* u1_b = up_b + ulong(row0 + 1u) * grp_per_row;
    device const half* u2_b = up_b + ulong(row0 + 2u) * grp_per_row;
    device const half* u3_b = up_b + ulong(row0 + 3u) * grp_per_row;

    float4 g_acc(0.0f);
    float4 u_acc(0.0f);

    for (uint it = 0u; it < k_iters; ++it) {
        const uint u4_idx  = it * 32u + lane;
        const uint grp_idx = it * 16u + (lane >> 1u);
        device const half4* x_ptr = reinterpret_cast<device const half4*>(x_in + u4_idx * 32u);

        const half4 xh0 = x_ptr[0];
        const half4 xh1 = x_ptr[1];
        const half4 xh2 = x_ptr[2];
        const half4 xh3 = x_ptr[3];
        const half4 xh4 = x_ptr[4];
        const half4 xh5 = x_ptr[5];
        const half4 xh6 = x_ptr[6];
        const half4 xh7 = x_ptr[7];

        const float4 x_s4  = ((float4(xh0) + float4(xh1)) + (float4(xh2) + float4(xh3))) +
                             ((float4(xh4) + float4(xh5)) + (float4(xh6) + float4(xh7)));
        const float  x_sum = (x_s4.x + x_s4.y) + (x_s4.z + x_s4.w);

        const uint4 qg0 = g0_w[u4_idx];
        const uint4 qg1 = g1_w[u4_idx];
        const uint4 qg2 = g2_w[u4_idx];
        const uint4 qg3 = g3_w[u4_idx];

        const uint4 qu0 = u0_w[u4_idx];
        const uint4 qu1 = u1_w[u4_idx];
        const uint4 qu2 = u2_w[u4_idx];
        const uint4 qu3 = u3_w[u4_idx];

        g_acc.x = fma(float(g0_s[grp_idx]), sq38_dot_u4_xh8(qg0, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                      fma(float(g0_b[grp_idx]), x_sum, g_acc.x));
        g_acc.y = fma(float(g1_s[grp_idx]), sq38_dot_u4_xh8(qg1, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                      fma(float(g1_b[grp_idx]), x_sum, g_acc.y));
        g_acc.z = fma(float(g2_s[grp_idx]), sq38_dot_u4_xh8(qg2, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                      fma(float(g2_b[grp_idx]), x_sum, g_acc.z));
        g_acc.w = fma(float(g3_s[grp_idx]), sq38_dot_u4_xh8(qg3, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                      fma(float(g3_b[grp_idx]), x_sum, g_acc.w));

        u_acc.x = fma(float(u0_s[grp_idx]), sq38_dot_u4_xh8(qu0, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                      fma(float(u0_b[grp_idx]), x_sum, u_acc.x));
        u_acc.y = fma(float(u1_s[grp_idx]), sq38_dot_u4_xh8(qu1, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                      fma(float(u1_b[grp_idx]), x_sum, u_acc.y));
        u_acc.z = fma(float(u2_s[grp_idx]), sq38_dot_u4_xh8(qu2, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                      fma(float(u2_b[grp_idx]), x_sum, u_acc.z));
        u_acc.w = fma(float(u3_s[grp_idx]), sq38_dot_u4_xh8(qu3, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7),
                      fma(float(u3_b[grp_idx]), x_sum, u_acc.w));
    }

    float4 gv(simd_sum(g_acc.x), simd_sum(g_acc.y), simd_sum(g_acc.z), simd_sum(g_acc.w));
    float4 uv(simd_sum(u_acc.x), simd_sum(u_acc.y), simd_sum(u_acc.z), simd_sum(u_acc.w));

    if (lane == 0u) {
        float4 prod = (gv / (float4(1.0f) + exp(-gv))) * uv;
        *reinterpret_cast<device half4*>(mid_out + row0) = sq38_clamp_f16x4(prod);
    }
}

struct Sq38LinInGemvParams {
    uint qkv_rows; // 10240
    uint z_rows;   // 6144
    uint ab_rows;  // 64 (padded from 48)
    uint cols;     // 5120
};

// Single-Dispatch Fused GatedDeltaNet Input GEMV (`128` threads = `4` SGs = `16` rows/TG)
kernel void sq38_q4_gemv_fused_lin_in(
    device const uint4* qkv_w [[buffer(0)]],
    device const half*  qkv_s [[buffer(1)]],
    device const half*  qkv_b [[buffer(2)]],
    device const uint4* z_w   [[buffer(3)]],
    device const half*  z_s   [[buffer(4)]],
    device const half*  z_b   [[buffer(5)]],
    device const uint4* a_w   [[buffer(6)]],
    device const half*  a_s   [[buffer(7)]],
    device const half*  a_b   [[buffer(8)]],
    device const uint4* b_w   [[buffer(9)]],
    device const half*  b_s   [[buffer(10)]],
    device const half*  b_b   [[buffer(11)]],
    device const half*  x_in  [[buffer(12)]],
    device half*        qkv_o [[buffer(13)]],
    device half*        z_o   [[buffer(14)]],
    device float*       a_o   [[buffer(15)]],
    device float*       b_o   [[buffer(16)]],
    constant Sq38LinInGemvParams& p [[buffer(17)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow = (tgid * 4u + sgid) * 4u;
    const uint b1   = p.qkv_rows;
    const uint b2   = b1 + p.z_rows;
    const uint b3   = b2 + p.ab_rows;
    const uint b4   = b3 + p.ab_rows;
    if (vrow >= b4) return;

    if (vrow < b1) {
        float4 res = sq38_gemv4_rows(qkv_w, qkv_s, qkv_b, x_in, vrow, p.cols, lane);
        if (lane == 0u) *reinterpret_cast<device half4*>(qkv_o + vrow) = sq38_clamp_f16x4(res);
    } else if (vrow < b2) {
        uint r = vrow - b1;
        float4 res = sq38_gemv4_rows(z_w, z_s, z_b, x_in, r, p.cols, lane);
        if (lane == 0u) *reinterpret_cast<device half4*>(z_o + r) = sq38_clamp_f16x4(res);
    } else if (vrow < b3) {
        uint r = vrow - b2;
        float4 res = sq38_gemv4_rows(a_w, a_s, a_b, x_in, r, p.cols, lane);
        if (lane == 0u) *reinterpret_cast<device float4*>(a_o + r) = res;
    } else {
        uint r = vrow - b3;
        float4 res = sq38_gemv4_rows(b_w, b_s, b_b, x_in, r, p.cols, lane);
        if (lane == 0u) *reinterpret_cast<device float4*>(b_o + r) = res;
    }
}

struct Sq38AttnQkvGemvParams {
    uint q_rows;  // 12288
    uint kv_rows; // 1024
    uint cols;    // 5120
    uint pad;
};

// Single-Dispatch Fused Full-Attention QKV GEMV (`128` threads = `4` SGs = `16` rows/TG)
kernel void sq38_q4_gemv_fused_attn_qkv(
    device const uint4* q_w   [[buffer(0)]],
    device const half*  q_s   [[buffer(1)]],
    device const half*  q_b   [[buffer(2)]],
    device const uint4* k_w   [[buffer(3)]],
    device const half*  k_s   [[buffer(4)]],
    device const half*  k_b   [[buffer(5)]],
    device const uint4* v_w   [[buffer(6)]],
    device const half*  v_s   [[buffer(7)]],
    device const half*  v_b   [[buffer(8)]],
    device const half*  x_in  [[buffer(9)]],
    device half*        q_out [[buffer(10)]],
    device half*        k_out [[buffer(11)]],
    device half*        v_out [[buffer(12)]],
    constant Sq38AttnQkvGemvParams& p [[buffer(13)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow = (tgid * 4u + sgid) * 4u;
    const uint b1   = p.q_rows;
    const uint b2   = b1 + p.kv_rows;
    const uint b3   = b2 + p.kv_rows;
    if (vrow >= b3) return;

    if (vrow < b1) {
        float4 res = sq38_gemv4_rows(q_w, q_s, q_b, x_in, vrow, p.cols, lane);
        if (lane == 0u) *reinterpret_cast<device half4*>(q_out + vrow) = sq38_clamp_f16x4(res);
    } else if (vrow < b2) {
        uint r = vrow - b1;
        float4 res = sq38_gemv4_rows(k_w, k_s, k_b, x_in, r, p.cols, lane);
        if (lane == 0u) *reinterpret_cast<device half4*>(k_out + r) = sq38_clamp_f16x4(res);
    } else {
        uint r = vrow - b2;
        float4 res = sq38_gemv4_rows(v_w, v_s, v_b, x_in, r, p.cols, lane);
        if (lane == 0u) *reinterpret_cast<device half4*>(v_out + r) = sq38_clamp_f16x4(res);
    }
}

/* ============================================================================
 * 4. Batched Prefill Metal 4 `NAX` (`mpp::tensor_ops::matmul2d`) 4-Bit Affine GEMM
 *    - 8.0 KiB Threadgroup Tile (`W_tile[64][64]`) -> 4 TGs/core across all 40 M5 Max cores!
 *    - `out_mode`: 0 -> `half` output, 1 -> `float` output, 2 -> fused `+ float` residual!
 * ============================================================================ */

#define SQ38_GEMM_OUT_F16     0u
#define SQ38_GEMM_OUT_F32     1u
#define SQ38_GEMM_OUT_RES_F32 2u

struct Sq38NaxGemmParams {
    uint M_valid;
    uint M_padded;
    uint N;         // rows_padded (multiple of 64)
    uint K;         // cols (multiple of 1024)
    uint out_mode;  // SQ38_GEMM_OUT_*
    uint pad0;
    uint pad1;
    uint pad2;
};

#if defined(__HAVE_TENSOR__)
template <int M_TILE>
inline void sq38_q4_gemm_nax_impl(
    device const uint*       w_u32,
    device const half*       scales,
    device const half*       biases,
    device const half*       x_in,
    device half*             y_f16,
    device float*            y_f32,
    constant Sq38NaxGemmParams& p,
    threadgroup half         (&W_tile)[64][64],
    uint2                    tgid,
    uint                     tid
) {
    const uint m0 = tgid.y * uint(M_TILE);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.N) return;

    device half* x_mut = const_cast<device half*>(x_in);
    tensor<device half, dextents<int, 2>, tensor_inline> tX(x_mut, dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup half, extents<int, 64, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 64, 64>());

    constexpr auto desc = matmul2d_descriptor(
        M_TILE, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto sX0 = tX.template slice<64, M_TILE>(0, int(m0));
    auto cT  = op.template get_destination_cooperative_tensor<decltype(sX0), decltype(tW), float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint u32_per_row = p.K >> 3u;
    const uint grp_per_row = p.K >> 6u;
    const uint r_grp       = tid >> 3u; // 0..15
    const uint sub         = tid & 7u;  // 0..7 (8 uint32s = 64 weights contiguous per row!)

    for (uint grp = 0u; grp < grp_per_row; ++grp) {
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 4u; ++q) {
            const uint  r_loc  = q * 16u + r_grp;
            const uint  r_glob = n0 + r_loc;
            const uint  u32_v  = w_u32[ulong(r_glob) * u32_per_row + grp * 8u + sub];
            const float sc     = float(scales[ulong(r_glob) * grp_per_row + grp]);
            const float bi     = float(biases[ulong(r_glob) * grp_per_row + grp]);

            half4 n_lo, n_hi;
            sq38_unpack_u32_nibbles_h4(u32_v, n_lo, n_hi);

            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 8u]);
            dst[0] = sq38_clamp_f16x4(fma(float4(n_lo), float4(sc), float4(bi)));
            dst[1] = sq38_clamp_f16x4(fma(float4(n_hi), float4(sc), float4(bi)));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX.template slice<64, M_TILE>(int(grp * 64u), int(m0));
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_valid && n_glob < p.N) {
                ulong out_idx = ulong(m_glob) * p.N + n_glob;
                if (p.out_mode == SQ38_GEMM_OUT_F16) {
                    y_f16[out_idx] = sq38_clamp_f16(cT[i]);
                } else if (p.out_mode == SQ38_GEMM_OUT_F32) {
                    y_f32[out_idx] = cT[i];
                } else {
                    y_f32[out_idx] += cT[i];
                }
            }
        }
    }
}
#endif

kernel void sq38_q4_gemm_nax_m16(
    device const uint*          w_u32  [[buffer(0)]],
    device const half*          scales [[buffer(1)]],
    device const half*          biases [[buffer(2)]],
    device const half*          x_in   [[buffer(3)]],
    device half*                y_f16  [[buffer(4)]],
    device float*               y_f32  [[buffer(5)]],
    constant Sq38NaxGemmParams& p      [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sq38_q4_gemm_nax_impl<16>(w_u32, scales, biases, x_in, y_f16, y_f32, p, W_tile, tgid, tid);
#else
    (void)w_u32; (void)scales; (void)biases; (void)x_in; (void)y_f16; (void)y_f32; (void)p; (void)W_tile; (void)tgid; (void)tid;
#endif
}

kernel void sq38_q4_gemm_nax_m64(
    device const uint*          w_u32  [[buffer(0)]],
    device const half*          scales [[buffer(1)]],
    device const half*          biases [[buffer(2)]],
    device const half*          x_in   [[buffer(3)]],
    device half*                y_f16  [[buffer(4)]],
    device float*               y_f32  [[buffer(5)]],
    constant Sq38NaxGemmParams& p      [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sq38_q4_gemm_nax_impl<64>(w_u32, scales, biases, x_in, y_f16, y_f32, p, W_tile, tgid, tid);
#else
    (void)w_u32; (void)scales; (void)biases; (void)x_in; (void)y_f16; (void)y_f32; (void)p; (void)W_tile; (void)tgid; (void)tid;
#endif
}

struct Sq38SwigluParams {
    uint M_valid;
    uint M_padded;
    uint d_ff;
    uint pad;
};

// Vectorized SwiGLU (`silu(gate_f32) * up_f32 -> mid_f16`, zero-padding `M_valid .. M_padded - 1`)
kernel void sq38_swiglu_f32_to_f16(
    device const float*      gate_f32 [[buffer(0)]], // [M_valid, d_ff]
    device const float*      up_f32   [[buffer(1)]], // [M_valid, d_ff]
    device half*             mid_f16  [[buffer(2)]], // [M_padded, d_ff]
    constant Sq38SwigluParams& p      [[buffer(3)]],
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
    reinterpret_cast<device half4*>(mid_f16)[off4] = sq38_clamp_f16x4(prod);
}

/* ============================================================================
 * 5. 48-Layer Gated DeltaNet (`linear_attention`) Kernels (Both Prefill & M=1 Decode!)
 * ============================================================================ */

struct Sq38Conv1dParams {
    uint M;
    uint channels; // 10240
};

// Exact Causal Depthwise `conv1d(k=4) + SiLU` updating persistent `conv_state [3, 10240]`
kernel void sq38_deltanet_conv1d_silu(
    device const half*       qkv_in     [[buffer(0)]], // [M, 10240]
    device const float4*     conv_w     [[buffer(1)]], // [10240] (each float4 = taps [t-3, t-2, t-1, t])
    device half*             conv_state [[buffer(2)]], // [3, 10240] for this linear layer
    device half*             qkv_out    [[buffer(3)]], // [M, 10240]
    constant Sq38Conv1dParams& p        [[buffer(4)]],
    uint gid [[thread_position_in_grid]]               // 0..10239
) {
    if (gid >= p.channels) return;

    const uint C  = p.channels;
    const float4 w = conv_w[gid];

    float s0 = float(conv_state[0u * C + gid]);
    float s1 = float(conv_state[1u * C + gid]);
    float s2 = float(conv_state[2u * C + gid]);

    for (uint t = 0u; t < p.M; ++t) {
        const ulong idx = ulong(t) * C + gid;
        const float s3  = float(qkv_in[idx]);
        const float raw = fma(w.x, s0, fma(w.y, s1, fma(w.z, s2, w.w * s3)));
        qkv_out[idx]    = sq38_clamp_f16(sq38_silu(raw));
        s0 = s1;
        s1 = s2;
        s2 = s3;
    }

    conv_state[0u * C + gid] = sq38_clamp_f16(s0);
    conv_state[1u * C + gid] = sq38_clamp_f16(s1);
    conv_state[2u * C + gid] = sq38_clamp_f16(s2);
}

struct Sq38DeltaNetParams {
    uint  M_valid;
    uint  M_padded;
    float eps;
    uint  ab_stride; // 64 (rows_padded of in_proj_a / in_proj_b)
};

// Exact 48-Head Recurrent State Update (`FP32 [48, 128, 128]`) + L2 QK Norm + Gates + `RMSNormGated`!
// Grid: 48 Threadgroups (`tgid = hv = 0..47`), 512 threads/TG (`qtr = tid >> 7` in 0..3, `dv = tid & 127`).
// Keeps `S` in `float s_reg[32]` registers across all `M_valid` tokens of a prefill chunk!
kernel void sq38_deltanet_recurrent_gated_norm(
    device const half*         qkv_conv  [[buffer(0)]], // [M_valid, 10240] -> q[2048], k[2048], v[6144]
    device const half*         z_in      [[buffer(1)]], // [M_valid, 6144]
    device const float*        a_in      [[buffer(2)]], // [M_valid, 64]
    device const float*        b_in      [[buffer(3)]], // [M_valid, 64]
    device const float*        A_log     [[buffer(4)]], // [64]
    device const float*        dt_bias   [[buffer(5)]], // [64]
    device const float*        lin_norm  [[buffer(6)]], // [128]
    device float*              rec_state [[buffer(7)]], // [48, 128, 128] for this linear layer
    device half*               y_out     [[buffer(8)]], // [M_padded, 6144]
    constant Sq38DeltaNetParams& p       [[buffer(9)]],
    uint hv   [[threadgroup_position_in_grid]],         // 0..47
    uint tid  [[thread_index_in_threadgroup]],          // 0..511
    uint sgid [[simdgroup_index_in_threadgroup]],       // 0..15
    uint lane [[thread_index_in_simdgroup]]             // 0..31
) {
    if (hv >= 48u) return;

    const uint hk      = hv / 3u;     // GQA broadcast 16 -> 48
    const uint dv      = tid & 127u;  // 0..127
    const uint qtr     = tid >> 7u;   // 0..3
    const uint dk_base = qtr * 32u;   // 0, 32, 64, 96

    threadgroup float q_hat[128];
    threadgroup float k_hat[128];
    threadgroup float tg_red[4][128];
    threadgroup float tg_qk_sq[8];
    threadgroup float tg_scalars[4]; // [0]=decay, [1]=beta, [2]=q_inv, [3]=k_inv

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
        device const half* qkv_t = qkv_conv + ulong(t) * 10240ul;
        device const half* q_raw = qkv_t + hk * 128u;
        device const half* k_raw = qkv_t + 2048u + hk * 128u;
        device const half* v_raw = qkv_t + 4096u + hv * 128u;
        device const half* z_raw = z_in + ulong(t) * 6144ul + hv * 128u;

        // 1. Load q[hk], k[hk] & compute L2 norms + gates
        float q_v = 0.0f;
        float k_v = 0.0f;
        if (sgid < 4u) {
            q_v = float(q_raw[tid]);
            q_hat[tid] = q_v;
            float q_sq = simd_sum(q_v * q_v);
            if (lane == 0u) tg_qk_sq[sgid] = q_sq;
        } else if (sgid < 8u) {
            uint idx = tid - 128u;
            k_v = float(k_raw[idx]);
            k_hat[idx] = k_v;
            float k_sq = simd_sum(k_v * k_v);
            if (lane == 0u) tg_qk_sq[sgid] = k_sq;
        }
        if (tid == 256u) {
            float a_val = a_in[ulong(t) * p.ab_stride + hv];
            float b_val = b_in[ulong(t) * p.ab_stride + hv];
            float g     = -exp(a_log_hv) * sq38_softplus(a_val + dt_bias_hv);
            tg_scalars[0] = exp(g);
            tg_scalars[1] = sq38_sigmoid(b_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float q_tot = (lane < 4u) ? tg_qk_sq[lane] : 0.0f;
            float k_tot = (lane < 4u) ? tg_qk_sq[4u + lane] : 0.0f;
            q_tot = simd_sum(q_tot);
            k_tot = simd_sum(k_tot);
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

        // 2. Decay S & compute partial `kv_mem = S^T @ k_hat`
        float kv_part = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 32u; ++i) {
            float sv = s_reg[i] * decay;
            s_reg[i] = sv;
            kv_part  = fma(sv, k_hat[dk_base + i], kv_part);
        }
        tg_red[qtr][dv] = kv_part;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float kv_mem = (tg_red[0][dv] + tg_red[1][dv]) + (tg_red[2][dv] + tg_red[3][dv]);
        const float delta  = beta * (float(v_raw[dv]) - kv_mem);

        // 3. Rank-1 delta update `S += k_hat (outer) delta` & partial `out = S^T @ q_hat`
        float out_part = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 32u; ++i) {
            float sv = fma(k_hat[dk_base + i], delta, s_reg[i]);
            s_reg[i] = sv;
            out_part = fma(sv, q_hat[dk_base + i], out_part);
        }
        tg_red[qtr][dv] = out_part;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. `Qwen3NextRMSNormGated(128, eps)` * `silu(z)` (uniform barriers across all 512 threads!)
        float o_val = 0.0f;
        if (sgid < 4u) {
            o_val = (tg_red[0][dv] + tg_red[1][dv]) + (tg_red[2][dv] + tg_red[3][dv]);
            q_hat[dv] = o_val; // reuse q_hat[128] to hold o_val[128]
            float o_sq = simd_sum(o_val * o_val);
            if (lane == 0u) tg_qk_sq[sgid] = o_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float o_tot = (lane < 4u) ? tg_qk_sq[lane] : 0.0f;
            o_tot = simd_sum(o_tot);
            if (lane == 0u) {
                tg_scalars[0] = rsqrt(o_tot * (1.0f / 128.0f) + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 128u) {
            float normed = q_hat[dv] * tg_scalars[0] * norm_w_dv;
            float gated  = normed * sq38_silu(float(z_raw[dv]));
            y_out[ulong(t) * 6144ul + hv * 128u + dv] = sq38_clamp_f16(gated);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Zero-pad tail rows `M_valid .. M_padded - 1` so NAX `lin_out_proj` has clean zeroed tail
    if (tid < 128u) {
        for (uint t = p.M_valid; t < p.M_padded; ++t) {
            y_out[ulong(t) * 6144ul + hv * 128u + dv] = half(0.0h);
        }
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 32u; ++i) {
        state_hv[(dk_base + i) * 128u + dv] = s_reg[i];
    }
}

// Fast Zero-Barrier Batched Prefill DeltaNet Pipeline (`M > 1`):
//   Stage A: Parallel Q/K L2-norm + Gate precompute across all `(33, M)` threadgroups
//   Stage B: 192-TG (`4` TGs/head x `48` heads) Zero-Barrier `simd_sum` Recurrence (`0 B` TG SRAM!)
//   Stage C: Parallel `RMSNormGated * silu(z)` across all `(48, M_padded)` threadgroups

kernel void sq38_deltanet_prep_batch(
    device const half*          qkv_conv   [[buffer(0)]], // [M_valid, 10240]
    device const float*         a_in       [[buffer(1)]], // [M_valid, ab_stride]
    device const float*         b_in       [[buffer(2)]], // [M_valid, ab_stride]
    device const float*         A_log      [[buffer(3)]], // [64]
    device const float*         dt_bias    [[buffer(4)]], // [64]
    device float*               qk_hat_out [[buffer(5)]], // [M_valid, 4096] (q_hat[2048] + k_hat[2048])
    device float*               gates_out  [[buffer(6)]], // [M_valid, 96]   (decay[48] + beta[48])
    constant Sq38DeltaNetParams& p         [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],          // (0..32, 0..M_valid-1)
    uint  tid  [[thread_index_in_threadgroup]],           // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]],        // 0..3
    uint  lane [[thread_index_in_simdgroup]]              // 0..31
) {
    const uint item = tgid.x;
    const uint t    = tgid.y;
    if (item >= 33u || t >= p.M_valid) return;

    if (item == 32u) {
        if (tid < 48u) {
            float a_val = a_in[ulong(t) * p.ab_stride + tid];
            float b_val = b_in[ulong(t) * p.ab_stride + tid];
            float g     = -exp(A_log[tid]) * sq38_softplus(a_val + dt_bias[tid]);
            gates_out[ulong(t) * 96ul + tid]       = exp(g);
            gates_out[ulong(t) * 96ul + 48ul + tid]= sq38_sigmoid(b_val);
        }
        return;
    }

    threadgroup float tg_sq[4];
    const bool is_q = (item < 16u);
    const uint hk   = is_q ? item : (item - 16u);
    const float v   = float(qkv_conv[ulong(t) * 10240ul + (is_q ? 0ul : 2048ul) + hk * 128u + tid]);

    float sg_sq = simd_sum(v * v);
    if (lane == 0u) tg_sq[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = (lane < 4u) ? tg_sq[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) {
            constexpr float kInvSqrt128 = 0.08838834764831845f;
            float inv = rsqrt(tot + p.eps);
            tg_sq[0]  = is_q ? (inv * kInvSqrt128) : inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    qk_hat_out[ulong(t) * 4096ul + (is_q ? 0ul : 2048ul) + hk * 128u + tid] = v * tg_sq[0];
}

kernel void sq38_deltanet_rec_batch(
    device const half*          qkv_conv  [[buffer(0)]], // [M_valid, 10240] (reads v[4096..10239])
    device const float4*        qk_hat    [[buffer(1)]], // [M_valid, 1024]  (float4: q_hat[512] + k_hat[512])
    device const float*         gates_in  [[buffer(2)]], // [M_valid, 96]
    device float*               rec_state [[buffer(3)]], // [48, 128, 128]
    device float*               raw_out   [[buffer(4)]], // [M_valid, 6144]
    constant Sq38DeltaNetParams& p        [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],         // (0..3, 0..47) = 192 TGs!
    uint  sgid [[simdgroup_index_in_threadgroup]],       // 0..3
    uint  lane [[thread_index_in_simdgroup]]             // 0..31
) {
    const uint col_blk = tgid.x;
    const uint hv      = tgid.y;
    if (col_blk >= 4u || hv >= 48u) return;

    const uint hk  = hv / 3u;
    const uint dv0 = col_blk * 32u + sgid * 8u; // 8 dv columns owned by this SIMD-group
    const uint dk0 = lane * 4u;                 // 4 dk rows owned by this lane (32 * 4 = 128!)

    device float* state_hv = rec_state + ulong(hv) * 16384ul;

    float4 s_col[8];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        uint dv = dv0 + c;
        s_col[c] = float4(state_hv[(dk0 + 0u) * 128u + dv],
                          state_hv[(dk0 + 1u) * 128u + dv],
                          state_hv[(dk0 + 2u) * 128u + dv],
                          state_hv[(dk0 + 3u) * 128u + dv]);
    }

    for (uint t = 0u; t < p.M_valid; ++t) {
        const float4 q4   = qk_hat[ulong(t) * 1024ul + hk * 32u + lane];
        const float4 k4   = qk_hat[ulong(t) * 1024ul + 512ul + hk * 32u + lane];
        const float decay = gates_in[ulong(t) * 96ul + hv];
        const float beta  = gates_in[ulong(t) * 96ul + 48ul + hv];
        device const half* v_ptr = qkv_conv + ulong(t) * 10240ul + 4096ul + hv * 128u + dv0;
        device float*      o_ptr = raw_out + ulong(t) * 6144ul + hv * 128u + dv0;

        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; ++c) {
            float4 sc    = s_col[c] * decay;
            float kv_mem = simd_sum(dot(sc, k4));
            float delta  = beta * (float(v_ptr[c]) - kv_mem);
            sc           = fma(k4, float4(delta), sc);
            s_col[c]     = sc;
            float ov     = simd_sum(dot(sc, q4));
            if (lane == 0u) {
                o_ptr[c] = ov;
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        uint dv = dv0 + c;
        state_hv[(dk0 + 0u) * 128u + dv] = s_col[c].x;
        state_hv[(dk0 + 1u) * 128u + dv] = s_col[c].y;
        state_hv[(dk0 + 2u) * 128u + dv] = s_col[c].z;
        state_hv[(dk0 + 3u) * 128u + dv] = s_col[c].w;
    }
}

kernel void sq38_deltanet_post_batch(
    device const float*         raw_out  [[buffer(0)]], // [M_valid, 6144]
    device const half*          z_in     [[buffer(1)]], // [M_valid, 6144]
    device const float*         lin_norm [[buffer(2)]], // [128]
    device half*                y_out    [[buffer(3)]], // [M_padded, 6144]
    constant Sq38DeltaNetParams& p       [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],        // (0..47, 0..M_padded-1)
    uint  tid  [[thread_index_in_threadgroup]],         // 0..127
    uint  sgid [[simdgroup_index_in_threadgroup]],      // 0..3
    uint  lane [[thread_index_in_simdgroup]]            // 0..31
) {
    const uint hv = tgid.x;
    const uint t  = tgid.y;
    if (hv >= 48u || t >= p.M_padded) return;

    const ulong idx = ulong(t) * 6144ul + hv * 128u + tid;
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
        float tot = (lane < 4u) ? tg_sq[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot * (1.0f / 128.0f) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float normed = ov * tg_sq[0] * lin_norm[tid];
    const float gated  = normed * sq38_silu(float(z_in[idx]));
    y_out[idx]         = sq38_clamp_f16(gated);
}

/* ============================================================================
 * 6. 16-Layer Gated Full Attention (`full_attention`, `l % 4 == 3`) Kernels
 *    - Per-head `q_proj` `[24, 512] -> q[256], gate[256]`
 *    - Per-head `q_norm`/`k_norm` (256) + Partial NeoX RoPE (64/256, `theta=1e7`, IEEE precise trig!)
 *    - GQA-Grouped Split-KV Flash-Decoding (`32K-64K`) + Metal 4 `NAX` Tiled FlashAttention!
 *    - `attn_out * sigmoid(gate)`!
 * ============================================================================ */

struct Sq38AttnPrepParams {
    uint  M_valid;
    uint  M_padded;
    uint  seq_pos_start;
    uint  max_seq_alloc;
    float eps;
    uint  kv_zero_end; // Align `(seq_pos_start + M_valid)` up to 32 for NAX KV slice safety
    uint  pad0;
    uint  pad1;
};

// Grid: (32, M_padded, 1), 256 threads/TG (`tid = d = 0..255`):
//   - `tgid.x = 0..23`: Q head `hq` (`q_norm` + RoPE -> `q_rope_out [24, M_padded, 256]`)
//   - `tgid.x = 24..27`: K head `hkv` (`k_norm` + RoPE -> `k_cache [4, max_seq_alloc, 256]`)
//   - `tgid.x = 28..31`: V head `hkv` (copy -> `v_cache [4, max_seq_alloc, 256]`)
kernel void sq38_attn_qkv_prep(
    device const half*       q_gate_in  [[buffer(0)]], // [M_valid, 12288] ([M_valid, 24, 512])
    device const half*       k_raw_in   [[buffer(1)]], // [M_valid, 1024]  ([M_valid, 4, 256])
    device const half*       v_raw_in   [[buffer(2)]], // [M_valid, 1024]  ([M_valid, 4, 256])
    device const float*      q_norm_w   [[buffer(3)]], // [256]
    device const float*      k_norm_w   [[buffer(4)]], // [256]
    constant float*          inv_freq   [[buffer(5)]], // [32]
    device half*             q_rope_out [[buffer(6)]], // [24, M_padded, 256]
    device half*             k_cache    [[buffer(7)]], // [4, max_seq_alloc, 256] for this full layer
    device half*             v_cache    [[buffer(8)]], // [4, max_seq_alloc, 256] for this full layer
    constant Sq38AttnPrepParams& p      [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],        // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],      // 0..7
    uint  lane [[thread_index_in_simdgroup]]            // 0..31
) {
    const uint item = tgid.x;
    const uint m    = tgid.y;
    if (item >= 32u || m >= p.M_padded) return;

    // Zero-pad `q_rope_out` and KV tail for static NAX tensor tiles
    if (m >= p.M_valid) {
        if (item < 24u) {
            q_rope_out[(ulong(item) * p.M_padded + m) * 256ul + tid] = half(0.0h);
        } else if (item < 28u) {
            uint hkv = item - 24u;
            uint pos = p.seq_pos_start + m;
            if (pos < p.kv_zero_end && pos < p.max_seq_alloc) {
                k_cache[(ulong(hkv) * p.max_seq_alloc + pos) * 256ul + tid] = half(0.0h);
            }
        } else {
            uint hkv = item - 28u;
            uint pos = p.seq_pos_start + m;
            if (pos < p.kv_zero_end && pos < p.max_seq_alloc) {
                v_cache[(ulong(hkv) * p.max_seq_alloc + pos) * 256ul + tid] = half(0.0h);
            }
        }
        return;
    }

    const uint pos = p.seq_pos_start + m;

    // V head copy (`item = 28..31`)
    if (item >= 28u) {
        uint hkv = item - 28u;
        half v   = v_raw_in[(ulong(m) * 4ul + hkv) * 256ul + tid];
        v_cache[(ulong(hkv) * p.max_seq_alloc + pos) * 256ul + tid] = v;
        if (m + 1u == p.M_valid) {
            for (uint zp = pos + 1u; zp < p.kv_zero_end && zp < p.max_seq_alloc; ++zp) {
                v_cache[(ulong(hkv) * p.max_seq_alloc + zp) * 256ul + tid] = half(0.0h);
            }
        }
        return;
    }

    threadgroup float tg_sq[8];
    threadgroup float tg_vec[64]; // First 64 rotary dims after RMSNorm

    const bool is_q = (item < 24u);
    float raw_val   = is_q ? float(q_gate_in[ulong(m) * 12288ul + ulong(item) * 512ul + tid])
                           : float(k_raw_in[(ulong(m) * 4ul + (item - 24u)) * 256ul + tid]);
    float nw        = is_q ? q_norm_w[tid] : k_norm_w[tid];

    float sg_sq = simd_sum(raw_val * raw_val);
    if (lane == 0u) tg_sq[sgid] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float tot = (lane < 8u) ? tg_sq[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) {
            tg_sq[0] = rsqrt(tot * (1.0f / 256.0f) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float normed = raw_val * tg_sq[0] * nw;
    if (tid < 64u) {
        tg_vec[tid] = normed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_val = normed;
    if (tid < 32u) {
        float angle = float(pos) * inv_freq[tid];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_val     = tg_vec[tid] * c - tg_vec[tid + 32u] * s;
    } else if (tid < 64u) {
        uint  i     = tid - 32u;
        float angle = float(pos) * inv_freq[i];
        float c     = precise::cos(angle);
        float s     = precise::sin(angle);
        out_val     = tg_vec[tid] * c + tg_vec[i] * s;
    }

    half out_h = sq38_clamp_f16(out_val);
    if (is_q) {
        q_rope_out[(ulong(item) * p.M_padded + m) * 256ul + tid] = out_h;
    } else {
        uint hkv = item - 24u;
        k_cache[(ulong(hkv) * p.max_seq_alloc + pos) * 256ul + tid] = out_h;
        if (m + 1u == p.M_valid) {
            for (uint zp = pos + 1u; zp < p.kv_zero_end && zp < p.max_seq_alloc; ++zp) {
                k_cache[(ulong(hkv) * p.max_seq_alloc + zp) * 256ul + tid] = half(0.0h);
            }
        }
    }
}

struct Sq38FlashDecodeParams {
    uint total_kv;
    uint max_seq_alloc;
    uint num_splits;
    uint kv_per_split;
};

// GQA-Grouped (`6` Q-heads per KV-head simultaneously!) Split-KV Flash-Decoding (`M = 1`):
// Reads `k_cache` and `v_cache` ONCE (instead of 6x!) across `32K-64K` context!
// Grid: `(num_splits, 4, 1)`, 256 threads/TG (`8` SIMD-groups, `lane = 0..31`, 8 dims/lane).
kernel void sq38_flash_decode_splitkv(
    device const half*          q_rope      [[buffer(0)]], // [24, M_padded(1), 256]
    device const half*          k_cache     [[buffer(1)]], // [4, max_seq_alloc, 256]
    device const half*          v_cache     [[buffer(2)]], // [4, max_seq_alloc, 256]
    device float*               part_m      [[buffer(3)]], // [num_splits, 24]
    device float*               part_l      [[buffer(4)]], // [num_splits, 24]
    device float*               part_o      [[buffer(5)]], // [num_splits, 24, 256]
    constant Sq38FlashDecodeParams& p       [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],             // 0..255
    uint  sgid [[simdgroup_index_in_threadgroup]],          // 0..7
    uint  lane [[thread_index_in_simdgroup]]                // 0..31
) {
    const uint split_idx = tgid.x;
    const uint hkv       = tgid.y;
    if (split_idx >= p.num_splits || hkv >= 4u) return;

    const uint kv_beg  = min(p.total_kv, split_idx * p.kv_per_split);
    const uint kv_end  = min(p.total_kv, kv_beg + p.kv_per_split);
    const uint hq_base = hkv * 6u;

    threadgroup float tg_q[6][256];  // 6.0 KiB
    threadgroup float tg_m[6][8];
    threadgroup float tg_l[6][8];
    threadgroup float tg_o[8][256];  // 8.0 KiB (reused across g=0..5 in epilogue)

    #pragma clang loop unroll(full)
    for (uint g = 0u; g < 6u; ++g) {
        tg_q[g][tid] = float(q_rope[ulong(hq_base + g) * 256ul + tid]) * 0.0625f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint d_base = lane * 8u;
    float q_reg[6][8];
    #pragma clang loop unroll(full)
    for (uint g = 0u; g < 6u; ++g) {
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) {
            q_reg[g][j] = tg_q[g][d_base + j];
        }
    }

    float m_sg[6];
    float l_sg[6];
    float o_sg[6][8];
    #pragma clang loop unroll(full)
    for (uint g = 0u; g < 6u; ++g) {
        m_sg[g] = -INFINITY;
        l_sg[g] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) o_sg[g][j] = 0.0f;
    }

    device const half* k_head = k_cache + ulong(hkv) * p.max_seq_alloc * 256ul;
    device const half* v_head = v_cache + ulong(hkv) * p.max_seq_alloc * 256ul;

    for (uint pos = kv_beg + sgid; pos < kv_end; pos += 8u) {
        device const half4* kp = reinterpret_cast<device const half4*>(k_head + ulong(pos) * 256ul + d_base);
        device const half4* vp = reinterpret_cast<device const half4*>(v_head + ulong(pos) * 256ul + d_base);

        const float4 k0 = float4(kp[0]);
        const float4 k1 = float4(kp[1]);
        const float4 v0 = float4(vp[0]);
        const float4 v1 = float4(vp[1]);

        float vf[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};

        #pragma clang loop unroll(full)
        for (uint g = 0u; g < 6u; ++g) {
            float dot_part = fma(q_reg[g][0], k0.x, fma(q_reg[g][1], k0.y,
                             fma(q_reg[g][2], k0.z, fma(q_reg[g][3], k0.w,
                             fma(q_reg[g][4], k1.x, fma(q_reg[g][5], k1.y,
                             fma(q_reg[g][6], k1.z, q_reg[g][7] * k1.w)))))));
            float score  = simd_sum(dot_part);
            float m_prev = m_sg[g];
            float m_new  = max(m_prev, score);
            float alpha  = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);
            float weight = exp(score - m_new);
            m_sg[g]      = m_new;
            l_sg[g]      = fma(l_sg[g], alpha, weight);
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 8u; ++j) {
                o_sg[g][j] = fma(o_sg[g][j], alpha, weight * vf[j]);
            }
        }
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint g = 0u; g < 6u; ++g) {
            tg_m[g][sgid] = m_sg[g];
            tg_l[g][sgid] = l_sg[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint g = 0u; g < 6u; ++g) {
        const uint hq = hq_base + g;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) {
            tg_o[sgid][d_base + j] = o_sg[g][j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
            part_m[ulong(split_idx) * 24ul + hq] = m_max;
            part_l[ulong(split_idx) * 24ul + hq] = l_tot;
        }
        part_o[(ulong(split_idx) * 24ul + hq) * 256ul + tid] = o_tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Combines `num_splits` online-softmax partitions AND fuses `* sigmoid(gate)` (`[6144]` output)!
kernel void sq38_flash_decode_reduce(
    device const float*         part_m         [[buffer(0)]], // [num_splits, 24]
    device const float*         part_l         [[buffer(1)]], // [num_splits, 24]
    device const float*         part_o         [[buffer(2)]], // [num_splits, 24, 256]
    device const half*          q_gate_in      [[buffer(3)]], // [1, 12288] ([24, 512])
    device half*                gated_attn_out [[buffer(4)]], // [1, 6144]
    constant uint&              num_splits     [[buffer(5)]],
    uint hq  [[threadgroup_position_in_grid]],                // 0..23
    uint tid [[thread_index_in_threadgroup]]                  // 0..255
) {
    if (hq >= 24u || tid >= 256u) return;

    float m_max = -INFINITY;
    for (uint s = 0u; s < num_splits; ++s) {
        m_max = max(m_max, part_m[ulong(s) * 24ul + hq]);
    }

    float l_tot = 0.0f;
    float o_tot = 0.0f;
    if (m_max > -INFINITY) {
        for (uint s = 0u; s < num_splits; ++s) {
            float ms = part_m[ulong(s) * 24ul + hq];
            if (ms > -INFINITY) {
                float w = exp(ms - m_max);
                l_tot = fma(part_l[ulong(s) * 24ul + hq], w, l_tot);
                o_tot = fma(part_o[(ulong(s) * 24ul + hq) * 256ul + tid], w, o_tot);
            }
        }
    }

    float attn_val = (l_tot > 0.0f) ? (o_tot / l_tot) : 0.0f;
    float g_val    = float(q_gate_in[ulong(hq) * 512ul + 256ul + tid]);
    gated_attn_out[ulong(hq) * 256ul + tid] = sq38_clamp_f16(attn_val * sq38_sigmoid(g_val));
}

struct Sq38FlashPrefillParams {
    uint M_valid;
    uint M_padded;      // Multiple of 64
    uint seq_pos_start;
    uint total_kv;      // seq_pos_start + M_valid
    uint max_seq_alloc; // Multiple of 64
    uint pad0;
    uint pad1;
    uint pad2;
};

// Metal 4 `NAX` Cooperative-Tensor Causal FlashAttention (`B_r=16`/SG, `B_c=32`, `D=256`) + `* sigmoid(gate)`!
// Grid: `((M_valid + 31) / 32, 24, 1)`, 64 threads/TG (2 independent SIMD-groups, 6.5 KiB TG SRAM -> 5 TGs/core!)
kernel void sq38_flash_prefill_nax(
    device const half*           q_rope         [[buffer(0)]], // [24, M_padded, 256]
    device const half*           k_cache        [[buffer(1)]], // [4, max_seq_alloc, 256]
    device const half*           v_cache        [[buffer(2)]], // [4, max_seq_alloc, 256]
    device const half*           q_gate_in      [[buffer(3)]], // [M_valid, 12288] ([M_valid, 24, 512])
    device half*                 gated_attn_out [[buffer(4)]], // [M_padded, 6144]
    constant Sq38FlashPrefillParams& p          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],             // 0..1
    uint  lane [[thread_index_in_simdgroup]]                   // 0..31
) {
    const uint hq = tgid.y;
    const uint m0 = tgid.x * 32u + sgid * 16u;
    if (hq >= 24u || m0 >= p.M_padded) return;

    if (m0 >= p.M_valid) {
        for (uint r = 0u; r < 16u && (m0 + r) < p.M_padded; ++r) {
            for (uint d = lane; d < 256u; d += 32u) {
                gated_attn_out[ulong(m0 + r) * 6144ul + ulong(hq) * 256ul + d] = half(0.0h);
            }
        }
        return;
    }

    const uint hkv       = hq / 6u;
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

    // Hoist Q [16, 256] into NAX cooperative registers ONCE before the 32K KV loop!
    auto sQ   = tQ.slice<256, 16>(0, int(m0));
    auto qCT  = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(sQ);

    auto sV0  = tV.slice<64, 32>(0, 0);
    auto oCT0 = op_pv.get_destination_cooperative_tensor<decltype(tP), decltype(sV0), float>();
    auto oCT1 = op_pv.get_destination_cooperative_tensor<decltype(tP), decltype(sV0), float>();
    auto oCT2 = op_pv.get_destination_cooperative_tensor<decltype(tP), decltype(sV0), float>();
    auto oCT3 = op_pv.get_destination_cooperative_tensor<decltype(tP), decltype(sV0), float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            oCT0[i] = 0.0f;
            oCT1[i] = 0.0f;
            oCT2[i] = 0.0f;
            oCT3[i] = 0.0f;
        }
    }

    for (uint kv_pos = 0u; kv_pos < kv_limit; kv_pos += 32u) {
        auto sK  = tK.slice<256, 32>(0, int(kv_pos));
        auto sCT = op_qk.get_destination_cooperative_tensor<decltype(qCT), decltype(sK), float>();
        op_qk.run(qCT, sK, sCT);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < sCT.get_capacity(); ++i) {
            if (sCT.is_valid_element(i)) {
                auto idx = sCT.get_multidimensional_index(i);
                S_tg[sgid][uint(idx[1])][uint(idx[0])] = sCT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online causal softmax across 32 KV columns per row `r = 0..15`
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
                oCT0[i] *= a;
                oCT1[i] *= a;
                oCT2[i] *= a;
                oCT3[i] *= a;
            }
        }

        auto sV_0 = tV.slice<64, 32>(0,   int(kv_pos));
        auto sV_1 = tV.slice<64, 32>(64,  int(kv_pos));
        auto sV_2 = tV.slice<64, 32>(128, int(kv_pos));
        auto sV_3 = tV.slice<64, 32>(192, int(kv_pos));
        op_pv.run(tP, sV_0, oCT0);
        op_pv.run(tP, sV_1, oCT1);
        op_pv.run(tP, sV_2, oCT2);
        op_pv.run(tP, sV_3, oCT3);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize by `1 / L_tg` and fuse `* sigmoid(gate)` (plus zero-fill padded tail rows)
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            auto idx    = oCT0.get_multidimensional_index(i);
            uint r      = uint(idx[1]);
            uint c      = uint(idx[0]);
            uint m_glob = m0 + r;
            if (m_glob < p.M_valid) {
                float inv_l = (L_tg[sgid][r] > 0.0f) ? (1.0f / L_tg[sgid][r]) : 0.0f;
                ulong g_base = ulong(m_glob) * 12288ul + ulong(hq) * 512ul + 256ul;
                ulong o_base = ulong(m_glob) * 6144ul + ulong(hq) * 256ul;

                gated_attn_out[o_base + 0u   + c] = sq38_clamp_f16((oCT0[i] * inv_l) * sq38_sigmoid(float(q_gate_in[g_base + 0u   + c])));
                gated_attn_out[o_base + 64u  + c] = sq38_clamp_f16((oCT1[i] * inv_l) * sq38_sigmoid(float(q_gate_in[g_base + 64u  + c])));
                gated_attn_out[o_base + 128u + c] = sq38_clamp_f16((oCT2[i] * inv_l) * sq38_sigmoid(float(q_gate_in[g_base + 128u + c])));
                gated_attn_out[o_base + 192u + c] = sq38_clamp_f16((oCT3[i] * inv_l) * sq38_sigmoid(float(q_gate_in[g_base + 192u + c])));
            } else if (m_glob < p.M_padded) {
                ulong o_base = ulong(m_glob) * 6144ul + ulong(hq) * 256ul;
                gated_attn_out[o_base + 0u   + c] = half(0.0h);
                gated_attn_out[o_base + 64u  + c] = half(0.0h);
                gated_attn_out[o_base + 128u + c] = half(0.0h);
                gated_attn_out[o_base + 192u + c] = half(0.0h);
            }
        }
    }
#else
    (void)q_rope; (void)k_cache; (void)v_cache; (void)q_gate_in; (void)gated_attn_out; (void)p; (void)lane;
#endif
}

/* ============================================================================
 * 7. GPU `248,320`-Vocab Fast Argmax & Top-K / Top-P Sampling (`4-byte` Readback!)
 * ============================================================================ */

struct Sq38SampleKernelParams {
    uint  vocab_size; // 248320
    uint  greedy;     // 1 => argmax, 0 => stochastic
    float inv_temp;   // 1.0f / max(temperature, 1e-5f)
    uint  top_k;      // 0 => default 64, else min(top_k, 64)
    float top_p;      // 0 < top_p < 1 => nucleus filter
    uint  rng_lo;
    uint  rng_hi;
    uint  pad;
};

inline float sq38_rand_u01(uint lo, uint hi) {
    ulong z = (ulong(hi) << 32ul) | ulong(lo);
    z += 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30ul)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27ul)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31ul);
    uint u24 = uint(z >> 40ul) & 0xFFFFFFu;
    return (float(u24) + 0.5f) * (1.0f / 16777216.0f);
}

// Single-TG (`1,024` threads = `32` SGs) SLC-Resident Argmax & Exact Top-K/Top-P Sampler!
// Scans `248,320` FP32 logits (`970 KiB` already hot in 48 MiB SLC right after `lm_head`!) in < 5 us!
kernel void sq38_sample_vocab(
    device const float*            logits   [[buffer(0)]], // [vocab_size]
    device uint*                   out_tok  [[buffer(1)]], // [1] (4-byte next_token_id!)
    constant Sq38SampleKernelParams& p      [[buffer(2)]],
    uint tid  [[thread_index_in_threadgroup]],             // 0..1023
    uint sgid [[simdgroup_index_in_threadgroup]],          // 0..31
    uint lane [[thread_index_in_simdgroup]]                // 0..31
) {
    threadgroup float tg_val[32];
    threadgroup uint  tg_idx[32];
    threadgroup float top_vals[64];
    threadgroup uint  top_ids[64];

    // Fast path 1: Greedy Argmax (deterministic lowest-ID tie-break!)
    if (p.greedy != 0u) {
        float best_v = -INFINITY;
        uint  best_i = 0u;
        for (uint i = tid; i < p.vocab_size; i += 1024u) {
            float v = logits[i];
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }

        // Exact SIMD-group argmax with lowest-ID tie-break via shuffle_xor
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
            float v = tg_val[lane];
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
                out_tok[0] = idx;
            }
        }
        return;
    }

    // Fast path 2: Stochastic Top-K (`<= 64`, default `20` in Qwen3.8!) + Top-P + Temperature
    const uint K = (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u;

    for (uint k = 0u; k < K; ++k) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;

        for (uint i = tid; i < p.vocab_size; i += 1024u) {
            float v = logits[i];
            bool skip = false;
            for (uint prev = 0u; prev < k; ++prev) {
                if (top_ids[prev] == i) {
                    skip = true;
                    break;
                }
            }
            if (!skip && (v > best_v || (v == best_v && i < best_i))) {
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
                top_vals[k] = v;
                top_ids[k]  = (idx < p.vocab_size) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 performs exact temperature softmax + nucleus `top_p` truncation + inverse-CDF draw
    if (tid == 0u) {
        const float max_v = top_vals[0];
        float probs[64];
        float sum_p = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float w = exp((top_vals[k] - max_v) * p.inv_temp);
            probs[k] = w;
            sum_p += w;
        }
        const float inv_sum = (sum_p > 0.0f) ? (1.0f / sum_p) : 1.0f;
        for (uint k = 0u; k < K; ++k) {
            probs[k] *= inv_sum;
        }

        uint  keep_k = K;
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

        const float u      = sq38_rand_u01(p.rng_lo, p.rng_hi) * kept_mass;
        float       cdf    = 0.0f;
        uint        chosen = top_ids[0];
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += probs[k];
            if (u <= cdf || k + 1u == keep_k) {
                chosen = top_ids[k];
                break;
            }
        }
        out_tok[0] = chosen;
    }
}

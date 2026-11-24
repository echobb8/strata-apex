#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#if defined(__HAVE_TENSOR__)
#include <metal_tensor>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
#if defined(__HAVE_PACKED_NUMERIC__)
#include <metal_packed_numeric>
#endif

// Inlined by strata_inkling2_gpu.mm at runtime from ../../include/strata_inkling2.h:
// @@STRATA_INKLING2_HEADER@@
#ifndef STRATA_INKLING2_H
#include "../../include/strata_inkling2.h"
#endif

#ifndef STRATA_INKLING2_DTYPE_Q8_0
#define STRATA_INKLING2_DTYPE_Q8_0 5u
#endif
#ifndef STRATA_INKLING2_DTYPE_Q4_A
#define STRATA_INKLING2_DTYPE_Q4_A 6u
#endif
#ifndef STRATA_INKLING2_KV_DTYPE_F16
#define STRATA_INKLING2_KV_DTYPE_F16 0u
#endif
#ifndef STRATA_INKLING2_KV_DTYPE_Q8_0
#define STRATA_INKLING2_KV_DTYPE_Q8_0 1u
#endif

using namespace metal;
#if defined(__HAVE_TENSOR__)
using namespace mpp::tensor_ops;
#endif

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 410)
template <typename T>
using sik_strip_as = metal::remove_addrspace_t<T>;
#else
template <typename T>
using sik_strip_as = T;
#endif

// Metal Function Constant for zero-branch exact-M (1..8) and MoE max-B specialization
constant uint SIK_CONST_M [[function_constant(0)]];

#define SIK_OUT_F16         0u
#define SIK_OUT_F32         1u
#define SIK_OUT_F32_SOFTCAP 2u
#define SIK_OUT_RES_F32     3u

// Dynamic SWA ring stride is passed per-kernel via `p.kv_stride` (power-of-2: 2048 or 4096).

/* ============================================================================
 * 1. Exact IEEE-754 Math, Metal 4.1 HW Unpack, NVFP4, MXFP8, Q8_0 & Q4_A Helpers
 * ============================================================================ */

constant half SIK_E2M1_H[16] = {
     0.0h,  0.5h,  1.0h,  1.5h,  2.0h,  3.0h,  4.0h,  6.0h,
    -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h
};

constant half2 SIK_E2M1_PAIR[256] = {
    half2(0.0h,0.0h), half2(0.5h,0.0h), half2(1.0h,0.0h), half2(1.5h,0.0h), half2(2.0h,0.0h), half2(3.0h,0.0h), half2(4.0h,0.0h), half2(6.0h,0.0h),
    half2(-0.0h,0.0h), half2(-0.5h,0.0h), half2(-1.0h,0.0h), half2(-1.5h,0.0h), half2(-2.0h,0.0h), half2(-3.0h,0.0h), half2(-4.0h,0.0h), half2(-6.0h,0.0h),
    half2(0.0h,0.5h), half2(0.5h,0.5h), half2(1.0h,0.5h), half2(1.5h,0.5h), half2(2.0h,0.5h), half2(3.0h,0.5h), half2(4.0h,0.5h), half2(6.0h,0.5h),
    half2(-0.0h,0.5h), half2(-0.5h,0.5h), half2(-1.0h,0.5h), half2(-1.5h,0.5h), half2(-2.0h,0.5h), half2(-3.0h,0.5h), half2(-4.0h,0.5h), half2(-6.0h,0.5h),
    half2(0.0h,1.0h), half2(0.5h,1.0h), half2(1.0h,1.0h), half2(1.5h,1.0h), half2(2.0h,1.0h), half2(3.0h,1.0h), half2(4.0h,1.0h), half2(6.0h,1.0h),
    half2(-0.0h,1.0h), half2(-0.5h,1.0h), half2(-1.0h,1.0h), half2(-1.5h,1.0h), half2(-2.0h,1.0h), half2(-3.0h,1.0h), half2(-4.0h,1.0h), half2(-6.0h,1.0h),
    half2(0.0h,1.5h), half2(0.5h,1.5h), half2(1.0h,1.5h), half2(1.5h,1.5h), half2(2.0h,1.5h), half2(3.0h,1.5h), half2(4.0h,1.5h), half2(6.0h,1.5h),
    half2(-0.0h,1.5h), half2(-0.5h,1.5h), half2(-1.0h,1.5h), half2(-1.5h,1.5h), half2(-2.0h,1.5h), half2(-3.0h,1.5h), half2(-4.0h,1.5h), half2(-6.0h,1.5h),
    half2(0.0h,2.0h), half2(0.5h,2.0h), half2(1.0h,2.0h), half2(1.5h,2.0h), half2(2.0h,2.0h), half2(3.0h,2.0h), half2(4.0h,2.0h), half2(6.0h,2.0h),
    half2(-0.0h,2.0h), half2(-0.5h,2.0h), half2(-1.0h,2.0h), half2(-1.5h,2.0h), half2(-2.0h,2.0h), half2(-3.0h,2.0h), half2(-4.0h,2.0h), half2(-6.0h,2.0h),
    half2(0.0h,3.0h), half2(0.5h,3.0h), half2(1.0h,3.0h), half2(1.5h,3.0h), half2(2.0h,3.0h), half2(3.0h,3.0h), half2(4.0h,3.0h), half2(6.0h,3.0h),
    half2(-0.0h,3.0h), half2(-0.5h,3.0h), half2(-1.0h,3.0h), half2(-1.5h,3.0h), half2(-2.0h,3.0h), half2(-3.0h,3.0h), half2(-4.0h,3.0h), half2(-6.0h,3.0h),
    half2(0.0h,4.0h), half2(0.5h,4.0h), half2(1.0h,4.0h), half2(1.5h,4.0h), half2(2.0h,4.0h), half2(3.0h,4.0h), half2(4.0h,4.0h), half2(6.0h,4.0h),
    half2(-0.0h,4.0h), half2(-0.5h,4.0h), half2(-1.0h,4.0h), half2(-1.5h,4.0h), half2(-2.0h,4.0h), half2(-3.0h,4.0h), half2(-4.0h,4.0h), half2(-6.0h,4.0h),
    half2(0.0h,6.0h), half2(0.5h,6.0h), half2(1.0h,6.0h), half2(1.5h,6.0h), half2(2.0h,6.0h), half2(3.0h,6.0h), half2(4.0h,6.0h), half2(6.0h,6.0h),
    half2(-0.0h,6.0h), half2(-0.5h,6.0h), half2(-1.0h,6.0h), half2(-1.5h,6.0h), half2(-2.0h,6.0h), half2(-3.0h,6.0h), half2(-4.0h,6.0h), half2(-6.0h,6.0h),
    half2(0.0h,-0.0h), half2(0.5h,-0.0h), half2(1.0h,-0.0h), half2(1.5h,-0.0h), half2(2.0h,-0.0h), half2(3.0h,-0.0h), half2(4.0h,-0.0h), half2(6.0h,-0.0h),
    half2(-0.0h,-0.0h), half2(-0.5h,-0.0h), half2(-1.0h,-0.0h), half2(-1.5h,-0.0h), half2(-2.0h,-0.0h), half2(-3.0h,-0.0h), half2(-4.0h,-0.0h), half2(-6.0h,-0.0h),
    half2(0.0h,-0.5h), half2(0.5h,-0.5h), half2(1.0h,-0.5h), half2(1.5h,-0.5h), half2(2.0h,-0.5h), half2(3.0h,-0.5h), half2(4.0h,-0.5h), half2(6.0h,-0.5h),
    half2(-0.0h,-0.5h), half2(-0.5h,-0.5h), half2(-1.0h,-0.5h), half2(-1.5h,-0.5h), half2(-2.0h,-0.5h), half2(-3.0h,-0.5h), half2(-4.0h,-0.5h), half2(-6.0h,-0.5h),
    half2(0.0h,-1.0h), half2(0.5h,-1.0h), half2(1.0h,-1.0h), half2(1.5h,-1.0h), half2(2.0h,-1.0h), half2(3.0h,-1.0h), half2(4.0h,-1.0h), half2(6.0h,-1.0h),
    half2(-0.0h,-1.0h), half2(-0.5h,-1.0h), half2(-1.0h,-1.0h), half2(-1.5h,-1.0h), half2(-2.0h,-1.0h), half2(-3.0h,-1.0h), half2(-4.0h,-1.0h), half2(-6.0h,-1.0h),
    half2(0.0h,-1.5h), half2(0.5h,-1.5h), half2(1.0h,-1.5h), half2(1.5h,-1.5h), half2(2.0h,-1.5h), half2(3.0h,-1.5h), half2(4.0h,-1.5h), half2(6.0h,-1.5h),
    half2(-0.0h,-1.5h), half2(-0.5h,-1.5h), half2(-1.0h,-1.5h), half2(-1.5h,-1.5h), half2(-2.0h,-1.5h), half2(-3.0h,-1.5h), half2(-4.0h,-1.5h), half2(-6.0h,-1.5h),
    half2(0.0h,-2.0h), half2(0.5h,-2.0h), half2(1.0h,-2.0h), half2(1.5h,-2.0h), half2(2.0h,-2.0h), half2(3.0h,-2.0h), half2(4.0h,-2.0h), half2(6.0h,-2.0h),
    half2(-0.0h,-2.0h), half2(-0.5h,-2.0h), half2(-1.0h,-2.0h), half2(-1.5h,-2.0h), half2(-2.0h,-2.0h), half2(-3.0h,-2.0h), half2(-4.0h,-2.0h), half2(-6.0h,-2.0h),
    half2(0.0h,-3.0h), half2(0.5h,-3.0h), half2(1.0h,-3.0h), half2(1.5h,-3.0h), half2(2.0h,-3.0h), half2(3.0h,-3.0h), half2(4.0h,-3.0h), half2(6.0h,-3.0h),
    half2(-0.0h,-3.0h), half2(-0.5h,-3.0h), half2(-1.0h,-3.0h), half2(-1.5h,-3.0h), half2(-2.0h,-3.0h), half2(-3.0h,-3.0h), half2(-4.0h,-3.0h), half2(-6.0h,-3.0h),
    half2(0.0h,-4.0h), half2(0.5h,-4.0h), half2(1.0h,-4.0h), half2(1.5h,-4.0h), half2(2.0h,-4.0h), half2(3.0h,-4.0h), half2(4.0h,-4.0h), half2(6.0h,-4.0h),
    half2(-0.0h,-4.0h), half2(-0.5h,-4.0h), half2(-1.0h,-4.0h), half2(-1.5h,-4.0h), half2(-2.0h,-4.0h), half2(-3.0h,-4.0h), half2(-4.0h,-4.0h), half2(-6.0h,-4.0h),
    half2(0.0h,-6.0h), half2(0.5h,-6.0h), half2(1.0h,-6.0h), half2(1.5h,-6.0h), half2(2.0h,-6.0h), half2(3.0h,-6.0h), half2(4.0h,-6.0h), half2(6.0h,-6.0h),
    half2(-0.0h,-6.0h), half2(-0.5h,-6.0h), half2(-1.0h,-6.0h), half2(-1.5h,-6.0h), half2(-2.0h,-6.0h), half2(-3.0h,-6.0h), half2(-4.0h,-6.0h), half2(-6.0h,-6.0h)
};


inline half sik_clamp_f16(float x) {
    return half(clamp(x, -65504.0f, 65504.0f));
}

inline half4 sik_clamp_f16x4(float4 x) {
    return half4(clamp(x, float4(-65504.0f), float4(65504.0f)));
}

inline float sik_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float sik_silu(float x) {
    return x / (1.0f + exp(-x));
}

inline float sik_safe_tanh(float x) {
    return tanh(clamp(x, -20.0f, 20.0f));
}

inline float sik_softcap(float x, float cap) {
    return (cap > 0.0f) ? (cap * sik_safe_tanh(x / cap)) : x;
}

// Softcapped SiTU SwiGLU: sigmoid sees RAW un-capped gate; softcaps bound linear gate and up
inline float sik_situ_swiglu(float g, float u, float gate_cap, float up_cap) {
    const float cg = gate_cap * sik_safe_tanh(g / gate_cap);
    const float sg = sik_sigmoid(g);
    const float cu = up_cap * sik_safe_tanh(u / up_cap);
    return (cg * sg) * cu;
}

inline float sik_reduce16(float v) {
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 4u);
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

inline float sik_reduce8(float v) {
    v += simd_shuffle_xor(v, 4u);
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

inline float sik_reduce4(float v) {
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

// Bit-exact finite FP8 E4M3FN -> IEEE FP16 (all 254 finite E4M3FN values are exact in FP16!)
inline half sik_e4m3_to_f16(uchar c) {
    const ushort sign = ushort(c & 0x80u) << 8u;
    const ushort mag  = ushort(c & 0x7Fu);
    const ushort norm_bits = ushort((mag << 7u) + 0x2000u);
    const ushort sub_bits  = as_type<ushort>(half(mag) * half(0.001953125h));
    const ushort bits      = select(norm_bits, sub_bits, mag < 8u) | sign;
    return as_type<half>(bits);
}

inline half4 sik_e4m3x4_to_f16x4(uchar4 c) {
    const ushort4 sign      = (ushort4(c) & ushort4(0x80u)) << 8u;
    const ushort4 mag       = ushort4(c) & ushort4(0x7Fu);
    const ushort4 norm_bits = (mag << 7u) + ushort4(0x2000u);
    const ushort4 sub_bits  = as_type<ushort4>(half4(mag) * half4(0.001953125h));
    const ushort4 bits      = select(norm_bits, sub_bits, mag < ushort4(8u)) | sign;
    return as_type<half4>(bits);
}

// Metal 4.1 Hardware FP8 E4M3 unpack (+20% faster for MXFP8 GEMV & NAX!)
inline void sik_e4m3x8_to_f16x4x2(uint2 v, thread half4& lo, thread half4& hi) {
#if defined(__HAVE_PACKED_NUMERIC__) && defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 410)
    vec<half, 8> u = unpack<half>(packed_metal_fp8_e4m3<8>(as_type<packed_vec<uchar, 8>>(v)));
    lo = half4(u[0], u[1], u[2], u[3]);
    hi = half4(u[4], u[5], u[6], u[7]);
#else
    lo = sik_e4m3x4_to_f16x4(as_type<uchar4>(v.x));
    hi = sik_e4m3x4_to_f16x4(as_type<uchar4>(v.y));
#endif
}

// Bit-exact UE8M0 exponent byte -> 2^(s - 127) in IEEE FP32
inline float sik_ue8m0_to_f32(uchar s) {
    return (s == 0u) ? as_type<float>(0x00400000u) : as_type<float>(uint(s) << 23u);
}

// Unpack 8 E2M1 nibbles (low nibble = even col, high nibble = odd col) via fast 16-entry LUT
// (Do NOT use Metal 4.1 packed_metal_fp4_e2m1 — SW-emulated on M5 Max per #149!)
inline void sik_unpack_nvfp4_u32_h4(uint w, thread half4& lo, thread half4& hi) {
    const half2 p0 = SIK_E2M1_PAIR[w & 0xFFu];
    const half2 p1 = SIK_E2M1_PAIR[(w >> 8u) & 0xFFu];
    const half2 p2 = SIK_E2M1_PAIR[(w >> 16u) & 0xFFu];
    const half2 p3 = SIK_E2M1_PAIR[w >> 24u];
    lo = half4(p0, p1);
    hi = half4(p2, p3);
}

inline void sik_unpack_nvfp4_u32_tg(uint w, threadgroup const half2* lut, thread half4& lo, thread half4& hi) {
    const half2 p0 = lut[w & 0xFFu];
    const half2 p1 = lut[(w >> 8u) & 0xFFu];
    const half2 p2 = lut[(w >> 16u) & 0xFFu];
    const half2 p3 = lut[w >> 24u];
    lo = half4(p0, p1);
    hi = half4(p2, p3);
}

// Unpack 8 unsigned 4-bit Q4_A nibbles (0..15) via branchless IEEE binary16 0x6400 trick!
// (Benchmarked 1.35-1.60x faster than Metal 4.1 packed_uint4b<8> on Apple M5 Max!)
inline void sik_unpack_u4_u32_h4(uint w, thread half4& lo, thread half4& hi) {
    const ushort4 n_lo(ushort(w & 0xFu),
                       ushort((w >> 4u) & 0xFu),
                       ushort((w >> 8u) & 0xFu),
                       ushort((w >> 12u) & 0xFu));
    const ushort4 n_hi(ushort((w >> 16u) & 0xFu),
                       ushort((w >> 20u) & 0xFu),
                       ushort((w >> 24u) & 0xFu),
                       ushort((w >> 28u) & 0xFu));
    lo = as_type<half4>(ushort4(0x6400u) | n_lo) - half4(1024.0h);
    hi = as_type<half4>(ushort4(0x6400u) | n_hi) - half4(1024.0h);
}

inline float sik_dot_h4x4_fast(
    half4 w0, half4 w1, half4 w2, half4 w3,
    half4 x0, half4 x1, half4 x2, half4 x3
) {
    const half4 dh = fma(w0, x0, w1 * x1) + fma(w2, x2, w3 * x3);
    return (float(dh.x) + float(dh.y)) + (float(dh.z) + float(dh.w));
}

inline float sik_dot_f32_h4x4(
    half4 w0, half4 w1, half4 w2, half4 w3,
    half4 x0, half4 x1, half4 x2, half4 x3
) {
    return (dot(float4(w0), float4(x0)) + dot(float4(w1), float4(x1))) +
           (dot(float4(w2), float4(x2)) + dot(float4(w3), float4(x3)));
}

inline float sik_sum_f32_h4x4(half4 x0, half4 x1, half4 x2, half4 x3) {
    const float4 s = (float4(x0) + float4(x1)) + (float4(x2) + float4(x3));
    return (s.x + s.y) + (s.z + s.w);
}

/* ============================================================================
 * 2. Utility Kernels: Zero, Copy, Cast/Pad, Tap Capture & SWA Checkpoint Copy
 * ============================================================================ */

kernel void sik_zero_u32(
    device uint4*  buf      [[buffer(0)]],
    constant uint& count_u4 [[buffer(1)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid < count_u4) buf[gid] = uint4(0u);
}

kernel void sik_copy_u32(
    device const uint4* src      [[buffer(0)]],
    device uint4*       dst      [[buffer(1)]],
    constant uint&      count_u4 [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid < count_u4) dst[gid] = src[gid];
}

struct SikPadCastParams {
    uint M_valid;
    uint M_padded;
    uint D;
    uint pad;
};

kernel void sik_f32_to_f16_pad(
    device const float*       src_f32 [[buffer(0)]],
    device half*              dst_f16 [[buffer(1)]],
    constant SikPadCastParams& p      [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
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
    const float4 v = reinterpret_cast<device const float4*>(src_f32)[off4];
    reinterpret_cast<device half4*>(dst_f16)[off4] = sik_clamp_f16x4(v);
}

struct SikCaptureTapParams {
    uint M_valid;
    uint D;
    uint tap_idx;  // 0..3
    uint num_taps; // 4
};

kernel void sik_capture_tap(
    device const float4*        src       [[buffer(0)]], // [M_valid, D/4]
    device float4*              taps_out  [[buffer(1)]], // [M_valid, 4, D/4]
    constant SikCaptureTapParams& p       [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint m      = tgid.y;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (m >= p.M_valid || idx4 >= d4_cnt) return;
    taps_out[(ulong(m) * p.num_taps + p.tap_idx) * d4_cnt + idx4] = src[ulong(m) * d4_cnt + idx4];
}

struct SikSwaCkptParams {
    uint num_layers;
    uint n_kv;
    uint head_dim;
    uint start_pos;
    uint count;       // <= 1024
    uint ring_len;    // 2048
    uint ckpt_stride; // 1024
    uint is_save;     // 1 => ring -> ckpt, 0 => ckpt -> ring
};

kernel void sik_swa_ring_copy(
    device half4*              k_ring [[buffer(0)]],
    device half4*              v_ring [[buffer(1)]],
    device half4*              k_ckpt [[buffer(2)]],
    device half4*              v_ckpt [[buffer(3)]],
    constant SikSwaCkptParams& p      [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint i  = tgid.x;
    const uint lh = tgid.y;
    if (i >= p.count || lh >= p.num_layers * p.n_kv) return;

    const uint  hd4   = p.head_dim >> 2u;
    const uint  rslot = (p.start_pos + i) & (p.ring_len - 1u);
    const ulong roff  = (ulong(lh) * p.ring_len + rslot) * hd4;
    const ulong coff  = (ulong(lh) * p.ckpt_stride + i) * hd4;

    for (uint d = tid; d < hd4; d += 32u) {
        if (p.is_save != 0u) {
            k_ckpt[coff + d] = k_ring[roff + d];
            v_ckpt[coff + d] = v_ring[roff + d];
        } else {
            k_ring[roff + d] = k_ckpt[coff + d];
            v_ring[roff + d] = v_ckpt[coff + d];
        }
    }
}

/* ============================================================================
 * 3. Embedding (FP16 & Q4_A), RMSNorm, Fast 2-Barrier Block AttnRes (+ Fused
 *    AttnRes+RMSNorm!), & High-Occupancy Decode/Prefill NAX Loop Controllers
 * ============================================================================ */

struct SikEmbedParams {
    uint  M_valid;
    uint  D;
    uint  vocab_size;
    float eps;
};

// Grid: `(M_valid, 1, 1)`, `256` threads/TG (`8` SGs)
kernel void sik_embed_rmsnorm(
    device const half*       embed_w    [[buffer(0)]], // [vocab_size, D]
    device const float*      embed_norm [[buffer(1)]], // [D]
    device const uint*       token_ids  [[buffer(2)]], // [M_valid]
    device float*            cur_out    [[buffer(3)]], // [M_valid, D]
    constant SikEmbedParams& p          [[buffer(4)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_valid) return;
    const uint tok    = min(token_ids[m], p.vocab_size - 1u);
    const uint d4_cnt = p.D >> 2u;

    device const half4*  src4 = reinterpret_cast<device const half4*>(embed_w + ulong(tok) * p.D);
    device const float4* nw4  = reinterpret_cast<device const float4*>(embed_norm);
    device float4*       dst4 = reinterpret_cast<device float4*>(cur_out + ulong(m) * p.D);

    threadgroup float tg_sq[8];
    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 v = float4(src4[i]);
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }
    const float sg_s = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    for (uint i = tid; i < d4_cnt; i += 256u) {
        dst4[i] = (float4(src4[i]) * inv_rms) * nw4[i];
    }
}

// Q4_A Planar Affine Embedding + RMSNorm (`-fast` bundle)
// Grid: `(M_valid, 1, 1)`, `256` threads/TG
kernel void sik_embed_rmsnorm_q4a(
    device const uint*       embed_q    [[buffer(0)]], // [vocab_size, D / 8]
    device const half*       embed_sc   [[buffer(1)]], // [vocab_size, D / 64]
    device const half*       embed_bi   [[buffer(2)]], // [vocab_size, D / 64]
    device const float*      embed_norm [[buffer(3)]], // [D]
    device const uint*       token_ids  [[buffer(4)]], // [M_valid]
    device float*            cur_out    [[buffer(5)]], // [M_valid, D]
    constant SikEmbedParams& p          [[buffer(6)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_valid) return;
    const uint tok        = min(token_ids[m], p.vocab_size - 1u);
    const uint u32_per_r  = p.D >> 3u;
    const uint grp_per_r  = p.D >> 6u;

    device const uint*   wq  = embed_q  + ulong(tok) * u32_per_r;
    device const half*   wsc = embed_sc + ulong(tok) * grp_per_r;
    device const half*   wbi = embed_bi + ulong(tok) * grp_per_r;
    device const float4* nw4 = reinterpret_cast<device const float4*>(embed_norm);
    device float4*       dst = reinterpret_cast<device float4*>(cur_out + ulong(m) * p.D);

    threadgroup float tg_sq[8];
    float sq_acc = 0.0f;
    for (uint u = tid; u < u32_per_r; u += 256u) {
        const uint grp = u >> 3u; // 8 * 8 = 64
        const float sc = float(wsc[grp]);
        const float bi = float(wbi[grp]);
        half4 n0, n1;
        sik_unpack_u4_u32_h4(wq[u], n0, n1);
        const float4 v0 = fma(float4(n0), float4(sc), float4(bi));
        const float4 v1 = fma(float4(n1), float4(sc), float4(bi));
        sq_acc = fma(v0.x, v0.x, fma(v0.y, v0.y, fma(v0.z, v0.z, fma(v0.w, v0.w, sq_acc))));
        sq_acc = fma(v1.x, v1.x, fma(v1.y, v1.y, fma(v1.z, v1.z, fma(v1.w, v1.w, sq_acc))));
    }
    const float sg_s = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = tg_sq[0];
    for (uint u = tid; u < u32_per_r; u += 256u) {
        const uint grp = u >> 3u;
        const float sc = float(wsc[grp]);
        const float bi = float(wbi[grp]);
        half4 n0, n1;
        sik_unpack_u4_u32_h4(wq[u], n0, n1);
        const float4 v0 = fma(float4(n0), float4(sc), float4(bi));
        const float4 v1 = fma(float4(n1), float4(sc), float4(bi));
        dst[u * 2u + 0u] = (v0 * inv_rms) * nw4[u * 2u + 0u];
        dst[u * 2u + 1u] = (v1 * inv_rms) * nw4[u * 2u + 1u];
    }
}

struct SikRmsNormParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    float eps;
    uint  has_weight;
    uint  write_f32;
    uint  tap_capture;
    uint  tap_idx;
};

// Grid: `(M_padded, 1, 1)`, `256` threads/TG (`8` SGs)
kernel void sik_rmsnorm(
    device const float*        x_in     [[buffer(0)]],
    device const float*        norm_w   [[buffer(1)]],
    device half*               out_f16  [[buffer(2)]],
    device float*              out_f32  [[buffer(3)]],
    device float*              taps_out [[buffer(4)]],
    constant SikRmsNormParams& p        [[buffer(5)]],
    device float*              xsum_out [[buffer(6)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_padded) return;
    const uint  d4_cnt  = p.D >> 2u;
    const ulong row_off = ulong(m) * p.D;

    if (m >= p.M_valid) {
        device half4* dst_h = reinterpret_cast<device half4*>(out_f16 + row_off);
        for (uint i = tid; i < d4_cnt; i += 256u) dst_h[i] = half4(0.0h);
        return;
    }

    threadgroup float tg_sq[8];
    device const float4* src4 = reinterpret_cast<device const float4*>(x_in + row_off);
    device const float4* w4   = reinterpret_cast<device const float4*>(norm_w);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 v = src4[i];
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }
    const float sg_s = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float    inv_rms = tg_sq[0];
    const uint     nblk    = p.D >> 6u;
    const bool     do_xs   = (p.M_valid <= 64u);
    device half4*  dst_h   = reinterpret_cast<device half4*>(out_f16 + row_off);
    device float4* dst_f   = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + row_off) : nullptr;
    device float4* dst_t   = (p.tap_capture != 0u)
        ? reinterpret_cast<device float4*>(taps_out + (ulong(m) * 4ul + p.tap_idx) * p.D)
        : nullptr;

    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 normed = src4[i] * inv_rms;
        if (p.has_weight != 0u) normed *= w4[i];
        const half4 h = sik_clamp_f16x4(normed);
        dst_h[i] = h;
        if (dst_f) dst_f[i] = normed;
        if (dst_t) dst_t[i] = normed;
        if (do_xs) {
            const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
            if ((lane & 15u) == 0u) xsum_out[ulong(m) * nblk + (i >> 4u)] = bs;
        }
    }
}

// Batched 3-way Input RMSNorm (`j = 0, 1, 2` in 1 dispatch!)
// Grid: `(M_padded, 3, 1)`, `256` threads/TG (`8` SGs)
kernel void sik_rmsnorm3(
    device const float*        x0      [[buffer(0)]],
    device const float*        x1      [[buffer(1)]],
    device const float*        x2      [[buffer(2)]],
    device const float*        w0      [[buffer(3)]],
    device const float*        w1      [[buffer(4)]],
    device const float*        w2      [[buffer(5)]],
    device half*               out0    [[buffer(6)]],
    device half*               out1    [[buffer(7)]],
    device half*               out2    [[buffer(8)]],
    constant SikRmsNormParams& p       [[buffer(9)]],
    device float*              xs0     [[buffer(10)]],
    device float*              xs1     [[buffer(11)]],
    device float*              xs2     [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m = tgid.x;
    const uint j = tgid.y;
    if (m >= p.M_padded || j >= 3u) return;

    device const float* x_in   = (j == 0u) ? x0   : ((j == 1u) ? x1   : x2);
    device const float* norm_w = (j == 0u) ? w0   : ((j == 1u) ? w1   : w2);
    device half*        out_h  = (j == 0u) ? out0 : ((j == 1u) ? out1 : out2);
    device float*       xsum_o = (j == 0u) ? xs0  : ((j == 1u) ? xs1  : xs2);

    const uint  d4_cnt  = p.D >> 2u;
    const ulong row_off = ulong(m) * p.D;

    if (m >= p.M_valid) {
        device half4* dst_h = reinterpret_cast<device half4*>(out_h + row_off);
        for (uint i = tid; i < d4_cnt; i += 256u) dst_h[i] = half4(0.0h);
        return;
    }

    threadgroup float tg_sq[8];
    device const float4* src4 = reinterpret_cast<device const float4*>(x_in + row_off);
    device const float4* w4   = reinterpret_cast<device const float4*>(norm_w);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 v = src4[i];
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }
    const float sg_s = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float   inv_rms = tg_sq[0];
    const uint    nblk    = p.D >> 6u;
    const bool    do_xs   = (p.M_valid <= 64u);
    device half4* dst_h   = reinterpret_cast<device half4*>(out_h + row_off);
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const half4 h = sik_clamp_f16x4((src4[i] * inv_rms) * w4[i]);
        dst_h[i] = h;
        if (do_xs) {
            const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
            if ((lane & 15u) == 0u) xsum_o[ulong(m) * nblk + (i >> 4u)] = bs;
        }
    }
}

struct SikAttnResParams {
    uint  M_valid;
    uint  D;
    uint  num_blocks;   // 0..7 completed blocks in `blocks_buf` (total stacked S = num_blocks + 1 <= 8)
    uint  block_stride; // elements per block slot in `blocks_buf` (`kMaxPrefillChunk * D`)
    uint  num_queries;  // 1, 2, or 3
    float eps;
    uint  pad0;
    uint  pad1;
};

// Rewritten 2-Barrier Block Attention Residual (down from up to 64 barriers!)
// Grid: `(M_valid, 1, 1)`, `256` threads/TG (`8` SGs)
kernel void sik_attn_res(
    device const float*        blocks_buf  [[buffer(0)]],
    device const float*        partial_buf [[buffer(1)]],
    device const float*        query_buf   [[buffer(2)]],
    device float*              out0        [[buffer(3)]],
    device float*              out1        [[buffer(4)]],
    device float*              out2        [[buffer(5)]],
    constant SikAttnResParams& p           [[buffer(6)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_valid) return;
    const uint  S       = min(p.num_blocks + 1u, 8u);
    const uint  Q       = min(p.num_queries, 3u);
    const uint  d4_cnt  = p.D >> 2u;
    const float inv_sqD = rsqrt(float(p.D));

    threadgroup float tg_sq[8][8];
    threadgroup float tg_dot[3][8][8];
    threadgroup float tg_inv_rms[8];
    threadgroup float tg_prob[3][8];

    device const float4* q0_ptr = reinterpret_cast<device const float4*>(query_buf);
    device const float4* q1_ptr = reinterpret_cast<device const float4*>(query_buf + ulong(p.D));
    device const float4* q2_ptr = reinterpret_cast<device const float4*>(query_buf + ulong(2u * p.D));

    // Pass 1: accumulate all S vectors' partial sums with ZERO barriers inside the loop!
    for (uint s = 0u; s < S; ++s) {
        device const float4* v_ptr = (s < p.num_blocks)
            ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + ulong(m) * p.D)
            : reinterpret_cast<device const float4*>(partial_buf + ulong(m) * p.D);

        float sq = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 v = v_ptr[i];
            sq = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq))));
            const float4 q0 = q0_ptr[i];
            d0 = fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, fma(v.w, q0.w, d0))));
            if (Q >= 2u) {
                const float4 q1 = q1_ptr[i];
                d1 = fma(v.x, q1.x, fma(v.y, q1.y, fma(v.z, q1.z, fma(v.w, q1.w, d1))));
            }
            if (Q >= 3u) {
                const float4 q2 = q2_ptr[i];
                d2 = fma(v.x, q2.x, fma(v.y, q2.y, fma(v.z, q2.z, fma(v.w, q2.w, d2))));
            }
        }
        const float s_sq = simd_sum(sq);
        const float s_d0 = simd_sum(d0);
        const float s_d1 = (Q >= 2u) ? simd_sum(d1) : 0.0f;
        const float s_d2 = (Q >= 3u) ? simd_sum(d2) : 0.0f;
        if (lane == 0u) {
            tg_sq[s][sgid]     = s_sq;
            tg_dot[0][s][sgid] = s_d0;
            if (Q >= 2u) tg_dot[1][s][sgid] = s_d1;
            if (Q >= 3u) tg_dot[2][s][sgid] = s_d2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Barrier 1 complete: SG0 reduces all S inv_rms + scores + softmax in registers!
    if (sgid == 0u) {
        float sc[3][8];
        for (uint s = 0u; s < S; ++s) {
            const float tot_sq = simd_sum((lane < 8u) ? tg_sq[s][lane] : 0.0f);
            const float irms   = rsqrt(tot_sq / float(p.D) + p.eps);
            if (lane == 0u) tg_inv_rms[s] = irms;
            for (uint q = 0u; q < Q; ++q) {
                const float tot_d = simd_sum((lane < 8u) ? tg_dot[q][s][lane] : 0.0f);
                sc[q][s] = (tot_d * irms) * inv_sqD;
            }
        }
        if (lane < Q) {
            float mx = -INFINITY;
            for (uint s = 0u; s < S; ++s) mx = max(mx, sc[lane][s]);
            float sm = 0.0f;
            for (uint s = 0u; s < S; ++s) {
                const float e = exp(sc[lane][s] - mx);
                tg_prob[lane][s] = e;
                sm += e;
            }
            const float inv_sm = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
            for (uint s = 0u; s < S; ++s) tg_prob[lane][s] *= inv_sm;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: weighted sum of unweighted-RMSNorm'd `v[s]`
    device float4* dst0 = reinterpret_cast<device float4*>(out0 + ulong(m) * p.D);
    device float4* dst1 = (Q >= 2u) ? reinterpret_cast<device float4*>(out1 + ulong(m) * p.D) : nullptr;
    device float4* dst2 = (Q >= 3u) ? reinterpret_cast<device float4*>(out2 + ulong(m) * p.D) : nullptr;

    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 a0(0.0f), a1(0.0f), a2(0.0f);
        for (uint s = 0u; s < S; ++s) {
            device const float4* v_ptr = (s < p.num_blocks)
                ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + ulong(m) * p.D)
                : reinterpret_cast<device const float4*>(partial_buf + ulong(m) * p.D);
            const float4 vn = v_ptr[i] * tg_inv_rms[s];
            a0 = fma(float4(tg_prob[0][s]), vn, a0);
            if (Q >= 2u) a1 = fma(float4(tg_prob[1][s]), vn, a1);
            if (Q >= 3u) a2 = fma(float4(tg_prob[2][s]), vn, a2);
        }
        dst0[i] = a0;
        if (dst1) dst1[i] = a1;
        if (dst2) dst2[i] = a2;
    }
}

struct SikAttnResNormParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    uint  num_blocks;
    uint  block_stride;
    float eps;
    uint  write_f32;
    uint  tap_capture;
    uint  tap_idx;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

// Single-Dispatch Fused `Block AttnRes (1 query) + RMSNorm`!
// Eliminates intermediate `b_inputs[0]` DRAM roundtrip AND 49 dispatches/step!
// Grid: `(M_padded, 1, 1)`, `256` threads/TG (`8` SGs)
kernel void sik_attn_res_rmsnorm(
    device const float*           blocks_buf  [[buffer(0)]],
    device const float*           partial_buf [[buffer(1)]],
    device const float*           query_buf   [[buffer(2)]],
    device const float*           norm_w      [[buffer(3)]],
    device half*                  out_f16     [[buffer(4)]],
    device float*                 out_f32     [[buffer(5)]],
    device float*                 taps_out    [[buffer(6)]],
    constant SikAttnResNormParams& p          [[buffer(7)]],
    device float*                 xsum_out    [[buffer(8)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_padded) return;
    const uint  d4_cnt  = p.D >> 2u;
    const ulong row_off = ulong(m) * p.D;

    if (m >= p.M_valid) {
        device half4* dst_h = reinterpret_cast<device half4*>(out_f16 + row_off);
        for (uint i = tid; i < d4_cnt; i += 256u) dst_h[i] = half4(0.0h);
        return;
    }

    const uint  S       = min(p.num_blocks + 1u, 8u);
    const float inv_sqD = rsqrt(float(p.D));

    threadgroup float tg_sq[8][8];
    threadgroup float tg_dot[8][8];
    threadgroup float tg_inv_rms[8];
    threadgroup float tg_prob[8];
    threadgroup float tg_out_sq[8];

    device const float4* q0_ptr = reinterpret_cast<device const float4*>(query_buf);
    device const float4* w4     = reinterpret_cast<device const float4*>(norm_w);

    for (uint s = 0u; s < S; ++s) {
        device const float4* v_ptr = (s < p.num_blocks)
            ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
            : reinterpret_cast<device const float4*>(partial_buf + row_off);
        float sq = 0.0f, d0 = 0.0f;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 v  = v_ptr[i];
            const float4 q0 = q0_ptr[i];
            sq = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq))));
            d0 = fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, fma(v.w, q0.w, d0))));
        }
        const float s_sq = simd_sum(sq);
        const float s_d0 = simd_sum(d0);
        if (lane == 0u) { tg_sq[s][sgid] = s_sq; tg_dot[s][sgid] = s_d0; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float sc[8];
        for (uint s = 0u; s < S; ++s) {
            const float tot_sq = simd_sum((lane < 8u) ? tg_sq[s][lane] : 0.0f);
            const float tot_d  = simd_sum((lane < 8u) ? tg_dot[s][lane] : 0.0f);
            const float irms   = rsqrt(tot_sq / float(p.D) + p.eps);
            if (lane == 0u) tg_inv_rms[s] = irms;
            sc[s] = (tot_d * irms) * inv_sqD;
        }
        if (lane == 0u) {
            float mx = -INFINITY;
            for (uint s = 0u; s < S; ++s) mx = max(mx, sc[s]);
            float sm = 0.0f;
            for (uint s = 0u; s < S; ++s) {
                const float e = exp(sc[s] - mx);
                tg_prob[s] = e;
                sm += e;
            }
            const float inv_sm = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
            for (uint s = 0u; s < S; ++s) tg_prob[s] *= inv_sm;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Keep combined `a0` in registers (`d4_cnt = 576 => <= 3` iters/thread!)
    float4 reg_a[4];
    float out_sq = 0.0f;
    uint  k_idx  = 0u;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 a0(0.0f);
        for (uint s = 0u; s < S; ++s) {
            device const float4* v_ptr = (s < p.num_blocks)
                ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
                : reinterpret_cast<device const float4*>(partial_buf + row_off);
            a0 = fma(float4(tg_prob[s] * tg_inv_rms[s]), v_ptr[i], a0);
        }
        if (k_idx < 4u) reg_a[k_idx] = a0;
        ++k_idx;
        out_sq = fma(a0.x, a0.x, fma(a0.y, a0.y, fma(a0.z, a0.z, fma(a0.w, a0.w, out_sq))));
    }
    const float sg_o = simd_sum(out_sq);
    if (lane == 0u) tg_out_sq[sgid] = sg_o;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_out_sq[lane] : 0.0f);
        if (lane == 0u) tg_out_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float    inv_rms_out = tg_out_sq[0];
    device half4*  dst_h = reinterpret_cast<device half4*>(out_f16 + row_off);
    device float4* dst_f = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + row_off) : nullptr;
    device float4* dst_t = (p.tap_capture != 0u)
        ? reinterpret_cast<device float4*>(taps_out + (ulong(m) * 4ul + p.tap_idx) * p.D)
        : nullptr;

    k_idx = 0u;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 a0;
        if (k_idx < 4u) {
            a0 = reg_a[k_idx];
        } else {
            a0 = float4(0.0f);
            for (uint s = 0u; s < S; ++s) {
                device const float4* v_ptr = (s < p.num_blocks)
                    ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
                    : reinterpret_cast<device const float4*>(partial_buf + row_off);
                a0 = fma(float4(tg_prob[s] * tg_inv_rms[s]), v_ptr[i], a0);
            }
        }
        ++k_idx;
        const float4 normed = (a0 * inv_rms_out) * w4[i];
        const half4  h      = sik_clamp_f16x4(normed);
        dst_h[i] = h;
        if (dst_f) dst_f[i] = normed;
        if (dst_t) dst_t[i] = normed;
        if (p.M_valid <= 64u) {
            const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
            if ((lane & 15u) == 0u) xsum_out[ulong(m) * (p.D >> 6u) + (i >> 4u)] = bs;
        }
    }
}

// Single-Dispatch Fused `Block AttnRes (3 queries: q/k/v) + 3-Way RMSNorm` for Global Layers without msgs!
// Grid: `(M_padded, 1, 1)`, `256` threads/TG (`8` SGs)
kernel void sik_attn_res3_rmsnorm3(
    device const float*           blocks_buf  [[buffer(0)]],
    device const float*           partial_buf [[buffer(1)]],
    device const float*           query_buf   [[buffer(2)]], // [3, D]
    device const float*           nw0         [[buffer(3)]],
    device const float*           nw1         [[buffer(4)]],
    device const float*           nw2         [[buffer(5)]],
    device half*                  out0        [[buffer(6)]],
    device half*                  out1        [[buffer(7)]],
    device half*                  out2        [[buffer(8)]],
    constant SikAttnResNormParams& p          [[buffer(9)]],
    device float*                 xs0         [[buffer(10)]],
    device float*                 xs1         [[buffer(11)]],
    device float*                 xs2         [[buffer(12)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_padded) return;
    const uint  d4_cnt  = p.D >> 2u;
    const ulong row_off = ulong(m) * p.D;

    if (m >= p.M_valid) {
        device half4* h0 = reinterpret_cast<device half4*>(out0 + row_off);
        device half4* h1 = reinterpret_cast<device half4*>(out1 + row_off);
        device half4* h2 = reinterpret_cast<device half4*>(out2 + row_off);
        for (uint i = tid; i < d4_cnt; i += 256u) {
            h0[i] = half4(0.0h); h1[i] = half4(0.0h); h2[i] = half4(0.0h);
        }
        return;
    }

    const uint  S       = min(p.num_blocks + 1u, 8u);
    const float inv_sqD = rsqrt(float(p.D));

    threadgroup float tg_sq[8][8];
    threadgroup float tg_dot[3][8][8];
    threadgroup float tg_inv_rms[8];
    threadgroup float tg_prob[3][8];
    threadgroup float tg_out_sq[3][8];
    threadgroup float tg_out_irms[3];

    device const float4* q0_ptr = reinterpret_cast<device const float4*>(query_buf);
    device const float4* q1_ptr = reinterpret_cast<device const float4*>(query_buf + ulong(p.D));
    device const float4* q2_ptr = reinterpret_cast<device const float4*>(query_buf + ulong(2u * p.D));

    for (uint s = 0u; s < S; ++s) {
        device const float4* v_ptr = (s < p.num_blocks)
            ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
            : reinterpret_cast<device const float4*>(partial_buf + row_off);
        float sq = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 v  = v_ptr[i];
            const float4 q0 = q0_ptr[i];
            const float4 q1 = q1_ptr[i];
            const float4 q2 = q2_ptr[i];
            sq = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq))));
            d0 = fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, fma(v.w, q0.w, d0))));
            d1 = fma(v.x, q1.x, fma(v.y, q1.y, fma(v.z, q1.z, fma(v.w, q1.w, d1))));
            d2 = fma(v.x, q2.x, fma(v.y, q2.y, fma(v.z, q2.z, fma(v.w, q2.w, d2))));
        }
        const float s_sq = simd_sum(sq);
        const float s_d0 = simd_sum(d0);
        const float s_d1 = simd_sum(d1);
        const float s_d2 = simd_sum(d2);
        if (lane == 0u) {
            tg_sq[s][sgid]     = s_sq;
            tg_dot[0][s][sgid] = s_d0;
            tg_dot[1][s][sgid] = s_d1;
            tg_dot[2][s][sgid] = s_d2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float sc[3][8];
        for (uint s = 0u; s < S; ++s) {
            const float tot_sq = simd_sum((lane < 8u) ? tg_sq[s][lane] : 0.0f);
            const float irms   = rsqrt(tot_sq / float(p.D) + p.eps);
            if (lane == 0u) tg_inv_rms[s] = irms;
            for (uint q = 0u; q < 3u; ++q) {
                const float tot_d = simd_sum((lane < 8u) ? tg_dot[q][s][lane] : 0.0f);
                sc[q][s] = (tot_d * irms) * inv_sqD;
            }
        }
        if (lane < 3u) {
            float mx = -INFINITY;
            for (uint s = 0u; s < S; ++s) mx = max(mx, sc[lane][s]);
            float sm = 0.0f;
            for (uint s = 0u; s < S; ++s) {
                const float e = exp(sc[lane][s] - mx);
                tg_prob[lane][s] = e;
                sm += e;
            }
            const float inv_sm = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
            for (uint s = 0u; s < S; ++s) tg_prob[lane][s] *= inv_sm;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 r0[3], r1[3], r2[3];
    float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
    uint k_idx = 0u;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 a0(0.0f), a1(0.0f), a2(0.0f);
        for (uint s = 0u; s < S; ++s) {
            device const float4* v_ptr = (s < p.num_blocks)
                ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
                : reinterpret_cast<device const float4*>(partial_buf + row_off);
            const float4 vn = v_ptr[i] * tg_inv_rms[s];
            a0 = fma(float4(tg_prob[0][s]), vn, a0);
            a1 = fma(float4(tg_prob[1][s]), vn, a1);
            a2 = fma(float4(tg_prob[2][s]), vn, a2);
        }
        if (k_idx < 3u) { r0[k_idx] = a0; r1[k_idx] = a1; r2[k_idx] = a2; }
        ++k_idx;
        sq0 = fma(a0.x, a0.x, fma(a0.y, a0.y, fma(a0.z, a0.z, fma(a0.w, a0.w, sq0))));
        sq1 = fma(a1.x, a1.x, fma(a1.y, a1.y, fma(a1.z, a1.z, fma(a1.w, a1.w, sq1))));
        sq2 = fma(a2.x, a2.x, fma(a2.y, a2.y, fma(a2.z, a2.z, fma(a2.w, a2.w, sq2))));
    }
    const float s0 = simd_sum(sq0);
    const float s1 = simd_sum(sq1);
    const float s2 = simd_sum(sq2);
    if (lane == 0u) { tg_out_sq[0][sgid] = s0; tg_out_sq[1][sgid] = s1; tg_out_sq[2][sgid] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        for (uint q = 0u; q < 3u; ++q) {
            const float tot = simd_sum((lane < 8u) ? tg_out_sq[q][lane] : 0.0f);
            if (lane == 0u) tg_out_irms[q] = rsqrt(tot / float(p.D) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float ir0 = tg_out_irms[0], ir1 = tg_out_irms[1], ir2 = tg_out_irms[2];
    device const float4* w0 = reinterpret_cast<device const float4*>(nw0);
    device const float4* w1 = reinterpret_cast<device const float4*>(nw1);
    device const float4* w2 = reinterpret_cast<device const float4*>(nw2);
    device half4* h0 = reinterpret_cast<device half4*>(out0 + row_off);
    device half4* h1 = reinterpret_cast<device half4*>(out1 + row_off);
    device half4* h2 = reinterpret_cast<device half4*>(out2 + row_off);

    k_idx = 0u;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 a0, a1, a2;
        if (k_idx < 3u) {
            a0 = r0[k_idx]; a1 = r1[k_idx]; a2 = r2[k_idx];
        } else {
            a0 = float4(0.0f); a1 = float4(0.0f); a2 = float4(0.0f);
            for (uint s = 0u; s < S; ++s) {
                device const float4* v_ptr = (s < p.num_blocks)
                    ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
                    : reinterpret_cast<device const float4*>(partial_buf + row_off);
                const float4 vn = v_ptr[i] * tg_inv_rms[s];
                a0 = fma(float4(tg_prob[0][s]), vn, a0);
                a1 = fma(float4(tg_prob[1][s]), vn, a1);
                a2 = fma(float4(tg_prob[2][s]), vn, a2);
            }
        }
        ++k_idx;
        const half4 hh0 = sik_clamp_f16x4((a0 * ir0) * w0[i]);
        const half4 hh1 = sik_clamp_f16x4((a1 * ir1) * w1[i]);
        const half4 hh2 = sik_clamp_f16x4((a2 * ir2) * w2[i]);
        h0[i] = hh0;
        h1[i] = hh1;
        h2[i] = hh2;
        if (p.M_valid <= 64u) {
            const uint nblk = p.D >> 6u;
            const float bs0 = sik_reduce16((float(hh0.x) + float(hh0.y)) + (float(hh0.z) + float(hh0.w)));
            const float bs1 = sik_reduce16((float(hh1.x) + float(hh1.y)) + (float(hh1.z) + float(hh1.w)));
            const float bs2 = sik_reduce16((float(hh2.x) + float(hh2.y)) + (float(hh2.z) + float(hh2.w)));
            if ((lane & 15u) == 0u) {
                const ulong xidx = ulong(m) * nblk + (i >> 4u);
                xs0[xidx] = bs0;
                xs1[xidx] = bs1;
                xs2[xidx] = bs2;
            }
        }
    }
}

// 576-Thread (`18` SGs = `1` float4/thread!) 1-Pass Register-Resident `AttnRes(1Q) + RMSNorm` (`D == 2304`, 2 barriers, 0 second block read!)
kernel void sik_attn_res_rmsnorm_576(
    device const float*            blocks_buf  [[buffer(0)]],
    device const float*            partial_buf [[buffer(1)]],
    device const float*            query_buf   [[buffer(2)]],
    device const float*            norm_w      [[buffer(3)]],
    device half*                   out_f16     [[buffer(4)]],
    device float*                  out_f32     [[buffer(5)]],
    device float*                  taps_out    [[buffer(6)]],
    constant SikAttnResNormParams& p           [[buffer(7)]],
    device float*                  xsum_out    [[buffer(8)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_padded || tid >= 576u) return;
    const ulong row_off = ulong(m) * p.D;
    if (m >= p.M_valid) {
        reinterpret_cast<device half4*>(out_f16 + row_off)[tid] = half4(0.0h);
        return;
    }

    const uint  S       = min(p.num_blocks + 1u, 8u);
    const float inv_sqD = rsqrt(float(p.D));
    threadgroup float tg_sq[8][18];
    threadgroup float tg_dot[8][18];
    threadgroup float tg_osq[18];

    // Prefetch BOTH q0 and norm_w in flight alongside v[0..S-1]!
    const float4 q0 = reinterpret_cast<device const float4*>(query_buf)[tid];
    const float4 wv = reinterpret_cast<device const float4*>(norm_w)[tid];
    float4 vreg[8];
    for (uint s = 0u; s < S; ++s) {
        device const float4* vp = (s < p.num_blocks)
            ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
            : reinterpret_cast<device const float4*>(partial_buf + row_off);
        const float4 v = vp[tid];
        vreg[s] = v;
        const float ss = simd_sum(fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, v.w * v.w))));
        const float sd = simd_sum(fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, v.w * q0.w))));
        if (lane == 0u) { tg_sq[s][sgid] = ss; tg_dot[s][sgid] = sd; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sc[8], ir[8];
    for (uint s = 0u; s < S; ++s) {
        const float tsq = simd_sum((lane < 18u) ? tg_sq[s][lane] : 0.0f);
        const float td  = simd_sum((lane < 18u) ? tg_dot[s][lane] : 0.0f);
        ir[s] = rsqrt(tsq / float(p.D) + p.eps);
        sc[s] = (td * ir[s]) * inv_sqD;
    }
    float mx = -INFINITY, sm = 0.0f, prob[8];
    for (uint s = 0u; s < S; ++s) mx = max(mx, sc[s]);
    for (uint s = 0u; s < S; ++s) { const float e = exp(sc[s] - mx); prob[s] = e; sm += e; }
    const float inv_sm = (sm > 0.0f) ? (1.0f / sm) : 0.0f;

    float4 a0 = float4(0.0f);
    for (uint s = 0u; s < S; ++s) {
        a0 = fma(float4((prob[s] * inv_sm) * ir[s]), vreg[s], a0);
    }
    const float so = simd_sum(fma(a0.x, a0.x, fma(a0.y, a0.y, fma(a0.z, a0.z, a0.w * a0.w))));
    if (lane == 0u) tg_osq[sgid] = so;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float  tot         = simd_sum((lane < 18u) ? tg_osq[lane] : 0.0f);
    const float  inv_rms_out = rsqrt(tot / float(p.D) + p.eps);
    const float4 normed      = (a0 * inv_rms_out) * wv;
    const half4  h           = sik_clamp_f16x4(normed);
    reinterpret_cast<device half4*>(out_f16 + row_off)[tid] = h;
    if (p.write_f32 != 0u) reinterpret_cast<device float4*>(out_f32 + row_off)[tid] = normed;
    if (p.tap_capture != 0u) {
        reinterpret_cast<device float4*>(taps_out + (ulong(m) * 4ul + p.tap_idx) * p.D)[tid] = normed;
    }
    if (p.M_valid <= 64u) {
        const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
        if ((lane & 15u) == 0u) xsum_out[ulong(m) * (p.D >> 6u) + (tid >> 4u)] = bs;
    }
}

// 576-Thread (`18` SGs = `1` float4/thread!) 1-Pass Register-Resident `AttnRes(3Q) + 3-Way RMSNorm` (`D == 2304`, 2 barriers!)
kernel void sik_attn_res3_rmsnorm3_576(
    device const float*            blocks_buf  [[buffer(0)]],
    device const float*            partial_buf [[buffer(1)]],
    device const float*            query_buf   [[buffer(2)]], // [3, D]
    device const float*            nw0         [[buffer(3)]],
    device const float*            nw1         [[buffer(4)]],
    device const float*            nw2         [[buffer(5)]],
    device half*                   out0        [[buffer(6)]],
    device half*                   out1        [[buffer(7)]],
    device half*                   out2        [[buffer(8)]],
    constant SikAttnResNormParams& p           [[buffer(9)]],
    device float*                  xs0         [[buffer(10)]],
    device float*                  xs1         [[buffer(11)]],
    device float*                  xs2         [[buffer(12)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_padded || tid >= 576u) return;
    const ulong row_off = ulong(m) * p.D;
    if (m >= p.M_valid) {
        reinterpret_cast<device half4*>(out0 + row_off)[tid] = half4(0.0h);
        reinterpret_cast<device half4*>(out1 + row_off)[tid] = half4(0.0h);
        reinterpret_cast<device half4*>(out2 + row_off)[tid] = half4(0.0h);
        return;
    }

    const uint  S       = min(p.num_blocks + 1u, 8u);
    const float inv_sqD = rsqrt(float(p.D));
    threadgroup float tg_sq[8][18];
    threadgroup float tg_dot[3][8][18];
    threadgroup float tg_osq[3][18];

    const float4 q0 = reinterpret_cast<device const float4*>(query_buf)[tid];
    const float4 q1 = reinterpret_cast<device const float4*>(query_buf + ulong(p.D))[tid];
    const float4 q2 = reinterpret_cast<device const float4*>(query_buf + ulong(2u * p.D))[tid];

    float4 vreg[8];
    for (uint s = 0u; s < S; ++s) {
        device const float4* vp = (s < p.num_blocks)
            ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
            : reinterpret_cast<device const float4*>(partial_buf + row_off);
        const float4 v = vp[tid];
        vreg[s] = v;
        const float ss = simd_sum(fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, v.w * v.w))));
        const float d0 = simd_sum(fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, v.w * q0.w))));
        const float d1 = simd_sum(fma(v.x, q1.x, fma(v.y, q1.y, fma(v.z, q1.z, v.w * q1.w))));
        const float d2 = simd_sum(fma(v.x, q2.x, fma(v.y, q2.y, fma(v.z, q2.z, v.w * q2.w))));
        if (lane == 0u) {
            tg_sq[s][sgid]     = ss;
            tg_dot[0][s][sgid] = d0;
            tg_dot[1][s][sgid] = d1;
            tg_dot[2][s][sgid] = d2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sc[3][8], ir[8];
    for (uint s = 0u; s < S; ++s) {
        const float tsq = simd_sum((lane < 18u) ? tg_sq[s][lane] : 0.0f);
        ir[s] = rsqrt(tsq / float(p.D) + p.eps);
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 3u; ++q) {
            const float td = simd_sum((lane < 18u) ? tg_dot[q][s][lane] : 0.0f);
            sc[q][s] = (td * ir[s]) * inv_sqD;
        }
    }
    float prob[3][8];
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < 3u; ++q) {
        float mx = -INFINITY, sm = 0.0f;
        for (uint s = 0u; s < S; ++s) mx = max(mx, sc[q][s]);
        for (uint s = 0u; s < S; ++s) { const float e = exp(sc[q][s] - mx); prob[q][s] = e; sm += e; }
        const float inv = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
        for (uint s = 0u; s < S; ++s) prob[q][s] *= inv;
    }

    float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f);
    for (uint s = 0u; s < S; ++s) {
        const float4 vn = vreg[s] * ir[s];
        a0 = fma(float4(prob[0][s]), vn, a0);
        a1 = fma(float4(prob[1][s]), vn, a1);
        a2 = fma(float4(prob[2][s]), vn, a2);
    }
    const float ss0 = simd_sum(fma(a0.x, a0.x, fma(a0.y, a0.y, fma(a0.z, a0.z, a0.w * a0.w))));
    const float ss1 = simd_sum(fma(a1.x, a1.x, fma(a1.y, a1.y, fma(a1.z, a1.z, a1.w * a1.w))));
    const float ss2 = simd_sum(fma(a2.x, a2.x, fma(a2.y, a2.y, fma(a2.z, a2.z, a2.w * a2.w))));
    if (lane == 0u) { tg_osq[0][sgid] = ss0; tg_osq[1][sgid] = ss1; tg_osq[2][sgid] = ss2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float ir0 = rsqrt(simd_sum((lane < 18u) ? tg_osq[0][lane] : 0.0f) / float(p.D) + p.eps);
    const float ir1 = rsqrt(simd_sum((lane < 18u) ? tg_osq[1][lane] : 0.0f) / float(p.D) + p.eps);
    const float ir2 = rsqrt(simd_sum((lane < 18u) ? tg_osq[2][lane] : 0.0f) / float(p.D) + p.eps);

    const half4 hh0 = sik_clamp_f16x4((a0 * ir0) * reinterpret_cast<device const float4*>(nw0)[tid]);
    const half4 hh1 = sik_clamp_f16x4((a1 * ir1) * reinterpret_cast<device const float4*>(nw1)[tid]);
    const half4 hh2 = sik_clamp_f16x4((a2 * ir2) * reinterpret_cast<device const float4*>(nw2)[tid]);
    reinterpret_cast<device half4*>(out0 + row_off)[tid] = hh0;
    reinterpret_cast<device half4*>(out1 + row_off)[tid] = hh1;
    reinterpret_cast<device half4*>(out2 + row_off)[tid] = hh2;

    if (p.M_valid <= 64u) {
        const uint  nblk = p.D >> 6u;
        const float bs0  = sik_reduce16((float(hh0.x) + float(hh0.y)) + (float(hh0.z) + float(hh0.w)));
        const float bs1  = sik_reduce16((float(hh1.x) + float(hh1.y)) + (float(hh1.z) + float(hh1.w)));
        const float bs2  = sik_reduce16((float(hh2.x) + float(hh2.y)) + (float(hh2.z) + float(hh2.w)));
        if ((lane & 15u) == 0u) {
            const ulong xidx = ulong(m) * nblk + (tid >> 4u);
            xs0[xidx] = bs0;
            xs1[xidx] = bs1;
            xs2[xidx] = bs2;
        }
    }
}

// Rewritten 2-Barrier 4-Message Cross-Pass AttnRes (down from 40 barriers!)
// Grid: `(M_valid, 1, 1)`, `256` threads/TG (`8` SGs)
kernel void sik_cross_pass_attn_res4(
    device const float*        blocks_buf  [[buffer(0)]],
    device const float*        partial_buf [[buffer(1)]],
    device const float*        q_am_buf    [[buffer(2)]],
    device const float*        q_kv_buf    [[buffer(3)]],
    device float*              out_attn    [[buffer(4)]],
    device float*              out_k       [[buffer(5)]],
    device float*              out_v       [[buffer(6)]],
    device float*              out_mlp     [[buffer(7)]],
    constant SikAttnResParams& p           [[buffer(8)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_valid) return;
    const uint  S       = min(p.num_blocks + 1u, 8u);
    const uint  d4_cnt  = p.D >> 2u;
    const float inv_sqD = rsqrt(float(p.D));

    threadgroup float tg_sq[8][8];
    threadgroup float tg_dot[4][8][8];
    threadgroup float tg_inv_rms[8];
    threadgroup float tg_prob[4][8];

    device const float4* q0_ptr = reinterpret_cast<device const float4*>(q_am_buf);
    device const float4* q1_ptr = reinterpret_cast<device const float4*>(q_kv_buf);
    device const float4* q2_ptr = reinterpret_cast<device const float4*>(q_kv_buf + ulong(p.D));
    device const float4* q3_ptr = reinterpret_cast<device const float4*>(q_am_buf + ulong(p.D));

    for (uint s = 0u; s < S; ++s) {
        device const float4* v_ptr = (s < p.num_blocks)
            ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + ulong(m) * p.D)
            : reinterpret_cast<device const float4*>(partial_buf + ulong(m) * p.D);

        float sq = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 v  = v_ptr[i];
            const float4 q0 = q0_ptr[i];
            const float4 q1 = q1_ptr[i];
            const float4 q2 = q2_ptr[i];
            const float4 q3 = q3_ptr[i];
            sq = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq))));
            d0 = fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, fma(v.w, q0.w, d0))));
            d1 = fma(v.x, q1.x, fma(v.y, q1.y, fma(v.z, q1.z, fma(v.w, q1.w, d1))));
            d2 = fma(v.x, q2.x, fma(v.y, q2.y, fma(v.z, q2.z, fma(v.w, q2.w, d2))));
            d3 = fma(v.x, q3.x, fma(v.y, q3.y, fma(v.z, q3.z, fma(v.w, q3.w, d3))));
        }
        const float s_sq = simd_sum(sq);
        const float s_d0 = simd_sum(d0);
        const float s_d1 = simd_sum(d1);
        const float s_d2 = simd_sum(d2);
        const float s_d3 = simd_sum(d3);
        if (lane == 0u) {
            tg_sq[s][sgid]     = s_sq;
            tg_dot[0][s][sgid] = s_d0;
            tg_dot[1][s][sgid] = s_d1;
            tg_dot[2][s][sgid] = s_d2;
            tg_dot[3][s][sgid] = s_d3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float sc[4][8];
        for (uint s = 0u; s < S; ++s) {
            const float tot_sq = simd_sum((lane < 8u) ? tg_sq[s][lane] : 0.0f);
            const float ir     = rsqrt(tot_sq / float(p.D) + p.eps);
            if (lane == 0u) tg_inv_rms[s] = ir;
            #pragma clang loop unroll(full)
            for (uint q = 0u; q < 4u; ++q) {
                const float tot_d = simd_sum((lane < 8u) ? tg_dot[q][s][lane] : 0.0f);
                sc[q][s] = (tot_d * ir) * inv_sqD;
            }
        }
        if (lane < 4u) {
            float mx = -INFINITY;
            for (uint s = 0u; s < S; ++s) mx = max(mx, sc[lane][s]);
            float sum = 0.0f;
            for (uint s = 0u; s < S; ++s) {
                const float ex = exp(sc[lane][s] - mx);
                tg_prob[lane][s] = ex;
                sum += ex;
            }
            const float inv = 1.0f / max(sum, 1e-30f);
            for (uint s = 0u; s < S; ++s) tg_prob[lane][s] *= inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float4* dst0 = reinterpret_cast<device float4*>(out_attn + ulong(m) * p.D);
    device float4* dst1 = reinterpret_cast<device float4*>(out_k    + ulong(m) * p.D);
    device float4* dst2 = reinterpret_cast<device float4*>(out_v    + ulong(m) * p.D);
    device float4* dst3 = reinterpret_cast<device float4*>(out_mlp  + ulong(m) * p.D);

    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 a0(0.0f), a1(0.0f), a2(0.0f), a3(0.0f);
        for (uint s = 0u; s < S; ++s) {
            device const float4* v_ptr = (s < p.num_blocks)
                ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + ulong(m) * p.D)
                : reinterpret_cast<device const float4*>(partial_buf + ulong(m) * p.D);
            const float4 vn = v_ptr[i] * tg_inv_rms[s];
            a0 = fma(float4(tg_prob[0][s]), vn, a0);
            a1 = fma(float4(tg_prob[1][s]), vn, a1);
            a2 = fma(float4(tg_prob[2][s]), vn, a2);
            a3 = fma(float4(tg_prob[3][s]), vn, a3);
        }
        dst0[i] = a0; dst1[i] = a1; dst2[i] = a2; dst3[i] = a3;
    }
}

// FUSED PARALLEL 1-DISPATCH `cross_pass_attn_res4 + attn_res(1 or 3) + loop_ctrl_prep3` (decode `M <= 8`)!
// Eliminates 48 serial 1-TG dispatches/step AND runs `cross_pass` (`tgid.y==0`) + `attn_res` (`tgid.y==1`) IN PARALLEL!
// Grid: `(M_valid, 2, 1)`, `256` threads/TG (`8` SGs)
struct SikLoopPreAttnParams {
    uint  M_valid;
    uint  D;
    uint  bank_count;
    uint  num_blocks;
    uint  block_stride;
    uint  num_attn_q; // 1 (local) or 3 (global)
    float eps;
    uint  pad0;
};

kernel void sik_loop_pre_attn_prep3(
    device const float*            bank_buf     [[buffer(0)]],
    device const float*            bank_last    [[buffer(1)]],
    device const float*            q_am_buf     [[buffer(2)]],
    device const float*            q_kv_buf     [[buffer(3)]],
    device const float*            blocks_buf   [[buffer(4)]],
    device const float*            cur_buf      [[buffer(5)]],
    device const float*            attn_res_q   [[buffer(6)]],
    device float*                  msg0         [[buffer(7)]],
    device float*                  msg1         [[buffer(8)]],
    device float*                  msg2         [[buffer(9)]],
    device float*                  msg3         [[buffer(10)]],
    device float*                  inp0         [[buffer(11)]],
    device float*                  inp1         [[buffer(12)]],
    device float*                  inp2         [[buffer(13)]],
    device half*                   cat_out      [[buffer(14)]], // [3, M_valid, 2 * D]
    constant SikLoopPreAttnParams& p            [[buffer(15)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m    = tgid.x;
    const uint side = tgid.y;
    if (m >= p.M_valid || side >= 2u) return;

    const uint  d4_cnt  = p.D >> 2u;
    const ulong row_off = ulong(m) * p.D;
    const float inv_sqD = rsqrt(float(p.D));

    threadgroup float tg_sq[8][8];
    threadgroup float tg_dot[4][8][8];
    threadgroup float tg_inv_rms[8];
    threadgroup float tg_prob[4][8];
    threadgroup float tg_osq[3][8];
    threadgroup float tg_oirms[3];

    if (side == 0u) {
        // Side 0: Cross-Pass AttnRes (4 msgs) + unweighted RMSNorm -> right half `[D .. 2*D-1]` of `cat_out[0..2]`!
        const uint S = min(p.bank_count + 1u, 8u);
        device const float4* q0_ptr = reinterpret_cast<device const float4*>(q_am_buf);
        device const float4* q1_ptr = reinterpret_cast<device const float4*>(q_kv_buf);
        device const float4* q2_ptr = reinterpret_cast<device const float4*>(q_kv_buf + ulong(p.D));
        device const float4* q3_ptr = reinterpret_cast<device const float4*>(q_am_buf + ulong(p.D));

        for (uint s = 0u; s < S; ++s) {
            device const float4* v_ptr = (s < p.bank_count)
                ? reinterpret_cast<device const float4*>(bank_buf + ulong(s) * p.block_stride + row_off)
                : reinterpret_cast<device const float4*>(bank_last + row_off);
            float sq = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (uint i = tid; i < d4_cnt; i += 256u) {
                const float4 v = v_ptr[i];
                const float4 q0 = q0_ptr[i], q1 = q1_ptr[i], q2 = q2_ptr[i], q3 = q3_ptr[i];
                sq = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq))));
                d0 = fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, fma(v.w, q0.w, d0))));
                d1 = fma(v.x, q1.x, fma(v.y, q1.y, fma(v.z, q1.z, fma(v.w, q1.w, d1))));
                d2 = fma(v.x, q2.x, fma(v.y, q2.y, fma(v.z, q2.z, fma(v.w, q2.w, d2))));
                d3 = fma(v.x, q3.x, fma(v.y, q3.y, fma(v.z, q3.z, fma(v.w, q3.w, d3))));
            }
            const float s_sq = simd_sum(sq);
            const float s_d0 = simd_sum(d0), s_d1 = simd_sum(d1), s_d2 = simd_sum(d2), s_d3 = simd_sum(d3);
            if (lane == 0u) {
                tg_sq[s][sgid] = s_sq;
                tg_dot[0][s][sgid] = s_d0; tg_dot[1][s][sgid] = s_d1;
                tg_dot[2][s][sgid] = s_d2; tg_dot[3][s][sgid] = s_d3;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float sc[4][8];
            for (uint s = 0u; s < S; ++s) {
                const float tot_sq = simd_sum((lane < 8u) ? tg_sq[s][lane] : 0.0f);
                const float ir     = rsqrt(tot_sq / float(p.D) + p.eps);
                if (lane == 0u) tg_inv_rms[s] = ir;
                #pragma clang loop unroll(full)
                for (uint q = 0u; q < 4u; ++q) {
                    const float tot_d = simd_sum((lane < 8u) ? tg_dot[q][s][lane] : 0.0f);
                    sc[q][s] = (tot_d * ir) * inv_sqD;
                }
            }
            if (lane < 4u) {
                float mx = -INFINITY;
                for (uint s = 0u; s < S; ++s) mx = max(mx, sc[lane][s]);
                float sm = 0.0f;
                for (uint s = 0u; s < S; ++s) { const float e = exp(sc[lane][s] - mx); tg_prob[lane][s] = e; sm += e; }
                const float inv = 1.0f / max(sm, 1e-30f);
                for (uint s = 0u; s < S; ++s) tg_prob[lane][s] *= inv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float4* dm0 = reinterpret_cast<device float4*>(msg0 + row_off);
        device float4* dm1 = reinterpret_cast<device float4*>(msg1 + row_off);
        device float4* dm2 = reinterpret_cast<device float4*>(msg2 + row_off);
        device float4* dm3 = reinterpret_cast<device float4*>(msg3 + row_off);

        float4 r0[3], r1[3], r2[3];
        float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
        uint k_idx = 0u;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            float4 a0(0.0f), a1(0.0f), a2(0.0f), a3(0.0f);
            for (uint s = 0u; s < S; ++s) {
                device const float4* v_ptr = (s < p.bank_count)
                    ? reinterpret_cast<device const float4*>(bank_buf + ulong(s) * p.block_stride + row_off)
                    : reinterpret_cast<device const float4*>(bank_last + row_off);
                const float4 vn = v_ptr[i] * tg_inv_rms[s];
                a0 = fma(float4(tg_prob[0][s]), vn, a0);
                a1 = fma(float4(tg_prob[1][s]), vn, a1);
                a2 = fma(float4(tg_prob[2][s]), vn, a2);
                a3 = fma(float4(tg_prob[3][s]), vn, a3);
            }
            dm0[i] = a0; dm1[i] = a1; dm2[i] = a2; dm3[i] = a3;
            if (k_idx < 3u) { r0[k_idx] = a0; r1[k_idx] = a1; r2[k_idx] = a2; }
            ++k_idx;
            sq0 = fma(a0.x, a0.x, fma(a0.y, a0.y, fma(a0.z, a0.z, fma(a0.w, a0.w, sq0))));
            sq1 = fma(a1.x, a1.x, fma(a1.y, a1.y, fma(a1.z, a1.z, fma(a1.w, a1.w, sq1))));
            sq2 = fma(a2.x, a2.x, fma(a2.y, a2.y, fma(a2.z, a2.z, fma(a2.w, a2.w, sq2))));
        }
        const float ss0 = simd_sum(sq0), ss1 = simd_sum(sq1), ss2 = simd_sum(sq2);
        if (lane == 0u) { tg_osq[0][sgid] = ss0; tg_osq[1][sgid] = ss1; tg_osq[2][sgid] = ss2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            for (uint q = 0u; q < 3u; ++q) {
                const float tot = simd_sum((lane < 8u) ? tg_osq[q][lane] : 0.0f);
                if (lane == 0u) tg_oirms[q] = rsqrt(tot / float(p.D) + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float ir0 = tg_oirms[0], ir1 = tg_oirms[1], ir2 = tg_oirms[2];
        device half4* c0 = reinterpret_cast<device half4*>(cat_out + (0ul * p.M_valid + m) * (2ul * p.D) + p.D);
        device half4* c1 = reinterpret_cast<device half4*>(cat_out + (1ul * p.M_valid + m) * (2ul * p.D) + p.D);
        device half4* c2 = reinterpret_cast<device half4*>(cat_out + (2ul * p.M_valid + m) * (2ul * p.D) + p.D);
        k_idx = 0u;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 a0 = (k_idx < 3u) ? r0[k_idx] : dm0[i];
            const float4 a1 = (k_idx < 3u) ? r1[k_idx] : dm1[i];
            const float4 a2 = (k_idx < 3u) ? r2[k_idx] : dm2[i];
            ++k_idx;
            c0[i] = sik_clamp_f16x4(a0 * ir0);
            c1[i] = sik_clamp_f16x4(a1 * ir1);
            c2[i] = sik_clamp_f16x4(a2 * ir2);
        }
    } else {
        // Side 1: Pre-Attention AttnRes (`1` or `3` queries) + unweighted RMSNorm -> left half `[0 .. D-1]` of `cat_out[0..2]`!
        const uint S = min(p.num_blocks + 1u, 8u);
        const uint Q = p.num_attn_q; // 1 or 3
        device const float4* q0_ptr = reinterpret_cast<device const float4*>(attn_res_q);
        device const float4* q1_ptr = reinterpret_cast<device const float4*>(attn_res_q + ulong(p.D));
        device const float4* q2_ptr = reinterpret_cast<device const float4*>(attn_res_q + ulong(2u * p.D));

        for (uint s = 0u; s < S; ++s) {
            device const float4* v_ptr = (s < p.num_blocks)
                ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
                : reinterpret_cast<device const float4*>(cur_buf + row_off);
            float sq = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            for (uint i = tid; i < d4_cnt; i += 256u) {
                const float4 v  = v_ptr[i];
                const float4 q0 = q0_ptr[i];
                sq = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq))));
                d0 = fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, fma(v.w, q0.w, d0))));
                if (Q >= 2u) {
                    const float4 q1 = q1_ptr[i], q2 = q2_ptr[i];
                    d1 = fma(v.x, q1.x, fma(v.y, q1.y, fma(v.z, q1.z, fma(v.w, q1.w, d1))));
                    d2 = fma(v.x, q2.x, fma(v.y, q2.y, fma(v.z, q2.z, fma(v.w, q2.w, d2))));
                }
            }
            const float s_sq = simd_sum(sq);
            const float s_d0 = simd_sum(d0);
            const float s_d1 = (Q >= 2u) ? simd_sum(d1) : 0.0f;
            const float s_d2 = (Q >= 2u) ? simd_sum(d2) : 0.0f;
            if (lane == 0u) {
                tg_sq[s][sgid] = s_sq; tg_dot[0][s][sgid] = s_d0;
                if (Q >= 2u) { tg_dot[1][s][sgid] = s_d1; tg_dot[2][s][sgid] = s_d2; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgid == 0u) {
            float sc[3][8];
            for (uint s = 0u; s < S; ++s) {
                const float tot_sq = simd_sum((lane < 8u) ? tg_sq[s][lane] : 0.0f);
                const float ir     = rsqrt(tot_sq / float(p.D) + p.eps);
                if (lane == 0u) tg_inv_rms[s] = ir;
                for (uint q = 0u; q < Q; ++q) {
                    const float tot_d = simd_sum((lane < 8u) ? tg_dot[q][s][lane] : 0.0f);
                    sc[q][s] = (tot_d * ir) * inv_sqD;
                }
            }
            if (lane < Q) {
                float mx = -INFINITY;
                for (uint s = 0u; s < S; ++s) mx = max(mx, sc[lane][s]);
                float sm = 0.0f;
                for (uint s = 0u; s < S; ++s) { const float e = exp(sc[lane][s] - mx); tg_prob[lane][s] = e; sm += e; }
                const float inv = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
                for (uint s = 0u; s < S; ++s) tg_prob[lane][s] *= inv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float4* di0 = reinterpret_cast<device float4*>(inp0 + row_off);
        device float4* di1 = (Q >= 2u) ? reinterpret_cast<device float4*>(inp1 + row_off) : nullptr;
        device float4* di2 = (Q >= 2u) ? reinterpret_cast<device float4*>(inp2 + row_off) : nullptr;

        float4 r0[3], r1[3], r2[3];
        float sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f;
        uint k_idx = 0u;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            float4 a0(0.0f), a1(0.0f), a2(0.0f);
            for (uint s = 0u; s < S; ++s) {
                device const float4* v_ptr = (s < p.num_blocks)
                    ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
                    : reinterpret_cast<device const float4*>(cur_buf + row_off);
                const float4 vn = v_ptr[i] * tg_inv_rms[s];
                a0 = fma(float4(tg_prob[0][s]), vn, a0);
                if (Q >= 2u) {
                    a1 = fma(float4(tg_prob[1][s]), vn, a1);
                    a2 = fma(float4(tg_prob[2][s]), vn, a2);
                }
            }
            di0[i] = a0;
            if (di1) di1[i] = a1;
            if (di2) di2[i] = a2;
            if (k_idx < 3u) { r0[k_idx] = a0; r1[k_idx] = a1; r2[k_idx] = a2; }
            ++k_idx;
            sq0 = fma(a0.x, a0.x, fma(a0.y, a0.y, fma(a0.z, a0.z, fma(a0.w, a0.w, sq0))));
            if (Q >= 2u) {
                sq1 = fma(a1.x, a1.x, fma(a1.y, a1.y, fma(a1.z, a1.z, fma(a1.w, a1.w, sq1))));
                sq2 = fma(a2.x, a2.x, fma(a2.y, a2.y, fma(a2.z, a2.z, fma(a2.w, a2.w, sq2))));
            }
        }
        const float ss0 = simd_sum(sq0);
        const float ss1 = (Q >= 2u) ? simd_sum(sq1) : 0.0f;
        const float ss2 = (Q >= 2u) ? simd_sum(sq2) : 0.0f;
        if (lane == 0u) {
            tg_osq[0][sgid] = ss0;
            if (Q >= 2u) { tg_osq[1][sgid] = ss1; tg_osq[2][sgid] = ss2; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            for (uint q = 0u; q < Q; ++q) {
                const float tot = simd_sum((lane < 8u) ? tg_osq[q][lane] : 0.0f);
                if (lane == 0u) tg_oirms[q] = rsqrt(tot / float(p.D) + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float ir0 = tg_oirms[0];
        const float ir1 = (Q >= 2u) ? tg_oirms[1] : ir0;
        const float ir2 = (Q >= 2u) ? tg_oirms[2] : ir0;
        device half4* c0 = reinterpret_cast<device half4*>(cat_out + (0ul * p.M_valid + m) * (2ul * p.D));
        device half4* c1 = reinterpret_cast<device half4*>(cat_out + (1ul * p.M_valid + m) * (2ul * p.D));
        device half4* c2 = reinterpret_cast<device half4*>(cat_out + (2ul * p.M_valid + m) * (2ul * p.D));
        k_idx = 0u;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 a0 = (k_idx < 3u) ? r0[k_idx] : di0[i];
            const half4  h0 = sik_clamp_f16x4(a0 * ir0);
            c0[i] = h0;
            if (Q >= 2u) {
                const float4 a1 = (k_idx < 3u) ? r1[k_idx] : di1[i];
                const float4 a2 = (k_idx < 3u) ? r2[k_idx] : di2[i];
                c1[i] = sik_clamp_f16x4(a1 * ir1);
                c2[i] = sik_clamp_f16x4(a2 * ir2);
            } else {
                c1[i] = h0;
                c2[i] = h0;
            }
            ++k_idx;
        }
    }
}

// FUSED 1-DISPATCH `attn_res(1 query) + loop_ctrl_prep1` for Pre-MLP Looped Visits (decode `M <= 8`)!
// Eliminates 24 serial 1-TG dispatches/step & intermediate `b_inputs[0]` DRAM read!
// Grid: `(M_valid, 1, 1)`, `256` threads/TG (`8` SGs)
kernel void sik_attn_res_loop_prep1(
    device const float*            blocks_buf   [[buffer(0)]],
    device const float*            cur_buf      [[buffer(1)]],
    device const float*            query_buf    [[buffer(2)]],
    device const float*            msg3_in      [[buffer(3)]],
    device float*                  inp0_out     [[buffer(4)]],
    device half*                   cat_out      [[buffer(5)]], // [M_valid, 2 * D]
    constant SikAttnResParams&     p            [[buffer(6)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_valid) return;
    const uint  S       = min(p.num_blocks + 1u, 8u);
    const uint  d4_cnt  = p.D >> 2u;
    const ulong row_off = ulong(m) * p.D;
    const float inv_sqD = rsqrt(float(p.D));

    threadgroup float tg_sq[8][8];
    threadgroup float tg_dot[8][8];
    threadgroup float tg_msq[8];
    threadgroup float tg_inv_rms[8];
    threadgroup float tg_prob[8];
    threadgroup float tg_mirms[1];
    threadgroup float tg_osq[8];
    threadgroup float tg_oirms[1];

    device const float4* q0_ptr = reinterpret_cast<device const float4*>(query_buf);
    device const float4* m3_ptr = reinterpret_cast<device const float4*>(msg3_in + row_off);

    for (uint s = 0u; s < S; ++s) {
        device const float4* v_ptr = (s < p.num_blocks)
            ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
            : reinterpret_cast<device const float4*>(cur_buf + row_off);
        float sq = 0.0f, d0 = 0.0f;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 v  = v_ptr[i];
            const float4 q0 = q0_ptr[i];
            sq = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq))));
            d0 = fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, fma(v.w, q0.w, d0))));
        }
        const float s_sq = simd_sum(sq), s_d0 = simd_sum(d0);
        if (lane == 0u) { tg_sq[s][sgid] = s_sq; tg_dot[s][sgid] = s_d0; }
    }
    float msq = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 mv = m3_ptr[i];
        msq = fma(mv.x, mv.x, fma(mv.y, mv.y, fma(mv.z, mv.z, fma(mv.w, mv.w, msq))));
    }
    const float s_msq = simd_sum(msq);
    if (lane == 0u) tg_msq[sgid] = s_msq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float sc[8];
        for (uint s = 0u; s < S; ++s) {
            const float tot_sq = simd_sum((lane < 8u) ? tg_sq[s][lane] : 0.0f);
            const float tot_d  = simd_sum((lane < 8u) ? tg_dot[s][lane] : 0.0f);
            const float ir     = rsqrt(tot_sq / float(p.D) + p.eps);
            if (lane == 0u) tg_inv_rms[s] = ir;
            sc[s] = (tot_d * ir) * inv_sqD;
        }
        const float tot_m = simd_sum((lane < 8u) ? tg_msq[lane] : 0.0f);
        if (lane == 0u) {
            tg_mirms[0] = rsqrt(tot_m / float(p.D) + p.eps);
            float mx = -INFINITY;
            for (uint s = 0u; s < S; ++s) mx = max(mx, sc[s]);
            float sm = 0.0f;
            for (uint s = 0u; s < S; ++s) { const float e = exp(sc[s] - mx); tg_prob[s] = e; sm += e; }
            const float inv = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
            for (uint s = 0u; s < S; ++s) tg_prob[s] *= inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float4* di0 = reinterpret_cast<device float4*>(inp0_out + row_off);
    device half4*  c_l = reinterpret_cast<device half4*>(cat_out + ulong(m) * (2ul * p.D));
    device half4*  c_r = reinterpret_cast<device half4*>(cat_out + ulong(m) * (2ul * p.D) + p.D);
    const float mir    = tg_mirms[0];

    float4 r0[3];
    float osq = 0.0f;
    uint k_idx = 0u;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 a0(0.0f);
        for (uint s = 0u; s < S; ++s) {
            device const float4* v_ptr = (s < p.num_blocks)
                ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
                : reinterpret_cast<device const float4*>(cur_buf + row_off);
            a0 = fma(float4(tg_prob[s] * tg_inv_rms[s]), v_ptr[i], a0);
        }
        di0[i] = a0;
        c_r[i] = sik_clamp_f16x4(m3_ptr[i] * mir);
        if (k_idx < 3u) r0[k_idx] = a0;
        ++k_idx;
        osq = fma(a0.x, a0.x, fma(a0.y, a0.y, fma(a0.z, a0.z, fma(a0.w, a0.w, osq))));
    }
    const float s_osq = simd_sum(osq);
    if (lane == 0u) tg_osq[sgid] = s_osq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_osq[lane] : 0.0f);
        if (lane == 0u) tg_oirms[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float oir = tg_oirms[0];
    k_idx = 0u;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 a0 = (k_idx < 3u) ? r0[k_idx] : di0[i];
        ++k_idx;
        c_l[i] = sik_clamp_f16x4(a0 * oir);
    }
}

// Batch-12 Parallel Cross-Pass AttnRes + Unweighted RMSNorm at Loop Pass 1 & Pass 2 Start (`M <= 8`, `D == 2304`)!
// Computes ALL 12 core layers' `msg[0..3]` AND `rms(msg[0..3])` in ONE 12-TG dispatch sharing `b_bank` in L2!
// Replaces 24 serial Side-0 runs AND 24 `msg3` RMSNorms (`~0.03 ms` total vs `~0.26 ms`!).
// Grid: `(M_valid, num_core_layers /* 12 */, 1)`, `576` threads/TG (`18` SGs)
kernel void sik_cross_pass_batch12_576(
    device const float*            bank_buf     [[buffer(0)]],
    device const float*            bank_last    [[buffer(1)]],
    device const float*            q_am_all     [[buffer(2)]], // [12 * 2, D]
    device const float*            q_kv_all     [[buffer(3)]], // [12 * 2, D]
    device float*                  msg_bank12   [[buffer(4)]], // [12, 4, M_valid, D]
    device half*                   msg_norm12   [[buffer(5)]], // [12, 4, M_valid, D]
    constant SikAttnResParams&     p            [[buffer(6)]], // num_blocks = bank_count, num_queries = num_core_layers
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m  = tgid.x;
    const uint li = tgid.y;
    if (m >= p.M_valid || li >= p.num_queries || tid >= 576u) return;

    const uint  S       = min(p.num_blocks + 1u, 8u);
    const ulong row_off = ulong(m) * p.D;
    const float inv_sqD = rsqrt(float(p.D));

    threadgroup float tg_sq[8][18];
    threadgroup float tg_dot[4][8][18];
    threadgroup float tg_osq[4][18];

    const ulong qrow = ulong(2u * li) * p.D;
    const float4 q0 = reinterpret_cast<device const float4*>(q_am_all + qrow)[tid];
    const float4 q1 = reinterpret_cast<device const float4*>(q_kv_all + qrow)[tid];
    const float4 q2 = reinterpret_cast<device const float4*>(q_kv_all + qrow + p.D)[tid];
    const float4 q3 = reinterpret_cast<device const float4*>(q_am_all + qrow + p.D)[tid];

    float4 vreg[8];
    for (uint s = 0u; s < S; ++s) {
        device const float4* vp = (s < p.num_blocks)
            ? reinterpret_cast<device const float4*>(bank_buf + ulong(s) * p.block_stride + row_off)
            : reinterpret_cast<device const float4*>(bank_last + row_off);
        const float4 v = vp[tid];
        vreg[s] = v;
        const float ss = simd_sum(fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, v.w * v.w))));
        const float d0 = simd_sum(fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, v.w * q0.w))));
        const float d1 = simd_sum(fma(v.x, q1.x, fma(v.y, q1.y, fma(v.z, q1.z, v.w * q1.w))));
        const float d2 = simd_sum(fma(v.x, q2.x, fma(v.y, q2.y, fma(v.z, q2.z, v.w * q2.w))));
        const float d3 = simd_sum(fma(v.x, q3.x, fma(v.y, q3.y, fma(v.z, q3.z, v.w * q3.w))));
        if (lane == 0u) {
            tg_sq[s][sgid]     = ss;
            tg_dot[0][s][sgid] = d0; tg_dot[1][s][sgid] = d1;
            tg_dot[2][s][sgid] = d2; tg_dot[3][s][sgid] = d3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sc[4][8], ir[8];
    for (uint s = 0u; s < S; ++s) {
        const float tsq = simd_sum((lane < 18u) ? tg_sq[s][lane] : 0.0f);
        ir[s] = rsqrt(tsq / float(p.D) + p.eps);
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 4u; ++q) {
            const float td = simd_sum((lane < 18u) ? tg_dot[q][s][lane] : 0.0f);
            sc[q][s] = (td * ir[s]) * inv_sqD;
        }
    }
    float prob[4][8];
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < 4u; ++q) {
        float mx = -INFINITY, sm = 0.0f;
        for (uint s = 0u; s < S; ++s) mx = max(mx, sc[q][s]);
        for (uint s = 0u; s < S; ++s) { const float e = exp(sc[q][s] - mx); prob[q][s] = e; sm += e; }
        const float inv = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
        for (uint s = 0u; s < S; ++s) prob[q][s] *= inv;
    }

    float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
    for (uint s = 0u; s < S; ++s) {
        const float4 vn = vreg[s] * ir[s];
        a0 = fma(float4(prob[0][s]), vn, a0);
        a1 = fma(float4(prob[1][s]), vn, a1);
        a2 = fma(float4(prob[2][s]), vn, a2);
        a3 = fma(float4(prob[3][s]), vn, a3);
    }
    const ulong base_slot = ulong(li) * 4ul;
    reinterpret_cast<device float4*>(msg_bank12 + ((base_slot + 0ul) * p.M_valid + m) * p.D)[tid] = a0;
    reinterpret_cast<device float4*>(msg_bank12 + ((base_slot + 1ul) * p.M_valid + m) * p.D)[tid] = a1;
    reinterpret_cast<device float4*>(msg_bank12 + ((base_slot + 2ul) * p.M_valid + m) * p.D)[tid] = a2;
    reinterpret_cast<device float4*>(msg_bank12 + ((base_slot + 3ul) * p.M_valid + m) * p.D)[tid] = a3;

    const float ss0 = simd_sum(fma(a0.x, a0.x, fma(a0.y, a0.y, fma(a0.z, a0.z, a0.w * a0.w))));
    const float ss1 = simd_sum(fma(a1.x, a1.x, fma(a1.y, a1.y, fma(a1.z, a1.z, a1.w * a1.w))));
    const float ss2 = simd_sum(fma(a2.x, a2.x, fma(a2.y, a2.y, fma(a2.z, a2.z, a2.w * a2.w))));
    const float ss3 = simd_sum(fma(a3.x, a3.x, fma(a3.y, a3.y, fma(a3.z, a3.z, a3.w * a3.w))));
    if (lane == 0u) {
        tg_osq[0][sgid] = ss0; tg_osq[1][sgid] = ss1;
        tg_osq[2][sgid] = ss2; tg_osq[3][sgid] = ss3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float ir0 = rsqrt(simd_sum((lane < 18u) ? tg_osq[0][lane] : 0.0f) / float(p.D) + p.eps);
    const float ir1 = rsqrt(simd_sum((lane < 18u) ? tg_osq[1][lane] : 0.0f) / float(p.D) + p.eps);
    const float ir2 = rsqrt(simd_sum((lane < 18u) ? tg_osq[2][lane] : 0.0f) / float(p.D) + p.eps);
    const float ir3 = rsqrt(simd_sum((lane < 18u) ? tg_osq[3][lane] : 0.0f) / float(p.D) + p.eps);

    reinterpret_cast<device half4*>(msg_norm12 + ((base_slot + 0ul) * p.M_valid + m) * p.D)[tid] = sik_clamp_f16x4(a0 * ir0);
    reinterpret_cast<device half4*>(msg_norm12 + ((base_slot + 1ul) * p.M_valid + m) * p.D)[tid] = sik_clamp_f16x4(a1 * ir1);
    reinterpret_cast<device half4*>(msg_norm12 + ((base_slot + 2ul) * p.M_valid + m) * p.D)[tid] = sik_clamp_f16x4(a2 * ir2);
    reinterpret_cast<device half4*>(msg_norm12 + ((base_slot + 3ul) * p.M_valid + m) * p.D)[tid] = sik_clamp_f16x4(a3 * ir3);
}

// 576-Thread 1-Pass Register-Resident Block AttnRes + Left-Half Unweighted RMSNorm (`Q = 1` or `3`, `D == 2304`)!
// Writes `inp[0..Q-1]` F32 AND compact `cat_left[0..cat_ways-1]` F16 (stride `D`, half bandwidth of `2*D`!).
// Used for BOTH PreAttn looped (`cat_ways = 3`) AND PreMoE looped (`cat_ways = 1`) after Batch-12 CrossPass!
// Grid: `(M_valid, 1, 1)`, `576` threads/TG (`18` SGs)
kernel void sik_attn_res_prep_left_576(
    device const float*        blocks_buf [[buffer(0)]],
    device const float*        cur_buf    [[buffer(1)]],
    device const float*        query_buf  [[buffer(2)]],
    device float*              inp0       [[buffer(3)]],
    device float*              inp1       [[buffer(4)]],
    device float*              inp2       [[buffer(5)]],
    device half*               cat_left   [[buffer(6)]], // [cat_ways, M_valid, D]
    constant SikAttnResParams& p          [[buffer(7)]], // num_queries = 1 or 3, pad0 = cat_ways (1 or 3)
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_valid || tid >= 576u) return;
    const uint  S       = min(p.num_blocks + 1u, 8u);
    const uint  Q       = p.num_queries; // 1 or 3
    const ulong row_off = ulong(m) * p.D;
    const float inv_sqD = rsqrt(float(p.D));

    threadgroup float tg_sq[8][18];
    threadgroup float tg_dot[3][8][18];
    threadgroup float tg_osq[3][18];

    const float4 q0 = reinterpret_cast<device const float4*>(query_buf)[tid];
    const float4 q1 = (Q >= 2u) ? reinterpret_cast<device const float4*>(query_buf + ulong(p.D))[tid] : float4(0.0f);
    const float4 q2 = (Q >= 2u) ? reinterpret_cast<device const float4*>(query_buf + ulong(2u * p.D))[tid] : float4(0.0f);

    float4 vreg[8];
    for (uint s = 0u; s < S; ++s) {
        device const float4* vp = (s < p.num_blocks)
            ? reinterpret_cast<device const float4*>(blocks_buf + ulong(s) * p.block_stride + row_off)
            : reinterpret_cast<device const float4*>(cur_buf + row_off);
        const float4 v = vp[tid];
        vreg[s] = v;
        const float ss = simd_sum(fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, v.w * v.w))));
        const float d0 = simd_sum(fma(v.x, q0.x, fma(v.y, q0.y, fma(v.z, q0.z, v.w * q0.w))));
        const float d1 = (Q >= 2u) ? simd_sum(fma(v.x, q1.x, fma(v.y, q1.y, fma(v.z, q1.z, v.w * q1.w)))) : 0.0f;
        const float d2 = (Q >= 2u) ? simd_sum(fma(v.x, q2.x, fma(v.y, q2.y, fma(v.z, q2.z, v.w * q2.w)))) : 0.0f;
        if (lane == 0u) {
            tg_sq[s][sgid]     = ss;
            tg_dot[0][s][sgid] = d0;
            if (Q >= 2u) { tg_dot[1][s][sgid] = d1; tg_dot[2][s][sgid] = d2; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sc[3][8], ir[8];
    for (uint s = 0u; s < S; ++s) {
        const float tsq = simd_sum((lane < 18u) ? tg_sq[s][lane] : 0.0f);
        ir[s] = rsqrt(tsq / float(p.D) + p.eps);
        for (uint q = 0u; q < Q; ++q) {
            const float td = simd_sum((lane < 18u) ? tg_dot[q][s][lane] : 0.0f);
            sc[q][s] = (td * ir[s]) * inv_sqD;
        }
    }
    float prob[3][8];
    for (uint q = 0u; q < Q; ++q) {
        float mx = -INFINITY, sm = 0.0f;
        for (uint s = 0u; s < S; ++s) mx = max(mx, sc[q][s]);
        for (uint s = 0u; s < S; ++s) { const float e = exp(sc[q][s] - mx); prob[q][s] = e; sm += e; }
        const float inv = (sm > 0.0f) ? (1.0f / sm) : 0.0f;
        for (uint s = 0u; s < S; ++s) prob[q][s] *= inv;
    }

    float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f);
    for (uint s = 0u; s < S; ++s) {
        const float4 vn = vreg[s] * ir[s];
        a0 = fma(float4(prob[0][s]), vn, a0);
        if (Q >= 2u) {
            a1 = fma(float4(prob[1][s]), vn, a1);
            a2 = fma(float4(prob[2][s]), vn, a2);
        }
    }
    reinterpret_cast<device float4*>(inp0 + row_off)[tid] = a0;
    if (Q >= 2u) {
        reinterpret_cast<device float4*>(inp1 + row_off)[tid] = a1;
        reinterpret_cast<device float4*>(inp2 + row_off)[tid] = a2;
    }
    const float ss0 = simd_sum(fma(a0.x, a0.x, fma(a0.y, a0.y, fma(a0.z, a0.z, a0.w * a0.w))));
    const float ss1 = (Q >= 2u) ? simd_sum(fma(a1.x, a1.x, fma(a1.y, a1.y, fma(a1.z, a1.z, a1.w * a1.w)))) : 0.0f;
    const float ss2 = (Q >= 2u) ? simd_sum(fma(a2.x, a2.x, fma(a2.y, a2.y, fma(a2.z, a2.z, a2.w * a2.w)))) : 0.0f;
    if (lane == 0u) {
        tg_osq[0][sgid] = ss0;
        if (Q >= 2u) { tg_osq[1][sgid] = ss1; tg_osq[2][sgid] = ss2; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float ir0 = rsqrt(simd_sum((lane < 18u) ? tg_osq[0][lane] : 0.0f) / float(p.D) + p.eps);
    const half4 h0  = sik_clamp_f16x4(a0 * ir0);
    reinterpret_cast<device half4*>(cat_left + (0ul * p.M_valid + m) * p.D)[tid] = h0;
    if (p.pad0 >= 3u) {
        if (Q >= 2u) {
            const float ir1 = rsqrt(simd_sum((lane < 18u) ? tg_osq[1][lane] : 0.0f) / float(p.D) + p.eps);
            const float ir2 = rsqrt(simd_sum((lane < 18u) ? tg_osq[2][lane] : 0.0f) / float(p.D) + p.eps);
            reinterpret_cast<device half4*>(cat_left + (1ul * p.M_valid + m) * p.D)[tid] = sik_clamp_f16x4(a1 * ir1);
            reinterpret_cast<device half4*>(cat_left + (2ul * p.M_valid + m) * p.D)[tid] = sik_clamp_f16x4(a2 * ir2);
        } else {
            reinterpret_cast<device half4*>(cat_left + (1ul * p.M_valid + m) * p.D)[tid] = h0;
            reinterpret_cast<device half4*>(cat_left + (2ul * p.M_valid + m) * p.D)[tid] = h0;
        }
    }
}

// Fused 1-Dispatch Pass-Boundary Bank Copy (`bank[0..bank_count-1] = blocks[src_start..]` + `bank_last = cur` + `cur = prelude_cur`)!
// Eliminates 10 serial copy dispatches per token with ZERO L2 message eviction!
struct SikPassCopyParams {
    uint M_valid;
    uint D;
    uint bank_count;
    uint src_start;
    uint block_stride;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void sik_pass_boundary_copy(
    device const float*       blocks_buf  [[buffer(0)]],
    device const float*       prelude_cur [[buffer(1)]],
    device float*             cur_io      [[buffer(2)]],
    device float*             bank_buf    [[buffer(3)]],
    device float*             bank_last   [[buffer(4)]],
    constant SikPassCopyParams& p         [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint m    = tgid.y;
    const uint d4   = tgid.x * 256u + tid;
    const uint d4_n = p.D >> 2u;
    if (m >= p.M_valid || d4 >= d4_n) return;
    const ulong row_off = ulong(m) * p.D;
    for (uint bi = 0u; bi < p.bank_count; ++bi) {
        device const float4* src = reinterpret_cast<device const float4*>(blocks_buf + ulong(p.src_start + bi) * p.block_stride + row_off);
        device float4*       dst = reinterpret_cast<device float4*>(bank_buf + ulong(bi) * p.block_stride + row_off);
        dst[d4] = src[d4];
    }
    device float4*       c4  = reinterpret_cast<device float4*>(cur_io + row_off);
    device float4*       bl4 = reinterpret_cast<device float4*>(bank_last + row_off);
    device const float4* pc4 = reinterpret_cast<device const float4*>(prelude_cur + row_off);
    bl4[d4] = c4[d4];
    c4[d4]  = pc4[d4];
}

struct SikLoopCtrlPrepParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    float eps;
};

kernel void sik_loop_ctrl_prep(
    device const float*             cur_in  [[buffer(0)]],
    device const float*             msg_in  [[buffer(1)]],
    device half*                    out_f16 [[buffer(2)]],
    constant SikLoopCtrlPrepParams& p       [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint half_idx = tgid.x;
    const uint m        = tgid.y;
    if (half_idx >= 2u || m >= p.M_padded) return;

    const uint  d4_cnt  = p.D >> 2u;
    const ulong dst_off = ulong(m) * (2ul * p.D) + ulong(half_idx) * p.D;

    if (m >= p.M_valid) {
        device half4* dst_h = reinterpret_cast<device half4*>(out_f16 + dst_off);
        for (uint i = tid; i < d4_cnt; i += 256u) dst_h[i] = half4(0.0h);
        return;
    }

    threadgroup float tg_sq[8];
    device const float4* src4 = reinterpret_cast<device const float4*>(
        ((half_idx == 0u) ? cur_in : msg_in) + ulong(m) * p.D);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 v = src4[i];
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }
    const float sg_s = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float   inv_rms = tg_sq[0];
    device half4* dst_h   = reinterpret_cast<device half4*>(out_f16 + dst_off);
    for (uint i = tid; i < d4_cnt; i += 256u) {
        dst_h[i] = sik_clamp_f16x4(src4[i] * inv_rms);
    }
}

// Batched 3-Way Loop Controller Prep (`j = 0, 1, 2`, works for BOTH decode and prefill!)
// Grid: `(2, M_padded, 3)`, `256` threads/TG
kernel void sik_loop_ctrl_prep3(
    device const float*             cur0    [[buffer(0)]],
    device const float*             cur1    [[buffer(1)]],
    device const float*             cur2    [[buffer(2)]],
    device const float*             msg0    [[buffer(3)]],
    device const float*             msg1    [[buffer(4)]],
    device const float*             msg2    [[buffer(5)]],
    device half*                    cat_out [[buffer(6)]], // [3, M_padded, 2 * D]
    constant SikLoopCtrlPrepParams& p       [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint half_idx = tgid.x;
    const uint m        = tgid.y;
    const uint j        = tgid.z;
    if (half_idx >= 2u || m >= p.M_padded || j >= 3u) return;

    const uint  d4_cnt  = p.D >> 2u;
    const ulong dst_off = (ulong(j) * p.M_padded + m) * (2ul * p.D) + ulong(half_idx) * p.D;

    if (m >= p.M_valid) {
        device half4* dst_h = reinterpret_cast<device half4*>(cat_out + dst_off);
        for (uint i = tid; i < d4_cnt; i += 256u) dst_h[i] = half4(0.0h);
        return;
    }

    device const float* cur_in = (j == 0u) ? cur0 : ((j == 1u) ? cur1 : cur2);
    device const float* msg_in = (j == 0u) ? msg0 : ((j == 1u) ? msg1 : msg2);
    threadgroup float tg_sq[8];
    device const float4* src4 = reinterpret_cast<device const float4*>(
        ((half_idx == 0u) ? cur_in : msg_in) + ulong(m) * p.D);

    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 v = src4[i];
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }
    const float sg_s = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float   inv_rms = tg_sq[0];
    device half4* dst_h   = reinterpret_cast<device half4*>(cat_out + dst_off);
    for (uint i = tid; i < d4_cnt; i += 256u) {
        dst_h[i] = sik_clamp_f16x4(src4[i] * inv_rms);
    }
}

// Fused Opt-In `--int8-prefill` Loop Controller Prep (`[rms(cur); rms(msg)] -> int8_t cat_i8 + half cat_sc`, 0 FP16 DRAM roundtrip!)
// Grid: `(M_padded, num_ctrls, 1)`, `256` threads/TG
kernel void sik_loop_ctrl_prep_i8(
    device const float*             cur0    [[buffer(0)]],
    device const float*             cur1    [[buffer(1)]],
    device const float*             cur2    [[buffer(2)]],
    device const float*             msg0    [[buffer(3)]],
    device const float*             msg1    [[buffer(4)]],
    device const float*             msg2    [[buffer(5)]],
    device int8_t*                  cat_i8  [[buffer(6)]], // [num_ctrls, M_padded, 2 * D]
    device half*                    cat_sc  [[buffer(7)]], // [num_ctrls, M_padded]
    constant SikLoopCtrlPrepParams& p       [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m = tgid.x;
    const uint j = tgid.y;
    if (m >= p.M_padded || j >= 3u) return;

    const uint  d4_cnt  = p.D >> 2u;
    const ulong row_idx = ulong(j) * p.M_padded + m;
    device char4* dst0  = reinterpret_cast<device char4*>(cat_i8 + row_idx * (2ul * p.D));
    device char4* dst1  = dst0 + d4_cnt;

    if (m >= p.M_valid) {
        for (uint i = tid; i < d4_cnt; i += 256u) { dst0[i] = char4(0); dst1[i] = char4(0); }
        if (tid == 0u) cat_sc[row_idx] = 0.0h;
        return;
    }

    device const float4* c4 = reinterpret_cast<device const float4*>(((j == 0u) ? cur0 : ((j == 1u) ? cur1 : cur2)) + ulong(m) * p.D);
    device const float4* m4 = reinterpret_cast<device const float4*>(((j == 0u) ? msg0 : ((j == 1u) ? msg1 : msg2)) + ulong(m) * p.D);

    threadgroup float tg_sq0[8], tg_sq1[8], tg_mx0[8], tg_mx1[8];
    float sq0 = 0.0f, sq1 = 0.0f, mx0 = 0.0f, mx1 = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 vc = c4[i];
        const float4 vm = m4[i];
        sq0 = fma(vc.x, vc.x, fma(vc.y, vc.y, fma(vc.z, vc.z, fma(vc.w, vc.w, sq0))));
        sq1 = fma(vm.x, vm.x, fma(vm.y, vm.y, fma(vm.z, vm.z, fma(vm.w, vm.w, sq1))));
        mx0 = max(mx0, max(max(abs(vc.x), abs(vc.y)), max(abs(vc.z), abs(vc.w))));
        mx1 = max(mx1, max(max(abs(vm.x), abs(vm.y)), max(abs(vm.z), abs(vm.w))));
    }
    const float s0 = simd_sum(sq0), s1 = simd_sum(sq1);
    const float x0 = simd_max(mx0), x1 = simd_max(mx1);
    if (lane == 0u) { tg_sq0[sgid] = s0; tg_sq1[sgid] = s1; tg_mx0[sgid] = x0; tg_mx1[sgid] = x1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const float tot0 = simd_sum((lane < 8u) ? tg_sq0[lane] : 0.0f);
        const float tot1 = simd_sum((lane < 8u) ? tg_sq1[lane] : 0.0f);
        const float am0  = simd_max((lane < 8u) ? tg_mx0[lane] : 0.0f);
        const float am1  = simd_max((lane < 8u) ? tg_mx1[lane] : 0.0f);
        if (lane == 0u) {
            const float ir0  = rsqrt(tot0 / float(p.D) + p.eps);
            const float ir1  = rsqrt(tot1 / float(p.D) + p.eps);
            const float amax = max(am0 * ir0, am1 * ir1);
            const float inv  = (amax > 1e-8f) ? (127.0f / amax) : 0.0f;
            tg_sq0[0] = ir0 * inv;
            tg_sq1[0] = ir1 * inv;
            cat_sc[row_idx] = sik_clamp_f16((amax > 1e-8f) ? (amax * (1.0f / 127.0f)) : 1.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sc0 = tg_sq0[0], sc1 = tg_sq1[0];
    for (uint i = tid; i < d4_cnt; i += 256u) {
        dst0[i] = char4(int4(clamp(rint(c4[i] * sc0), float4(-127.0f), float4(127.0f))));
        dst1[i] = char4(int4(clamp(rint(m4[i] * sc1), float4(-127.0f), float4(127.0f))));
    }
}

struct SikLoopCtrlApplyParams {
    uint M_valid;
    uint M_padded;
    uint D;         // 2304
    uint gate_rank; // 64
};

// Decode FP16 Loop Controller Apply (8 lanes/row, 4 rows/SG, 16 rows/TG)
kernel void sik_loop_ctrl_apply(
    device const half*               w_gain   [[buffer(0)]],
    device const half*               w_forget [[buffer(1)]],
    device const float*              h_in     [[buffer(2)]],
    device const float*              cur_in   [[buffer(3)]],
    device const float*              msg_in   [[buffer(4)]],
    device float*                    out_f32  [[buffer(5)]],
    constant SikLoopCtrlApplyParams& p        [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m0 = tgid.y * 8u;
    const uint mb = min(8u, (m0 < p.M_valid) ? (p.M_valid - m0) : 0u);
    if (mb == 0u) return;

    threadgroup float tg_h[8][64];
    const uint total_h = mb * p.gate_rank;
    for (uint idx = tid; idx < total_h; idx += 128u) {
        tg_h[idx / p.gate_rank][idx % p.gate_rank] = h_in[ulong(m0 + idx / p.gate_rank) * p.gate_rank + (idx % p.gate_rank)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row   = (tgid.x * 4u + sgid) * 4u + (lane >> 3u);
    const uint lane8 = lane & 7u;
    if (row >= p.D) return;

    const uint c8_cnt = p.gate_rank >> 3u;
    device const half4* wg4 = reinterpret_cast<device const half4*>(w_gain   + ulong(row) * p.gate_rank);
    device const half4* wf4 = reinterpret_cast<device const half4*>(w_forget + ulong(row) * p.gate_rank);

    float ga[8], fa[8];
    for (uint b = 0u; b < 8u; ++b) { ga[b] = 0.0f; fa[b] = 0.0f; }

    for (uint c8 = lane8; c8 < c8_cnt; c8 += 8u) {
        const float4 gw0 = float4(wg4[c8 * 2u + 0u]), gw1 = float4(wg4[c8 * 2u + 1u]);
        const float4 fw0 = float4(wf4[c8 * 2u + 0u]), fw1 = float4(wf4[c8 * 2u + 1u]);
        const uint k0 = c8 * 8u;
        for (uint b = 0u; b < mb; ++b) {
            const float4 h0(tg_h[b][k0 + 0u], tg_h[b][k0 + 1u], tg_h[b][k0 + 2u], tg_h[b][k0 + 3u]);
            const float4 h1(tg_h[b][k0 + 4u], tg_h[b][k0 + 5u], tg_h[b][k0 + 6u], tg_h[b][k0 + 7u]);
            ga[b] += dot(gw0, h0) + dot(gw1, h1);
            fa[b] += dot(fw0, h0) + dot(fw1, h1);
        }
    }
    for (uint b = 0u; b < mb; ++b) {
        const float g = sik_reduce8(ga[b]);
        const float f = sik_reduce8(fa[b]);
        if (lane8 == 0u) {
            const ulong off = ulong(m0 + b) * p.D + row;
            out_f32[off] = (1.0f - sik_safe_tanh(f)) * cur_in[off] + sik_safe_tanh(g) * msg_in[off];
        }
    }
}

// Decode Q8_0 Loop Controller Apply (`4` lanes/row `uint4` = 16B single load, `32` rows/TG -> 2.6x faster!)
template <uint M_FIX>
inline void sik_loop_ctrl_apply_q8_fixed(
    device const char*               wg_q,
    device const half*               wg_s,
    device const char*               wf_q,
    device const half*               wf_s,
    device const float*              h_in,
    device const float*              cur_in,
    device const float*              msg_in,
    device float*                    out_f32,
    device float*                    rms_sq_out,
    constant SikLoopCtrlApplyParams& p,
    threadgroup float                (&tg_h)[8][64],
    threadgroup float                (&tg_sq)[8][4],
    uint2 tgid, uint tid, uint sgid, uint lane
) {
    const uint m0      = tgid.y * 8u;
    const uint total_h = M_FIX * p.gate_rank;
    for (uint idx = tid; idx < total_h; idx += 128u) {
        tg_h[idx / p.gate_rank][idx % p.gate_rank] = h_in[ulong(m0 + idx / p.gate_rank) * p.gate_rank + (idx % p.gate_rank)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row    = (tgid.x * 4u + sgid) * 8u + (lane >> 2u);
    const uint l4     = lane & 3u;
    const bool active = (row < p.D);

    const uint c16_cnt = p.gate_rank >> 4u;
    const uint nblk    = p.gate_rank >> 6u;
    const uint safe_r  = active ? row : (p.D - 1u);
    device const uint4* wg4 = reinterpret_cast<device const uint4*>(wg_q + ulong(safe_r) * p.gate_rank);
    device const uint4* wf4 = reinterpret_cast<device const uint4*>(wf_q + ulong(safe_r) * p.gate_rank);
    device const half*  sg  = wg_s + ulong(safe_r) * nblk;
    device const half*  sf  = wf_s + ulong(safe_r) * nblk;

    float ga[M_FIX], fa[M_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < M_FIX; ++b) { ga[b] = 0.0f; fa[b] = 0.0f; }

    if (active) {
        for (uint c16 = l4; c16 < c16_cnt; c16 += 4u) {
            const float gs = float(sg[c16 >> 2u]);
            const float fs = float(sf[c16 >> 2u]);
            const uint4 qg = wg4[c16];
            const uint4 qf = wf4[c16];
            const float4 gw0 = float4(as_type<char4>(qg.x)), gw1 = float4(as_type<char4>(qg.y));
            const float4 gw2 = float4(as_type<char4>(qg.z)), gw3 = float4(as_type<char4>(qg.w));
            const float4 fw0 = float4(as_type<char4>(qf.x)), fw1 = float4(as_type<char4>(qf.y));
            const float4 fw2 = float4(as_type<char4>(qf.z)), fw3 = float4(as_type<char4>(qf.w));
            const uint k0 = c16 * 16u;
            #pragma clang loop unroll(full)
            for (uint b = 0u; b < M_FIX; ++b) {
                const float4 h0(tg_h[b][k0 +  0u], tg_h[b][k0 +  1u], tg_h[b][k0 +  2u], tg_h[b][k0 +  3u]);
                const float4 h1(tg_h[b][k0 +  4u], tg_h[b][k0 +  5u], tg_h[b][k0 +  6u], tg_h[b][k0 +  7u]);
                const float4 h2(tg_h[b][k0 +  8u], tg_h[b][k0 +  9u], tg_h[b][k0 + 10u], tg_h[b][k0 + 11u]);
                const float4 h3(tg_h[b][k0 + 12u], tg_h[b][k0 + 13u], tg_h[b][k0 + 14u], tg_h[b][k0 + 15u]);
                ga[b] = fma(gs, (dot(gw0, h0) + dot(gw1, h1)) + (dot(gw2, h2) + dot(gw3, h3)), ga[b]);
                fa[b] = fma(fs, (dot(fw0, h0) + dot(fw1, h1)) + (dot(fw2, h2) + dot(fw3, h3)), fa[b]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < M_FIX; ++b) {
        const float g = sik_reduce4(ga[b]);
        const float f = sik_reduce4(fa[b]);
        float sq_lane = 0.0f;
        if (active && l4 == 0u) {
            const ulong off = ulong(m0 + b) * p.D + row;
            const float ov  = (1.0f - sik_safe_tanh(f)) * cur_in[off] + sik_safe_tanh(g) * msg_in[off];
            out_f32[off] = ov;
            sq_lane = ov * ov;
        }
        const float sg_sq = simd_sum(sq_lane);
        if (lane == 0u) tg_sq[b][sgid] = sg_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const uint grid_x = (p.D + 31u) >> 5u;
        #pragma clang loop unroll(full)
        for (uint b = 0u; b < M_FIX; ++b) {
            const float tot = simd_sum((lane < 4u) ? tg_sq[b][lane] : 0.0f);
            if (lane == 0u) rms_sq_out[ulong(m0 + b) * grid_x + tgid.x] = tot;
        }
    }
}

kernel void sik_loop_ctrl_apply_q8(
    device const char*               wg_q       [[buffer(0)]],
    device const half*               wg_s       [[buffer(1)]],
    device const char*               wf_q       [[buffer(2)]],
    device const half*               wf_s       [[buffer(3)]],
    device const float*              h_in       [[buffer(4)]],
    device const float*              cur_in     [[buffer(5)]],
    device const float*              msg_in     [[buffer(6)]],
    device float*                    out_f32    [[buffer(7)]],
    constant SikLoopCtrlApplyParams& p          [[buffer(8)]],
    device float*                    rms_sq_out [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_h[8][64];
    threadgroup float tg_sq[8][4];
    #define SIK_APQ8_CASE(MF) { sik_loop_ctrl_apply_q8_fixed<MF>(wg_q, wg_s, wf_q, wf_s, h_in, cur_in, msg_in, out_f32, rms_sq_out, p, tg_h, tg_sq, tgid, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_APQ8_CASE(1u) case 2u: SIK_APQ8_CASE(2u) case 3u: SIK_APQ8_CASE(3u) case 4u: SIK_APQ8_CASE(4u)
        case 5u: SIK_APQ8_CASE(5u) case 6u: SIK_APQ8_CASE(6u) case 7u: SIK_APQ8_CASE(7u) case 8u: SIK_APQ8_CASE(8u)
        default: break;
    }
    #undef SIK_APQ8_CASE
}

// Batched 3-Way Decode FP16 Loop Controller Apply (`j = 0, 1, 2`)
kernel void sik_loop_ctrl_apply3(
    device const half*               wg0  [[buffer(0)]],
    device const half*               wg1  [[buffer(1)]],
    device const half*               wg2  [[buffer(2)]],
    device const half*               wf0  [[buffer(3)]],
    device const half*               wf1  [[buffer(4)]],
    device const half*               wf2  [[buffer(5)]],
    device const float*              h_in [[buffer(6)]],
    device const float*              c0   [[buffer(7)]],
    device const float*              c1   [[buffer(8)]],
    device const float*              c2   [[buffer(9)]],
    device const float*              m0_b [[buffer(10)]],
    device const float*              m1_b [[buffer(11)]],
    device const float*              m2_b [[buffer(12)]],
    device float*                    o0   [[buffer(13)]],
    device float*                    o1   [[buffer(14)]],
    device float*                    o2   [[buffer(15)]],
    constant SikLoopCtrlApplyParams& p    [[buffer(16)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint j  = tgid.z;
    const uint m0 = tgid.y * 8u;
    const uint mb = min(8u, (m0 < p.M_valid) ? (p.M_valid - m0) : 0u);
    if (j >= 3u || mb == 0u) return;

    device const float* h_base = h_in + ulong(j) * ulong(p.M_valid) * p.gate_rank;
    threadgroup float tg_h[8][64];
    const uint total_h = mb * p.gate_rank;
    for (uint idx = tid; idx < total_h; idx += 128u) {
        tg_h[idx / p.gate_rank][idx % p.gate_rank] = h_base[ulong(m0 + idx / p.gate_rank) * p.gate_rank + (idx % p.gate_rank)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row   = (tgid.x * 4u + sgid) * 4u + (lane >> 3u);
    const uint lane8 = lane & 7u;
    if (row >= p.D) return;

    device const half*  w_gain   = (j == 0u) ? wg0  : ((j == 1u) ? wg1  : wg2);
    device const half*  w_forget = (j == 0u) ? wf0  : ((j == 1u) ? wf1  : wf2);
    device const float* cur_in   = (j == 0u) ? c0   : ((j == 1u) ? c1   : c2);
    device const float* msg_in   = (j == 0u) ? m0_b : ((j == 1u) ? m1_b : m2_b);
    device float*       out_f32  = (j == 0u) ? o0   : ((j == 1u) ? o1   : o2);

    const uint c8_cnt = p.gate_rank >> 3u;
    device const half4* wg4 = reinterpret_cast<device const half4*>(w_gain   + ulong(row) * p.gate_rank);
    device const half4* wf4 = reinterpret_cast<device const half4*>(w_forget + ulong(row) * p.gate_rank);

    float ga[8], fa[8];
    for (uint b = 0u; b < 8u; ++b) { ga[b] = 0.0f; fa[b] = 0.0f; }

    for (uint c8 = lane8; c8 < c8_cnt; c8 += 8u) {
        const float4 gw0 = float4(wg4[c8 * 2u + 0u]), gw1 = float4(wg4[c8 * 2u + 1u]);
        const float4 fw0 = float4(wf4[c8 * 2u + 0u]), fw1 = float4(wf4[c8 * 2u + 1u]);
        const uint k0 = c8 * 8u;
        for (uint b = 0u; b < mb; ++b) {
            const float4 h0(tg_h[b][k0 + 0u], tg_h[b][k0 + 1u], tg_h[b][k0 + 2u], tg_h[b][k0 + 3u]);
            const float4 h1(tg_h[b][k0 + 4u], tg_h[b][k0 + 5u], tg_h[b][k0 + 6u], tg_h[b][k0 + 7u]);
            ga[b] += dot(gw0, h0) + dot(gw1, h1);
            fa[b] += dot(fw0, h0) + dot(fw1, h1);
        }
    }
    for (uint b = 0u; b < mb; ++b) {
        const float g = sik_reduce8(ga[b]);
        const float f = sik_reduce8(fa[b]);
        if (lane8 == 0u) {
            const ulong off = ulong(m0 + b) * p.D + row;
            out_f32[off] = (1.0f - sik_safe_tanh(f)) * cur_in[off] + sik_safe_tanh(g) * msg_in[off];
        }
    }
}

struct SikLoopCtrl3Offsets {
    ulong w_off[3];
    ulong s_off[3];
    ulong f_off[3];
    ulong fs_off[3];
    uint  M_valid;
    uint  M_padded;
    uint  D;
    uint  gate_rank;
    uint  num_ctrls; // 1 or 3
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

// Batched 3-Way Decode Q8_0 Loop Controller Apply (`4` lanes/row `uint4`, `32` rows/TG -> 2.6x faster!)
template <uint M_FIX>
inline void sik_loop_ctrl_apply3_q8_fixed(
    device const uchar*           uma,
    device const float*           h_in,
    device const float*           c0,
    device const float*           c1,
    device const float*           c2,
    device const float*           m0_b,
    device const float*           m1_b,
    device const float*           m2_b,
    device float*                 o0,
    device float*                 o1,
    device float*                 o2,
    device float*                 rms_sq_out,
    constant SikLoopCtrl3Offsets& p,
    threadgroup float             (&tg_h)[8][64],
    threadgroup float             (&tg_sq)[8][4],
    uint3 tgid, uint tid, uint sgid, uint lane
) {
    const uint j  = tgid.z;
    const uint m0 = tgid.y * 8u;
    if (j >= p.num_ctrls) return;

    device const float* h_base = h_in + ulong(j) * ulong(p.M_valid) * p.gate_rank;
    const uint total_h = M_FIX * p.gate_rank;
    for (uint idx = tid; idx < total_h; idx += 128u) {
        tg_h[idx / p.gate_rank][idx % p.gate_rank] = h_base[ulong(m0 + idx / p.gate_rank) * p.gate_rank + (idx % p.gate_rank)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row    = (tgid.x * 4u + sgid) * 8u + (lane >> 2u);
    const uint l4     = lane & 3u;
    const bool active = (row < p.D);
    const uint safe_r = active ? row : (p.D - 1u);

    device const char*  wg_q    = reinterpret_cast<device const char*>(uma + p.w_off[j]);
    device const half*  wg_s    = reinterpret_cast<device const half*>(uma + p.s_off[j]);
    device const char*  wf_q    = reinterpret_cast<device const char*>(uma + p.f_off[j]);
    device const half*  wf_s    = reinterpret_cast<device const half*>(uma + p.fs_off[j]);
    device const float* cur_in  = (j == 0u) ? c0   : ((j == 1u) ? c1   : c2);
    device const float* msg_in  = (j == 0u) ? m0_b : ((j == 1u) ? m1_b : m2_b);
    device float*       out_f32 = (j == 0u) ? o0   : ((j == 1u) ? o1   : o2);

    const uint c16_cnt = p.gate_rank >> 4u;
    const uint nblk    = p.gate_rank >> 6u;
    device const uint4* wg4 = reinterpret_cast<device const uint4*>(wg_q + ulong(safe_r) * p.gate_rank);
    device const uint4* wf4 = reinterpret_cast<device const uint4*>(wf_q + ulong(safe_r) * p.gate_rank);
    device const half*  sg  = wg_s + ulong(safe_r) * nblk;
    device const half*  sf  = wf_s + ulong(safe_r) * nblk;

    float ga[M_FIX], fa[M_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < M_FIX; ++b) { ga[b] = 0.0f; fa[b] = 0.0f; }

    if (active) {
        for (uint c16 = l4; c16 < c16_cnt; c16 += 4u) {
            const float gs = float(sg[c16 >> 2u]);
            const float fs = float(sf[c16 >> 2u]);
            const uint4 qg = wg4[c16];
            const uint4 qf = wf4[c16];
            const float4 gw0 = float4(as_type<char4>(qg.x)), gw1 = float4(as_type<char4>(qg.y));
            const float4 gw2 = float4(as_type<char4>(qg.z)), gw3 = float4(as_type<char4>(qg.w));
            const float4 fw0 = float4(as_type<char4>(qf.x)), fw1 = float4(as_type<char4>(qf.y));
            const float4 fw2 = float4(as_type<char4>(qf.z)), fw3 = float4(as_type<char4>(qf.w));
            const uint k0 = c16 * 16u;
            #pragma clang loop unroll(full)
            for (uint b = 0u; b < M_FIX; ++b) {
                const float4 h0(tg_h[b][k0 +  0u], tg_h[b][k0 +  1u], tg_h[b][k0 +  2u], tg_h[b][k0 +  3u]);
                const float4 h1(tg_h[b][k0 +  4u], tg_h[b][k0 +  5u], tg_h[b][k0 +  6u], tg_h[b][k0 +  7u]);
                const float4 h2(tg_h[b][k0 +  8u], tg_h[b][k0 +  9u], tg_h[b][k0 + 10u], tg_h[b][k0 + 11u]);
                const float4 h3(tg_h[b][k0 + 12u], tg_h[b][k0 + 13u], tg_h[b][k0 + 14u], tg_h[b][k0 + 15u]);
                ga[b] = fma(gs, (dot(gw0, h0) + dot(gw1, h1)) + (dot(gw2, h2) + dot(gw3, h3)), ga[b]);
                fa[b] = fma(fs, (dot(fw0, h0) + dot(fw1, h1)) + (dot(fw2, h2) + dot(fw3, h3)), fa[b]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < M_FIX; ++b) {
        const float g = sik_reduce4(ga[b]);
        const float f = sik_reduce4(fa[b]);
        float sq_lane = 0.0f;
        if (active && l4 == 0u) {
            const ulong off = ulong(m0 + b) * p.D + row;
            const float ov  = (1.0f - sik_safe_tanh(f)) * cur_in[off] + sik_safe_tanh(g) * msg_in[off];
            out_f32[off] = ov;
            sq_lane = ov * ov;
        }
        const float sg_sq = simd_sum(sq_lane);
        if (lane == 0u) tg_sq[b][sgid] = sg_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const uint grid_x = (p.D + 31u) >> 5u;
        #pragma clang loop unroll(full)
        for (uint b = 0u; b < M_FIX; ++b) {
            const float tot = simd_sum((lane < 4u) ? tg_sq[b][lane] : 0.0f);
            if (lane == 0u) rms_sq_out[(ulong(j) * p.M_valid + (m0 + b)) * grid_x + tgid.x] = tot;
        }
    }
}

kernel void sik_loop_ctrl_apply3_q8(
    device const uchar*           uma        [[buffer(0)]],
    device const float*           h_in       [[buffer(1)]],
    device const float*           c0         [[buffer(2)]],
    device const float*           c1         [[buffer(3)]],
    device const float*           c2         [[buffer(4)]],
    device const float*           m0_b       [[buffer(5)]],
    device const float*           m1_b       [[buffer(6)]],
    device const float*           m2_b       [[buffer(7)]],
    device float*                 o0         [[buffer(8)]],
    device float*                 o1         [[buffer(9)]],
    device float*                 o2         [[buffer(10)]],
    constant SikLoopCtrl3Offsets& p          [[buffer(11)]],
    device float*                 rms_sq_out [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_h[8][64];
    threadgroup float tg_sq[8][4];
    #define SIK_AP3Q8_CASE(MF) { sik_loop_ctrl_apply3_q8_fixed<MF>(uma, h_in, c0, c1, c2, m0_b, m1_b, m2_b, o0, o1, o2, rms_sq_out, p, tg_h, tg_sq, tgid, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_AP3Q8_CASE(1u) case 2u: SIK_AP3Q8_CASE(2u) case 3u: SIK_AP3Q8_CASE(3u) case 4u: SIK_AP3Q8_CASE(4u)
        case 5u: SIK_AP3Q8_CASE(5u) case 6u: SIK_AP3Q8_CASE(6u) case 7u: SIK_AP3Q8_CASE(7u) case 8u: SIK_AP3Q8_CASE(8u)
        default: break;
    }
    #undef SIK_AP3Q8_CASE
}

// Zero-Barrier 1-SG (`32` th/TG, 2 elems/lane!) 108-TG 3-Way RMSNorm + `xsum` consuming `rms_sq_out` from `apply3_q8`!
// Replaces 3-TG 2-pass `sik_rmsnorm3` on looped PreAttn visits (`~2.0 us` vs `4.3 us`), 100% bit-exact `M = 1..8`!
// Grid: `(D / 64 /* 36 */, 3, M_valid)`, `32` threads/TG (`1` SG, ZERO TG mem, ZERO barriers!)
kernel void sik_rmsnorm3_fast(
    device const float*        in0    [[buffer(0)]],
    device const float*        in1    [[buffer(1)]],
    device const float*        in2    [[buffer(2)]],
    device const float*        nw0    [[buffer(3)]],
    device const float*        nw1    [[buffer(4)]],
    device const float*        nw2    [[buffer(5)]],
    device const float*        rms_sq [[buffer(6)]],
    device half*               out0   [[buffer(7)]],
    device half*               out1   [[buffer(8)]],
    device half*               out2   [[buffer(9)]],
    device float*              xs0    [[buffer(10)]],
    device float*              xs1    [[buffer(11)]],
    device float*              xs2    [[buffer(12)]],
    constant SikRmsNormParams& p      [[buffer(13)]], // has_weight = n_sq_tiles ((D+31)/32 = 72)
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint blk = tgid.x;
    const uint j   = tgid.y;
    const uint m   = tgid.z;
    const uint nblk = p.D >> 6u;
    if (blk >= nblk || j >= 3u || m >= p.M_valid) return;

    const uint n_sq = p.has_weight;
    device const float* sq_row = rms_sq + (ulong(j) * p.M_valid + m) * n_sq;
    float part = 0.0f;
    for (uint t = lane; t < n_sq; t += 32u) part += sq_row[t];
    const float inv_rms = rsqrt(simd_sum(part) / float(p.D) + p.eps);

    device const float* src = (j == 0u) ? in0  : ((j == 1u) ? in1  : in2);
    device const float* nw  = (j == 0u) ? nw0  : ((j == 1u) ? nw1  : nw2);
    device half*        dst = (j == 0u) ? out0 : ((j == 1u) ? out1 : out2);
    device float*       xso = (j == 0u) ? xs0  : ((j == 1u) ? xs1  : xs2);

    const uint   idx2 = blk * 32u + lane;
    const ulong  off2 = ulong(m) * (p.D >> 1u) + idx2;
    const float2 v2   = reinterpret_cast<device const float2*>(src)[off2];
    const float2 w2   = reinterpret_cast<device const float2*>(nw)[idx2];
    const float2 n2   = (v2 * inv_rms) * w2;
    const half2  h2   = half2(sik_clamp_f16(n2.x), sik_clamp_f16(n2.y));
    reinterpret_cast<device half2*>(dst)[off2] = h2;
    const float bs = simd_sum(float(h2.x) + float(h2.y));
    if (lane == 0u) xso[ulong(m) * nblk + blk] = bs;
}

// Zero-Barrier 1-SG-per-Block 1-Pass `RMSNorm1` (emits `out_f16` + `out_f32` for `M = 2..8` MoE Router!)
// Consumes `b_rms_sq` already emitted by `sik_loop_ctrl_apply_q8`; 100.000000% bit-exact with `sik_moe_router_f32_rms`!
// Grid: `(D / 64, M_valid, 1)`, `32` threads/TG (`1` SG, ZERO barriers!)
kernel void sik_rmsnorm1_fast(
    device const float*        src     [[buffer(0)]],
    device const float*        nw      [[buffer(1)]],
    device const float*        rms_sq  [[buffer(2)]],
    device half*               out_f16 [[buffer(3)]],
    device float*              out_f32 [[buffer(4)]],
    constant SikRmsNormParams& p       [[buffer(5)]], // has_weight = n_sq_tiles ((D+31)/32 = 72)
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint blk  = tgid.x;
    const uint m    = tgid.y;
    const uint nblk = p.D >> 6u;
    if (blk >= nblk || m >= p.M_valid) return;

    const uint n_sq = p.has_weight;
    device const float* sq_row = rms_sq + ulong(m) * n_sq;
    float part = 0.0f;
    for (uint t = lane; t < n_sq; t += 32u) part += sq_row[t];
    const float inv_rms = rsqrt(simd_sum(part) / float(p.D) + p.eps);

    const uint   idx2 = blk * 32u + lane;
    const ulong  off2 = ulong(m) * (p.D >> 1u) + idx2;
    const float2 v2   = reinterpret_cast<device const float2*>(src)[off2];
    const float2 w2   = reinterpret_cast<device const float2*>(nw)[idx2];
    const float2 n2   = (v2 * inv_rms) * w2;
    reinterpret_cast<device half2*>(out_f16)[off2]  = half2(sik_clamp_f16(n2.x), sik_clamp_f16(n2.y));
    reinterpret_cast<device float2*>(out_f32)[off2] = n2;
}

// Prefill High-Occupancy Batched NAX Loop Controller `down_weight [64, 4608]` (`1` or `3` ctrls in 1 dispatch!)
// Upgraded to BK=128 (`W_tile[32][128] = 8 KiB`, `uint4` 16B loads) to eliminate M5 Max's 2.17x K=64 NAX cliff!
#if defined(__HAVE_TENSOR__)
template <int BM, int SG>
inline void sik_loop_ctrl_down_nax_f16_impl(
    device const uchar*           uma,
    device const half*            cat_f16,
    device half*                  h_out_f16,
    constant SikLoopCtrl3Offsets& p,
    uint3                         tgid
) {
    const uint j  = tgid.z;
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 32u;
    if (j >= p.num_ctrls || m0 >= p.M_padded || n0 >= p.gate_rank) return;

    const uint K = 2u * p.D;
    device half* x_ptr = const_cast<device half*>(cat_f16) + ulong(j) * ulong(p.M_padded) * K;
    device half* w_ptr = reinterpret_cast<device half*>(const_cast<device uchar*>(uma) + p.w_off[j]);
    device half* h_ptr = h_out_f16 + ulong(j) * ulong(p.M_padded) * p.gate_rank;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(x_ptr, dextents<int, 2>(int(K), int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tW(w_ptr, dextents<int, 2>(int(K), int(p.gate_rank)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 32, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX = tX.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW = tW.template slice<dynamic_extent, 32>(0, int(n0));
    auto cT = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sX)>, sik_strip_as<decltype(sW)>, float>();
    op.run(sX, sW, cT);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            const uint mg = m0 + uint(idx[1]);
            const uint ng = n0 + uint(idx[0]);
            if (mg < p.M_padded && ng < p.gate_rank) {
                h_ptr[ulong(mg) * p.gate_rank + ng] = (mg < p.M_valid) ? sik_clamp_f16(cT[i]) : 0.0h;
            }
        }
    }
}

template <int BM, int SG>
inline void sik_loop_ctrl_down_nax_q8_impl(
    device const uchar*           uma,
    device const half*            cat_f16,
    device half*                  h_out_f16,
    constant SikLoopCtrl3Offsets& p,
    threadgroup half              (&W_tile)[32][128],
    uint3                         tgid,
    uint                          tid
) {
    const uint j  = tgid.z;
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 32u;
    if (j >= p.num_ctrls || m0 >= p.M_padded || n0 >= p.gate_rank) return;

    const uint K = 2u * p.D;
    device half*        x_ptr = const_cast<device half*>(cat_f16) + ulong(j) * ulong(p.M_padded) * K;
    device const uint4* w_u4  = reinterpret_cast<device const uint4*>(uma + p.w_off[j]);
    device const half*  w_s   = reinterpret_cast<device const half*>(uma + p.s_off[j]);
    device half*        h_ptr = h_out_f16 + ulong(j) * ulong(p.M_padded) * p.gate_rank;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(x_ptr, dextents<int, 2>(int(K), int(p.M_padded)));
    tensor<threadgroup half, extents<int, 128, 32>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 32>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<128, BM>(0, int(m0));
    auto cT  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sX0)>, sik_strip_as<decltype(tW)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint k_tiles  = K >> 7u; // K / 128
    const uint grp_row  = K >> 6u; // K / 64
    const uint u4_row   = K >> 4u; // K / 16
    constexpr uint r_step  = (uint(SG) * 32u) >> 3u; // 8 (SG=2) or 16 (SG=4)
    constexpr uint q_iters = 32u / r_step;
    const uint r_grp    = tid >> 3u;
    const uint sub      = tid & 7u;  // 0..7 (16 int8s = uint4)
    const uint sub_g    = sub >> 2u; // 0 or 1

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const uint grp = kt * 2u + sub_g;
        const uint u4c = kt * 8u + sub;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint r_loc  = q * r_step + r_grp;
            const uint r_glob = n0 + r_loc;
            const half sc     = w_s[ulong(r_glob) * grp_row + grp];
            const uint4 q16   = w_u4[ulong(r_glob) * u4_row + u4c];
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
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
            auto idx = cT.get_multidimensional_index(i);
            const uint mg = m0 + uint(idx[1]);
            const uint ng = n0 + uint(idx[0]);
            if (mg < p.M_padded && ng < p.gate_rank) {
                h_ptr[ulong(mg) * p.gate_rank + ng] = (mg < p.M_valid) ? sik_clamp_f16(cT[i]) : 0.0h;
            }
        }
    }
}
#endif

kernel void sik_loop_ctrl_down_nax_f16(
    device const uchar*           uma       [[buffer(0)]],
    device const half*            cat_f16   [[buffer(1)]],
    device half*                  h_out_f16 [[buffer(2)]],
    constant SikLoopCtrl3Offsets& p         [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_down_nax_f16_impl<16, 2>(uma, cat_f16, h_out_f16, p, tgid);
#else
    (void)uma; (void)cat_f16; (void)h_out_f16; (void)p; (void)tgid;
#endif
}

kernel void sik_loop_ctrl_down_nax_f16_m32(
    device const uchar*           uma       [[buffer(0)]],
    device const half*            cat_f16   [[buffer(1)]],
    device half*                  h_out_f16 [[buffer(2)]],
    constant SikLoopCtrl3Offsets& p         [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_down_nax_f16_impl<32, 2>(uma, cat_f16, h_out_f16, p, tgid);
#else
    (void)uma; (void)cat_f16; (void)h_out_f16; (void)p; (void)tgid;
#endif
}

kernel void sik_loop_ctrl_down_nax_q8(
    device const uchar*           uma       [[buffer(0)]],
    device const half*            cat_f16   [[buffer(1)]],
    device half*                  h_out_f16 [[buffer(2)]],
    constant SikLoopCtrl3Offsets& p         [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[32][128];
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_down_nax_q8_impl<16, 2>(uma, cat_f16, h_out_f16, p, W_tile, tgid, tid);
#else
    (void)uma; (void)cat_f16; (void)h_out_f16; (void)p; (void)W_tile; (void)tgid; (void)tid;
#endif
}

kernel void sik_loop_ctrl_down_nax_q8_m32(
    device const uchar*           uma       [[buffer(0)]],
    device const half*            cat_f16   [[buffer(1)]],
    device half*                  h_out_f16 [[buffer(2)]],
    constant SikLoopCtrl3Offsets& p         [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[32][128];
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_down_nax_q8_impl<32, 2>(uma, cat_f16, h_out_f16, p, W_tile, tgid, tid);
#else
    (void)uma; (void)cat_f16; (void)h_out_f16; (void)p; (void)W_tile; (void)tgid; (void)tid;
#endif
}

// Prefill NAX Loop Controller Apply (`gain & forget` `dynamic_extent` direct-device static_slice — escapes K=64 cliff!)
// Grid: `(D / 64, M_padded / BM, num_ctrls)`, `SG * 32` threads/TG
#if defined(__HAVE_TENSOR__)
template <int BM, int SG>
inline void sik_loop_ctrl_apply_nax_f16_impl(
    device const uchar*           uma,
    device const half*            h_in_f16,
    device const float*           c0,
    device const float*           c1,
    device const float*           c2,
    device const float*           m0_b,
    device const float*           m1_b,
    device const float*           m2_b,
    device float*                 o0,
    device float*                 o1,
    device float*                 o2,
    constant SikLoopCtrl3Offsets& p,
    uint3                         tgid
) {
    const uint j  = tgid.z;
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (j >= p.num_ctrls || m0 >= p.M_padded || n0 >= p.D) return;

    device half* h_ptr  = const_cast<device half*>(h_in_f16) + ulong(j) * ulong(p.M_padded) * p.gate_rank;
    device half* wg_ptr = reinterpret_cast<device half*>(const_cast<device uchar*>(uma) + p.w_off[j]);
    device half* wf_ptr = reinterpret_cast<device half*>(const_cast<device uchar*>(uma) + p.f_off[j]);
    device const float* cur_in  = (j == 0u) ? c0   : ((j == 1u) ? c1   : c2);
    device const float* msg_in  = (j == 0u) ? m0_b : ((j == 1u) ? m1_b : m2_b);
    device float*       out_f32 = (j == 0u) ? o0   : ((j == 1u) ? o1   : o2);

    tensor<device half, dextents<int, 2>, tensor_inline> tH(h_ptr,  dextents<int, 2>(int(p.gate_rank), int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tG(wg_ptr, dextents<int, 2>(int(p.gate_rank), int(p.D)));
    tensor<device half, dextents<int, 2>, tensor_inline> tF(wf_ptr, dextents<int, 2>(int(p.gate_rank), int(p.D)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sH = tH.template slice<dynamic_extent, BM>(0, int(m0));
    auto sG = tG.template slice<dynamic_extent, 64>(0, int(n0));
    auto sF = tF.template slice<dynamic_extent, 64>(0, int(n0));
    auto cG = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sH)>, sik_strip_as<decltype(sG)>, float>();
    auto cF = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sH)>, sik_strip_as<decltype(sF)>, float>();
    op.run(sH, sG, cG);
    op.run(sH, sF, cF);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cG.get_capacity(); ++i) {
        if (cG.is_valid_element(i)) {
            auto idx = cG.get_multidimensional_index(i);
            const uint mg = m0 + uint(idx[1]);
            const uint ng = n0 + uint(idx[0]);
            if (mg < p.M_valid && ng < p.D) {
                const ulong off = ulong(mg) * p.D + ng;
                out_f32[off] = (1.0f - sik_safe_tanh(cF[i])) * cur_in[off] + sik_safe_tanh(cG[i]) * msg_in[off];
            }
        }
    }
}

template <int BM, int SG>
inline void sik_loop_ctrl_apply_nax_q8_impl(
    device const uchar*           uma,
    device const half*            h_in_f16,
    device const float*           c0,
    device const float*           c1,
    device const float*           c2,
    device const float*           m0_b,
    device const float*           m1_b,
    device const float*           m2_b,
    device float*                 o0,
    device float*                 o1,
    device float*                 o2,
    constant SikLoopCtrl3Offsets& p,
    threadgroup half              (&W_tile)[64][64],
    uint3                         tgid,
    uint                          tid
) {
    const uint j  = tgid.z;
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (j >= p.num_ctrls || m0 >= p.M_padded || n0 >= p.D) return;

    device half*        h_ptr   = const_cast<device half*>(h_in_f16) + ulong(j) * ulong(p.M_padded) * p.gate_rank;
    device const uint4* wg_u4   = reinterpret_cast<device const uint4*>(uma + p.w_off[j]);
    device const half*  wg_s    = reinterpret_cast<device const half*>(uma + p.s_off[j]);
    device const uint4* wf_u4   = reinterpret_cast<device const uint4*>(uma + p.f_off[j]);
    device const half*  wf_s    = reinterpret_cast<device const half*>(uma + p.fs_off[j]);
    device const float* cur_in  = (j == 0u) ? c0   : ((j == 1u) ? c1   : c2);
    device const float* msg_in  = (j == 0u) ? m0_b : ((j == 1u) ? m1_b : m2_b);
    device float*       out_f32 = (j == 0u) ? o0   : ((j == 1u) ? o1   : o2);

    // 16B `uint4` loads: 4 threads/row (`64` int8s), 16 rows/step (SG=2) or 32 rows/step (SG=4) -> 8 KiB single W_tile!
    constexpr uint r_step  = (uint(SG) * 32u) >> 2u; // 16 (SG=2) or 32 (SG=4)
    constexpr uint q_iters = 64u / r_step;
    const uint r_grp       = tid >> 2u;
    const uint sub         = tid & 3u; // 0..3 (16 int8s each = 64)

    tensor<device half, dextents<int, 2>, tensor_inline> tH(h_ptr, dextents<int, 2>(int(p.gate_rank), int(p.M_padded)));
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> tW(&W_tile[0][0], dextents<int, 2>(64, 64));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sH = tH.template slice<dynamic_extent, BM>(0, int(m0));
    auto sW = tW.template slice<dynamic_extent, 64>(0, 0);
    auto cG = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sH)>, sik_strip_as<decltype(sW)>, float>();
    auto cF = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sH)>, sik_strip_as<decltype(sW)>, float>();

    // Pass 1: Gain
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < q_iters; ++q) {
        const uint r_loc  = q * r_step + r_grp;
        const uint r_glob = n0 + r_loc;
        const half scg    = wg_s[r_glob];
        const uint4 q16   = wg_u4[ulong(r_glob) * 4ul + sub];
        threadgroup half4* dg = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
        dg[0] = half4(as_type<char4>(q16.x)) * scg;
        dg[1] = half4(as_type<char4>(q16.y)) * scg;
        dg[2] = half4(as_type<char4>(q16.z)) * scg;
        dg[3] = half4(as_type<char4>(q16.w)) * scg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(sH, sW, cG);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Forget (reusing exact same 8 KiB W_tile -> 2x occupancy!)
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < q_iters; ++q) {
        const uint r_loc  = q * r_step + r_grp;
        const uint r_glob = n0 + r_loc;
        const half scf    = wf_s[r_glob];
        const uint4 q16   = wf_u4[ulong(r_glob) * 4ul + sub];
        threadgroup half4* df = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
        df[0] = half4(as_type<char4>(q16.x)) * scf;
        df[1] = half4(as_type<char4>(q16.y)) * scf;
        df[2] = half4(as_type<char4>(q16.z)) * scf;
        df[3] = half4(as_type<char4>(q16.w)) * scf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(sH, sW, cF);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cG.get_capacity(); ++i) {
        if (cG.is_valid_element(i)) {
            auto idx = cG.get_multidimensional_index(i);
            const uint mg = m0 + uint(idx[1]);
            const uint ng = n0 + uint(idx[0]);
            if (mg < p.M_valid && ng < p.D) {
                const ulong off = ulong(mg) * p.D + ng;
                out_f32[off] = (1.0f - sik_safe_tanh(cF[i])) * cur_in[off] + sik_safe_tanh(cG[i]) * msg_in[off];
            }
        }
    }
}
#endif

kernel void sik_loop_ctrl_apply_nax_f16_m64(
    device const uchar* uma [[buffer(0)]], device const half* h_in [[buffer(1)]],
    device const float* c0 [[buffer(2)]], device const float* c1 [[buffer(3)]], device const float* c2 [[buffer(4)]],
    device const float* m0 [[buffer(5)]], device const float* m1 [[buffer(6)]], device const float* m2 [[buffer(7)]],
    device float* o0 [[buffer(8)]], device float* o1 [[buffer(9)]], device float* o2 [[buffer(10)]],
    constant SikLoopCtrl3Offsets& p [[buffer(11)]], uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_apply_nax_f16_impl<64, 4>(uma, h_in, c0, c1, c2, m0, m1, m2, o0, o1, o2, p, tgid);
#else
    (void)uma;(void)h_in;(void)c0;(void)c1;(void)c2;(void)m0;(void)m1;(void)m2;(void)o0;(void)o1;(void)o2;(void)p;(void)tgid;
#endif
}

kernel void sik_loop_ctrl_apply_nax_f16_m16(
    device const uchar* uma [[buffer(0)]], device const half* h_in [[buffer(1)]],
    device const float* c0 [[buffer(2)]], device const float* c1 [[buffer(3)]], device const float* c2 [[buffer(4)]],
    device const float* m0 [[buffer(5)]], device const float* m1 [[buffer(6)]], device const float* m2 [[buffer(7)]],
    device float* o0 [[buffer(8)]], device float* o1 [[buffer(9)]], device float* o2 [[buffer(10)]],
    constant SikLoopCtrl3Offsets& p [[buffer(11)]], uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_apply_nax_f16_impl<16, 2>(uma, h_in, c0, c1, c2, m0, m1, m2, o0, o1, o2, p, tgid);
#else
    (void)uma;(void)h_in;(void)c0;(void)c1;(void)c2;(void)m0;(void)m1;(void)m2;(void)o0;(void)o1;(void)o2;(void)p;(void)tgid;
#endif
}

kernel void sik_loop_ctrl_apply_nax_q8_m64(
    device const uchar* uma [[buffer(0)]], device const half* h_in [[buffer(1)]],
    device const float* c0 [[buffer(2)]], device const float* c1 [[buffer(3)]], device const float* c2 [[buffer(4)]],
    device const float* m0 [[buffer(5)]], device const float* m1 [[buffer(6)]], device const float* m2 [[buffer(7)]],
    device float* o0 [[buffer(8)]], device float* o1 [[buffer(9)]], device float* o2 [[buffer(10)]],
    constant SikLoopCtrl3Offsets& p [[buffer(11)]], uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_apply_nax_q8_impl<64, 4>(uma, h_in, c0, c1, c2, m0, m1, m2, o0, o1, o2, p, W_tile, tgid, tid);
#else
    (void)uma;(void)h_in;(void)c0;(void)c1;(void)c2;(void)m0;(void)m1;(void)m2;(void)o0;(void)o1;(void)o2;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sik_loop_ctrl_apply_nax_q8_m16(
    device const uchar* uma [[buffer(0)]], device const half* h_in [[buffer(1)]],
    device const float* c0 [[buffer(2)]], device const float* c1 [[buffer(3)]], device const float* c2 [[buffer(4)]],
    device const float* m0 [[buffer(5)]], device const float* m1 [[buffer(6)]], device const float* m2 [[buffer(7)]],
    device float* o0 [[buffer(8)]], device float* o1 [[buffer(9)]], device float* o2 [[buffer(10)]],
    constant SikLoopCtrl3Offsets& p [[buffer(11)]], uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup half W_tile[64][64];
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_apply_nax_q8_impl<16, 2>(uma, h_in, c0, c1, c2, m0, m1, m2, o0, o1, o2, p, W_tile, tgid, tid);
#else
    (void)uma;(void)h_in;(void)c0;(void)c1;(void)c2;(void)m0;(void)m1;(void)m2;(void)o0;(void)o1;(void)o2;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

// Opt-In `--int8-prefill` Direct-Device INT8xINT8->INT32 Loop Controller `down` (`86.8 TOPS`, 0B TG, 0 barriers!)
#if defined(__HAVE_TENSOR__)
template <int BM, int SG, int CAP>
inline void sik_loop_ctrl_down_nax_q8_i8_impl(
    device const uchar*           uma,
    device const int8_t*          cat_i8,
    device const half*            cat_sc,
    device half*                  h_out_f16,
    constant SikLoopCtrl3Offsets& p,
    uint3                         tgid
) {
    const uint j  = tgid.z;
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (j >= p.num_ctrls || m0 >= p.M_padded || n0 >= p.gate_rank) return;

    const uint K    = 2u * p.D;
    const uint ngrp = K >> 6u;
    device int8_t*     x_ptr  = const_cast<device int8_t*>(cat_i8) + ulong(j) * ulong(p.M_padded) * K;
    device const half* xs_ptr = cat_sc + ulong(j) * ulong(p.M_padded);
    device int8_t*     w_ptr  = reinterpret_cast<device int8_t*>(const_cast<device uchar*>(uma) + p.w_off[j]);
    device const half* ws_ptr = reinterpret_cast<device const half*>(uma + p.s_off[j]);
    device half*       h_ptr  = h_out_f16 + ulong(j) * ulong(p.M_padded) * p.gate_rank;

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tX(x_ptr, dextents<int, 2>(int(K), int(p.M_padded)));
    tensor<device int8_t, dextents<int, 2>, tensor_inline> tW(w_ptr, dextents<int, 2>(int(K), int(p.gate_rank)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<64, BM>(0, int(m0));
    auto sW0 = tW.template slice<64, 64>(0, int(n0));
    auto cI  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sX0)>, sik_strip_as<decltype(sW0)>, int32_t>();

    float acc[CAP];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) acc[i] = 0.0f;

    for (uint g = 0u; g < ngrp; ++g) {
        auto sX = tX.template slice<64, BM>(int(g * 64u), int(m0));
        auto sW = tW.template slice<64, 64>(int(g * 64u), int(n0));
        op.run(sX, sW, cI);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < ushort(CAP); ++i) {
            if (cI.is_valid_element(i)) {
                auto idx = cI.get_multidimensional_index(i);
                const uint ng = min(n0 + uint(idx[0]), p.gate_rank - 1u);
                acc[i] = fma(float(cI[i]), float(ws_ptr[ulong(ng) * ngrp + g]), acc[i]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx = cI.get_multidimensional_index(i);
            const uint mg = m0 + uint(idx[1]);
            const uint ng = n0 + uint(idx[0]);
            if (mg < p.M_padded && ng < p.gate_rank) {
                h_ptr[ulong(mg) * p.gate_rank + ng] = (mg < p.M_valid) ? sik_clamp_f16(acc[i] * float(xs_ptr[mg])) : 0.0h;
            }
        }
    }
}

// Opt-In `--int8-prefill` Direct-Device Single-Tile K=64 INT8xINT8->INT32 Loop Controller `apply` (0 loops, 0 barriers, 0B TG!)
template <int BM, int SG, int CAP>
inline void sik_loop_ctrl_apply_nax_q8_i8_impl(
    device const uchar*           uma,
    device const int8_t*          h_i8,
    device const half*            h_sc,
    device const float*           c0,
    device const float*           c1,
    device const float*           c2,
    device const float*           m0_b,
    device const float*           m1_b,
    device const float*           m2_b,
    device float*                 o0,
    device float*                 o1,
    device float*                 o2,
    constant SikLoopCtrl3Offsets& p,
    uint3                         tgid
) {
    const uint j  = tgid.z;
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (j >= p.num_ctrls || m0 >= p.M_padded || n0 >= p.D) return;

    device int8_t*      h_ptr   = const_cast<device int8_t*>(h_i8) + ulong(j) * ulong(p.M_padded) * p.gate_rank;
    device const half*  hs_ptr  = h_sc + ulong(j) * ulong(p.M_padded);
    device int8_t*      wg_ptr  = reinterpret_cast<device int8_t*>(const_cast<device uchar*>(uma) + p.w_off[j]);
    device const half*  wgs_ptr = reinterpret_cast<device const half*>(uma + p.s_off[j]);
    device int8_t*      wf_ptr  = reinterpret_cast<device int8_t*>(const_cast<device uchar*>(uma) + p.f_off[j]);
    device const half*  wfs_ptr = reinterpret_cast<device const half*>(uma + p.fs_off[j]);
    device const float* cur_in  = (j == 0u) ? c0   : ((j == 1u) ? c1   : c2);
    device const float* msg_in  = (j == 0u) ? m0_b : ((j == 1u) ? m1_b : m2_b);
    device float*       out_f32 = (j == 0u) ? o0   : ((j == 1u) ? o1   : o2);

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tH(h_ptr,  dextents<int, 2>(64, int(p.M_padded)));
    tensor<device int8_t, dextents<int, 2>, tensor_inline> tG(wg_ptr, dextents<int, 2>(64, int(p.D)));
    tensor<device int8_t, dextents<int, 2>, tensor_inline> tF(wf_ptr, dextents<int, 2>(64, int(p.D)));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sH = tH.template slice<64, BM>(0, int(m0));
    auto sG = tG.template slice<64, 64>(0, int(n0));
    auto sF = tF.template slice<64, 64>(0, int(n0));
    auto cG = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sH)>, sik_strip_as<decltype(sG)>, int32_t>();
    auto cF = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sH)>, sik_strip_as<decltype(sF)>, int32_t>();
    op.run(sH, sG, cG);
    op.run(sH, sF, cF);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) {
        if (cG.is_valid_element(i)) {
            auto idx = cG.get_multidimensional_index(i);
            const uint mg = m0 + uint(idx[1]);
            const uint ng = n0 + uint(idx[0]);
            if (mg < p.M_valid && ng < p.D) {
                const float hsc = float(hs_ptr[mg]);
                const float gv  = float(cG[i]) * (hsc * float(wgs_ptr[ng]));
                const float fv  = float(cF[i]) * (hsc * float(wfs_ptr[ng]));
                const ulong off = ulong(mg) * p.D + ng;
                out_f32[off] = (1.0f - sik_safe_tanh(fv)) * cur_in[off] + sik_safe_tanh(gv) * msg_in[off];
            }
        }
    }
}
#endif

kernel void sik_loop_ctrl_down_nax_q8_i8_m64(
    device const uchar* uma [[buffer(0)]], device const int8_t* cat_i8 [[buffer(1)]],
    device const half* cat_sc [[buffer(2)]], device half* h_out_f16 [[buffer(3)]],
    constant SikLoopCtrl3Offsets& p [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_down_nax_q8_i8_impl<64, 4, 32>(uma, cat_i8, cat_sc, h_out_f16, p, tgid);
#else
    (void)uma;(void)cat_i8;(void)cat_sc;(void)h_out_f16;(void)p;(void)tgid;
#endif
}

kernel void sik_loop_ctrl_down_nax_q8_i8_m32(
    device const uchar* uma [[buffer(0)]], device const int8_t* cat_i8 [[buffer(1)]],
    device const half* cat_sc [[buffer(2)]], device half* h_out_f16 [[buffer(3)]],
    constant SikLoopCtrl3Offsets& p [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_down_nax_q8_i8_impl<32, 2, 32>(uma, cat_i8, cat_sc, h_out_f16, p, tgid);
#else
    (void)uma;(void)cat_i8;(void)cat_sc;(void)h_out_f16;(void)p;(void)tgid;
#endif
}

kernel void sik_loop_ctrl_apply_nax_q8_i8_m64(
    device const uchar* uma [[buffer(0)]], device const int8_t* h_i8 [[buffer(1)]], device const half* h_sc [[buffer(2)]],
    device const float* c0 [[buffer(3)]], device const float* c1 [[buffer(4)]], device const float* c2 [[buffer(5)]],
    device const float* m0 [[buffer(6)]], device const float* m1 [[buffer(7)]], device const float* m2 [[buffer(8)]],
    device float* o0 [[buffer(9)]], device float* o1 [[buffer(10)]], device float* o2 [[buffer(11)]],
    constant SikLoopCtrl3Offsets& p [[buffer(12)]], uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_apply_nax_q8_i8_impl<64, 4, 32>(uma, h_i8, h_sc, c0, c1, c2, m0, m1, m2, o0, o1, o2, p, tgid);
#else
    (void)uma;(void)h_i8;(void)h_sc;(void)c0;(void)c1;(void)c2;(void)m0;(void)m1;(void)m2;(void)o0;(void)o1;(void)o2;(void)p;(void)tgid;
#endif
}

kernel void sik_loop_ctrl_apply_nax_q8_i8_m32(
    device const uchar* uma [[buffer(0)]], device const int8_t* h_i8 [[buffer(1)]], device const half* h_sc [[buffer(2)]],
    device const float* c0 [[buffer(3)]], device const float* c1 [[buffer(4)]], device const float* c2 [[buffer(5)]],
    device const float* m0 [[buffer(6)]], device const float* m1 [[buffer(7)]], device const float* m2 [[buffer(8)]],
    device float* o0 [[buffer(9)]], device float* o1 [[buffer(10)]], device float* o2 [[buffer(11)]],
    constant SikLoopCtrl3Offsets& p [[buffer(12)]], uint3 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_loop_ctrl_apply_nax_q8_i8_impl<32, 2, 32>(uma, h_i8, h_sc, c0, c1, c2, m0, m1, m2, o0, o1, o2, p, tgid);
#else
    (void)uma;(void)h_i8;(void)h_sc;(void)c0;(void)c1;(void)c2;(void)m0;(void)m1;(void)m2;(void)o0;(void)o1;(void)o2;(void)p;(void)tgid;
#endif
}

/* ============================================================================
 * 4. Coalesced 16-Lane Dense GEMV (`FP16`, `Q8_0`, `Q4_A`, `M = 1..8`) &
 *    Metal 4.1 Direct/Quantized NAX Prefill GEMMs (`M > 8`)
 * ============================================================================ */

struct SikGemvBatchParams {
    uint  M;
    uint  rows;
    uint  cols;
    uint  out_mode; // SIK_OUT_*
    float softcap;  // >0 when SIK_OUT_F32_SOFTCAP
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

template <uint M_FIX>
inline void sik_store_gemv1(
    device half*  y_f16,
    device float* y_f32,
    uint row,
    uint rows,
    uint out_mode,
    float softcap,
    thread const float (&res)[M_FIX]
) {
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const ulong idx = ulong(m) * rows + row;
        float v = res[m];
        if (out_mode == SIK_OUT_F32_SOFTCAP && softcap > 0.0f) {
            v = sik_softcap(v, softcap);
        }
        if (out_mode == SIK_OUT_F16)          y_f16[idx]  = sik_clamp_f16(v);
        else if (out_mode == SIK_OUT_RES_F32) y_f32[idx] += v;
        else                                  y_f32[idx]  = v;
    }
}

// 16-lane/row Coalesced FP16 GEMV (1 row/half-SG => ZERO register pressure at M=8, 490-521 GB/s!)
template <uint M_FIX>
inline void sik_f16_gemv_l16_fixed(
    device const half* w_f16,
    device const half* x_in,
    device half*       y_f16,
    device float*      y_f32,
    uint row,
    uint rows,
    uint cols,
    uint out_mode,
    float softcap,
    uint lane16
) {
    const uint c8_cnt = cols >> 3u;
    device const uint4* w16 = reinterpret_cast<device const uint4*>(w_f16 + ulong(row) * cols);

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint c8 = lane16; c8 < c8_cnt; c8 += 16u) {
        const uint4 v = w16[c8];
        const half4 a0 = as_type<half4>(v.xy);
        const half4 a1 = as_type<half4>(v.zw);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + c8 * 8u);
            acc[m] += dot(float4(a0), float4(x4[0])) + dot(float4(a1), float4(x4[1]));
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = sik_reduce16(acc[m]);
    if (lane16 == 0u) sik_store_gemv1<M_FIX>(y_f16, y_f32, row, rows, out_mode, softcap, acc);
}

// 16-lane/row Coalesced Q8_0 (gs=64) GEMV (`16B/lane = 256B` contiguous, 522+ GB/s!)
template <uint M_FIX>
inline void sik_q8_gemv_l16_fixed(
    device const char* w_q,
    device const half* w_s,
    device const half* x_in,
    device half*       y_f16,
    device float*      y_f32,
    uint row,
    uint rows,
    uint cols,
    uint out_mode,
    float softcap,
    uint lane16
) {
    const uint k16  = cols >> 4u;
    const uint nblk = cols >> 6u;
    device const uint4* w16 = reinterpret_cast<device const uint4*>(w_q + ulong(row) * cols);
    device const half*  sc  = w_s + ulong(row) * nblk;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = lane16; i < k16; i += 16u) {
        const float  s  = float(sc[i >> 2u]);
        const uint4  q  = w16[i];
        const float4 f0 = float4(as_type<char4>(q.x)), f1 = float4(as_type<char4>(q.y));
        const float4 f2 = float4(as_type<char4>(q.z)), f3 = float4(as_type<char4>(q.w));
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + i * 16u);
            const float d = (dot(f0, float4(xm[0])) + dot(f1, float4(xm[1])))
                          + (dot(f2, float4(xm[2])) + dot(f3, float4(xm[3])));
            acc[m] = fma(s, d, acc[m]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = sik_reduce16(acc[m]);
    if (lane16 == 0u) sik_store_gemv1<M_FIX>(y_f16, y_f32, row, rows, out_mode, softcap, acc);
}

// 16-lane/row Coalesced Q4_A (gs=64) GEMV (`uint2` = 16 cols/iter, low register pressure!)
template <uint M_FIX>
inline void sik_q4a_gemv_l16_fixed(
    device const uint2* w_u2,
    device const half*  w_s,
    device const half*  w_b,
    device const half*  x_in,
    device half*        y_f16,
    device float*       y_f32,
    uint row,
    uint rows,
    uint cols,
    uint out_mode,
    float softcap,
    uint lane16
) {
    const uint k16  = cols >> 4u;
    const uint nblk = cols >> 6u;
    device const uint2* rw = w_u2 + ulong(row) * k16;
    device const half*  rs = w_s  + ulong(row) * nblk;
    device const half*  rb = w_b  + ulong(row) * nblk;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = lane16; i < k16; i += 16u) {
        const uint2 q   = rw[i];
        const uint  grp = i >> 2u;
        const float sc  = float(rs[grp]);
        const float bi  = float(rb[grp]);
        half4 n0, n1, n2, n3;
        sik_unpack_u4_u32_h4(q.x, n0, n1);
        sik_unpack_u4_u32_h4(q.y, n2, n3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + i * 16u);
            const half4 x0 = xm[0], x1 = xm[1], x2 = xm[2], x3 = xm[3];
            const half4 dh = fma(n0, x0, n1 * x1) + fma(n2, x2, n3 * x3);
            const half4 sh = (x0 + x1) + (x2 + x3);
            const float d  = (float(dh.x) + float(dh.y)) + (float(dh.z) + float(dh.w));
            const float xs = (float(sh.x) + float(sh.y)) + (float(sh.z) + float(sh.w));
            acc[m] = fma(sc, d, fma(bi, xs, acc[m]));
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = sik_reduce16(acc[m]);
    if (lane16 == 0u) sik_store_gemv1<M_FIX>(y_f16, y_f32, row, rows, out_mode, softcap, acc);
}

// 16-lane/row Coalesced Q4_A (gs=64) GEMV with pre-computed `x_blk_sum` (479 GB/s @ M=1, 394 GB/s @ M=5!)
template <uint M_FIX>
inline void sik_q4a_gemv_l16_xsum_fixed(
    device const uint2* w_u2,
    device const half*  w_s,
    device const half*  w_b,
    device const half*  x_in,
    device const float* x_sum,
    device half*        y_f16,
    device float*       y_f32,
    uint row,
    uint rows,
    uint cols,
    uint out_mode,
    float softcap,
    uint lane16
) {
    const uint k16     = cols >> 4u;
    const uint nblk    = cols >> 6u;
    const bool do_bias = ((lane16 & 3u) == 0u);
    device const uint2* rw = w_u2 + ulong(row) * k16;
    device const half*  rs = w_s  + ulong(row) * nblk;
    device const half*  rb = w_b  + ulong(row) * nblk;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = lane16; i < k16; i += 16u) {
        const uint2 q   = rw[i];
        const uint  grp = i >> 2u;
        const float sc  = float(rs[grp]);
        half4 n0, n1, n2, n3;
        sik_unpack_u4_u32_h4(q.x, n0, n1);
        sik_unpack_u4_u32_h4(q.y, n2, n3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + i * 16u);
            const half4 dh = fma(n0, xm[0], n1 * xm[1]) + fma(n2, xm[2], n3 * xm[3]);
            acc[m] = fma(sc, (float(dh.x) + float(dh.y)) + (float(dh.z) + float(dh.w)), acc[m]);
        }
        if (do_bias) {
            const float bi = float(rb[grp]);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                acc[m] = fma(bi, x_sum[ulong(m) * nblk + grp], acc[m]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = sik_reduce16(acc[m]);
    if (lane16 == 0u) sik_store_gemv1<M_FIX>(y_f16, y_f32, row, rows, out_mode, softcap, acc);
}

template <uint M_FIX>
inline void sik_store_gemv2(
    device half*  y_f16,
    device float* y_f32,
    uint r0,
    uint rows,
    uint out_mode,
    float softcap,
    thread const float2 (&res)[M_FIX]
) {
    const bool has_r1 = (r0 + 1u < rows);
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const ulong idx0 = ulong(m) * rows + r0;
        float v0 = res[m].x;
        float v1 = res[m].y;
        if (out_mode == SIK_OUT_F32_SOFTCAP && softcap > 0.0f) {
            v0 = sik_softcap(v0, softcap);
            v1 = sik_softcap(v1, softcap);
        }
        if (out_mode == SIK_OUT_F16) {
            y_f16[idx0] = sik_clamp_f16(v0);
            if (has_r1) y_f16[idx0 + 1ul] = sik_clamp_f16(v1);
        } else if (out_mode == SIK_OUT_RES_F32) {
            y_f32[idx0] += v0;
            if (has_r1) y_f32[idx0 + 1ul] += v1;
        } else {
            y_f32[idx0] = v0;
            if (has_r1) y_f32[idx0 + 1ul] = v1;
        }
    }
}

// 32-lane/row, 2-row/SG Coalesced FP16 GEMV (compact regs for mixed-projection dispatch!)
template <uint M_FIX>
inline void sik_f16_gemv_l32_2r_fixed(
    device const half* w_f16,
    device const half* x_in,
    device half*       y_f16,
    device float*      y_f32,
    uint r0,
    uint rows,
    uint cols,
    uint out_mode,
    float softcap,
    uint lane
) {
    const uint r1     = min(r0 + 1u, rows - 1u);
    const uint c8_cnt = cols >> 3u;
    device const uint4* w0 = reinterpret_cast<device const uint4*>(w_f16 + ulong(r0) * cols);
    device const uint4* w1 = reinterpret_cast<device const uint4*>(w_f16 + ulong(r1) * cols);

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint c8 = lane; c8 < c8_cnt; c8 += 32u) {
        const uint4 v0 = w0[c8], v1 = w1[c8];
        const half4 a0 = as_type<half4>(v0.xy), a1 = as_type<half4>(v0.zw);
        const half4 b0 = as_type<half4>(v1.xy), b1 = as_type<half4>(v1.zw);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + c8 * 8u);
            const float4 fx0 = float4(x4[0]), fx1 = float4(x4[1]);
            acc[m].x += dot(float4(a0), fx0) + dot(float4(a1), fx1);
            acc[m].y += dot(float4(b0), fx0) + dot(float4(b1), fx1);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        acc[m] = float2(simd_sum(acc[m].x), simd_sum(acc[m].y));
    }
    if (lane == 0u) sik_store_gemv2<M_FIX>(y_f16, y_f32, r0, rows, out_mode, softcap, acc);
}

// 32-lane/row, 2-row/SG Coalesced Q8_0 (gs=64) GEMV (compact regs for mixed-projection dispatch!)
template <uint M_FIX>
inline void sik_q8_gemv_l32_2r_fixed(
    device const char* w_q,
    device const half* w_s,
    device const half* x_in,
    device half*       y_f16,
    device float*      y_f32,
    uint r0,
    uint rows,
    uint cols,
    uint out_mode,
    float softcap,
    uint lane
) {
    const uint r1   = min(r0 + 1u, rows - 1u);
    const uint k16  = cols >> 4u;
    const uint nblk = cols >> 6u;
    device const uint4* w0  = reinterpret_cast<device const uint4*>(w_q + ulong(r0) * cols);
    device const uint4* w1  = reinterpret_cast<device const uint4*>(w_q + ulong(r1) * cols);
    device const half*  sc0 = w_s + ulong(r0) * nblk;
    device const half*  sc1 = w_s + ulong(r1) * nblk;

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint i = lane; i < k16; i += 32u) {
        const uint   grp = i >> 2u;
        const float2 s   = float2(float(sc0[grp]), float(sc1[grp]));
        const uint4  q0  = w0[i], q1 = w1[i];
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + i * 16u);
            const float4 x0 = float4(xm[0]), x1 = float4(xm[1]), x2 = float4(xm[2]), x3 = float4(xm[3]);
            const float d0 = (dot(float4(as_type<char4>(q0.x)), x0) + dot(float4(as_type<char4>(q0.y)), x1))
                           + (dot(float4(as_type<char4>(q0.z)), x2) + dot(float4(as_type<char4>(q0.w)), x3));
            const float d1 = (dot(float4(as_type<char4>(q1.x)), x0) + dot(float4(as_type<char4>(q1.y)), x1))
                           + (dot(float4(as_type<char4>(q1.z)), x2) + dot(float4(as_type<char4>(q1.w)), x3));
            acc[m].x = fma(s.x, d0, acc[m].x);
            acc[m].y = fma(s.y, d1, acc[m].y);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        acc[m] = float2(simd_sum(acc[m].x), simd_sum(acc[m].y));
    }
    if (lane == 0u) sik_store_gemv2<M_FIX>(y_f16, y_f32, r0, rows, out_mode, softcap, acc);
}

// 32-lane/row, 2-row/SG Coalesced Q4_A (gs=64) GEMV (256B full-SG bursts + 2x x-reuse!)
template <uint M_FIX>
inline void sik_q4a_gemv_l32_2r_fixed(
    device const uint2* w_u2,
    device const half*  w_s,
    device const half*  w_b,
    device const half*  x_in,
    device half*        y_f16,
    device float*       y_f32,
    uint r0,
    uint rows,
    uint cols,
    uint out_mode,
    float softcap,
    uint lane
) {
    const uint r1   = min(r0 + 1u, rows - 1u);
    const uint k16  = cols >> 4u;
    const uint nblk = cols >> 6u;
    device const uint2* rw0 = w_u2 + ulong(r0) * k16;
    device const uint2* rw1 = w_u2 + ulong(r1) * k16;
    device const half*  rs0 = w_s  + ulong(r0) * nblk;
    device const half*  rs1 = w_s  + ulong(r1) * nblk;
    device const half*  rb0 = w_b  + ulong(r0) * nblk;
    device const half*  rb1 = w_b  + ulong(r1) * nblk;

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint i = lane; i < k16; i += 32u) {
        const uint2  q0  = rw0[i], q1 = rw1[i];
        const uint   grp = i >> 2u;
        const float2 sc  = float2(float(rs0[grp]), float(rs1[grp]));
        const float2 bi  = float2(float(rb0[grp]), float(rb1[grp]));
        half4 a0, a1, a2, a3, b0, b1, b2, b3;
        sik_unpack_u4_u32_h4(q0.x, a0, a1); sik_unpack_u4_u32_h4(q0.y, a2, a3);
        sik_unpack_u4_u32_h4(q1.x, b0, b1); sik_unpack_u4_u32_h4(q1.y, b2, b3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + i * 16u);
            const half4 x0 = xm[0], x1 = xm[1], x2 = xm[2], x3 = xm[3];
            const half4 dh0 = fma(a0, x0, a1 * x1) + fma(a2, x2, a3 * x3);
            const half4 dh1 = fma(b0, x0, b1 * x1) + fma(b2, x2, b3 * x3);
            const half4 sh  = (x0 + x1) + (x2 + x3);
            const float d0  = (float(dh0.x) + float(dh0.y)) + (float(dh0.z) + float(dh0.w));
            const float d1  = (float(dh1.x) + float(dh1.y)) + (float(dh1.z) + float(dh1.w));
            const float xs  = (float(sh.x) + float(sh.y)) + (float(sh.z) + float(sh.w));
            acc[m].x = fma(sc.x, d0, fma(bi.x, xs, acc[m].x));
            acc[m].y = fma(sc.y, d1, fma(bi.y, xs, acc[m].y));
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        acc[m] = float2(simd_sum(acc[m].x), simd_sum(acc[m].y));
    }
    if (lane == 0u) sik_store_gemv2<M_FIX>(y_f16, y_f32, r0, rows, out_mode, softcap, acc);
}

// 32-lane/row, 2-row/SG Coalesced Q4_A (gs=64) GEMV with pre-computed `x_blk_sum` (458-500 GB/s @ M=1, +25-46%!)
template <uint M_FIX>
inline void sik_q4a_gemv_l32_2r_xsum_fixed(
    device const uint2* w_u2,
    device const half*  w_s,
    device const half*  w_b,
    device const half*  x_in,
    device const float* x_sum,
    device half*        y_f16,
    device float*       y_f32,
    uint r0,
    uint rows,
    uint cols,
    uint out_mode,
    float softcap,
    uint lane
) {
    const uint r1      = min(r0 + 1u, rows - 1u);
    const uint k16     = cols >> 4u;
    const uint nblk    = cols >> 6u;
    const bool do_bias = ((lane & 3u) == 0u);
    device const uint2* rw0 = w_u2 + ulong(r0) * k16;
    device const uint2* rw1 = w_u2 + ulong(r1) * k16;
    device const half*  rs0 = w_s  + ulong(r0) * nblk;
    device const half*  rs1 = w_s  + ulong(r1) * nblk;
    device const half*  rb0 = w_b  + ulong(r0) * nblk;
    device const half*  rb1 = w_b  + ulong(r1) * nblk;

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint i = lane; i < k16; i += 32u) {
        const uint2  q0  = rw0[i], q1 = rw1[i];
        const uint   grp = i >> 2u;
        const float2 sc  = float2(float(rs0[grp]), float(rs1[grp]));
        half4 a0, a1, a2, a3, b0, b1, b2, b3;
        sik_unpack_u4_u32_h4(q0.x, a0, a1); sik_unpack_u4_u32_h4(q0.y, a2, a3);
        sik_unpack_u4_u32_h4(q1.x, b0, b1); sik_unpack_u4_u32_h4(q1.y, b2, b3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + i * 16u);
            const half4 x0 = xm[0], x1 = xm[1], x2 = xm[2], x3 = xm[3];
            const half4 dh0 = fma(a0, x0, a1 * x1) + fma(a2, x2, a3 * x3);
            const half4 dh1 = fma(b0, x0, b1 * x1) + fma(b2, x2, b3 * x3);
            acc[m].x = fma(sc.x, (float(dh0.x) + float(dh0.y)) + (float(dh0.z) + float(dh0.w)), acc[m].x);
            acc[m].y = fma(sc.y, (float(dh1.x) + float(dh1.y)) + (float(dh1.z) + float(dh1.w)), acc[m].y);
        }
        if (do_bias) {
            const float2 bi = float2(float(rb0[grp]), float(rb1[grp]));
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                acc[m] = fma(bi, float2(x_sum[ulong(m) * nblk + grp]), acc[m]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        acc[m] = float2(simd_sum(acc[m].x), simd_sum(acc[m].y));
    }
    if (lane == 0u) sik_store_gemv2<M_FIX>(y_f16, y_f32, r0, rows, out_mode, softcap, acc);
}

// 32-lane/row, 1-row/SG Coalesced Q4_A (gs=64) GEMV w/ pre-computed `x_blk_sum` (532.6 GB/s @ M=1, half regs @ M=8!)
template <uint M_FIX>
inline void sik_q4a_gemv_l32_1r_xsum_fixed(
    device const uint2* w_u2,
    device const half*  w_s,
    device const half*  w_b,
    device const half*  x_in,
    device const float* x_sum,
    device half*        y_f16,
    device float*       y_f32,
    uint row,
    uint rows,
    uint cols,
    uint out_mode,
    uint lane
) {
    const uint k16     = cols >> 4u;
    const uint nblk    = cols >> 6u;
    const bool do_bias = ((lane & 3u) == 0u);
    device const uint2* rw = w_u2 + ulong(row) * k16;
    device const half*  rs = w_s  + ulong(row) * nblk;
    device const half*  rb = w_b  + ulong(row) * nblk;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = lane; i < k16; i += 32u) {
        const uint2 q   = rw[i];
        const uint  grp = i >> 2u;
        const float sc  = float(rs[grp]);
        half4 n0, n1, n2, n3;
        sik_unpack_u4_u32_h4(q.x, n0, n1);
        sik_unpack_u4_u32_h4(q.y, n2, n3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + i * 16u);
            const half4 dh = fma(n0, xm[0], n1 * xm[1]) + fma(n2, xm[2], n3 * xm[3]);
            acc[m] = fma(sc, (float(dh.x) + float(dh.y)) + (float(dh.z) + float(dh.w)), acc[m]);
        }
        if (do_bias) {
            const float bi = float(rb[grp]);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                acc[m] = fma(bi, x_sum[ulong(m) * nblk + grp], acc[m]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0u) {
            const ulong idx = ulong(m) * rows + row;
            if (out_mode == SIK_OUT_F16) y_f16[idx] = sik_clamp_f16(v);
            else                         y_f32[idx] = v;
        }
    }
}

// Launch with `128` th/TG (8 rows/TG) OR `64` th/TG (4 rows/TG for `wo` -> 576 TGs!)
kernel void sik_f16_gemv_batch(
    device const half*           w_f16 [[buffer(0)]],
    device const half*           x_in  [[buffer(1)]],
    device half*                 y_f16 [[buffer(2)]],
    device float*                y_f32 [[buffer(3)]],
    constant SikGemvBatchParams& p     [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = (tgid * sgs + sgid) * 2u + (lane >> 4u);
    if (row >= p.rows) return;
    const uint l16 = lane & 15u;
    #define SIK_F16_CASE(MF) { sik_f16_gemv_l16_fixed<MF>(w_f16, x_in, y_f16, y_f32, row, p.rows, p.cols, p.out_mode, p.softcap, l16); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_F16_CASE(1u) case 2u: SIK_F16_CASE(2u) case 3u: SIK_F16_CASE(3u) case 4u: SIK_F16_CASE(4u)
        case 5u: SIK_F16_CASE(5u) case 6u: SIK_F16_CASE(6u) case 7u: SIK_F16_CASE(7u) case 8u: SIK_F16_CASE(8u)
        default: break;
    }
    #undef SIK_F16_CASE
}

kernel void sik_q8_gemv_batch(
    device const char*           w_q   [[buffer(0)]],
    device const half*           w_s   [[buffer(1)]],
    device const half*           x_in  [[buffer(2)]],
    device half*                 y_f16 [[buffer(3)]],
    device float*                y_f32 [[buffer(4)]],
    constant SikGemvBatchParams& p     [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = (tgid * sgs + sgid) * 2u + (lane >> 4u);
    if (row >= p.rows) return;
    const uint l16 = lane & 15u;
    #define SIK_Q8_CASE(MF) { sik_q8_gemv_l16_fixed<MF>(w_q, w_s, x_in, y_f16, y_f32, row, p.rows, p.cols, p.out_mode, p.softcap, l16); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_Q8_CASE(1u) case 2u: SIK_Q8_CASE(2u) case 3u: SIK_Q8_CASE(3u) case 4u: SIK_Q8_CASE(4u)
        case 5u: SIK_Q8_CASE(5u) case 6u: SIK_Q8_CASE(6u) case 7u: SIK_Q8_CASE(7u) case 8u: SIK_Q8_CASE(8u)
        default: break;
    }
    #undef SIK_Q8_CASE
}

kernel void sik_q4a_gemv_batch(
    device const uint2*          w_u2  [[buffer(0)]],
    device const half*           w_s   [[buffer(1)]],
    device const half*           w_b   [[buffer(2)]],
    device const half*           x_in  [[buffer(3)]],
    device half*                 y_f16 [[buffer(4)]],
    device float*                y_f32 [[buffer(5)]],
    constant SikGemvBatchParams& p     [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint r0 = (tgid * sgs + sgid) * 2u;
    if (r0 >= p.rows) return;
    #define SIK_Q4A_CASE(MF) { sik_q4a_gemv_l32_2r_fixed<MF>(w_u2, w_s, w_b, x_in, y_f16, y_f32, r0, p.rows, p.cols, p.out_mode, p.softcap, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_Q4A_CASE(1u) case 2u: SIK_Q4A_CASE(2u) case 3u: SIK_Q4A_CASE(3u) case 4u: SIK_Q4A_CASE(4u)
        case 5u: SIK_Q4A_CASE(5u) case 6u: SIK_Q4A_CASE(6u) case 7u: SIK_Q4A_CASE(7u) case 8u: SIK_Q4A_CASE(8u)
        default: break;
    }
    #undef SIK_Q4A_CASE
}

kernel void sik_q4a_gemv_batch_xsum(
    device const uint2*          w_u2  [[buffer(0)]],
    device const half*           w_s   [[buffer(1)]],
    device const half*           w_b   [[buffer(2)]],
    device const half*           x_in  [[buffer(3)]],
    device const float*          x_sum [[buffer(4)]],
    device half*                 y_f16 [[buffer(5)]],
    device float*                y_f32 [[buffer(6)]],
    constant SikGemvBatchParams& p     [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint r0 = (tgid * sgs + sgid) * 2u;
    if (r0 >= p.rows) return;
    #define SIK_Q4AX_CASE(MF) { sik_q4a_gemv_l32_2r_xsum_fixed<MF>(w_u2, w_s, w_b, x_in, x_sum, y_f16, y_f32, r0, p.rows, p.cols, p.out_mode, p.softcap, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_Q4AX_CASE(1u) case 2u: SIK_Q4AX_CASE(2u) case 3u: SIK_Q4AX_CASE(3u) case 4u: SIK_Q4AX_CASE(4u)
        case 5u: SIK_Q4AX_CASE(5u) case 6u: SIK_Q4AX_CASE(6u) case 7u: SIK_Q4AX_CASE(7u) case 8u: SIK_Q4AX_CASE(8u)
        default: break;
    }
    #undef SIK_Q4AX_CASE
}

// High-Occupancy `1` Row/TG (`128` th = `4` SGs) FP16 & Q8_0 GEMV for small-N projections (sink, sconv_proj, loop_down, wr)
template <uint M_FIX>
inline void sik_f16_row_tg128_impl(
    device const half* w_f16,
    device const half* x_in,
    device half*       y_f16,
    device float*      y_f32,
    threadgroup float  (&tg_red)[8][4],
    uint row, uint rows, uint cols, uint out_mode, float softcap,
    uint tid, uint sgid, uint lane
) {
    const uint c8_cnt = cols >> 3u;
    device const half4* w4 = reinterpret_cast<device const half4*>(w_f16 + ulong(row) * cols);

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint c8 = tid; c8 < c8_cnt; c8 += 128u) {
        const float4 fw0 = float4(w4[c8 * 2u + 0u]), fw1 = float4(w4[c8 * 2u + 1u]);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + c8 * 8u);
            acc[m] += dot(fw0, float4(x4[0])) + dot(fw1, float4(x4[1]));
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            float tot = simd_sum((lane < 4u) ? tg_red[m][lane] : 0.0f);
            if (lane == 0u) {
                if (out_mode == SIK_OUT_F32_SOFTCAP && softcap > 0.0f) tot = sik_softcap(tot, softcap);
                const ulong idx = ulong(m) * rows + row;
                if (out_mode == SIK_OUT_F16)          y_f16[idx]  = sik_clamp_f16(tot);
                else if (out_mode == SIK_OUT_RES_F32) y_f32[idx] += tot;
                else                                  y_f32[idx]  = tot;
            }
        }
    }
}

template <uint M_FIX>
inline void sik_q8_row_tg128_impl(
    device const char* w_q,
    device const half* w_s,
    device const half* x_in,
    device half*       y_f16,
    device float*      y_f32,
    threadgroup float  (&tg_red)[8][4],
    uint row, uint rows, uint cols, uint out_mode, float softcap,
    uint tid, uint sgid, uint lane
) {
    const uint k16  = cols >> 4u;
    const uint nblk = cols >> 6u;
    device const uint4* w16 = reinterpret_cast<device const uint4*>(w_q + ulong(row) * cols);
    device const half*  sc  = w_s + ulong(row) * nblk;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = tid; i < k16; i += 128u) {
        const float s  = float(sc[i >> 2u]);
        const uint4 q  = w16[i];
        const char4 c0 = as_type<char4>(q.x), c1 = as_type<char4>(q.y);
        const char4 c2 = as_type<char4>(q.z), c3 = as_type<char4>(q.w);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + i * 16u);
            const float d = (dot(float4(c0), float4(xm[0])) + dot(float4(c1), float4(xm[1])))
                          + (dot(float4(c2), float4(xm[2])) + dot(float4(c3), float4(xm[3])));
            acc[m] = fma(s, d, acc[m]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            float tot = simd_sum((lane < 4u) ? tg_red[m][lane] : 0.0f);
            if (lane == 0u) {
                if (out_mode == SIK_OUT_F32_SOFTCAP && softcap > 0.0f) tot = sik_softcap(tot, softcap);
                const ulong idx = ulong(m) * rows + row;
                if (out_mode == SIK_OUT_F16)          y_f16[idx]  = sik_clamp_f16(tot);
                else if (out_mode == SIK_OUT_RES_F32) y_f32[idx] += tot;
                else                                  y_f32[idx]  = tot;
            }
        }
    }
}

kernel void sik_f16_gemv_row_tg128(
    device const half*           w_f16 [[buffer(0)]],
    device const half*           x_in  [[buffer(1)]],
    device half*                 y_f16 [[buffer(2)]],
    device float*                y_f32 [[buffer(3)]],
    constant SikGemvBatchParams& p     [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][4];
    if (row >= p.rows) return;
    #define SIK_R128_CASE(MF) { sik_f16_row_tg128_impl<MF>(w_f16, x_in, y_f16, y_f32, tg_red, row, p.rows, p.cols, p.out_mode, p.softcap, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_R128_CASE(1u) case 2u: SIK_R128_CASE(2u) case 3u: SIK_R128_CASE(3u) case 4u: SIK_R128_CASE(4u)
        case 5u: SIK_R128_CASE(5u) case 6u: SIK_R128_CASE(6u) case 7u: SIK_R128_CASE(7u) case 8u: SIK_R128_CASE(8u)
        default: break;
    }
    #undef SIK_R128_CASE
}

kernel void sik_q8_gemv_row_tg128(
    device const char*           w_q   [[buffer(0)]],
    device const half*           w_s   [[buffer(1)]],
    device const half*           x_in  [[buffer(2)]],
    device half*                 y_f16 [[buffer(3)]],
    device float*                y_f32 [[buffer(4)]],
    constant SikGemvBatchParams& p     [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][4];
    if (row >= p.rows) return;
    #define SIK_QR128_CASE(MF) { sik_q8_row_tg128_impl<MF>(w_q, w_s, x_in, y_f16, y_f32, tg_red, row, p.rows, p.cols, p.out_mode, p.softcap, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_QR128_CASE(1u) case 2u: SIK_QR128_CASE(2u) case 3u: SIK_QR128_CASE(3u) case 4u: SIK_QR128_CASE(4u)
        case 5u: SIK_QR128_CASE(5u) case 6u: SIK_QR128_CASE(6u) case 7u: SIK_QR128_CASE(7u) case 8u: SIK_QR128_CASE(8u)
        default: break;
    }
    #undef SIK_QR128_CASE
}

kernel void sik_loop_ctrl_down3(
    device const half*           w0    [[buffer(0)]],
    device const half*           w1    [[buffer(1)]],
    device const half*           w2    [[buffer(2)]],
    device const half*           cat   [[buffer(3)]],
    device float*                h_out [[buffer(4)]],
    constant SikGemvBatchParams& p     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][4];
    const uint row = tgid.x, j = tgid.y;
    if (row >= p.rows || j >= 3u) return;
    device const half* w_f16 = (j == 0u) ? w0 : ((j == 1u) ? w1 : w2);
    device const half* x_in  = cat   + ulong(j) * ulong(p.M) * p.cols;
    device float*      y_f32 = h_out + ulong(j) * ulong(p.M) * p.rows;
    #define SIK_D3_CASE(MF) { sik_f16_row_tg128_impl<MF>(w_f16, x_in, nullptr, y_f32, tg_red, row, p.rows, p.cols, SIK_OUT_F32, 0.0f, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_D3_CASE(1u) case 2u: SIK_D3_CASE(2u) case 3u: SIK_D3_CASE(3u) case 4u: SIK_D3_CASE(4u)
        case 5u: SIK_D3_CASE(5u) case 6u: SIK_D3_CASE(6u) case 7u: SIK_D3_CASE(7u) case 8u: SIK_D3_CASE(8u)
        default: break;
    }
    #undef SIK_D3_CASE
}

// Zero-Barrier 1-SG (`32` th/TG) Q8_0 Loop Controller Down (`num_ctrls = 1` or `3`, 2.1x faster!)
template <uint M_FIX>
inline void sik_q8_row_tg32_impl(
    device const char* w_q,
    device const half* w_s,
    device const half* x_in,
    device float*      y_f32,
    uint row, uint rows, uint cols, uint lane
) {
    const uint k16  = cols >> 4u;
    const uint nblk = cols >> 6u;
    device const uint4* w16 = reinterpret_cast<device const uint4*>(w_q + ulong(row) * cols);
    device const half*  sc  = w_s + ulong(row) * nblk;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = lane; i < k16; i += 32u) {
        const float  s  = float(sc[i >> 2u]);
        const uint4  q  = w16[i];
        const float4 f0 = float4(as_type<char4>(q.x)), f1 = float4(as_type<char4>(q.y));
        const float4 f2 = float4(as_type<char4>(q.z)), f3 = float4(as_type<char4>(q.w));
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * cols + i * 16u);
            const float d = (dot(f0, float4(xm[0])) + dot(f1, float4(xm[1])))
                          + (dot(f2, float4(xm[2])) + dot(f3, float4(xm[3])));
            acc[m] = fma(s, d, acc[m]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float tot = simd_sum(acc[m]);
        if (lane == 0u) y_f32[ulong(m) * rows + row] = tot;
    }
}

kernel void sik_loop_ctrl_down3_q8(
    device const uchar*           uma   [[buffer(0)]],
    device const half*            cat   [[buffer(1)]],
    device float*                 h_out [[buffer(2)]],
    constant SikLoopCtrl3Offsets& p     [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid.x, j = tgid.y;
    if (row >= p.gate_rank || j >= p.num_ctrls) return;
    const uint cols = 2u * p.D;
    device const char* w_q   = reinterpret_cast<device const char*>(uma + p.w_off[j]);
    device const half* w_s   = reinterpret_cast<device const half*>(uma + p.s_off[j]);
    device const half* x_in  = cat   + ulong(j) * ulong(p.M_valid) * cols;
    device float*      y_f32 = h_out + ulong(j) * ulong(p.M_valid) * p.gate_rank;
    #define SIK_DQ3_CASE(MF) { sik_q8_row_tg32_impl<MF>(w_q, w_s, x_in, y_f32, row, p.gate_rank, cols, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_DQ3_CASE(1u) case 2u: SIK_DQ3_CASE(2u) case 3u: SIK_DQ3_CASE(3u) case 4u: SIK_DQ3_CASE(4u)
        case 5u: SIK_DQ3_CASE(5u) case 6u: SIK_DQ3_CASE(6u) case 7u: SIK_DQ3_CASE(7u) case 8u: SIK_DQ3_CASE(8u)
        default: break;
    }
    #undef SIK_DQ3_CASE
}

// Batch-12 Full-GPU `down_w_right @ rms(msg)` at Pass 1 & Pass 2 Start (`3,072` TGs -> `500 GB/s`, `~0.03 ms`!)
struct SikLoopBatch12Offsets {
    ulong w_off[12][4];
    ulong s_off[12][4];
    uint  M_valid;
    uint  D;
    uint  gate_rank;
    uint  num_layers;
};

template <uint M_FIX>
inline void sik_q8_half_row_tg32_impl(
    device const char*  w_q,
    device const half*  w_s,
    device const half*  x_in,
    device const float* h_add, // nullptr -> write directly; non-null -> add to result
    device float*       y_f32,
    uint row, uint rows, uint D, uint col_off, uint lane
) {
    const uint k16       = D >> 4u;
    const uint nblk_full = (2u * D) >> 6u;
    const uint blk_off   = col_off >> 6u;
    device const uint4* w16 = reinterpret_cast<device const uint4*>(w_q + ulong(row) * (2ul * D) + col_off);
    device const half*  sc  = w_s + ulong(row) * nblk_full + blk_off;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = lane; i < k16; i += 32u) {
        const float  s  = float(sc[i >> 2u]);
        const uint4  q  = w16[i];
        const float4 f0 = float4(as_type<char4>(q.x)), f1 = float4(as_type<char4>(q.y));
        const float4 f2 = float4(as_type<char4>(q.z)), f3 = float4(as_type<char4>(q.w));
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * D + i * 16u);
            const float d = (dot(f0, float4(xm[0])) + dot(f1, float4(xm[1])))
                          + (dot(f2, float4(xm[2])) + dot(f3, float4(xm[3])));
            acc[m] = fma(s, d, acc[m]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float tot = simd_sum(acc[m]);
        if (lane == 0u) {
            const ulong idx = ulong(m) * rows + row;
            y_f32[idx] = h_add ? (tot + h_add[idx]) : tot;
        }
    }
}

// Grid: `(gate_rank /* 64 */, 4, num_layers /* 12 */)`, `32` threads/TG (`1` SG, ZERO barriers!)
kernel void sik_loop_ctrl_down_right_batch12_q8(
    device const uchar*             uma        [[buffer(0)]],
    device const half*              msg_norm12 [[buffer(1)]], // [12, 4, M_valid, D]
    device float*                   h_msg12    [[buffer(2)]], // [12, 4, M_valid, R]
    constant SikLoopBatch12Offsets& p          [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid.x, j = tgid.y, li = tgid.z;
    if (row >= p.gate_rank || j >= 4u || li >= p.num_layers) return;
    device const char* w_q   = reinterpret_cast<device const char*>(uma + p.w_off[li][j]);
    device const half* w_s   = reinterpret_cast<device const half*>(uma + p.s_off[li][j]);
    const ulong slot = ulong(li) * 4ul + j;
    device const half* x_in  = msg_norm12 + slot * ulong(p.M_valid) * p.D;
    device float*      y_f32 = h_msg12    + slot * ulong(p.M_valid) * p.gate_rank;
    #define SIK_DR12_CASE(MF) { sik_q8_half_row_tg32_impl<MF>(w_q, w_s, x_in, nullptr, y_f32, row, p.gate_rank, p.D, p.D, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_DR12_CASE(1u) case 2u: SIK_DR12_CASE(2u) case 3u: SIK_DR12_CASE(3u) case 4u: SIK_DR12_CASE(4u)
        case 5u: SIK_DR12_CASE(5u) case 6u: SIK_DR12_CASE(6u) case 7u: SIK_DR12_CASE(7u) case 8u: SIK_DR12_CASE(8u)
        default: break;
    }
    #undef SIK_DR12_CASE
}

// Per-Layer Half-Width `down_w_left @ rms(cur) + h_msg` (`64` th/TG for `num_ctrls == 3`, `128` th/TG for `num_ctrls == 1`!)
template <uint M_FIX, uint NSG>
inline void sik_q8_left_add_impl(
    device const char*  w_q,
    device const half*  w_s,
    device const half*  x_in,
    device const float* h_msg,
    device float*       y_f32,
    threadgroup float   (&tg_red)[8][4],
    uint row, uint rows, uint D, uint tid, uint sgid, uint lane
) {
    const uint k16       = D >> 4u;
    const uint nblk_full = (2u * D) >> 6u;
    device const uint4* w16 = reinterpret_cast<device const uint4*>(w_q + ulong(row) * (2ul * D));
    device const half*  sc  = w_s + ulong(row) * nblk_full;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = tid; i < k16; i += (NSG * 32u)) {
        const float  s  = float(sc[i >> 2u]);
        const uint4  q  = w16[i];
        const float4 f0 = float4(as_type<char4>(q.x)), f1 = float4(as_type<char4>(q.y));
        const float4 f2 = float4(as_type<char4>(q.z)), f3 = float4(as_type<char4>(q.w));
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * D + i * 16u);
            const float d = (dot(f0, float4(xm[0])) + dot(f1, float4(xm[1])))
                          + (dot(f2, float4(xm[2])) + dot(f3, float4(xm[3])));
            acc[m] = fma(s, d, acc[m]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            const float tot = simd_sum((lane < NSG) ? tg_red[m][lane] : 0.0f);
            if (lane == 0u) {
                const ulong idx = ulong(m) * rows + row;
                y_f32[idx] = tot + h_msg[idx];
            }
        }
    }
}

kernel void sik_loop_ctrl_down3_left_q8(
    device const uchar*           uma      [[buffer(0)]],
    device const half*            cat_left [[buffer(1)]], // [3, M_valid, D]
    device const float*           h_msg    [[buffer(2)]], // [3, M_valid, R]
    device float*                 h_out    [[buffer(3)]], // [3, M_valid, R]
    constant SikLoopCtrl3Offsets& p        [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid.x, j = tgid.y;
    if (row >= p.gate_rank || j >= p.num_ctrls) return;
    threadgroup float tg_red[8][4];
    device const char*  w_q   = reinterpret_cast<device const char*>(uma + p.w_off[j]);
    device const half*  w_s   = reinterpret_cast<device const half*>(uma + p.s_off[j]);
    device const half*  x_in  = cat_left + ulong(j) * ulong(p.M_valid) * p.D;
    device const float* hm    = h_msg    + ulong(j) * ulong(p.M_valid) * p.gate_rank;
    device float*       y_f32 = h_out    + ulong(j) * ulong(p.M_valid) * p.gate_rank;
    #define SIK_DL3_CASE(MF) { sik_q8_left_add_impl<MF, 2u>(w_q, w_s, x_in, hm, y_f32, tg_red, row, p.gate_rank, p.D, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_DL3_CASE(1u) case 2u: SIK_DL3_CASE(2u) case 3u: SIK_DL3_CASE(3u) case 4u: SIK_DL3_CASE(4u)
        case 5u: SIK_DL3_CASE(5u) case 6u: SIK_DL3_CASE(6u) case 7u: SIK_DL3_CASE(7u) case 8u: SIK_DL3_CASE(8u)
        default: break;
    }
    #undef SIK_DL3_CASE
}

kernel void sik_loop_ctrl_down1_left_q8(
    device const uchar*           uma      [[buffer(0)]],
    device const half*            cat_left [[buffer(1)]], // [M_valid, D]
    device const float*           h_msg    [[buffer(2)]], // [M_valid, R]
    device float*                 h_out    [[buffer(3)]], // [M_valid, R]
    constant SikLoopCtrl3Offsets& p        [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row >= p.gate_rank) return;
    threadgroup float tg_red[8][4];
    device const char* w_q = reinterpret_cast<device const char*>(uma + p.w_off[0]);
    device const half* w_s = reinterpret_cast<device const half*>(uma + p.s_off[0]);
    #define SIK_DL1_CASE(MF) { sik_q8_left_add_impl<MF, 4u>(w_q, w_s, cat_left, h_msg, h_out, tg_red, row, p.gate_rank, p.D, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_DL1_CASE(1u) case 2u: SIK_DL1_CASE(2u) case 3u: SIK_DL1_CASE(3u) case 4u: SIK_DL1_CASE(4u)
        case 5u: SIK_DL1_CASE(5u) case 6u: SIK_DL1_CASE(6u) case 7u: SIK_DL1_CASE(7u) case 8u: SIK_DL1_CASE(8u)
        default: break;
    }
    #undef SIK_DL1_CASE
}

// Prefill small-N FP16 kernel: Grid `(rows, (M + 7)/8, 1)`, `128` threads/TG
kernel void sik_f16_small_n_prefill(
    device const half*           w_f16 [[buffer(0)]],
    device const half*           x_in  [[buffer(1)]],
    device half*                 y_f16 [[buffer(2)]],
    device float*                y_f32 [[buffer(3)]],
    constant SikGemvBatchParams& p     [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_red[8][4];
    const uint row = tgid.x;
    const uint m0  = tgid.y * 8u;
    if (row >= p.rows || m0 >= p.M) return;

    device const half* xb  = x_in  + ulong(m0) * p.cols;
    device half*       y16 = y_f16 + ulong(m0) * p.rows;
    device float*      y32 = y_f32 + ulong(m0) * p.rows;

    if (m0 + 8u <= p.M) {
        sik_f16_row_tg128_impl<8u>(w_f16, xb, y16, y32, tg_red, row, p.rows, p.cols, p.out_mode, p.softcap, tid, sgid, lane);
    } else {
        const uint mb = p.M - m0;
        for (uint m = 0u; m < mb; ++m) {
            sik_f16_row_tg128_impl<1u>(w_f16, xb + ulong(m) * p.cols,
                                       y16 + ulong(m) * p.rows, y32 + ulong(m) * p.rows,
                                       tg_red, row, p.rows, p.cols, p.out_mode, p.softcap, tid, sgid, lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

struct SikAttnQgrsMixedParams {
    ulong w_off[8]; // 0=wq, 1=gate, 2=wr, 3=sink, 4=wk, 5=wv, 6=ks_proj, 7=vs_proj
    ulong s_off[8];
    ulong b_off[8];
    uint  dtype[8]; // F16, Q8_0, or Q4_A
    uint  rows[8];
    uint  M;
    uint  cols;
    uint  pad0;
    uint  pad1;
};

// Single-Dispatch Fused ALL 6/8 Attention Projections across ANY mix of `FP16`, `Q8_0`, and `Q4_A`!
// 16 lanes/row (`2` rows/SG = `8` rows/TG) -> `1,083` TGs, half regs of l32_2r, ZERO extra serial barrier (493 GB/s @ M=1, 460 GB/s @ M=4, 263 GB/s @ M=8)!
template <uint M_FIX>
inline void sik_attn_qgrs_mixed_impl(
    device const uchar*             uma,
    device const half*              x_q,
    device const half*              x_k,
    device const half*              x_v,
    device const float*             xs_q,
    device const float*             xs_k,
    device const float*             xs_v,
    device float*                   q_out,
    device float*                   g_out,
    device float*                   r_out,
    device float*                   s_out,
    device float*                   k_out,
    device float*                   v_out,
    device float*                   ks_out,
    device float*                   vs_out,
    constant SikAttnQgrsMixedParams& p,
    uint vrow,
    uint l16
) {
    uint acc_r = 0u;
    uint proj  = 8u;
    uint rr    = 0u;
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
        const uint nr = p.rows[j];
        if (proj == 8u && vrow < acc_r + nr) {
            proj = j;
            rr   = vrow - acc_r;
        }
        acc_r += nr;
    }
    if (proj >= 8u) return;

    device const half*  xx = (proj < 4u) ? x_q  : ((proj == 4u || proj == 6u) ? x_k  : x_v);
    device const float* xs = (proj < 4u) ? xs_q : ((proj == 4u || proj == 6u) ? xs_k : xs_v);
    device float* oo = nullptr;
    switch (proj) {
        case 0u: oo = q_out;  break;
        case 1u: oo = g_out;  break;
        case 2u: oo = r_out;  break;
        case 3u: oo = s_out;  break;
        case 4u: oo = k_out;  break;
        case 5u: oo = v_out;  break;
        case 6u: oo = ks_out; break;
        default: oo = vs_out; break;
    }

    const uint dt = p.dtype[proj];
    const uint nr = p.rows[proj];
    if (dt == STRATA_INKLING2_DTYPE_Q4_A) {
        sik_q4a_gemv_l16_xsum_fixed<M_FIX>(
            reinterpret_cast<device const uint2*>(uma + p.w_off[proj]),
            reinterpret_cast<device const half*>(uma + p.s_off[proj]),
            reinterpret_cast<device const half*>(uma + p.b_off[proj]),
            xx, xs, nullptr, oo, rr, nr, p.cols, SIK_OUT_F32, 0.0f, l16);
    } else if (dt == STRATA_INKLING2_DTYPE_Q8_0) {
        sik_q8_gemv_l16_fixed<M_FIX>(
            reinterpret_cast<device const char*>(uma + p.w_off[proj]),
            reinterpret_cast<device const half*>(uma + p.s_off[proj]),
            xx, nullptr, oo, rr, nr, p.cols, SIK_OUT_F32, 0.0f, l16);
    } else {
        sik_f16_gemv_l16_fixed<M_FIX>(
            reinterpret_cast<device const half*>(uma + p.w_off[proj]),
            xx, nullptr, oo, rr, nr, p.cols, SIK_OUT_F32, 0.0f, l16);
    }
}

kernel void sik_gemv_attn_qgrs_mixed(
    device const uchar*              uma    [[buffer(0)]],
    device const half*               x_q    [[buffer(1)]],
    device const half*               x_k    [[buffer(2)]],
    device const half*               x_v    [[buffer(3)]],
    device float*                    q_out  [[buffer(4)]],
    device float*                    g_out  [[buffer(5)]],
    device float*                    r_out  [[buffer(6)]],
    device float*                    s_out  [[buffer(7)]],
    device float*                    k_out  [[buffer(8)]],
    device float*                    v_out  [[buffer(9)]],
    device float*                    ks_out [[buffer(10)]],
    device float*                    vs_out [[buffer(11)]],
    constant SikAttnQgrsMixedParams& p      [[buffer(12)]],
    device const float*              xs_q   [[buffer(13)]],
    device const float*              xs_k   [[buffer(14)]],
    device const float*              xs_v   [[buffer(15)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow = (tgid * 4u + sgid) * 2u + (lane >> 4u);
    const uint l16  = lane & 15u;
    #define SIK_QGRSM_CASE(MF) { sik_attn_qgrs_mixed_impl<MF>(uma, x_q, x_k, x_v, xs_q, xs_k, xs_v, q_out, g_out, r_out, s_out, k_out, v_out, ks_out, vs_out, p, vrow, l16); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_QGRSM_CASE(1u) case 2u: SIK_QGRSM_CASE(2u) case 3u: SIK_QGRSM_CASE(3u) case 4u: SIK_QGRSM_CASE(4u)
        case 5u: SIK_QGRSM_CASE(5u) case 6u: SIK_QGRSM_CASE(6u) case 7u: SIK_QGRSM_CASE(7u) case 8u: SIK_QGRSM_CASE(8u)
        default: break;
    }
    #undef SIK_QGRSM_CASE
}

// Pure-Q4_A 4-Projection Fused Attention Kernel (`wq + gate_attn + wk + wv`, 32-lane 1-row, ZERO Q8/FP16 reg bloat -> 532 GB/s!)
struct SikAttnQkvgQ4aParams {
    ulong w_off[4]; // 0=wq, 1=gate, 2=wk, 3=wv
    ulong s_off[4];
    ulong b_off[4];
    uint  rows[4];
    uint  cols;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

template <uint M_FIX>
inline void sik_attn_qkvg_q4a_impl(
    device const uchar*           uma,
    device const half*            x_q,
    device const half*            x_k,
    device const half*            x_v,
    device const float*           xs_q,
    device const float*           xs_k,
    device const float*           xs_v,
    device float*                 q_out,
    device float*                 g_out,
    device float*                 k_out,
    device float*                 v_out,
    constant SikAttnQkvgQ4aParams& p,
    uint vrow,
    uint lane
) {
    const uint b0 = p.rows[0], b1 = b0 + p.rows[1], b2 = b1 + p.rows[2], b3 = b2 + p.rows[3];
    if (vrow >= b3) return;
    const uint proj = (vrow < b0) ? 0u : ((vrow < b1) ? 1u : ((vrow < b2) ? 2u : 3u));
    const uint rr   = (proj == 0u) ? vrow : ((proj == 1u) ? (vrow - b0) : ((proj == 2u) ? (vrow - b1) : (vrow - b2)));
    device const half*  xx = (proj < 2u) ? x_q  : ((proj == 2u) ? x_k  : x_v);
    device const float* xs = (proj < 2u) ? xs_q : ((proj == 2u) ? xs_k : xs_v);
    device float*       oo = (proj == 0u) ? q_out : ((proj == 1u) ? g_out : ((proj == 2u) ? k_out : v_out));
    sik_q4a_gemv_l32_1r_xsum_fixed<M_FIX>(
        reinterpret_cast<device const uint2*>(uma + p.w_off[proj]),
        reinterpret_cast<device const half*>(uma + p.s_off[proj]),
        reinterpret_cast<device const half*>(uma + p.b_off[proj]),
        xx, xs, nullptr, oo, rr, p.rows[proj], p.cols, SIK_OUT_F32, lane);
}

kernel void sik_gemv_attn_qkvg_q4a(
    device const uchar*            uma   [[buffer(0)]],
    device const half*             x_q   [[buffer(1)]],
    device const half*             x_k   [[buffer(2)]],
    device const half*             x_v   [[buffer(3)]],
    device const float*            xs_q  [[buffer(4)]],
    device const float*            xs_k  [[buffer(5)]],
    device const float*            xs_v  [[buffer(6)]],
    device float*                  q_out [[buffer(7)]],
    device float*                  g_out [[buffer(8)]],
    device float*                  k_out [[buffer(9)]],
    device float*                  v_out [[buffer(10)]],
    constant SikAttnQkvgQ4aParams& p     [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow = tgid * 4u + sgid;
    #define SIK_QKVG_CASE(MF) { sik_attn_qkvg_q4a_impl<MF>(uma, x_q, x_k, x_v, xs_q, xs_k, xs_v, q_out, g_out, k_out, v_out, p, vrow, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_QKVG_CASE(1u) case 2u: SIK_QKVG_CASE(2u) case 3u: SIK_QKVG_CASE(3u) case 4u: SIK_QKVG_CASE(4u)
        case 5u: SIK_QKVG_CASE(5u) case 6u: SIK_QKVG_CASE(6u) case 7u: SIK_QKVG_CASE(7u) case 8u: SIK_QKVG_CASE(8u)
        default: break;
    }
    #undef SIK_QKVG_CASE
}

// Lean Q8_0/F16 Tail Projections (`wr + sink + optional ks_proj + vs_proj`, 32-lane 1-row, 4 rows/TG -> 102-118 TGs!)
struct SikAttnRsTailParams {
    ulong w_off[4]; // 0=wr, 1=sink, 2=ks_proj, 3=vs_proj
    ulong s_off[4];
    uint  dtype[4];
    uint  rows[4];
    uint  cols;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

template <uint M_FIX>
inline void sik_attn_rs_tail_impl(
    device const uchar*          uma,
    device const half*           x_q,
    device const half*           x_k,
    device const half*           x_v,
    device float*                r_out,
    device float*                s_out,
    device float*                ks_out,
    device float*                vs_out,
    constant SikAttnRsTailParams& p,
    uint vrow,
    uint lane
) {
    const uint b0 = p.rows[0], b1 = b0 + p.rows[1], b2 = b1 + p.rows[2], b3 = b2 + p.rows[3];
    if (vrow >= b3) return;
    const uint proj = (vrow < b0) ? 0u : ((vrow < b1) ? 1u : ((vrow < b2) ? 2u : 3u));
    const uint rr   = (proj == 0u) ? vrow : ((proj == 1u) ? (vrow - b0) : ((proj == 2u) ? (vrow - b1) : (vrow - b2)));
    const uint nr   = p.rows[proj];
    device const half* xx = (proj < 2u) ? x_q : ((proj == 2u) ? x_k : x_v);
    device float*      oo = (proj == 0u) ? r_out : ((proj == 1u) ? s_out : ((proj == 2u) ? ks_out : vs_out));

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    if (p.dtype[proj] == STRATA_INKLING2_DTYPE_Q8_0) {
        const uint k16  = p.cols >> 4u;
        const uint nblk = p.cols >> 6u;
        device const uint4* rw = reinterpret_cast<device const uint4*>(uma + p.w_off[proj] + ulong(rr) * p.cols);
        device const half*  rs = reinterpret_cast<device const half*>(uma + p.s_off[proj]) + ulong(rr) * nblk;
        for (uint i = lane; i < k16; i += 32u) {
            const float  s  = float(rs[i >> 2u]);
            const uint4  q  = rw[i];
            const float4 f0 = float4(as_type<char4>(q.x)), f1 = float4(as_type<char4>(q.y));
            const float4 f2 = float4(as_type<char4>(q.z)), f3 = float4(as_type<char4>(q.w));
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                device const half4* xm = reinterpret_cast<device const half4*>(xx + ulong(m) * p.cols + i * 16u);
                const float d = (dot(f0, float4(xm[0])) + dot(f1, float4(xm[1])))
                              + (dot(f2, float4(xm[2])) + dot(f3, float4(xm[3])));
                acc[m] = fma(s, d, acc[m]);
            }
        }
    } else {
        const uint c8 = p.cols >> 3u;
        device const uint4* rw = reinterpret_cast<device const uint4*>(uma + p.w_off[proj] + ulong(rr) * p.cols * 2ul);
        for (uint i = lane; i < c8; i += 32u) {
            const uint4 v  = rw[i];
            const half4 a0 = as_type<half4>(v.xy), a1 = as_type<half4>(v.zw);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                device const half4* xm = reinterpret_cast<device const half4*>(xx + ulong(m) * p.cols + i * 8u);
                acc[m] += dot(float4(a0), float4(xm[0])) + dot(float4(a1), float4(xm[1]));
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0u) oo[ulong(m) * nr + rr] = v;
    }
}

kernel void sik_gemv_attn_rs_tail(
    device const uchar*           uma    [[buffer(0)]],
    device const half*            x_q    [[buffer(1)]],
    device const half*            x_k    [[buffer(2)]],
    device const half*            x_v    [[buffer(3)]],
    device float*                 r_out  [[buffer(4)]],
    device float*                 s_out  [[buffer(5)]],
    device float*                 ks_out [[buffer(6)]],
    device float*                 vs_out [[buffer(7)]],
    constant SikAttnRsTailParams& p      [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow = tgid * 4u + sgid;
    #define SIK_RST_CASE(MF) { sik_attn_rs_tail_impl<MF>(uma, x_q, x_k, x_v, r_out, s_out, ks_out, vs_out, p, vrow, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_RST_CASE(1u) case 2u: SIK_RST_CASE(2u) case 3u: SIK_RST_CASE(3u) case 4u: SIK_RST_CASE(4u)
        case 5u: SIK_RST_CASE(5u) case 6u: SIK_RST_CASE(6u) case 7u: SIK_RST_CASE(7u) case 8u: SIK_RST_CASE(8u)
        default: break;
    }
    #undef SIK_RST_CASE
}

struct SikMtpQkvMixedParams {
    ulong w_off[3]; // 0=q, 1=k, 2=v
    ulong s_off[3];
    ulong b_off[3];
    uint  dtype[3];
    uint  rows[3];
    uint  cols;
    uint  pad0;
    uint  pad1;
};

// Pure-Q4_A MTP Block-8 `q_proj + k_proj + v_proj` (32-lane 1-row, ZERO Q8/FP16 reg bloat!)
kernel void sik_gemv_mtp_qkv_q4a(
    device const uchar*            uma   [[buffer(0)]],
    device const half*             x_in  [[buffer(1)]],
    device float*                  q_out [[buffer(2)]],
    device float*                  k_out [[buffer(3)]],
    device half*                   v_out [[buffer(4)]],
    constant SikMtpQkvMixedParams& p     [[buffer(5)]],
    device const float*            x_sum [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow = tgid * 4u + sgid;
    const uint b1 = p.rows[0], b2 = b1 + p.rows[1], b3 = b2 + p.rows[2];
    if (vrow >= b3) return;
    const uint proj  = (vrow < b1) ? 0u : ((vrow < b2) ? 1u : 2u);
    const uint rr    = (proj == 0u) ? vrow : ((proj == 1u) ? (vrow - b1) : (vrow - b2));
    device half*  y16 = (proj == 2u) ? v_out : nullptr;
    device float* y32 = (proj == 0u) ? q_out : ((proj == 1u) ? k_out : nullptr);
    const uint omode  = (proj == 2u) ? SIK_OUT_F16 : SIK_OUT_F32;
    sik_q4a_gemv_l32_1r_xsum_fixed<8u>(
        reinterpret_cast<device const uint2*>(uma + p.w_off[proj]),
        reinterpret_cast<device const half*>(uma + p.s_off[proj]),
        reinterpret_cast<device const half*>(uma + p.b_off[proj]),
        x_in, x_sum, y16, y32, rr, p.rows[proj], p.cols, omode, lane);
}

// Single-Dispatch Fused MTP Block-8 `q_proj + k_proj + v_proj` across `FP16`/`Q8_0`/`Q4_A` (32-lane 2-row w/ `x_sum`!)
kernel void sik_gemv_mtp_qkv_mixed(
    device const uchar*            uma   [[buffer(0)]],
    device const half*             x_in  [[buffer(1)]],
    device float*                  q_out [[buffer(2)]],
    device float*                  k_out [[buffer(3)]],
    device half*                   v_out [[buffer(4)]],
    constant SikMtpQkvMixedParams& p     [[buffer(5)]],
    device const float*            x_sum [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow0 = (tgid * 4u + sgid) * 2u;
    const uint b1 = p.rows[0], b2 = b1 + p.rows[1], b3 = b2 + p.rows[2];
    if (vrow0 >= b3) return;

    const uint proj = (vrow0 < b1) ? 0u : ((vrow0 < b2) ? 1u : 2u);
    const uint rr   = (proj == 0u) ? vrow0 : ((proj == 1u) ? (vrow0 - b1) : (vrow0 - b2));
    const uint nr   = p.rows[proj];
    const uint dt   = p.dtype[proj];
    device half*  y16 = (proj == 2u) ? v_out : nullptr;
    device float* y32 = (proj == 0u) ? q_out : ((proj == 1u) ? k_out : nullptr);
    const uint omode  = (proj == 2u) ? SIK_OUT_F16 : SIK_OUT_F32;

    if (dt == STRATA_INKLING2_DTYPE_Q4_A) {
        sik_q4a_gemv_l32_2r_xsum_fixed<8u>(
            reinterpret_cast<device const uint2*>(uma + p.w_off[proj]),
            reinterpret_cast<device const half*>(uma + p.s_off[proj]),
            reinterpret_cast<device const half*>(uma + p.b_off[proj]),
            x_in, x_sum, y16, y32, rr, nr, p.cols, omode, 0.0f, lane);
    } else if (dt == STRATA_INKLING2_DTYPE_Q8_0) {
        sik_q8_gemv_l32_2r_fixed<8u>(
            reinterpret_cast<device const char*>(uma + p.w_off[proj]),
            reinterpret_cast<device const half*>(uma + p.s_off[proj]),
            x_in, y16, y32, rr, nr, p.cols, omode, 0.0f, lane);
    } else {
        sik_f16_gemv_l32_2r_fixed<8u>(
            reinterpret_cast<device const half*>(uma + p.w_off[proj]),
            x_in, y16, y32, rr, nr, p.cols, omode, 0.0f, lane);
    }
}

// Single-Dispatch Fused MTP Context Update `k_proj_ctx + v_proj_ctx` (`M = 1..8`, 32-lane 2-row w/ `x_sum`!)
template <uint M_FIX>
inline void sik_mtp_kv_ctx_mixed_impl(
    device const uchar*            uma,
    device const half*             x_in,
    device const float*            x_sum,
    device float*                  k_out,
    device float*                  v_out,
    constant SikMtpQkvMixedParams& p,
    uint vrow0,
    uint lane
) {
    const uint b1 = p.rows[0], b2 = b1 + p.rows[1];
    if (vrow0 >= b2) return;
    const uint proj = (vrow0 < b1) ? 0u : 1u;
    const uint rr   = (proj == 0u) ? vrow0 : (vrow0 - b1);
    const uint nr   = p.rows[proj];
    const uint dt   = p.dtype[proj];
    device float* y32 = (proj == 0u) ? k_out : v_out;

    if (dt == STRATA_INKLING2_DTYPE_Q4_A) {
        sik_q4a_gemv_l32_2r_xsum_fixed<M_FIX>(
            reinterpret_cast<device const uint2*>(uma + p.w_off[proj]),
            reinterpret_cast<device const half*>(uma + p.s_off[proj]),
            reinterpret_cast<device const half*>(uma + p.b_off[proj]),
            x_in, x_sum, nullptr, y32, rr, nr, p.cols, SIK_OUT_F32, 0.0f, lane);
    } else if (dt == STRATA_INKLING2_DTYPE_Q8_0) {
        sik_q8_gemv_l32_2r_fixed<M_FIX>(
            reinterpret_cast<device const char*>(uma + p.w_off[proj]),
            reinterpret_cast<device const half*>(uma + p.s_off[proj]),
            x_in, nullptr, y32, rr, nr, p.cols, SIK_OUT_F32, 0.0f, lane);
    } else {
        sik_f16_gemv_l32_2r_fixed<M_FIX>(
            reinterpret_cast<device const half*>(uma + p.w_off[proj]),
            x_in, nullptr, y32, rr, nr, p.cols, SIK_OUT_F32, 0.0f, lane);
    }
}

kernel void sik_gemv_mtp_kv_ctx_mixed(
    device const uchar*            uma   [[buffer(0)]],
    device const half*             x_in  [[buffer(1)]],
    device float*                  k_out [[buffer(2)]],
    device float*                  v_out [[buffer(3)]],
    constant SikMtpQkvMixedParams& p     [[buffer(4)]],
    device const float*            x_sum [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint vrow0 = (tgid * 4u + sgid) * 2u;
    #define SIK_MKVC_CASE(MF) { sik_mtp_kv_ctx_mixed_impl<MF>(uma, x_in, x_sum, k_out, v_out, p, vrow0, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_MKVC_CASE(1u) case 2u: SIK_MKVC_CASE(2u) case 3u: SIK_MKVC_CASE(3u) case 4u: SIK_MKVC_CASE(4u)
        case 5u: SIK_MKVC_CASE(5u) case 6u: SIK_MKVC_CASE(6u) case 7u: SIK_MKVC_CASE(7u) case 8u: SIK_MKVC_CASE(8u)
        default: break;
    }
    #undef SIK_MKVC_CASE
}

struct SikBlkSumParams {
    uint M;
    uint cols;
    uint pad0;
    uint pad1;
};

// Zero-Barrier 64-Element Block-Sum Precomputation for `Q4_A` `xsum` GEMVs (`Grid(M, 1, 1)`, `256` th/TG)
kernel void sik_f16_blk64_sum(
    device const half*        x_in     [[buffer(0)]],
    device float*             xsum_out [[buffer(1)]],
    constant SikBlkSumParams& p        [[buffer(2)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M) return;
    const uint c4_cnt = p.cols >> 2u;
    const uint nblk   = p.cols >> 6u;
    device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols);
    for (uint i = tid; i < c4_cnt; i += 256u) {
        const half4 h  = x4[i];
        const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
        if ((lane & 15u) == 0u) xsum_out[ulong(m) * nblk + (i >> 4u)] = bs;
    }
}

struct SikNaxGemmParams {
    uint  M_valid;
    uint  M_padded;
    uint  N_logical;
    uint  N_padded;
    uint  K;
    uint  out_mode;
    float softcap;
    uint  pad0;
};

// ---- Direct-Device `static_slice` `K=dynamic_extent` FP16 NAX (`58.6 TF`! ZERO TG staging, ZERO barriers!) ----
#if defined(__HAVE_TENSOR__)
template <int BM, int BN, int SG>
inline void sik_f16_gemm_nax_direct_impl(
    device const half*         w_f16,
    device const half*         x_in,
    device half*               y_f16,
    device float*              y_f32,
    constant SikNaxGemmParams& p,
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

    // Zero-overhead hardware direct store when F32 and N is BN-aligned (`58.6 TF`!)
    if (p.out_mode == SIK_OUT_F32 && (p.N_logical % uint(BN)) == 0u) {
        tensor<device float, dextents<int, 2>, tensor_inline> tY(y_f32, dextents<int, 2>(int(p.N_logical), int(p.M_padded)));
        auto sY = tY.template slice<BN, BM>(int(n0), int(m0));
        op.run(sX, sW, sY);
        return;
    }

    auto cT = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sX)>, sik_strip_as<decltype(sW)>, float>();
    op.run(sX, sW, cT);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx    = cT.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_valid && n_glob < p.N_logical) {
                const ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                float val = cT[i];
                if (p.out_mode == SIK_OUT_F32_SOFTCAP && p.softcap > 0.0f) val = sik_softcap(val, p.softcap);
                if (p.out_mode == SIK_OUT_F16)          y_f16[out_idx]  = sik_clamp_f16(val);
                else if (p.out_mode == SIK_OUT_RES_F32) y_f32[out_idx] += val;
                else                                    y_f32[out_idx]  = val;
            }
        }
    }
}

// ---- Q4_A (gs=64) Prefill NAX Upgraded to BK=128 (`W_tile[64][128] = 16 KiB`, `uint2` loads -> escapes 2.17x K=64 cliff!) ----
template <int BM, int SG>
inline void sik_q4a_gemm_nax_impl(
    device const uint2*        w_u2,
    device const half*         w_s,
    device const half*         w_b,
    device const half*         x_in,
    device half*               y_f16,
    device float*              y_f32,
    constant SikNaxGemmParams& p,
    threadgroup half           (&W_tile)[64][128],
    uint2                      tgid,
    uint                       tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(const_cast<device half*>(x_in), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<128, BM>(0, int(m0));
    auto cT  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sX0)>, sik_strip_as<decltype(tW)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    const uint u2_r     = p.K >> 4u; // K / 16
    const uint grp_r    = p.K >> 6u; // K / 64
    const uint kt_cnt   = p.K >> 7u; // K / 128
    constexpr uint r_step  = (uint(SG) * 32u) >> 3u; // 8 (SG=2) or 16 (SG=4)
    constexpr uint q_iters = 64u / r_step;
    const uint r_grp    = tid >> 3u;
    const uint sub      = tid & 7u;  // 0..7 (16 nibbles = uint2 each)
    const uint sub_g    = sub >> 2u; // 0 or 1

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        const uint grp = kt * 2u + sub_g;
        const uint u2c = kt * 8u + sub;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint r_loc  = q * r_step + r_grp;
            const uint r_glob = n0 + r_loc;
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            if (r_glob < p.N_logical) {
                const uint2 v  = w_u2[ulong(r_glob) * u2_r + u2c];
                const half4 sc = half4(w_s[ulong(r_glob) * grp_r + grp]);
                const half4 bi = half4(w_b[ulong(r_glob) * grp_r + grp]);
                half4 n0h, n1h, n2h, n3h;
                sik_unpack_u4_u32_h4(v.x, n0h, n1h);
                sik_unpack_u4_u32_h4(v.y, n2h, n3h);
                dst[0] = fma(n0h, sc, bi);
                dst[1] = fma(n1h, sc, bi);
                dst[2] = fma(n2h, sc, bi);
                dst[3] = fma(n3h, sc, bi);
            } else {
                dst[0] = half4(0.0h); dst[1] = half4(0.0h);
                dst[2] = half4(0.0h); dst[3] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX.template slice<128, BM>(int(kt * 128u), int(m0));
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.out_mode == SIK_OUT_F32 && (p.N_logical % 64u) == 0u) {
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
                const ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                float val = cT[i];
                if (p.out_mode == SIK_OUT_F32_SOFTCAP && p.softcap > 0.0f) val = sik_softcap(val, p.softcap);
                if (p.out_mode == SIK_OUT_F16)          y_f16[out_idx]  = sik_clamp_f16(val);
                else if (p.out_mode == SIK_OUT_RES_F32) y_f32[out_idx] += val;
                else                                    y_f32[out_idx]  = val;
            }
        }
    }
}

// ---- Q8_0 (gs=64) Prefill NAX Upgraded to BK=128 (`W_tile[BN][128]`, `uint4` 16B loads!) ----
template <int BM, int BN, int SG>
inline void sik_q8_gemm_nax_impl(
    device const uint4*        w_u4,
    device const half*         w_s,
    device const half*         x_in,
    device half*               y_f16,
    device float*              y_f32,
    constant SikNaxGemmParams& p,
    threadgroup half           (&W_tile)[BN][128],
    uint2                      tgid,
    uint                       tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * uint(BN);
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    tensor<device half, dextents<int, 2>, tensor_inline> tX(const_cast<device half*>(x_in), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup half, extents<int, 128, BN>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, BN>());

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<128, BM>(0, int(m0));
    auto cT  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sX0)>, sik_strip_as<decltype(tW)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    const uint u4_r     = p.K >> 4u; // K / 16
    const uint grp_r    = p.K >> 6u; // K / 64
    const uint kt_cnt   = p.K >> 7u; // K / 128
    constexpr uint r_step  = (uint(SG) * 32u) >> 3u; // 8 (SG=2) or 16 (SG=4)
    constexpr uint q_iters = uint(BN) / r_step;
    const uint r_grp    = tid >> 3u;
    const uint sub      = tid & 7u;  // 0..7 (16 int8s = uint4)
    const uint sub_g    = sub >> 2u; // 0 or 1

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        const uint grp = kt * 2u + sub_g;
        const uint u4c = kt * 8u + sub;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint r_loc  = q * r_step + r_grp;
            const uint r_glob = n0 + r_loc;
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][sub * 16u]);
            if (r_glob < p.N_logical) {
                const half sc   = w_s[ulong(r_glob) * grp_r + grp];
                const uint4 q16 = w_u4[ulong(r_glob) * u4_r + u4c];
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
        auto sX = tX.template slice<128, BM>(int(kt * 128u), int(m0));
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.out_mode == SIK_OUT_F32 && (p.N_logical % uint(BN)) == 0u) {
        tensor<device float, dextents<int, 2>, tensor_inline> tY(y_f32, dextents<int, 2>(int(p.N_logical), int(p.M_padded)));
        auto sY = tY.template slice<BN, BM>(int(n0), int(m0));
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
                const ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                float val = cT[i];
                if (p.out_mode == SIK_OUT_F32_SOFTCAP && p.softcap > 0.0f) val = sik_softcap(val, p.softcap);
                if (p.out_mode == SIK_OUT_F16)          y_f16[out_idx]  = sik_clamp_f16(val);
                else if (p.out_mode == SIK_OUT_RES_F32) y_f32[out_idx] += val;
                else                                    y_f32[out_idx]  = val;
            }
        }
    }
}
#endif

kernel void sik_f16_gemm_nax_64x128_sg4(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* y16 [[buffer(2)]], device float* y32 [[buffer(3)]], constant SikNaxGemmParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sik_f16_gemm_nax_direct_impl<64, 128, 4>(w, x, y16, y32, p, tgid);
#else
    (void)w;(void)x;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sik_f16_gemm_nax_64x64_sg4(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* y16 [[buffer(2)]], device float* y32 [[buffer(3)]], constant SikNaxGemmParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sik_f16_gemm_nax_direct_impl<64, 64, 4>(w, x, y16, y32, p, tgid);
#else
    (void)w;(void)x;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sik_f16_gemm_nax_32x64_sg2(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* y16 [[buffer(2)]], device float* y32 [[buffer(3)]], constant SikNaxGemmParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sik_f16_gemm_nax_direct_impl<32, 64, 2>(w, x, y16, y32, p, tgid);
#else
    (void)w;(void)x;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sik_f16_gemm_nax_16x32_sg2(device const half* w [[buffer(0)]], device const half* x [[buffer(1)]], device half* y16 [[buffer(2)]], device float* y32 [[buffer(3)]], constant SikNaxGemmParams& p [[buffer(4)]], uint2 tgid [[threadgroup_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    sik_f16_gemm_nax_direct_impl<16, 32, 2>(w, x, y16, y32, p, tgid);
#else
    (void)w;(void)x;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sik_q4a_gemm_nax_m128(device const uint2* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]], device const half* x [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SikNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sik_q4a_gemm_nax_impl<128, 4>(wq, ws, wb, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)wb;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sik_q4a_gemm_nax_m64(device const uint2* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]], device const half* x [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SikNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sik_q4a_gemm_nax_impl<64, 4>(wq, ws, wb, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)wb;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sik_q4a_gemm_nax_m32(device const uint2* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]], device const half* x [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SikNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sik_q4a_gemm_nax_impl<32, 2>(wq, ws, wb, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)wb;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sik_q4a_gemm_nax_m16(device const uint2* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]], device const half* x [[buffer(3)]], device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]], constant SikNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sik_q4a_gemm_nax_impl<16, 2>(wq, ws, wb, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)wb;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sik_q8_gemm_nax_m128(device const uint4* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* x [[buffer(2)]], device half* y16 [[buffer(3)]], device float* y32 [[buffer(4)]], constant SikNaxGemmParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sik_q8_gemm_nax_impl<128, 64, 4>(wq, ws, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sik_q8_gemm_nax_m64(device const uint4* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* x [[buffer(2)]], device half* y16 [[buffer(3)]], device float* y32 [[buffer(4)]], constant SikNaxGemmParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[64][128];
#if defined(__HAVE_TENSOR__)
    sik_q8_gemm_nax_impl<64, 64, 4>(wq, ws, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sik_q8_gemm_nax_m32_n32(device const uint4* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* x [[buffer(2)]], device half* y16 [[buffer(3)]], device float* y32 [[buffer(4)]], constant SikNaxGemmParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[32][128];
#if defined(__HAVE_TENSOR__)
    sik_q8_gemm_nax_impl<32, 32, 2>(wq, ws, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sik_q8_gemm_nax_m16_n32(device const uint4* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* x [[buffer(2)]], device half* y16 [[buffer(3)]], device float* y32 [[buffer(4)]], constant SikNaxGemmParams& p [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half W_tile[32][128];
#if defined(__HAVE_TENSOR__)
    sik_q8_gemm_nax_impl<16, 32, 2>(wq, ws, x, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)x;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

/* ============================================================================
 * 4b. Opt-In `--int8-prefill` Metal 4.1 INT8xINT8->INT32 (`86.8 TOPS`) &
 *     INT8xINT4->INT32 (`63+ TOPS`) Prefill NAX GEMMs (M >= 32 ONLY!)
 * ============================================================================ */

struct SikQuantI8Params {
    uint M_valid;
    uint M_padded;
    uint K;
    uint emit_xsum;
    uint num_slices;
    uint pad0;
    uint pad1;
    uint pad2;
};

// Zero-Barrier 1-Simdgroup-Per-Row FP16 -> INT8 Row Quantizer (`> 650 GB/s`, 0B TG, 0 barriers!)
inline void sik_quant_row_sg_impl(
    device const half4* x_in,
    device int8_t*      x_i8,
    device half*        x_sc,
    device half*        x_sum,
    uint                row,
    uint                M_valid,
    uint                K,
    uint                emit_xsum,
    uint                lane
) {
    const uint k4   = K >> 2u;
    const uint ngrp = K >> 6u;
    device char4* di8 = reinterpret_cast<device char4*>(x_i8 + ulong(row) * K);

    if (row >= M_valid) {
        for (uint i = lane; i < k4; i += 32u) di8[i] = char4(0);
        if (lane == 0u) x_sc[row] = 0.0h;
        if (emit_xsum != 0u) {
            for (uint g = lane; g < ngrp; g += 32u) x_sum[ulong(row) * ngrp + g] = 0.0h;
        }
        return;
    }

    device const half4* xr = x_in + ulong(row) * k4;
    half4 am4 = half4(0.0h);
    if (emit_xsum == 0u) {
        for (uint i = lane; i < k4; i += 32u) {
            am4 = max(am4, abs(xr[i]));
        }
    } else {
        const uint half_lane = lane & 15u;
        const uint half_id   = lane >> 4u;
        for (uint g = 0u; g < ngrp; g += 2u) {
            const uint gi = g + half_id;
            float s = 0.0f;
            if (gi < ngrp) {
                const half4 hv = xr[gi * 16u + half_lane];
                am4 = max(am4, abs(hv));
                const float4 fv = float4(hv);
                s = (fv.x + fv.y) + (fv.z + fv.w);
            }
            s += simd_shuffle_xor(s, 1u);
            s += simd_shuffle_xor(s, 2u);
            s += simd_shuffle_xor(s, 4u);
            s += simd_shuffle_xor(s, 8u);
            if (half_lane == 0u && gi < ngrp) x_sum[ulong(row) * ngrp + gi] = sik_clamp_f16(s);
        }
    }

    const float amax = simd_max(float(max(max(am4.x, am4.y), max(am4.z, am4.w))));
    const float sc   = (amax > 1e-8f) ? (amax * (1.0f / 127.0f)) : 1.0f;
    const float inv  = (amax > 1e-8f) ? (127.0f / amax) : 0.0f;
    if (lane == 0u) x_sc[row] = sik_clamp_f16(sc);

    for (uint i = lane; i < k4; i += 32u) {
        di8[i] = char4(int4(clamp(rint(float4(xr[i]) * inv), float4(-127.0f), float4(127.0f))));
    }
}

// Grid: `((M_padded + 3) / 4, num_slices, 1)`, `128` threads/TG (4 rows/TG, 0 barriers!)
kernel void sik_quant_f16_to_i8_row(
    device const half4*      x_in  [[buffer(0)]],
    device int8_t*           x_i8  [[buffer(1)]],
    device half*             x_sc  [[buffer(2)]],
    device half*             x_sum [[buffer(3)]],
    constant SikQuantI8Params& p   [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid.x * 4u + sgid;
    const uint sl  = tgid.y;
    if (row >= p.M_padded || sl >= p.num_slices) return;
    const uint k4   = p.K >> 2u;
    const uint ngrp = p.K >> 6u;
    sik_quant_row_sg_impl(
        x_in  + ulong(sl) * ulong(p.M_padded) * k4,
        x_i8  + ulong(sl) * ulong(p.M_padded) * p.K,
        x_sc  + ulong(sl) * ulong(p.M_padded),
        x_sum + ulong(sl) * ulong(p.M_padded) * ngrp,
        row, p.M_valid, p.K, p.emit_xsum, lane);
}

// 3-Way Batched Quantize (`in_q, in_k, in_v` -> 1 dispatch!). Grid: `((M_padded + 3) / 4, 3, 1)`, `128` threads/TG
kernel void sik_quant_f16_to_i8_row3(
    device const half4*      x0   [[buffer(0)]],
    device const half4*      x1   [[buffer(1)]],
    device const half4*      x2   [[buffer(2)]],
    device int8_t*           o0   [[buffer(3)]],
    device int8_t*           o1   [[buffer(4)]],
    device int8_t*           o2   [[buffer(5)]],
    device half*             sc0  [[buffer(6)]],
    device half*             sc1  [[buffer(7)]],
    device half*             sc2  [[buffer(8)]],
    device half*             xs0  [[buffer(9)]],
    device half*             xs1  [[buffer(10)]],
    device half*             xs2  [[buffer(11)]],
    constant SikQuantI8Params& p  [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid.x * 4u + sgid;
    const uint sl  = tgid.y;
    if (row >= p.M_padded || sl >= 3u) return;
    device const half4* xi = (sl == 0u) ? x0  : ((sl == 1u) ? x1  : x2);
    device int8_t*      oi = (sl == 0u) ? o0  : ((sl == 1u) ? o1  : o2);
    device half*        si = (sl == 0u) ? sc0 : ((sl == 1u) ? sc1 : sc2);
    device half*        mi = (sl == 0u) ? xs0 : ((sl == 1u) ? xs1 : xs2);
    sik_quant_row_sg_impl(xi, oi, si, mi, row, p.M_valid, p.K, p.emit_xsum, lane);
}

#if defined(__HAVE_TENSOR__)
// Direct-Device Q8_0 (gs=64) INT8xINT8->INT32 NAX GEMM (`86.8 TOPS`!, 0 B TG mem, 0 barriers!)
template <int BM, int BN, int SG, int CAP>
inline void sik_q8_gemm_nax_i8_impl(
    device const int8_t*       w_i8,
    device const half*         w_s,
    device const int8_t*       x_i8,
    device const half*         x_sc,
    device half*               y_f16,
    device float*              y_f32,
    constant SikNaxGemmParams& p,
    uint2                      tgid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * uint(BN);
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tX(const_cast<device int8_t*>(x_i8), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<device int8_t, dextents<int, 2>, tensor_inline> tW(const_cast<device int8_t*>(w_i8), dextents<int, 2>(int(p.K), int(p.N_logical)));

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<64, BM>(0, int(m0));
    auto sW0 = tW.template slice<64, BN>(0, int(n0));
    auto cI  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sX0)>, sik_strip_as<decltype(sW0)>, int32_t>();

    float acc[CAP];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) acc[i] = 0.0f;

    const uint ngrp = p.K >> 6u;
    for (uint g = 0u; g < ngrp; ++g) {
        auto sX = tX.template slice<64, BM>(int(g * 64u), int(m0));
        auto sW = tW.template slice<64, BN>(int(g * 64u), int(n0));
        op.run(sX, sW, cI);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < ushort(CAP); ++i) {
            if (cI.is_valid_element(i)) {
                auto idx = cI.get_multidimensional_index(i);
                const uint ng = min(n0 + uint(idx[0]), p.N_logical - 1u);
                acc[i] = fma(float(cI[i]), float(w_s[ulong(ng) * ngrp + g]), acc[i]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx    = cI.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_valid && n_glob < p.N_logical) {
                const ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                float val = acc[i] * float(x_sc[m_glob]);
                if (p.out_mode == SIK_OUT_F32_SOFTCAP && p.softcap > 0.0f) val = sik_softcap(val, p.softcap);
                if (p.out_mode == SIK_OUT_F16)          y_f16[out_idx]  = sik_clamp_f16(val);
                else if (p.out_mode == SIK_OUT_RES_F32) y_f32[out_idx] += val;
                else                                    y_f32[out_idx]  = val;
            }
        }
    }
}

// Double-Buffered Q4_A (gs=64) INT8xINT4->INT32 NAX GEMM + NAX FP16 Bias Epilogue (`cos = 0.99991`!)
template <int BM, int SG, int CAP>
inline void sik_q4a_gemm_nax_i8i4_impl(
    device const uint2*        w_u2,
    device const half*         w_s,
    device const half*         w_b,
    device const int8_t*       x_i8,
    device const half*         x_sc,
    device const half*         x_sum,
    device half*               y_f16,
    device float*              y_f32,
    constant SikNaxGemmParams& p,
    threadgroup uint2          (&W_tile)[2][64][8],
    uint2                      tgid,
    uint                       tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    const uint u2_r   = p.K >> 4u;
    const uint ngrp   = p.K >> 6u;
    const uint kt_cnt = p.K >> 7u;

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tX(const_cast<device int8_t*>(x_i8), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup metal::int4b_format, dextents<int, 2>, tensor_inline> tW0(
        reinterpret_cast<threadgroup uint8_t*>(&W_tile[0][0][0]), dextents<int, 2>(128, 64));
    tensor<threadgroup metal::int4b_format, dextents<int, 2>, tensor_inline> tW1(
        reinterpret_cast<threadgroup uint8_t*>(&W_tile[1][0][0]), dextents<int, 2>(128, 64));

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<64, BM>(0, int(m0));
    auto sW0 = tW0.template slice<64, 64>(0, 0);
    auto cI  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sX0)>, sik_strip_as<decltype(sW0)>, int32_t>();

    float acc[CAP];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) acc[i] = 0.0f;

    constexpr uint r_step  = (uint(SG) * 32u) >> 3u; // 8 (SG=2) or 16 (SG=4)
    constexpr uint q_iters = 64u / r_step;
    const uint r_grp = tid >> 3u;
    const uint sub8  = tid & 7u;

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        const uint buf = kt & 1u;
        const uint g0  = kt * 2u;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint r_loc  = q * r_step + r_grp;
            const uint r_glob = n0 + r_loc;
            if (r_glob < p.N_logical) {
                W_tile[buf][r_loc][sub8] = w_u2[ulong(r_glob) * u2_r + kt * 8u + sub8] ^ uint2(0x88888888u);
            } else {
                W_tile[buf][r_loc][sub8] = uint2(0u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 2u; ++h) {
            const uint g = g0 + h;
            auto sX = tX.template slice<64, BM>(int(g * 64u), int(m0));
            if (buf == 0u) {
                auto sW = tW0.template slice<64, 64>(int(h * 64u), 0);
                op.run(sX, sW, cI);
            } else {
                auto sW = tW1.template slice<64, 64>(int(h * 64u), 0);
                op.run(sX, sW, cI);
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < ushort(CAP); ++i) {
                if (cI.is_valid_element(i)) {
                    auto idx = cI.get_multidimensional_index(i);
                    const uint ng = min(n0 + uint(idx[0]), p.N_logical - 1u);
                    acc[i] = fma(float(cI[i]), float(w_s[ulong(ng) * ngrp + g]), acc[i]);
                }
            }
        }
    }

    // Ultra-fast NAX FP16 epilogue for `sum_g x_sum[m,g] * (w_b[n,g] + 8 * w_s[n,g])` (~0.014 ms!)
    tensor<device half, dextents<int, 2>, tensor_inline> tXsum(const_cast<device half*>(x_sum), dextents<int, 2>(int(ngrp), int(p.M_padded)));
    tensor<device half, dextents<int, 2>, tensor_inline> tWb(const_cast<device half*>(w_b),     dextents<int, 2>(int(ngrp), int(p.N_logical)));
    tensor<device half, dextents<int, 2>, tensor_inline> tWs(const_cast<device half*>(w_s),     dextents<int, 2>(int(ngrp), int(p.N_logical)));
    constexpr auto desc_b = matmul2d_descriptor(
        BM, 64, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_b, execution_simdgroups<SG>> op_b;
    auto sXsum = tXsum.template slice<dynamic_extent, BM>(0, int(m0));
    auto sWb   = tWb.template slice<dynamic_extent, 64>(0, int(n0));
    auto sWs   = tWs.template slice<dynamic_extent, 64>(0, int(n0));
    auto cB    = op_b.template get_destination_cooperative_tensor<sik_strip_as<decltype(sXsum)>, sik_strip_as<decltype(sWb)>, float>();
    auto cS    = op_b.template get_destination_cooperative_tensor<sik_strip_as<decltype(sXsum)>, sik_strip_as<decltype(sWs)>, float>();
    op_b.run(sXsum, sWb, cB);
    op_b.run(sXsum, sWs, cS);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx    = cI.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_valid && n_glob < p.N_logical) {
                const ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                float val = fma(acc[i], float(x_sc[m_glob]), fma(8.0f, cS[i], cB[i]));
                if (p.out_mode == SIK_OUT_F32_SOFTCAP && p.softcap > 0.0f) val = sik_softcap(val, p.softcap);
                if (p.out_mode == SIK_OUT_F16)          y_f16[out_idx]  = sik_clamp_f16(val);
                else if (p.out_mode == SIK_OUT_RES_F32) y_f32[out_idx] += val;
                else                                    y_f32[out_idx]  = val;
            }
        }
    }
}
#endif

kernel void sik_q8_gemm_nax_i8_m64(
    device const int8_t* wq [[buffer(0)]], device const half* ws [[buffer(1)]],
    device const int8_t* x [[buffer(2)]], device const half* xsc [[buffer(3)]],
    device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]],
    constant SikNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_q8_gemm_nax_i8_impl<64, 64, 4, 32>(wq, ws, x, xsc, y16, y32, p, tgid);
#else
    (void)wq;(void)ws;(void)x;(void)xsc;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sik_q8_gemm_nax_i8_m32(
    device const int8_t* wq [[buffer(0)]], device const half* ws [[buffer(1)]],
    device const int8_t* x [[buffer(2)]], device const half* xsc [[buffer(3)]],
    device half* y16 [[buffer(4)]], device float* y32 [[buffer(5)]],
    constant SikNaxGemmParams& p [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    sik_q8_gemm_nax_i8_impl<32, 64, 2, 32>(wq, ws, x, xsc, y16, y32, p, tgid);
#else
    (void)wq;(void)ws;(void)x;(void)xsc;(void)y16;(void)y32;(void)p;(void)tgid;
#endif
}

kernel void sik_q4a_gemm_nax_i8i4_m64(
    device const uint2* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]],
    device const int8_t* x [[buffer(3)]], device const half* xsc [[buffer(4)]], device const half* xsum [[buffer(5)]],
    device half* y16 [[buffer(6)]], device float* y32 [[buffer(7)]],
    constant SikNaxGemmParams& p [[buffer(8)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup uint2 W_tile[2][64][8];
#if defined(__HAVE_TENSOR__)
    sik_q4a_gemm_nax_i8i4_impl<64, 4, 32>(wq, ws, wb, x, xsc, xsum, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)wb;(void)x;(void)xsc;(void)xsum;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

kernel void sik_q4a_gemm_nax_i8i4_m32(
    device const uint2* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]],
    device const int8_t* x [[buffer(3)]], device const half* xsc [[buffer(4)]], device const half* xsum [[buffer(5)]],
    device half* y16 [[buffer(6)]], device float* y32 [[buffer(7)]],
    constant SikNaxGemmParams& p [[buffer(8)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup uint2 W_tile[2][64][8];
#if defined(__HAVE_TENSOR__)
    sik_q4a_gemm_nax_i8i4_impl<32, 2, 32>(wq, ws, wb, x, xsc, xsum, y16, y32, p, W_tile, tgid, tid);
#else
    (void)wq;(void)ws;(void)wb;(void)x;(void)xsc;(void)xsum;(void)y16;(void)y32;(void)p;(void)W_tile;(void)tgid;(void)tid;
#endif
}

// High-Occupancy 8KiB Whole-Row Zero-Acc Q4_A -> INT8xINT8->INT32 NAX GEMM (`90.7+ TOPS`, 4 TGs/core, 0 x_sum!)
union SikQ4aTgI8 {
    int8_t W[64][128]; // 8,192 B (fits 4 TGs per 32 KiB M5 core!)
    float  ws[64];     //   256 B (used ONLY in epilogue after K-loop!)
};

#if defined(__HAVE_TENSOR__)
template <int BM, int SG, int CAP>
inline void sik_q4a_gemm_nax_i8_sb128_impl(
    device const uint4*        w_u4,
    device const half*         w_s,
    device const half*         w_b,
    device const int8_t*       x_i8,
    device const half*         x_sc,
    device half*               y_f16,
    device float*              y_f32,
    constant SikNaxGemmParams& p,
    threadgroup SikQ4aTgI8&    tg,
    uint2                      tgid,
    uint                       tid
) {
    const uint m0 = tgid.y * uint(BM);
    const uint n0 = tgid.x * 64u;
    if (m0 >= p.M_padded || n0 >= p.N_padded) return;

    const uint u4_r   = p.K >> 5u;
    const uint ngrp   = p.K >> 6u;
    const uint kt_cnt = p.K >> 7u;

    tensor<device int8_t, dextents<int, 2>, tensor_inline> tX(const_cast<device int8_t*>(x_i8), dextents<int, 2>(int(p.K), int(p.M_padded)));
    tensor<threadgroup int8_t, extents<int, 128, 64>, tensor_inline> tW(&tg.W[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto sX0 = tX.template slice<128, BM>(0, int(m0));
    auto cI  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(sX0)>, sik_strip_as<decltype(tW)>, int32_t>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) if (cI.is_valid_element(i)) cI[i] = 0;

    constexpr uint r_step  = (uint(SG) * 32u) >> 2u; // 32 (SG=4) or 16 (SG=2)
    constexpr uint q_iters = 64u / r_step;
    const uint r_grp       = tid >> 2u;
    const uint sub4        = tid & 3u;
    const uint lane_base   = (tid & 31u) & ~3u;

    float inv_arr[q_iters];
    float wsc_arr[q_iters];
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < q_iters; ++q) {
        const uint r_loc  = q * r_step + r_grp;
        const uint r_glob = n0 + r_loc;
        float amax = 0.0f;
        if (r_glob < p.N_logical) {
            for (uint g = sub4; g < ngrp; g += 4u) {
                const float s = float(w_s[ulong(r_glob) * ngrp + g]);
                const float b = float(w_b[ulong(r_glob) * ngrp + g]);
                amax = max(amax, max(abs(b), abs(fma(15.0f, s, b))));
            }
        }
        amax = max(amax, simd_shuffle_xor(amax, 1u));
        amax = max(amax, simd_shuffle_xor(amax, 2u));
        wsc_arr[q] = amax * (1.0f / 127.0f);
        inv_arr[q] = simd_shuffle((amax > 1e-12f) ? (127.0f / amax) : 0.0f, lane_base);
    }

    for (uint kt = 0u; kt < kt_cnt; ++kt) {
        const uint g = kt * 2u + (sub4 >> 1u);
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) {
            const uint r_loc  = q * r_step + r_grp;
            const uint r_glob = n0 + r_loc;
            threadgroup char4* d = reinterpret_cast<threadgroup char4*>(&tg.W[r_loc][sub4 * 32u]);
            if (r_glob < p.N_logical) {
                const half  sc = half(float(w_s[ulong(r_glob) * ngrp + g]) * inv_arr[q]);
                const half  bi = half(float(w_b[ulong(r_glob) * ngrp + g]) * inv_arr[q]);
                const uint4 pk = w_u4[ulong(r_glob) * u4_r + kt * 4u + sub4];
                half4 lo, hi;
                sik_unpack_u4_u32_h4(pk.x, lo, hi);
                d[0] = char4(clamp(rint(fma(lo, sc, bi)), half4(-127.0h), half4(127.0h)));
                d[1] = char4(clamp(rint(fma(hi, sc, bi)), half4(-127.0h), half4(127.0h)));
                sik_unpack_u4_u32_h4(pk.y, lo, hi);
                d[2] = char4(clamp(rint(fma(lo, sc, bi)), half4(-127.0h), half4(127.0h)));
                d[3] = char4(clamp(rint(fma(hi, sc, bi)), half4(-127.0h), half4(127.0h)));
                sik_unpack_u4_u32_h4(pk.z, lo, hi);
                d[4] = char4(clamp(rint(fma(lo, sc, bi)), half4(-127.0h), half4(127.0h)));
                d[5] = char4(clamp(rint(fma(hi, sc, bi)), half4(-127.0h), half4(127.0h)));
                sik_unpack_u4_u32_h4(pk.w, lo, hi);
                d[6] = char4(clamp(rint(fma(lo, sc, bi)), half4(-127.0h), half4(127.0h)));
                d[7] = char4(clamp(rint(fma(hi, sc, bi)), half4(-127.0h), half4(127.0h)));
            } else {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 8u; ++j) d[j] = char4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX.template slice<128, BM>(int(kt * 128u), int(m0));
        op.run(sX, tW, cI);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sub4 == 0u) {
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < q_iters; ++q) tg.ws[q * r_step + r_grp] = wsc_arr[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx    = cI.get_multidimensional_index(i);
            uint m_glob = m0 + uint(idx[1]);
            uint n_glob = n0 + uint(idx[0]);
            if (m_glob < p.M_valid && n_glob < p.N_logical) {
                const ulong out_idx = ulong(m_glob) * p.N_logical + n_glob;
                float val = float(cI[i]) * (tg.ws[uint(idx[0])] * float(x_sc[m_glob]));
                if (p.out_mode == SIK_OUT_F32_SOFTCAP && p.softcap > 0.0f) val = sik_softcap(val, p.softcap);
                if (p.out_mode == SIK_OUT_F16)          y_f16[out_idx]  = sik_clamp_f16(val);
                else if (p.out_mode == SIK_OUT_RES_F32) y_f32[out_idx] += val;
                else                                    y_f32[out_idx]  = val;
            }
        }
    }
}
#endif

kernel void sik_q4a_gemm_nax_i8_m128(
    device const uint4* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]],
    device const int8_t* x [[buffer(3)]], device const half* xsc [[buffer(4)]],
    device half* y16 [[buffer(5)]], device float* y32 [[buffer(6)]],
    constant SikNaxGemmParams& p [[buffer(7)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikQ4aTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_q4a_gemm_nax_i8_sb128_impl<128, 4, 64>(wq, ws, wb, x, xsc, y16, y32, p, tg, tgid, tid);
#else
    (void)wq;(void)ws;(void)wb;(void)x;(void)xsc;(void)y16;(void)y32;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_q4a_gemm_nax_i8_m64(
    device const uint4* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]],
    device const int8_t* x [[buffer(3)]], device const half* xsc [[buffer(4)]],
    device half* y16 [[buffer(5)]], device float* y32 [[buffer(6)]],
    constant SikNaxGemmParams& p [[buffer(7)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikQ4aTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_q4a_gemm_nax_i8_sb128_impl<64, 4, 32>(wq, ws, wb, x, xsc, y16, y32, p, tg, tgid, tid);
#else
    (void)wq;(void)ws;(void)wb;(void)x;(void)xsc;(void)y16;(void)y32;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_q4a_gemm_nax_i8_m32(
    device const uint4* wq [[buffer(0)]], device const half* ws [[buffer(1)]], device const half* wb [[buffer(2)]],
    device const int8_t* x [[buffer(3)]], device const half* xsc [[buffer(4)]],
    device half* y16 [[buffer(5)]], device float* y32 [[buffer(6)]],
    constant SikNaxGemmParams& p [[buffer(7)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikQ4aTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_q4a_gemm_nax_i8_sb128_impl<32, 2, 32>(wq, ws, wb, x, xsc, y16, y32, p, tg, tgid, tid);
#else
    (void)wq;(void)ws;(void)wb;(void)x;(void)xsc;(void)y16;(void)y32;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

/* ============================================================================
 * 5. Vectorized `float4` Causal Short Convolutions (Decode + 2D Parallel Prefill
 *    + Fused KV-SConv + QKVR-Prep!)
 * ============================================================================ */

struct SikSconvParams {
    uint M;
    uint channels;        // 1024 (K/V) or 2304 (attn/mlp)
    uint save_snap_mask;  // bit `t` (0..6) => save post-token `t` state into `snaps + t * snap_stride_f4`
    uint res_mode;        // 0 => out = conv; 1 => cur_inout += conv; 2 => block_dst = cur_inout, cur_inout = conv
    uint head_dim;        // 128
    uint n_kv;            // 8
    uint v_state_off_f4;  // float4 offset from K state to V state (prefill KV) OR `snap_stride_f4` (decode static/combine!)
    uint top_k;           // 8
};

// Decode Static Short-Conv (`M <= 8`, single contiguous `snaps` buffer -> 0 switch bloat!)
kernel void sik_sconv_static(
    device const float*      x_in       [[buffer(0)]],
    device const float4*     weight     [[buffer(1)]],
    device float4*           conv_state [[buffer(2)]], // [C] float4(h0, h1, h2, 0)
    device float*            out_or_cur [[buffer(3)]],
    device float*            block_dst  [[buffer(4)]],
    device float4*           snaps      [[buffer(5)]], // [7, snap_stride_f4] at `state_byte_off`
    constant SikSconvParams& p          [[buffer(6)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= p.channels) return;
    const uint   C      = p.channels;
    const float4 w      = weight[c];
    const float  inv_s  = 0.4472135955f;
    const ulong  sstr   = ulong(p.v_state_off_f4);

    float4 st = conv_state[c];
    float h0 = st.x, h1 = st.y, h2 = st.z;

    for (uint m = 0u; m < p.M; ++m) {
        const ulong off = ulong(m) * C + c;
        const float cur = x_in[off];
        const float sum = fma(h0, w.x, fma(h1, w.y, fma(h2, w.z, cur * w.w)));
        const float y   = (sum + cur) * inv_s;

        h0 = h1; h1 = h2; h2 = cur;
        if ((p.save_snap_mask & (1u << m)) != 0u && m < 7u) {
            snaps[ulong(m) * sstr + c] = float4(h0, h1, h2, 0.0f);
        }

        if (p.res_mode == 2u) {
            block_dst[off]  = out_or_cur[off];
            out_or_cur[off] = y;
        } else if (p.res_mode == 1u) {
            out_or_cur[off] = out_or_cur[off] + y;
        } else {
            out_or_cur[off] = y;
        }
    }
    conv_state[c] = float4(h0, h1, h2, 0.0f);
}

struct SikWoSconvParams {
    uint M;
    uint rows;           // D = 2304
    uint cols;           // n_q * hd = 3072
    uint save_snap_mask; // bit `t` (0..6) => save post-token `t` state into `snaps + t * snap_stride_f4`
    uint res_mode;       // 1 => cur += y; 2 => block_dst = cur, cur = y
    uint snap_stride_f4; // sconv_total_bytes / 16
    uint pad1;
    uint pad2;
};

// Fused 32-lane 2-row Q4_A `Wo` GEMV + `attn_sconv` + single-buffer snapshots + residual!
// Drops 17 buffers -> 11 buffers & eliminates 7-way snap switch (-0.30 ms @ M=8), 100% bit-exact M=1..8!
template <uint M_FIX>
inline void sik_q4a_wo_sconv_fixed(
    device const uint2*        w_u2,
    device const half*         w_s,
    device const half*         w_b,
    device const half*         x_in,
    device const float*        x_sum,
    device const float4*       sconv_w,
    device float4*             conv_state,
    device float*              out_or_cur,
    device float*              block_dst,
    device float4*             snaps,
    constant SikWoSconvParams& p,
    uint r0,
    uint lane
) {
    const uint r1      = min(r0 + 1u, p.rows - 1u);
    const uint k16     = p.cols >> 4u;
    const uint nblk    = p.cols >> 6u;
    const bool do_bias = ((lane & 3u) == 0u);
    device const uint2* rw0 = w_u2 + ulong(r0) * k16;
    device const uint2* rw1 = w_u2 + ulong(r1) * k16;
    device const half*  rs0 = w_s  + ulong(r0) * nblk;
    device const half*  rs1 = w_s  + ulong(r1) * nblk;
    device const half*  rb0 = w_b  + ulong(r0) * nblk;
    device const half*  rb1 = w_b  + ulong(r1) * nblk;

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint i = lane; i < k16; i += 32u) {
        const uint2  q0  = rw0[i], q1 = rw1[i];
        const uint   grp = i >> 2u;
        const float2 sc  = float2(float(rs0[grp]), float(rs1[grp]));
        half4 a0, a1, a2, a3, b0, b1, b2, b3;
        sik_unpack_u4_u32_h4(q0.x, a0, a1); sik_unpack_u4_u32_h4(q0.y, a2, a3);
        sik_unpack_u4_u32_h4(q1.x, b0, b1); sik_unpack_u4_u32_h4(q1.y, b2, b3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + i * 16u);
            const half4 x0 = xm[0], x1 = xm[1], x2 = xm[2], x3 = xm[3];
            const half4 dh0 = fma(a0, x0, a1 * x1) + fma(a2, x2, a3 * x3);
            const half4 dh1 = fma(b0, x0, b1 * x1) + fma(b2, x2, b3 * x3);
            acc[m].x = fma(sc.x, (float(dh0.x) + float(dh0.y)) + (float(dh0.z) + float(dh0.w)), acc[m].x);
            acc[m].y = fma(sc.y, (float(dh1.x) + float(dh1.y)) + (float(dh1.z) + float(dh1.w)), acc[m].y);
        }
        if (do_bias) {
            const float2 bi = float2(float(rb0[grp]), float(rb1[grp]));
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                acc[m] = fma(bi, float2(x_sum[ulong(m) * nblk + grp]), acc[m]);
            }
        }
    }

    float val[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float vx = simd_sum(acc[m].x);
        const float vy = simd_sum(acc[m].y);
        val[m] = (lane == 0u) ? vx : vy;
    }

    if (lane < 2u) {
        const uint c = r0 + lane;
        if (c < p.rows) {
            const uint   C     = p.rows;
            const ulong  sstr  = ulong(p.snap_stride_f4);
            const float4 w     = sconv_w[c];
            const float  inv_s = 0.4472135955f;
            float4 st = conv_state[c];
            float h0 = st.x, h1 = st.y, h2 = st.z;
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                const float cur = val[m];
                const float sum = fma(h0, w.x, fma(h1, w.y, fma(h2, w.z, cur * w.w)));
                const float y   = (sum + cur) * inv_s;
                h0 = h1; h1 = h2; h2 = cur;
                if ((p.save_snap_mask & (1u << m)) != 0u && m < 7u) {
                    snaps[ulong(m) * sstr + c] = float4(h0, h1, h2, 0.0f);
                }
                const ulong off = ulong(m) * C + c;
                if (p.res_mode == 2u) {
                    block_dst[off]  = out_or_cur[off];
                    out_or_cur[off] = y;
                } else if (p.res_mode == 1u) {
                    out_or_cur[off] = out_or_cur[off] + y;
                } else {
                    out_or_cur[off] = y;
                }
            }
            conv_state[c] = float4(h0, h1, h2, 0.0f);
        }
    }
}

kernel void sik_q4a_gemv_wo_sconv_xsum(
    device const uint2*        w_u2       [[buffer(0)]],
    device const half*         w_s        [[buffer(1)]],
    device const half*         w_b        [[buffer(2)]],
    device const half*         x_in       [[buffer(3)]],
    device const float*        x_sum      [[buffer(4)]],
    device const float4*       sconv_w    [[buffer(5)]],
    device float4*             conv_state [[buffer(6)]],
    device float*              out_or_cur [[buffer(7)]],
    device float*              block_dst  [[buffer(8)]],
    device float4*             snaps      [[buffer(9)]],
    constant SikWoSconvParams& p          [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint r0 = (tgid * sgs + sgid) * 2u;
    if (r0 >= p.rows) return;
    #define SIK_WOSC_CASE(MF) { sik_q4a_wo_sconv_fixed<MF>(w_u2, w_s, w_b, x_in, x_sum, sconv_w, conv_state, out_or_cur, block_dst, snaps, p, r0, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_WOSC_CASE(1u) case 2u: SIK_WOSC_CASE(2u) case 3u: SIK_WOSC_CASE(3u) case 4u: SIK_WOSC_CASE(4u)
        case 5u: SIK_WOSC_CASE(5u) case 6u: SIK_WOSC_CASE(6u) case 7u: SIK_WOSC_CASE(7u) case 8u: SIK_WOSC_CASE(8u)
        default: break;
    }
    #undef SIK_WOSC_CASE
}

// Lean 10-Buffer `M == 1` (`save_snap_mask == 0`) Q4_A `Wo + attn_sconv + residual` (`l32_2r`, 0 snap ptr reg bloat -> 444 GB/s!)
// 100.000000% bit-exact with `sik_q4a_gemv_wo_sconv_xsum` (`M = 1..8`)!
kernel void sik_q4a_gemv_wo_sconv_m1(
    device const uint2*        w_u2       [[buffer(0)]],
    device const half*         w_s        [[buffer(1)]],
    device const half*         w_b        [[buffer(2)]],
    device const half*         x_in       [[buffer(3)]],
    device const float*        x_sum      [[buffer(4)]],
    device const float4*       sconv_w    [[buffer(5)]],
    device float4*             conv_state [[buffer(6)]],
    device float*              out_or_cur [[buffer(7)]],
    device float*              block_dst  [[buffer(8)]],
    constant SikWoSconvParams& p          [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint r0 = (tgid * sgs + sgid) * 2u;
    if (r0 >= p.rows) return;
    const uint r1      = min(r0 + 1u, p.rows - 1u);
    const uint k16     = p.cols >> 4u;
    const uint nblk    = p.cols >> 6u;
    const bool do_bias = ((lane & 3u) == 0u);
    device const uint2* rw0 = w_u2 + ulong(r0) * k16;
    device const uint2* rw1 = w_u2 + ulong(r1) * k16;
    device const half*  rs0 = w_s  + ulong(r0) * nblk;
    device const half*  rs1 = w_s  + ulong(r1) * nblk;
    device const half*  rb0 = w_b  + ulong(r0) * nblk;
    device const half*  rb1 = w_b  + ulong(r1) * nblk;

    float2 acc = float2(0.0f);
    for (uint i = lane; i < k16; i += 32u) {
        const uint2  q0  = rw0[i], q1 = rw1[i];
        const uint   grp = i >> 2u;
        const float2 sc  = float2(float(rs0[grp]), float(rs1[grp]));
        half4 a0, a1, a2, a3, b0, b1, b2, b3;
        sik_unpack_u4_u32_h4(q0.x, a0, a1); sik_unpack_u4_u32_h4(q0.y, a2, a3);
        sik_unpack_u4_u32_h4(q1.x, b0, b1); sik_unpack_u4_u32_h4(q1.y, b2, b3);
        device const half4* xm = reinterpret_cast<device const half4*>(x_in + i * 16u);
        const half4 x0 = xm[0], x1 = xm[1], x2 = xm[2], x3 = xm[3];
        const half4 dh0 = fma(a0, x0, a1 * x1) + fma(a2, x2, a3 * x3);
        const half4 dh1 = fma(b0, x0, b1 * x1) + fma(b2, x2, b3 * x3);
        acc.x = fma(sc.x, (float(dh0.x) + float(dh0.y)) + (float(dh0.z) + float(dh0.w)), acc.x);
        acc.y = fma(sc.y, (float(dh1.x) + float(dh1.y)) + (float(dh1.z) + float(dh1.w)), acc.y);
        if (do_bias) {
            const float2 bi = float2(float(rb0[grp]), float(rb1[grp]));
            acc = fma(bi, float2(x_sum[grp]), acc);
        }
    }
    const float vx = simd_sum(acc.x);
    const float vy = simd_sum(acc.y);
    if (lane < 2u) {
        const uint c = r0 + lane;
        if (c < p.rows) {
            const float  cur   = (lane == 0u) ? vx : vy;
            const float4 w     = sconv_w[c];
            const float  inv_s = 0.4472135955f;
            const float4 st    = conv_state[c];
            const float  sum   = fma(st.x, w.x, fma(st.y, w.y, fma(st.z, w.z, cur * w.w)));
            const float  y     = (sum + cur) * inv_s;
            conv_state[c]      = float4(st.y, st.z, cur, 0.0f);
            if (p.res_mode == 2u) {
                block_dst[c]  = out_or_cur[c];
                out_or_cur[c] = y;
            } else if (p.res_mode == 1u) {
                out_or_cur[c] = out_or_cur[c] + y;
            } else {
                out_or_cur[c] = y;
            }
        }
    }
}

// 2D `(c, m)` Parallel FIR-4 Static Short-Conv for Prefill (`M > 8`, saturates all 40 GPU cores!)
// Does NOT modify `conv_state` (race-free!); followed by 9-TG `sik_sconv_save_tail`!
// Grid: `((channels + 255) / 256, (M + 15) / 16, 1)`, `256` threads/TG
kernel void sik_sconv_static_prefill(
    device const float*      x_in       [[buffer(0)]],
    device const float4*     weight     [[buffer(1)]],
    device const float4*     conv_state [[buffer(2)]], // [C] float4(x[-3], x[-2], x[-1], 0)
    device float*            out_or_cur [[buffer(3)]],
    device float*            block_dst  [[buffer(4)]],
    constant SikSconvParams& p          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint c  = tgid.x * 256u + tid;
    const uint m0 = tgid.y * 16u;
    if (c >= p.channels || m0 >= p.M) return;

    const uint   C     = p.channels;
    const float4 w     = weight[c];
    const float  inv_s = 0.4472135955f;
    const uint   m_end = min(m0 + 16u, p.M);

    float h0, h1, h2;
    if (m0 >= 3u) {
        h0 = x_in[ulong(m0 - 3u) * C + c];
        h1 = x_in[ulong(m0 - 2u) * C + c];
        h2 = x_in[ulong(m0 - 1u) * C + c];
    } else {
        const float4 st = conv_state[c];
        h0 = (m0 == 0u) ? st.x : ((m0 == 1u) ? st.y : st.z);
        h1 = (m0 == 0u) ? st.y : ((m0 == 1u) ? st.z : x_in[0ul * C + c]);
        h2 = (m0 == 0u) ? st.z : ((m0 == 1u) ? x_in[0ul * C + c] : x_in[1ul * C + c]);
    }

    for (uint m = m0; m < m_end; ++m) {
        const ulong off = ulong(m) * C + c;
        const float cur = x_in[off];
        const float sum = fma(h0, w.x, fma(h1, w.y, fma(h2, w.z, cur * w.w)));
        const float y   = (sum + cur) * inv_s;
        h0 = h1; h1 = h2; h2 = cur;

        if (p.res_mode == 2u) {
            block_dst[off]  = out_or_cur[off];
            out_or_cur[off] = y;
        } else if (p.res_mode == 1u) {
            out_or_cur[off] = out_or_cur[off] + y;
        } else {
            out_or_cur[off] = y;
        }
    }
}

// Tiny `<1 us` Tail State Saver for 2D Prefill Static SConv (`M >= 3`)
kernel void sik_sconv_save_tail(
    device const float*      x_in       [[buffer(0)]],
    device float4*           conv_state [[buffer(1)]],
    constant SikSconvParams& p          [[buffer(2)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= p.channels) return;
    const uint C = p.channels;
    const uint M = p.M;
    if (M >= 3u) {
        conv_state[c] = float4(x_in[ulong(M - 3u) * C + c],
                               x_in[ulong(M - 2u) * C + c],
                               x_in[ulong(M - 1u) * C + c], 0.0f);
    } else {
        float4 st = conv_state[c];
        float h0 = st.x, h1 = st.y, h2 = st.z;
        for (uint m = 0u; m < M; ++m) {
            h0 = h1; h1 = h2; h2 = x_in[ulong(m) * C + c];
        }
        conv_state[c] = float4(h0, h1, h2, 0.0f);
    }
}

// 2D Parallel Prefill Fused K+V Static Short-Conv (`tgid.z = 0 => K`, `1 => V`)
// Grid: `((channels + 255) / 256, (M + 15) / 16, 2)`, `256` threads/TG
kernel void sik_sconv_static_kv_prefill(
    device const float*      xk_in      [[buffer(0)]],
    device const float*      xv_in      [[buffer(1)]],
    device const float4*     wk         [[buffer(2)]],
    device const float4*     wv         [[buffer(3)]],
    device const float4*     conv_state [[buffer(4)]],
    device float*            ok_out     [[buffer(5)]],
    device float*            ov_out     [[buffer(6)]],
    constant SikSconvParams& p          [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint c    = tgid.x * 256u + tid;
    const uint m0   = tgid.y * 16u;
    const uint is_v = tgid.z;
    if (c >= p.channels || m0 >= p.M || is_v >= 2u) return;

    const uint   C     = p.channels;
    const uint   m_end = min(m0 + 16u, p.M);
    const ulong  s_off = (is_v != 0u) ? ulong(p.v_state_off_f4) : 0ul;
    device const float*  xin = (is_v != 0u) ? xv_in  : xk_in;
    device const float4* wt  = (is_v != 0u) ? wv     : wk;
    device float*        dst = (is_v != 0u) ? ov_out : ok_out;

    const float4 w     = wt[c];
    const float  inv_s = 0.4472135955f;

    float h0, h1, h2;
    if (m0 >= 3u) {
        h0 = xin[ulong(m0 - 3u) * C + c];
        h1 = xin[ulong(m0 - 2u) * C + c];
        h2 = xin[ulong(m0 - 1u) * C + c];
    } else {
        const float4 st = conv_state[s_off + c];
        h0 = (m0 == 0u) ? st.x : ((m0 == 1u) ? st.y : st.z);
        h1 = (m0 == 0u) ? st.y : ((m0 == 1u) ? st.z : xin[0ul * C + c]);
        h2 = (m0 == 0u) ? st.z : ((m0 == 1u) ? xin[0ul * C + c] : xin[1ul * C + c]);
    }

    for (uint m = m0; m < m_end; ++m) {
        const ulong off = ulong(m) * C + c;
        const float cur = xin[off];
        const float sum = fma(h0, w.x, fma(h1, w.y, fma(h2, w.z, cur * w.w)));
        dst[off] = (sum + cur) * inv_s;
        h0 = h1; h1 = h2; h2 = cur;
    }
}

// 2D Parallel Prefill Fused K+V Dynamic Short-Conv (`tgid.z = 0 => K`, `1 => V`)
// Grid: `((channels + 255) / 256, (M + 15) / 16, 2)`, `256` threads/TG
kernel void sik_sconv_dynamic_kv_prefill(
    device const float*      xk_in      [[buffer(0)]],
    device const float*      xv_in      [[buffer(1)]],
    device const float4*     sw_k       [[buffer(2)]],
    device const float4*     sw_v       [[buffer(3)]],
    device const float4*     dw_k       [[buffer(4)]],
    device const float4*     dw_v       [[buffer(5)]],
    device const float4*     conv_state [[buffer(6)]],
    device float*            ok_out     [[buffer(7)]],
    device float*            ov_out     [[buffer(8)]],
    constant SikSconvParams& p          [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint c    = tgid.x * 256u + tid;
    const uint m0   = tgid.y * 16u;
    const uint is_v = tgid.z;
    if (c >= p.channels || m0 >= p.M || is_v >= 2u) return;

    const uint   C     = p.channels;
    const uint   hkv   = c / p.head_dim;
    const uint   m_end = min(m0 + 16u, p.M);
    const ulong  s_off = (is_v != 0u) ? ulong(p.v_state_off_f4) : 0ul;
    device const float*  xin = (is_v != 0u) ? xv_in  : xk_in;
    device const float4* swt = (is_v != 0u) ? sw_v   : sw_k;
    device const float4* dwt = (is_v != 0u) ? dw_v   : dw_k;
    device float*        dst = (is_v != 0u) ? ov_out : ok_out;

    const float4 sw    = swt[c];
    const float  inv_s = 1.0f / 3.0f;

    float h0, h1, h2;
    if (m0 >= 3u) {
        h0 = xin[ulong(m0 - 3u) * C + c];
        h1 = xin[ulong(m0 - 2u) * C + c];
        h2 = xin[ulong(m0 - 1u) * C + c];
    } else {
        const float4 st = conv_state[s_off + c];
        h0 = (m0 == 0u) ? st.x : ((m0 == 1u) ? st.y : st.z);
        h1 = (m0 == 0u) ? st.y : ((m0 == 1u) ? st.z : xin[0ul * C + c]);
        h2 = (m0 == 0u) ? st.z : ((m0 == 1u) ? xin[0ul * C + c] : xin[1ul * C + c]);
    }

    for (uint m = m0; m < m_end; ++m) {
        const ulong  off = ulong(m) * C + c;
        const float  cur = xin[off];
        const float4 tap = sw + dwt[ulong(m) * p.n_kv + hkv];
        const float  sum = fma(cur, tap.x, fma(h2, tap.y, fma(h1, tap.z, h0 * tap.w)));
        dst[off] = (sum + cur) * inv_s;
        h0 = h1; h1 = h2; h2 = cur;
    }
}

// Tiny `<1 us` K+V Tail State Saver for 2D Prefill
kernel void sik_sconv_save_tail_kv(
    device const float*      xk_in      [[buffer(0)]],
    device const float*      xv_in      [[buffer(1)]],
    device float4*           conv_state [[buffer(2)]],
    constant SikSconvParams& p          [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint c    = tgid.x * 256u + tid;
    const uint is_v = tgid.y;
    if (c >= p.channels || is_v >= 2u) return;
    const uint  C     = p.channels;
    const uint  M     = p.M;
    const ulong s_off = (is_v != 0u) ? ulong(p.v_state_off_f4) : 0ul;
    device const float* xin = (is_v != 0u) ? xv_in : xk_in;
    if (M >= 3u) {
        conv_state[s_off + c] = float4(xin[ulong(M - 3u) * C + c],
                                       xin[ulong(M - 2u) * C + c],
                                       xin[ulong(M - 1u) * C + c], 0.0f);
    } else {
        float4 st = conv_state[s_off + c];
        float h0 = st.x, h1 = st.y, h2 = st.z;
        for (uint m = 0u; m < M; ++m) { h0 = h1; h1 = h2; h2 = xin[ulong(m) * C + c]; }
        conv_state[s_off + c] = float4(h0, h1, h2, 0.0f);
    }
}

// Decode Fused `MoE Top-8 Combine + Post-MoE mlp_sconv + Residual Add` (`M <= 8`, single-buffer snapshots + TG `topk_w`!)
// Grid: `((D + 63) / 64, 1, 1)`, `64` threads/TG (`2` SGs)
kernel void sik_moe_combine_sconv(
    device const float*      down_part  [[buffer(0)]],
    device const float*      topk_w     [[buffer(1)]],
    device const float4*     weight     [[buffer(2)]],
    device float4*           conv_state [[buffer(3)]],
    device float*            cur_io     [[buffer(4)]],
    device float4*           snaps      [[buffer(5)]],
    constant SikSconvParams& p          [[buffer(6)]],
    uint c   [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float tg_tw[64];
    const uint n_tw = min(p.M * p.top_k, 64u);
    if (tid < n_tw) tg_tw[tid] = topk_w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (c >= p.channels) return;
    const uint   C     = p.channels;
    const uint   K     = p.top_k;
    const ulong  sstr  = ulong(p.v_state_off_f4);
    const float4 w     = weight[c];
    const float  inv_s = 0.4472135955f;

    float4 st = conv_state[c];
    float h0 = st.x, h1 = st.y, h2 = st.z;

    for (uint m = 0u; m < p.M; ++m) {
        float cur = 0.0f;
        const ulong base_pair = ulong(m) * K;
        if (K == 8u && m < 8u) {
            const uint b = m * 8u;
            cur = fma(tg_tw[b + 0u], down_part[(base_pair + 0u) * C + c], cur);
            cur = fma(tg_tw[b + 1u], down_part[(base_pair + 1u) * C + c], cur);
            cur = fma(tg_tw[b + 2u], down_part[(base_pair + 2u) * C + c], cur);
            cur = fma(tg_tw[b + 3u], down_part[(base_pair + 3u) * C + c], cur);
            cur = fma(tg_tw[b + 4u], down_part[(base_pair + 4u) * C + c], cur);
            cur = fma(tg_tw[b + 5u], down_part[(base_pair + 5u) * C + c], cur);
            cur = fma(tg_tw[b + 6u], down_part[(base_pair + 6u) * C + c], cur);
            cur = fma(tg_tw[b + 7u], down_part[(base_pair + 7u) * C + c], cur);
        } else {
            for (uint k = 0u; k < K; ++k) {
                cur = fma(topk_w[base_pair + k], down_part[(base_pair + k) * C + c], cur);
            }
        }
        const float sum = fma(h0, w.x, fma(h1, w.y, fma(h2, w.z, cur * w.w)));
        const float y   = (sum + cur) * inv_s;

        h0 = h1; h1 = h2; h2 = cur;
        if ((p.save_snap_mask & (1u << m)) != 0u && m < 7u) {
            snaps[ulong(m) * sstr + c] = float4(h0, h1, h2, 0.0f);
        }
        const ulong off = ulong(m) * C + c;
        cur_io[off] = cur_io[off] + y;
    }
    conv_state[c] = float4(h0, h1, h2, 0.0f);
}


/* ============================================================================
 * 6. Fused Decode SConv-KV + QKVR Prep, Prefill Head-Major QKVR Prep,
 *    Metal 4.1 NAX FlashAttention Prefill, & Shared-KV Block-16 Decode Split-KV
 * ============================================================================ */

struct SikAttnPrepParams {
    uint  M_valid;
    uint  M_pad16;        // Multiple of 16 for head-major NAX prefill Q/R
    uint  n_q;            // 24
    uint  n_kv;           // 8
    uint  head_dim;       // 128
    uint  d_rel;          // 16
    uint  seq_pos_start;
    uint  kv_stride;      // 2048 (SWA ring) or max_seq_alloc (Global)
    uint  is_local;       // 1 => SWA ring + static sconv; 0 => Global linear + dynamic sconv + tau
    uint  save_snap_mask; // bit `t` (0..6) => save post-token `t` sconv state
    uint  v_state_off_f4; // float4 offset from K state to V state
    uint  kv_zero_tail;   // extra tail slots to zero up to mult of 32 (0..31)
    float qk_clip;        // 22.627417f
    float rel_clip;       // 8.0f
    float eps;            // 1e-6f
    float log_alpha;      // 0.1f
    float log_n_floor;    // 8192.0f
    float attn_scale;     // 1.0f / sqrt(128.0f)
    uint  kv_dtype;       // STRATA_INKLING2_KV_DTYPE_F16 (0) or STRATA_INKLING2_KV_DTYPE_Q8_0 (1)
    uint  snap_stride_f4; // sconv_total_bytes / 16
};

// Per-head symmetric Q8_0 KV quantizer (128 threads/head, exact `int8[-127..127]` + `half` scale)
inline void sik_store_kv_q8_head(
    float val_out,
    device char* dst_q,
    device half* dst_s,
    ulong hs_idx,
    uint head_dim,
    threadgroup float (&tg_sq)[4],
    uint tid,
    uint sgid,
    uint lane
) {
    const float am = simd_max((tid < head_dim) ? fabs(val_out) : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) tg_sq[sgid] = am;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot_am = simd_max((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = tot_am;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float absmax = tg_sq[0];
    const half  sc_h   = (absmax > 0.0f) ? half(max(absmax / 127.0f, 1e-7f)) : 0.0h;
    const float inv_sc = (float(sc_h) > 0.0f) ? (1.0f / float(sc_h)) : 0.0f;
    if (tid < head_dim) {
        dst_q[hs_idx * head_dim + tid] = char(clamp(rint(val_out * inv_sc), -127.0f, 127.0f));
    }
    if (tid == 0u) dst_s[hs_idx] = sc_h;
}

// Zero-Barrier 1-SG (`32` lanes x `float4`) Per-head symmetric Q8_0 KV quantizer
inline void sik_store_kv_q8_head_sg32(
    float4        val_out,
    device char4* dst_q4,
    device half*  dst_s,
    ulong         hs_idx,
    uint          lane
) {
    const float4 av    = fabs(val_out);
    const float  am    = simd_max(max(max(av.x, av.y), max(av.z, av.w)));
    const half   sc_h  = (am > 0.0f) ? half(max(am / 127.0f, 1e-7f)) : 0.0h;
    const float  inv_s = (float(sc_h) > 0.0f) ? (1.0f / float(sc_h)) : 0.0f;
    const float4 qv    = clamp(rint(val_out * inv_s), float4(-127.0f), float4(127.0f));
    dst_q4[hs_idx * 32ul + lane] = char4(char(qv.x), char(qv.y), char(qv.z), char(qv.w));
    if (lane == 0u) dst_s[hs_idx] = sc_h;
}

// ZERO-BARRIER 1-SG (`32` th/TG, `float4`/lane) SINGLE-DISPATCH FUSED Decode (`M <= 8`):
// K/V Static OR Dynamic SConv (+ float4 snapshots) + Q/K L2-Clip + V Unweighted RMSNorm + R L2-Clip + Global Tau + KV Store!
// Grid: `(n_q * M_valid + n_kv + n_kv + n_q * M_valid, 1, 1)`, `32` threads/TG (`1` SG, ZERO TG mem, ZERO barriers!)
kernel void sik_sconv_kv_qkvr_prep(
    device const float*         q_raw      [[buffer(0)]],  // [M_valid, n_q * 128]
    device const float*         k_raw      [[buffer(1)]],  // [M_valid, n_kv * 128]
    device const float*         v_raw      [[buffer(2)]],  // [M_valid, n_kv * 128]
    device const float*         r_raw      [[buffer(3)]],  // [M_valid, n_q * 16]
    device const float4*        k_sconv_w  [[buffer(4)]],  // [1024]
    device const float4*        v_sconv_w  [[buffer(5)]],  // [1024]
    device const float4*        k_dyn      [[buffer(6)]],  // [M_valid, 8] (global only)
    device const float4*        v_dyn      [[buffer(7)]],  // [M_valid, 8] (global only)
    device float4*              conv_state [[buffer(8)]],  // K at +0, V at +v_state_off_f4
    device const float*         q_scale    [[buffer(9)]],  // [24, 128]
    device const float*         r_scale    [[buffer(10)]], // [24, 16]
    device float*               q_out      [[buffer(11)]], // [M_valid, 24, 128]
    device float*               r_out      [[buffer(12)]], // [M_valid, 24, 16]
    device half*                k_cache    [[buffer(13)]], // [8, kv_stride, 128] (half or char)
    device half*                v_cache    [[buffer(14)]], // [8, kv_stride, 128] (half or char)
    device float4*              snaps      [[buffer(15)]], // [7, snap_stride_f4] at `k_sconv_off`
    constant SikAttnPrepParams& p          [[buffer(16)]],
    device half*                k_scales   [[buffer(17)]], // [8, kv_stride] half (Q8_0 global only)
    device half*                v_scales   [[buffer(18)]], // [8, kv_stride] half (Q8_0 global only)
    uint idx  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint  nq_m = p.n_q * p.M_valid;
    const uint  b1   = nq_m;
    const uint  b2   = b1 + p.n_kv;
    const uint  b3   = b2 + p.n_kv;
    const ulong sstr = ulong(p.snap_stride_f4);

    if (idx < b1) {
        // Q: 32 lanes x float4 = 128 dims, ZERO barriers!
        const uint  m    = idx / p.n_q;
        const uint  hq   = idx - m * p.n_q;
        const uint  pos  = p.seq_pos_start + m;
        const float tau  = (p.is_local == 0u && p.log_n_floor > 0.0f)
            ? (1.0f + p.log_alpha * log(max(1.0f, float(pos + 1u) / p.log_n_floor))) : 1.0f;
        const ulong off4 = (ulong(m) * p.n_q + hq) * 32ul + lane;
        const float4 qv  = reinterpret_cast<device const float4*>(q_raw)[off4]
                         * reinterpret_cast<device const float4*>(q_scale)[ulong(hq) * 32ul + lane];
        const float tot  = simd_sum(dot(qv, qv));
        const float fac  = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps)) * tau;
        reinterpret_cast<device float4*>(q_out)[off4] = qv * fac;
    } else if (idx < b2) {
        // K: 32 lanes x 4 channels = 128 dims, ZERO barriers!
        const uint   hkv = idx - b1;
        const uint   c0  = hkv * 128u + lane * 4u;
        const float4 wt0 = k_sconv_w[c0 + 0u], wt1 = k_sconv_w[c0 + 1u];
        const float4 wt2 = k_sconv_w[c0 + 2u], wt3 = k_sconv_w[c0 + 3u];
        float4 st0 = conv_state[c0 + 0u], st1 = conv_state[c0 + 1u];
        float4 st2 = conv_state[c0 + 2u], st3 = conv_state[c0 + 3u];

        device const float4* kraw4 = reinterpret_cast<device const float4*>(k_raw);
        device half4*        kc4   = reinterpret_cast<device half4*>(k_cache);
        device char4*        kcc4  = reinterpret_cast<device char4*>(k_cache);

        for (uint m = 0u; m < p.M_valid; ++m) {
            const uint   pos  = p.seq_pos_start + m;
            const uint   slot = (p.is_local != 0u) ? (pos & (p.kv_stride - 1u)) : pos;
            const float4 cur  = kraw4[(ulong(m) * p.n_kv + hkv) * 32ul + lane];
            float4 v4;
            if (p.is_local != 0u) {
                const float inv_s = 0.4472135955f;
                v4.x = (fma(st0.x, wt0.x, fma(st0.y, wt0.y, fma(st0.z, wt0.z, cur.x * wt0.w))) + cur.x) * inv_s;
                v4.y = (fma(st1.x, wt1.x, fma(st1.y, wt1.y, fma(st1.z, wt1.z, cur.y * wt1.w))) + cur.y) * inv_s;
                v4.z = (fma(st2.x, wt2.x, fma(st2.y, wt2.y, fma(st2.z, wt2.z, cur.z * wt2.w))) + cur.z) * inv_s;
                v4.w = (fma(st3.x, wt3.x, fma(st3.y, wt3.y, fma(st3.z, wt3.z, cur.w * wt3.w))) + cur.w) * inv_s;
            } else {
                const float4 dyn  = k_dyn[ulong(m) * p.n_kv + hkv];
                const float4 t0   = wt0 + dyn, t1 = wt1 + dyn, t2 = wt2 + dyn, t3 = wt3 + dyn;
                const float inv3  = 1.0f / 3.0f;
                v4.x = (fma(cur.x, t0.x, fma(st0.z, t0.y, fma(st0.y, t0.z, st0.x * t0.w))) + cur.x) * inv3;
                v4.y = (fma(cur.y, t1.x, fma(st1.z, t1.y, fma(st1.y, t1.z, st1.x * t1.w))) + cur.y) * inv3;
                v4.z = (fma(cur.z, t2.x, fma(st2.z, t2.y, fma(st2.y, t2.z, st2.x * t2.w))) + cur.z) * inv3;
                v4.w = (fma(cur.w, t3.x, fma(st3.z, t3.y, fma(st3.y, t3.z, st3.x * t3.w))) + cur.w) * inv3;
            }
            st0 = float4(st0.y, st0.z, cur.x, 0.0f);
            st1 = float4(st1.y, st1.z, cur.y, 0.0f);
            st2 = float4(st2.y, st2.z, cur.z, 0.0f);
            st3 = float4(st3.y, st3.z, cur.w, 0.0f);
            if ((p.save_snap_mask & (1u << m)) != 0u && m < 7u) {
                const ulong sb = ulong(m) * sstr + c0;
                snaps[sb + 0u] = st0; snaps[sb + 1u] = st1;
                snaps[sb + 2u] = st2; snaps[sb + 3u] = st3;
            }
            const float tot   = simd_sum(dot(v4, v4));
            const float k_fac = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps));
            const ulong hs    = ulong(hkv) * p.kv_stride + slot;
            if (p.kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0 && p.is_local == 0u) {
                sik_store_kv_q8_head_sg32(v4 * k_fac, kcc4, k_scales, hs, lane);
            } else {
                kc4[hs * 32ul + lane] = sik_clamp_f16x4(v4 * k_fac);
            }
        }
        conv_state[c0 + 0u] = st0; conv_state[c0 + 1u] = st1;
        conv_state[c0 + 2u] = st2; conv_state[c0 + 3u] = st3;
    } else if (idx < b3) {
        // V: 32 lanes x 4 channels = 128 dims, ZERO barriers!
        const uint   hkv   = idx - b2;
        const uint   c0    = hkv * 128u + lane * 4u;
        const ulong  s_off = ulong(p.v_state_off_f4);
        const float4 wt0 = v_sconv_w[c0 + 0u], wt1 = v_sconv_w[c0 + 1u];
        const float4 wt2 = v_sconv_w[c0 + 2u], wt3 = v_sconv_w[c0 + 3u];
        float4 st0 = conv_state[s_off + c0 + 0u], st1 = conv_state[s_off + c0 + 1u];
        float4 st2 = conv_state[s_off + c0 + 2u], st3 = conv_state[s_off + c0 + 3u];

        device const float4* vraw4 = reinterpret_cast<device const float4*>(v_raw);
        device half4*        vc4   = reinterpret_cast<device half4*>(v_cache);
        device char4*        vcc4  = reinterpret_cast<device char4*>(v_cache);

        for (uint m = 0u; m < p.M_valid; ++m) {
            const uint   pos  = p.seq_pos_start + m;
            const uint   slot = (p.is_local != 0u) ? (pos & (p.kv_stride - 1u)) : pos;
            const float4 cur  = vraw4[(ulong(m) * p.n_kv + hkv) * 32ul + lane];
            float4 v4;
            if (p.is_local != 0u) {
                const float inv_s = 0.4472135955f;
                v4.x = (fma(st0.x, wt0.x, fma(st0.y, wt0.y, fma(st0.z, wt0.z, cur.x * wt0.w))) + cur.x) * inv_s;
                v4.y = (fma(st1.x, wt1.x, fma(st1.y, wt1.y, fma(st1.z, wt1.z, cur.y * wt1.w))) + cur.y) * inv_s;
                v4.z = (fma(st2.x, wt2.x, fma(st2.y, wt2.y, fma(st2.z, wt2.z, cur.z * wt2.w))) + cur.z) * inv_s;
                v4.w = (fma(st3.x, wt3.x, fma(st3.y, wt3.y, fma(st3.z, wt3.z, cur.w * wt3.w))) + cur.w) * inv_s;
            } else {
                const float4 dyn  = v_dyn[ulong(m) * p.n_kv + hkv];
                const float4 t0   = wt0 + dyn, t1 = wt1 + dyn, t2 = wt2 + dyn, t3 = wt3 + dyn;
                const float inv3  = 1.0f / 3.0f;
                v4.x = (fma(cur.x, t0.x, fma(st0.z, t0.y, fma(st0.y, t0.z, st0.x * t0.w))) + cur.x) * inv3;
                v4.y = (fma(cur.y, t1.x, fma(st1.z, t1.y, fma(st1.y, t1.z, st1.x * t1.w))) + cur.y) * inv3;
                v4.z = (fma(cur.z, t2.x, fma(st2.z, t2.y, fma(st2.y, t2.z, st2.x * t2.w))) + cur.z) * inv3;
                v4.w = (fma(cur.w, t3.x, fma(st3.z, t3.y, fma(st3.y, t3.z, st3.x * t3.w))) + cur.w) * inv3;
            }
            st0 = float4(st0.y, st0.z, cur.x, 0.0f);
            st1 = float4(st1.y, st1.z, cur.y, 0.0f);
            st2 = float4(st2.y, st2.z, cur.z, 0.0f);
            st3 = float4(st3.y, st3.z, cur.w, 0.0f);
            if ((p.save_snap_mask & (1u << m)) != 0u && m < 7u) {
                const ulong sb = ulong(m) * sstr + s_off + c0;
                snaps[sb + 0u] = st0; snaps[sb + 1u] = st1;
                snaps[sb + 2u] = st2; snaps[sb + 3u] = st3;
            }
            const float tot   = simd_sum(dot(v4, v4));
            const float v_fac = rsqrt(tot / float(p.head_dim) + p.eps);
            const ulong hs    = ulong(hkv) * p.kv_stride + slot;
            if (p.kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0 && p.is_local == 0u) {
                sik_store_kv_q8_head_sg32(v4 * v_fac, vcc4, v_scales, hs, lane);
            } else {
                vc4[hs * 32ul + lane] = sik_clamp_f16x4(v4 * v_fac);
            }
        }
        conv_state[s_off + c0 + 0u] = st0; conv_state[s_off + c0 + 1u] = st1;
        conv_state[s_off + c0 + 2u] = st2; conv_state[s_off + c0 + 3u] = st3;
    } else if (idx < b3 + nq_m) {
        // R: 100% parallel across (hq, m), single SG, ZERO barriers!
        const uint  ri  = idx - b3;
        const uint  m   = ri / p.n_q;
        const uint  hq  = ri - m * p.n_q;
        const uint  pos = p.seq_pos_start + m;
        const float tau = (p.is_local == 0u && p.log_n_floor > 0.0f)
            ? (1.0f + p.log_alpha * log(max(1.0f, float(pos + 1u) / p.log_n_floor))) : 1.0f;
        const ulong off = (ulong(m) * p.n_q + hq) * p.d_rel + lane;
        const float val = (lane < p.d_rel) ? (r_raw[off] * r_scale[ulong(hq) * p.d_rel + lane]) : 0.0f;
        const float tot = simd_sum(val * val);
        const float fac = min(1.0f, p.rel_clip / (sqrt(tot) + p.eps)) * tau;
        if (lane < p.d_rel) r_out[off] = val * fac;
    }
}

// Prefill Head-Major QKVR Prep for NAX FlashAttention:
//   - Writes pre-scaled head-major FP16 `q_hm [n_q, M_pad16, 128]` (zero-padded for m >= M_valid)
//   - Writes head-major FP32 `r_hm [n_q, M_pad16, 16]`
//   - Writes `k_cache` & `v_cache` AND zero-pads tail slots `[M_valid .. M_valid + kv_zero_tail - 1]` so NAX PV never sees NaN!
// Grid: `(64, max(M_pad16, M_valid + kv_zero_tail), 1)`, `128` threads/TG
kernel void sik_attn_qkvr_prep_prefill(
    device const float*         q_raw    [[buffer(0)]],
    device const float*         k_conv   [[buffer(1)]],
    device const float*         v_conv   [[buffer(2)]],
    device const float*         r_raw    [[buffer(3)]],
    device const float*         q_scale  [[buffer(4)]],
    device const float*         r_scale  [[buffer(5)]],
    device half*                q_hm     [[buffer(6)]],  // [n_q, M_pad16, 128]
    device float*               r_hm     [[buffer(7)]],  // [n_q, M_pad16, 16]
    device half*                k_cache  [[buffer(8)]],  // [n_kv, kv_stride, 128] (half or char)
    device half*                v_cache  [[buffer(9)]],  // [n_kv, kv_stride, 128] (half or char)
    constant SikAttnPrepParams& p        [[buffer(10)]],
    device half*                k_scales [[buffer(11)]], // [n_kv, kv_stride] half (Q8_0 global only)
    device half*                v_scales [[buffer(12)]], // [n_kv, kv_stride] half (Q8_0 global only)
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint idx = tgid.x;
    const uint m   = tgid.y;
    const uint b1  = p.n_q;
    const uint b2  = b1 + p.n_kv;
    const uint b3  = b2 + p.n_kv;

    const uint  pos  = p.seq_pos_start + m;
    const uint  slot = (p.is_local != 0u) ? (pos & (p.kv_stride - 1u)) : pos;
    const float tau  = (p.is_local == 0u && p.log_n_floor > 0.0f)
        ? (1.0f + p.log_alpha * log(max(1.0f, float(pos + 1u) / p.log_n_floor))) : 1.0f;

    threadgroup float tg_sq[4];

    if (idx < b1) {
        if (m >= p.M_pad16) return;
        const uint  hq     = idx;
        const ulong hm_off = (ulong(hq) * p.M_pad16 + m) * p.head_dim + tid;
        if (m >= p.M_valid) {
            if (tid < p.head_dim) q_hm[hm_off] = 0.0h;
            return;
        }
        const ulong off = (ulong(m) * p.n_q + hq) * p.head_dim + tid;
        const float val = (tid < p.head_dim) ? (q_raw[off] * q_scale[ulong(hq) * p.head_dim + tid]) : 0.0f;
        const float s   = simd_sum(val * val);
        if (lane == 0u) tg_sq[sgid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps)) * tau * p.attn_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < p.head_dim) q_hm[hm_off] = sik_clamp_f16(val * tg_sq[0]);
    } else if (idx < b2) {
        if (m >= p.M_valid + p.kv_zero_tail) return;
        const uint  hkv    = idx - b1;
        const ulong hs_idx = ulong(hkv) * p.kv_stride + slot;
        const bool  is_q8  = (p.kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0 && p.is_local == 0u);
        if (m >= p.M_valid) {
            if (is_q8) {
                if (tid < p.head_dim) reinterpret_cast<device char*>(k_cache)[hs_idx * p.head_dim + tid] = char(0);
                if (tid == 0u) k_scales[hs_idx] = 0.0h;
            } else if (tid < p.head_dim) {
                k_cache[hs_idx * p.head_dim + tid] = 0.0h;
            }
            return;
        }
        const ulong off = (ulong(m) * p.n_kv + hkv) * p.head_dim + tid;
        const float val = (tid < p.head_dim) ? k_conv[off] : 0.0f;
        const float s   = simd_sum(val * val);
        if (lane == 0u) tg_sq[sgid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float k_fac = tg_sq[0];
        if (is_q8) {
            sik_store_kv_q8_head(val * k_fac, reinterpret_cast<device char*>(k_cache), k_scales,
                                 hs_idx, p.head_dim, tg_sq, tid, sgid, lane);
        } else if (tid < p.head_dim) {
            k_cache[hs_idx * p.head_dim + tid] = sik_clamp_f16(val * k_fac);
        }
    } else if (idx < b3) {
        if (m >= p.M_valid + p.kv_zero_tail) return;
        const uint  hkv    = idx - b2;
        const ulong hs_idx = ulong(hkv) * p.kv_stride + slot;
        const bool  is_q8  = (p.kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0 && p.is_local == 0u);
        if (m >= p.M_valid) {
            if (is_q8) {
                if (tid < p.head_dim) reinterpret_cast<device char*>(v_cache)[hs_idx * p.head_dim + tid] = char(0);
                if (tid == 0u) v_scales[hs_idx] = 0.0h;
            } else if (tid < p.head_dim) {
                v_cache[hs_idx * p.head_dim + tid] = 0.0h;
            }
            return;
        }
        const ulong off = (ulong(m) * p.n_kv + hkv) * p.head_dim + tid;
        const float val = (tid < p.head_dim) ? v_conv[off] : 0.0f;
        const float s   = simd_sum(val * val);
        if (lane == 0u) tg_sq[sgid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.head_dim) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float v_fac = tg_sq[0];
        if (is_q8) {
            sik_store_kv_q8_head(val * v_fac, reinterpret_cast<device char*>(v_cache), v_scales,
                                 hs_idx, p.head_dim, tg_sq, tid, sgid, lane);
        } else if (tid < p.head_dim) {
            v_cache[hs_idx * p.head_dim + tid] = sik_clamp_f16(val * v_fac);
        }
    } else if (idx < b3 + p.n_q) {
        if (sgid != 0u || m >= p.M_pad16) return;
        const uint  hq     = idx - b3;
        const ulong hm_off = (ulong(hq) * p.M_pad16 + m) * p.d_rel + lane;
        if (m >= p.M_valid) {
            if (lane < p.d_rel) r_hm[hm_off] = 0.0f;
            return;
        }
        const ulong off = (ulong(m) * p.n_q + hq) * p.d_rel + lane;
        const float val = (lane < p.d_rel) ? (r_raw[off] * r_scale[ulong(hq) * p.d_rel + lane]) : 0.0f;
        const float tot = simd_sum(val * val);
        const float fac = min(1.0f, p.rel_clip / (sqrt(tot) + p.eps)) * tau;
        if (lane < p.d_rel) r_hm[hm_off] = val * fac;
    }
}

struct SikAttnParams {
    uint  M_valid;
    uint  M_pad16;
    uint  n_q;           // 24
    uint  n_kv;          // 8
    uint  head_dim;      // 128
    uint  d_rel;         // 16
    uint  rel_extent;    // 1024
    uint  swa_win;       // 1024
    uint  is_local;      // 1 => SWA-1024 ring; 0 => Global full linear
    uint  seq_pos_start;
    uint  m_offset;      // output offset in part_m/l/o (0 normally)
    uint  kv_stride;     // 2048 (SWA) or max_seq_alloc (Global)
    uint  max_splits;    // 36
    uint  chunk_size;    // aligned KV chunk size B
    uint  c_batch_first; // first absolute chunk index for this dispatch
    float attn_scale;    // 1.0f / sqrt(128.0f)
};

// Deterministic absolute-position chunk size: GUARANTEES 100.0000% bit-exact `K>0 == K=0`
// and `my_splits <= 32 <= kMaxDecodeSplits (36)` across full native 1M (`1,048,576`) context!
inline uint sik_attn_chunk_size(uint q_pos, uint is_local) {
    if (q_pos < 160u)    return 16u;
    if (q_pos < 512u)    return 32u;
    if (is_local != 0u)  return 64u;
    if (q_pos < 2048u)   return 64u;
    if (q_pos < 4096u)   return 128u;
    if (q_pos < 8192u)   return 256u;
    if (q_pos < 16384u)  return 512u;
    if (q_pos < 32768u)  return 1024u;
    if (q_pos < 65536u)  return 2048u;
    if (q_pos < 262144u) return 8192u;
    if (q_pos < 524288u) return 16384u;
    return 32768u;
}

// Single-Pass Zero-Split GQA-3 Direct-SIMT Decode Attention (`q_pos < 160`, F16 KV, 0 splits, 0 reduce dispatch!)
// Fuses virtual sink + sigmoid output gate + `wo_xsum` (`Grid(8, mb, 1)`, `256` th/TG = 8 SGs/TG, 2.6-2.9x faster!)
kernel void sik_attn_decode_single_gqa3_sg8(
    device const float4*    q_in     [[buffer(0)]],
    device const float*     r_in     [[buffer(1)]],
    device const float*     rel_proj [[buffer(2)]],
    device const half4*     k_cache  [[buffer(3)]],
    device const half4*     v_cache  [[buffer(4)]],
    device const float*     sink_in  [[buffer(5)]],
    device const float4*    gate_in  [[buffer(6)]],
    device half4*           out_f16  [[buffer(7)]],
    device float*           wo_xsum  [[buffer(8)]],
    constant SikAttnParams& p        [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint hkv = tgid.x;
    const uint m   = tgid.y;
    if (hkv >= p.n_kv || m >= p.M_valid) return;

    const uint gm = p.m_offset + m;
    const uint qp = p.seq_pos_start + m;
    const uint ws = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;

    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        const ulong qrow = ulong(gm) * p.n_q + (hkv * 3u + h);
        qv[h] = q_in[qrow * 32ul + lane] * p.attn_scale;
        rl[h] = (lane < p.d_rel) ? r_in[qrow * p.d_rel + lane] : 0.0f;
    }

    device const half4* kh = k_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* vh = v_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    const uint mask = p.kv_stride - 1u;

    for (uint k = ws + sgid; k <= qp; k += 8u) {
        const uint   slot    = (p.is_local != 0u) ? (k & mask) : k;
        const ulong  off     = ulong(slot) * 32ul + lane;
        const float4 kv      = float4(kh[off]);
        const float4 vv      = float4(vh[off]);
        const uint   dist    = qp - k;
        const bool   has_rel = (dist < p.rel_extent);
        const float  relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    threadgroup float  tg_m[3][8], tg_l[3][8];
    threadgroup float4 tg_o[3][8][32];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        if (lane == 0u) { tg_m[h][sgid] = mv[h]; tg_l[h][sgid] = lv[h]; }
        tg_o[h][sgid][lane] = ov[h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid < 3u) {
        const uint  h      = sgid;
        const uint  hq     = hkv * 3u + h;
        const float sink_v = sink_in[ulong(gm) * p.n_q + hq];
        float m_max = sink_v;
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) m_max = max(m_max, tg_m[h][s]);
        float  l_tot = exp(sink_v - m_max);
        float4 o_tot = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) {
            if (tg_m[h][s] > -INFINITY) {
                const float w = exp(tg_m[h][s] - m_max);
                l_tot = fma(tg_l[h][s], w, l_tot);
                o_tot = fma(tg_o[h][s][lane], float4(w), o_tot);
            }
        }
        const float  inv_l   = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
        const ulong  out_off = (ulong(gm) * p.n_q + hq) * 32ul + lane;
        const float4 gv      = gate_in[out_off];
        const float4 sig_g   = float4(1.0f) / (float4(1.0f) + exp(-gv));
        const half4  oh      = sik_clamp_f16x4((o_tot * inv_l) * sig_g);
        out_f16[out_off]     = oh;
        if (gm < 8u) {
            const float bs = sik_reduce16((float(oh.x) + float(oh.y)) + (float(oh.z) + float(oh.w)));
            if ((lane & 15u) == 0u) {
                wo_xsum[ulong(gm) * (p.n_q * 2u) + hq * 2u + (lane >> 4u)] = bs;
            }
        }
    }
}

struct SikSconvAttnM1Params {
    uint  n_q;
    uint  n_kv;
    uint  head_dim;
    uint  d_rel;
    uint  rel_extent;
    uint  swa_win;
    uint  seq_pos;
    uint  kv_stride;
    uint  is_local;
    uint  v_state_off_f4;
    float qk_clip;
    float rel_clip;
    float eps;
    float log_alpha;
    float log_n_floor;
    float attn_scale;
    uint  chunk_size;
    uint  c_batch_first;
    uint  max_splits;
    uint  pad0;
};

// FUSED 1-DISPATCH `SConvKV + QKVR-Prep + Single-Pass GQA-3 Attention + Sink + Gate + wo_xsum` (`M == 1`, F16 KV, `pos < 160`)!
// Eliminates ALL 48 `sik_sconv_kv_qkvr_prep` dispatches AND `b_q_f32`/`b_r_f32` DRAM roundtrips (`-0.28 ms`), 100% bit-exact with M=2..8!
// Grid: `(8 /* n_kv */, 1, 1)`, `256` threads/TG (`8` SGs: SG0=K, SG1=V, SG2..4=Q0..2, SG5..7=R0..2 -> barrier -> 8-SG GQA-3 attention!)
kernel void sik_sconv_attn_single_gqa3_m1(
    device const float*           q_raw      [[buffer(0)]],
    device const float*           k_raw      [[buffer(1)]],
    device const float*           v_raw      [[buffer(2)]],
    device const float*           r_raw      [[buffer(3)]],
    device const float4*          k_sconv_w  [[buffer(4)]],
    device const float4*          v_sconv_w  [[buffer(5)]],
    device const float4*          k_dyn      [[buffer(6)]],
    device const float4*          v_dyn      [[buffer(7)]],
    device float4*                conv_state [[buffer(8)]],
    device const float*           q_scale    [[buffer(9)]],
    device const float*           r_scale    [[buffer(10)]],
    device const float*           rel_proj   [[buffer(11)]],
    device half4*                 k_cache    [[buffer(12)]],
    device half4*                 v_cache    [[buffer(13)]],
    device const float*           sink_in    [[buffer(14)]],
    device const float4*          gate_in    [[buffer(15)]],
    device half4*                 out_f16    [[buffer(16)]],
    device float*                 wo_xsum    [[buffer(17)]],
    constant SikSconvAttnM1Params& p         [[buffer(18)]],
    uint hkv  [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (hkv >= p.n_kv) return;
    const uint  qp   = p.seq_pos;
    const uint  slot = (p.is_local != 0u) ? (qp & (p.kv_stride - 1u)) : qp;
    const float tau  = (p.is_local == 0u && p.log_n_floor > 0.0f)
        ? (1.0f + p.log_alpha * log(max(1.0f, float(qp + 1u) / p.log_n_floor))) : 1.0f;

    threadgroup float4 tg_q[3][32];
    threadgroup float  tg_r[3][32];
    threadgroup half4  tg_k_new[32];
    threadgroup half4  tg_v_new[32];

    if (sgid == 0u) {
        // K head SConv + L2-clip + store to k_cache & tg_k_new (100% bit-exact with sik_sconv_kv_qkvr_prep!)
        const uint   c0  = hkv * 128u + lane * 4u;
        const float4 wt0 = k_sconv_w[c0 + 0u], wt1 = k_sconv_w[c0 + 1u];
        const float4 wt2 = k_sconv_w[c0 + 2u], wt3 = k_sconv_w[c0 + 3u];
        const float4 st0 = conv_state[c0 + 0u], st1 = conv_state[c0 + 1u];
        const float4 st2 = conv_state[c0 + 2u], st3 = conv_state[c0 + 3u];
        const float4 cur = reinterpret_cast<device const float4*>(k_raw)[ulong(hkv) * 32ul + lane];
        float4 v4;
        if (p.is_local != 0u) {
            const float inv_s = 0.4472135955f;
            v4.x = (fma(st0.x, wt0.x, fma(st0.y, wt0.y, fma(st0.z, wt0.z, cur.x * wt0.w))) + cur.x) * inv_s;
            v4.y = (fma(st1.x, wt1.x, fma(st1.y, wt1.y, fma(st1.z, wt1.z, cur.y * wt1.w))) + cur.y) * inv_s;
            v4.z = (fma(st2.x, wt2.x, fma(st2.y, wt2.y, fma(st2.z, wt2.z, cur.z * wt2.w))) + cur.z) * inv_s;
            v4.w = (fma(st3.x, wt3.x, fma(st3.y, wt3.y, fma(st3.z, wt3.z, cur.w * wt3.w))) + cur.w) * inv_s;
        } else {
            const float4 dyn  = k_dyn[hkv];
            const float4 t0   = wt0 + dyn, t1 = wt1 + dyn, t2 = wt2 + dyn, t3 = wt3 + dyn;
            const float  inv3 = 1.0f / 3.0f;
            v4.x = (fma(cur.x, t0.x, fma(st0.z, t0.y, fma(st0.y, t0.z, st0.x * t0.w))) + cur.x) * inv3;
            v4.y = (fma(cur.y, t1.x, fma(st1.z, t1.y, fma(st1.y, t1.z, st1.x * t1.w))) + cur.y) * inv3;
            v4.z = (fma(cur.z, t2.x, fma(st2.z, t2.y, fma(st2.y, t2.z, st2.x * t2.w))) + cur.z) * inv3;
            v4.w = (fma(cur.w, t3.x, fma(st3.z, t3.y, fma(st3.y, t3.z, st3.x * t3.w))) + cur.w) * inv3;
        }
        conv_state[c0 + 0u] = float4(st0.y, st0.z, cur.x, 0.0f);
        conv_state[c0 + 1u] = float4(st1.y, st1.z, cur.y, 0.0f);
        conv_state[c0 + 2u] = float4(st2.y, st2.z, cur.z, 0.0f);
        conv_state[c0 + 3u] = float4(st3.y, st3.z, cur.w, 0.0f);
        const float tot   = simd_sum(dot(v4, v4));
        const float k_fac = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps));
        const half4 kh4   = sik_clamp_f16x4(v4 * k_fac);
        k_cache[(ulong(hkv) * p.kv_stride + slot) * 32ul + lane] = kh4;
        tg_k_new[lane] = kh4;
    } else if (sgid == 1u) {
        // V head SConv + RMSNorm + store to v_cache & tg_v_new (100% bit-exact!)
        const uint   c0    = hkv * 128u + lane * 4u;
        const ulong  s_off = ulong(p.v_state_off_f4);
        const float4 wt0 = v_sconv_w[c0 + 0u], wt1 = v_sconv_w[c0 + 1u];
        const float4 wt2 = v_sconv_w[c0 + 2u], wt3 = v_sconv_w[c0 + 3u];
        const float4 st0 = conv_state[s_off + c0 + 0u], st1 = conv_state[s_off + c0 + 1u];
        const float4 st2 = conv_state[s_off + c0 + 2u], st3 = conv_state[s_off + c0 + 3u];
        const float4 cur = reinterpret_cast<device const float4*>(v_raw)[ulong(hkv) * 32ul + lane];
        float4 v4;
        if (p.is_local != 0u) {
            const float inv_s = 0.4472135955f;
            v4.x = (fma(st0.x, wt0.x, fma(st0.y, wt0.y, fma(st0.z, wt0.z, cur.x * wt0.w))) + cur.x) * inv_s;
            v4.y = (fma(st1.x, wt1.x, fma(st1.y, wt1.y, fma(st1.z, wt1.z, cur.y * wt1.w))) + cur.y) * inv_s;
            v4.z = (fma(st2.x, wt2.x, fma(st2.y, wt2.y, fma(st2.z, wt2.z, cur.z * wt2.w))) + cur.z) * inv_s;
            v4.w = (fma(st3.x, wt3.x, fma(st3.y, wt3.y, fma(st3.z, wt3.z, cur.w * wt3.w))) + cur.w) * inv_s;
        } else {
            const float4 dyn  = v_dyn[hkv];
            const float4 t0   = wt0 + dyn, t1 = wt1 + dyn, t2 = wt2 + dyn, t3 = wt3 + dyn;
            const float  inv3 = 1.0f / 3.0f;
            v4.x = (fma(cur.x, t0.x, fma(st0.z, t0.y, fma(st0.y, t0.z, st0.x * t0.w))) + cur.x) * inv3;
            v4.y = (fma(cur.y, t1.x, fma(st1.z, t1.y, fma(st1.y, t1.z, st1.x * t1.w))) + cur.y) * inv3;
            v4.z = (fma(cur.z, t2.x, fma(st2.z, t2.y, fma(st2.y, t2.z, st2.x * t2.w))) + cur.z) * inv3;
            v4.w = (fma(cur.w, t3.x, fma(st3.z, t3.y, fma(st3.y, t3.z, st3.x * t3.w))) + cur.w) * inv3;
        }
        conv_state[s_off + c0 + 0u] = float4(st0.y, st0.z, cur.x, 0.0f);
        conv_state[s_off + c0 + 1u] = float4(st1.y, st1.z, cur.y, 0.0f);
        conv_state[s_off + c0 + 2u] = float4(st2.y, st2.z, cur.z, 0.0f);
        conv_state[s_off + c0 + 3u] = float4(st3.y, st3.z, cur.w, 0.0f);
        const float tot   = simd_sum(dot(v4, v4));
        const float v_fac = rsqrt(tot / float(p.head_dim) + p.eps);
        const half4 vh4   = sik_clamp_f16x4(v4 * v_fac);
        v_cache[(ulong(hkv) * p.kv_stride + slot) * 32ul + lane] = vh4;
        tg_v_new[lane] = vh4;
    } else if (sgid < 5u) {
        // Q heads h = 0, 1, 2
        const uint   h    = sgid - 2u;
        const uint   hq   = hkv * 3u + h;
        const ulong  off4 = ulong(hq) * 32ul + lane;
        const float4 qv   = reinterpret_cast<device const float4*>(q_raw)[off4]
                          * reinterpret_cast<device const float4*>(q_scale)[off4];
        const float  tot  = simd_sum(dot(qv, qv));
        const float  fac  = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps)) * tau;
        tg_q[h][lane]     = (qv * fac) * p.attn_scale;
    } else {
        // R heads h = 0, 1, 2
        const uint  h   = sgid - 5u;
        const uint  hq  = hkv * 3u + h;
        const ulong off = ulong(hq) * p.d_rel + lane;
        const float val = (lane < p.d_rel) ? (r_raw[off] * r_scale[off]) : 0.0f;
        const float tot = simd_sum(val * val);
        const float fac = min(1.0f, p.rel_clip / (sqrt(tot) + p.eps)) * tau;
        tg_r[h][lane]   = (lane < p.d_rel) ? (val * fac) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Exact same 8-SG GQA-3 Single-Pass Attention as `sik_attn_decode_single_gqa3_sg8`!
    const uint ws = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        qv[h] = tg_q[h][lane];
        rl[h] = tg_r[h][lane];
    }

    device const half4* kh = k_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* vh = v_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    const uint mask = p.kv_stride - 1u;

    for (uint k = ws + sgid; k <= qp; k += 8u) {
        float4 kv, vv;
        if (k == qp) {
            kv = float4(tg_k_new[lane]);
            vv = float4(tg_v_new[lane]);
        } else {
            const uint  kslot = (p.is_local != 0u) ? (k & mask) : k;
            const ulong off   = ulong(kslot) * 32ul + lane;
            kv = float4(kh[off]);
            vv = float4(vh[off]);
        }
        const uint  dist    = qp - k;
        const bool  has_rel = (dist < p.rel_extent);
        const float relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    threadgroup float  tg_m[3][8], tg_l[3][8];
    threadgroup float4 tg_o[3][8][32];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        if (lane == 0u) { tg_m[h][sgid] = mv[h]; tg_l[h][sgid] = lv[h]; }
        tg_o[h][sgid][lane] = ov[h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid < 3u) {
        const uint  h      = sgid;
        const uint  hq     = hkv * 3u + h;
        const float sink_v = sink_in[hq];
        float m_max = sink_v;
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) m_max = max(m_max, tg_m[h][s]);
        float  l_tot = exp(sink_v - m_max);
        float4 o_tot = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) {
            if (tg_m[h][s] > -INFINITY) {
                const float w = exp(tg_m[h][s] - m_max);
                l_tot = fma(tg_l[h][s], w, l_tot);
                o_tot = fma(tg_o[h][s][lane], float4(w), o_tot);
            }
        }
        const float  inv_l   = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
        const ulong  out_off = ulong(hq) * 32ul + lane;
        const float4 gv      = gate_in[out_off];
        const float4 sig_g   = float4(1.0f) / (float4(1.0f) + exp(-gv));
        const half4  oh      = sik_clamp_f16x4((o_tot * inv_l) * sig_g);
        out_f16[out_off]     = oh;
        const float  bs      = sik_reduce16((float(oh.x) + float(oh.y)) + (float(oh.z) + float(oh.w)));
        if ((lane & 15u) == 0u) {
            wo_xsum[hq * 2u + (lane >> 4u)] = bs;
        }
    }
}

// FUSED `SConvKV + QKVR-Prep + Split-KV GQA-3 Attention` (`M == 1`, F16 KV, `pos >= 256` -> ALL context lengths up to 131K!)
// Ultra-high-occupancy 2.5 KiB aliased TG mem + rolled 8-reg SConv + `device` rel_proj + 1-exp softmax + 2-tok KV prefetch!
// 100% race-free & 100.0000% bit-exact with `sik_attn_decode_split_gqa3_reg` (`M=2..8`)!
// Grid: `(n_chunks, 8 /* n_kv */, 1)`, `128` threads/TG (`4` SGs)
kernel void sik_sconv_attn_split_gqa3_m1(
    device const float*            q_raw      [[buffer(0)]],
    device const float*            k_raw      [[buffer(1)]],
    device const float*            v_raw      [[buffer(2)]],
    device const float*            r_raw      [[buffer(3)]],
    device const float4*           k_sconv_w  [[buffer(4)]],
    device const float4*           v_sconv_w  [[buffer(5)]],
    device const float4*           k_dyn      [[buffer(6)]],
    device const float4*           v_dyn      [[buffer(7)]],
    device float4*                 conv_state [[buffer(8)]],
    device const float*            q_scale    [[buffer(9)]],
    device const float*            r_scale    [[buffer(10)]],
    device const float*            rel_proj   [[buffer(11)]],
    device half4*                  k_cache    [[buffer(12)]],
    device half4*                  v_cache    [[buffer(13)]],
    device float*                  part_m     [[buffer(14)]],
    device float*                  part_l     [[buffer(15)]],
    device float4*                 part_o     [[buffer(16)]],
    constant SikSconvAttnM1Params& p          [[buffer(17)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint c   = p.c_batch_first + tgid.x;
    const uint hkv = tgid.y;
    if (hkv >= p.n_kv) return;

    const uint B      = p.chunk_size;
    const uint qp     = p.seq_pos;
    const uint ws     = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
    const uint c_last = qp / B;
    if (c < (ws / B) || c > c_last) return;

    const float tau = (p.is_local == 0u && p.log_n_floor > 0.0f)
        ? (1.0f + p.log_alpha * log(max(1.0f, float(qp + 1u) / p.log_n_floor))) : 1.0f;

    // 2.5 KiB total TG memory (down from 8.5 KiB!): `tg_buf[4][32]` reused for Phase 1 Q/R AND Phase 2 epilogue!
    threadgroup float4 tg_buf[4][32];
    threadgroup half4  tg_k_new[32];
    threadgroup half4  tg_v_new[32];
    threadgroup float  tg_m[4], tg_l[4];

    if (sgid < 3u) {
        const uint   hq   = hkv * 3u + sgid;
        const ulong  off4 = ulong(hq) * 32ul + lane;
        const float4 qv   = reinterpret_cast<device const float4*>(q_raw)[off4]
                          * reinterpret_cast<device const float4*>(q_scale)[off4];
        const float  tot  = simd_sum(dot(qv, qv));
        const float  fac  = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps)) * tau;
        tg_buf[sgid][lane] = (qv * fac) * p.attn_scale;
    } else {
        float4 rpack(0.0f);
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            const uint  hq  = hkv * 3u + h;
            const ulong off = ulong(hq) * p.d_rel + lane;
            const float val = (lane < p.d_rel) ? (r_raw[off] * r_scale[off]) : 0.0f;
            const float tot = simd_sum(val * val);
            const float fac = min(1.0f, p.rel_clip / (sqrt(tot) + p.eps)) * tau;
            rpack[h]        = (lane < p.d_rel) ? (val * fac) : 0.0f;
        }
        tg_buf[3][lane] = rpack;
    }

    // ONLY the unique last chunk `c == c_last` writes K/V SConv state & KV cache slot `qp` (compact rolled loop -> zero reg bloat!)
    if (c == c_last && sgid < 2u) {
        const uint   slot  = (p.is_local != 0u) ? (qp & (p.kv_stride - 1u)) : qp;
        const uint   c0    = hkv * 128u + lane * 4u;
        const bool   is_v  = (sgid == 1u);
        const ulong  s_off = is_v ? ulong(p.v_state_off_f4) : 0ul;
        device const float4* sw_p  = is_v ? v_sconv_w : k_sconv_w;
        device const float4* raw_p = reinterpret_cast<device const float4*>(is_v ? v_raw : k_raw);
        const float4 cur = raw_p[ulong(hkv) * 32ul + lane];
        const float4 dyn = (p.is_local != 0u) ? float4(0.0f) : (is_v ? v_dyn[hkv] : k_dyn[hkv]);
        float4 v4;
        #pragma clang loop unroll(disable)
        for (uint u = 0u; u < 4u; ++u) {
            const float4 wt = sw_p[c0 + u];
            const float4 st = conv_state[s_off + c0 + u];
            const float  cv = cur[u];
            if (p.is_local != 0u) {
                v4[u] = (fma(st.x, wt.x, fma(st.y, wt.y, fma(st.z, wt.z, cv * wt.w))) + cv) * 0.4472135955f;
            } else {
                const float4 t = wt + dyn;
                v4[u] = (fma(cv, t.x, fma(st.z, t.y, fma(st.y, t.z, st.x * t.w))) + cv) * (1.0f / 3.0f);
            }
            conv_state[s_off + c0 + u] = float4(st.y, st.z, cv, 0.0f);
        }
        const float tot = simd_sum(dot(v4, v4));
        const float fac = is_v ? rsqrt(tot / float(p.head_dim) + p.eps)
                               : min(1.0f, p.qk_clip / (sqrt(tot) + p.eps));
        const half4 h4  = sik_clamp_f16x4(v4 * fac);
        if (is_v) {
            v_cache[(ulong(hkv) * p.kv_stride + slot) * 32ul + lane] = h4;
            tg_v_new[lane] = h4;
        } else {
            k_cache[(ulong(hkv) * p.kv_stride + slot) * 32ul + lane] = h4;
            tg_k_new[lane] = h4;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Exact same 4-SG GQA-3 Split-KV loop as `sik_attn_decode_split_gqa3_reg`!
    const uint kv_beg = c * B;
    const uint kv_end = min(kv_beg + B, qp + 1u);
    const uint kv_lo  = max(kv_beg, ws);

    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    const float4 rpack = tg_buf[3][lane];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        qv[h] = tg_buf[h][lane];
        rl[h] = rpack[h];
    }

    device const half4* kh = k_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* vh = v_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    const uint mask = p.kv_stride - 1u;

    uint k = kv_beg + sgid;
    // Fast 2-token prefetched inner loop when full chunk is inside window & below `qp` (`~95%` of chunks!)
    if (kv_beg >= kv_lo && c != c_last) {
        for (; k + 4u < kv_end; k += 8u) {
            const uint   s0   = (p.is_local != 0u) ? (k & mask) : k;
            const uint   s1   = (p.is_local != 0u) ? ((k + 4u) & mask) : (k + 4u);
            const ulong  o0   = ulong(s0) * 32ul + lane;
            const ulong  o1   = ulong(s1) * 32ul + lane;
            const float4 kv0  = float4(kh[o0]), vv0 = float4(vh[o0]);
            const float4 kv1  = float4(kh[o1]), vv1 = float4(vh[o1]);
            const uint   d0   = qp - k, d1 = d0 - 4u;
            const bool   hr0  = (d0 < p.rel_extent), hr1 = (d1 < p.rel_extent);
            const float  rw0  = (hr0 && lane < p.d_rel) ? rel_proj[ulong(d0) * p.d_rel + lane] : 0.0f;
            const float  rw1  = (hr1 && lane < p.d_rel) ? rel_proj[ulong(d1) * p.d_rel + lane] : 0.0f;
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 3u; ++h) {
                float p0 = dot(qv[h], kv0);
                if (hr0 && lane < p.d_rel) p0 = fma(rl[h], rw0, p0);
                const float sc0 = simd_sum(p0);
                const float mn0 = max(mv[h], sc0);
                const float al0 = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn0);
                const float w0  = exp(sc0 - mn0);
                mv[h] = mn0; lv[h] = fma(lv[h], al0, w0); ov[h] = fma(ov[h], al0, w0 * vv0);

                float p1 = dot(qv[h], kv1);
                if (hr1 && lane < p.d_rel) p1 = fma(rl[h], rw1, p1);
                const float sc1 = simd_sum(p1);
                const float mn1 = max(mv[h], sc1);
                const float al1 = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn1);
                const float w1  = exp(sc1 - mn1);
                mv[h] = mn1; lv[h] = fma(lv[h], al1, w1); ov[h] = fma(ov[h], al1, w1 * vv1);
            }
        }
    }
    for (; k < kv_end; k += 4u) {
        if (k < kv_lo) continue;
        float4 kv, vv;
        if (k == qp) {
            kv = float4(tg_k_new[lane]);
            vv = float4(tg_v_new[lane]);
        } else {
            const uint  kslot = (p.is_local != 0u) ? (k & mask) : k;
            const ulong off   = ulong(kslot) * 32ul + lane;
            kv = float4(kh[off]);
            vv = float4(vh[off]);
        }
        const uint  dist    = qp - k;
        const bool  has_rel = (dist < p.rel_extent);
        const float relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    const uint s_slot = c - (ws / B);
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) { tg_m[sgid] = mv[h]; tg_l[sgid] = lv[h]; }
        tg_buf[sgid][lane] = ov[h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float mm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
            float  ll = 0.0f;
            float4 oo(0.0f);
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 4u; ++s) {
                if (tg_m[s] > -INFINITY) {
                    const float w = exp(tg_m[s] - mm);
                    ll = fma(tg_l[s], w, ll);
                    oo = fma(tg_buf[s][lane], float4(w), oo);
                }
            }
            const uint  hq  = hkv * 3u + h;
            const ulong idx = ulong(hq) * p.max_splits + s_slot;
            if (lane == 0u) { part_m[idx] = mm; part_l[idx] = ll; }
            part_o[idx * 32ul + lane] = oo;
        }
    }
}

// 2 KiB TG-Mem Register-Direct GQA-3 Split-KV Decode Attention (`q_pos >= 256`, F16 KV, `M_SUB = 1`!):
// 100.0000% bit-exact with `sik_sconv_attn_split_gqa3_m1` (`M = 1..8`)!
// Grid: `(n_chunks, 8 /* n_kv */, mb)`, `128` threads/TG (`4` SGs)
kernel void sik_attn_decode_split_gqa3_reg(
    device const float4*    q_in     [[buffer(0)]],
    device const float*     r_in     [[buffer(1)]],
    device const float*     rel_proj [[buffer(2)]],
    device const half4*     k_cache  [[buffer(3)]],
    device const half4*     v_cache  [[buffer(4)]],
    device float*           part_m   [[buffer(5)]],
    device float*           part_l   [[buffer(6)]],
    device float4*          part_o   [[buffer(7)]],
    constant SikAttnParams& p        [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint c   = p.c_batch_first + tgid.x;
    const uint hkv = tgid.y;
    const uint m   = tgid.z;
    if (hkv >= p.n_kv || m >= p.M_valid) return;

    const uint B  = p.chunk_size;
    const uint gm = p.m_offset + m;
    const uint qp = p.seq_pos_start + m;
    const uint ws = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
    const uint c_last = qp / B;
    if (c < (ws / B) || c > c_last) return;

    const uint kv_beg = c * B;
    const uint kv_end = min(kv_beg + B, qp + 1u);
    const uint kv_lo  = max(kv_beg, ws);

    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        const ulong qrow = ulong(gm) * p.n_q + (hkv * 3u + h);
        qv[h] = q_in[qrow * 32ul + lane] * p.attn_scale;
        rl[h] = (lane < p.d_rel) ? r_in[qrow * p.d_rel + lane] : 0.0f;
    }

    device const half4* kh = k_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* vh = v_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    const uint mask = p.kv_stride - 1u;

    uint k = kv_beg + sgid;
    if (kv_beg >= kv_lo && c != c_last) {
        for (; k + 4u < kv_end; k += 8u) {
            const uint   s0   = (p.is_local != 0u) ? (k & mask) : k;
            const uint   s1   = (p.is_local != 0u) ? ((k + 4u) & mask) : (k + 4u);
            const ulong  o0   = ulong(s0) * 32ul + lane;
            const ulong  o1   = ulong(s1) * 32ul + lane;
            const float4 kv0  = float4(kh[o0]), vv0 = float4(vh[o0]);
            const float4 kv1  = float4(kh[o1]), vv1 = float4(vh[o1]);
            const uint   d0   = qp - k, d1 = d0 - 4u;
            const bool   hr0  = (d0 < p.rel_extent), hr1 = (d1 < p.rel_extent);
            const float  rw0  = (hr0 && lane < p.d_rel) ? rel_proj[ulong(d0) * p.d_rel + lane] : 0.0f;
            const float  rw1  = (hr1 && lane < p.d_rel) ? rel_proj[ulong(d1) * p.d_rel + lane] : 0.0f;
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 3u; ++h) {
                float p0 = dot(qv[h], kv0);
                if (hr0 && lane < p.d_rel) p0 = fma(rl[h], rw0, p0);
                const float sc0 = simd_sum(p0);
                const float mn0 = max(mv[h], sc0);
                const float al0 = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn0);
                const float w0  = exp(sc0 - mn0);
                mv[h] = mn0; lv[h] = fma(lv[h], al0, w0); ov[h] = fma(ov[h], al0, w0 * vv0);

                float p1 = dot(qv[h], kv1);
                if (hr1 && lane < p.d_rel) p1 = fma(rl[h], rw1, p1);
                const float sc1 = simd_sum(p1);
                const float mn1 = max(mv[h], sc1);
                const float al1 = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn1);
                const float w1  = exp(sc1 - mn1);
                mv[h] = mn1; lv[h] = fma(lv[h], al1, w1); ov[h] = fma(ov[h], al1, w1 * vv1);
            }
        }
    }
    for (; k < kv_end; k += 4u) {
        if (k < kv_lo) continue;
        const uint   slot    = (p.is_local != 0u) ? (k & mask) : k;
        const ulong  off     = ulong(slot) * 32ul + lane;
        const float4 kv      = float4(kh[off]);
        const float4 vv      = float4(vh[off]);
        const uint   dist    = qp - k;
        const bool   has_rel = (dist < p.rel_extent);
        const float  relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    threadgroup float  tg_m[4], tg_l[4];
    threadgroup float4 tg_o[4][32];
    const uint s_slot = c - (ws / B);
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        if (h > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) { tg_m[sgid] = mv[h]; tg_l[sgid] = lv[h]; }
        tg_o[sgid][lane] = ov[h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float mm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
            float  ll = 0.0f;
            float4 oo(0.0f);
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 4u; ++s) {
                if (tg_m[s] > -INFINITY) {
                    const float w = exp(tg_m[s] - mm);
                    ll = fma(tg_l[s], w, ll);
                    oo = fma(tg_o[s][lane], float4(w), oo);
                }
            }
            const uint  hq  = hkv * 3u + h;
            const ulong idx = (ulong(gm) * p.n_q + hq) * p.max_splits + s_slot;
            if (lane == 0u) { part_m[idx] = mm; part_l[idx] = ll; }
            part_o[idx * 32ul + lane] = oo;
        }
    }
}

// High-Speed Zero-Spill Block-16 Two-Pass Split-KV Decode (`M_SUB <= 2` along `tgid.z`):
//   - Fused single `simd_sum(dot(q, k) + r * rel)` (2x fewer reductions!)
//   - Two-pass Block-16 tile softmax (16x fewer o_vec rescalings, full SFU exp() pipelining!)
//   - Zero register spilling (`M_SUB <= 2`) -> 0.30 ms @ pos=108 M=1, 0.51 ms @ M=5!
//   - 100.0000% BIT-EXACT `K>0 == K=0`!
// Grid: `(num_batch_chunks, 8 /* n_kv */, M_groups)`, `96` threads/TG (`3` SGs = `3` Q heads/KV head!)
template <uint M_SUB, bool IS_Q8>
inline void sik_attn_decode_splitkv_sub_impl(
    device const float4*    q_in,
    device const float*     r_in,
    device const float*     rel_proj,
    device const uchar*     k_cache_raw,
    device const uchar*     v_cache_raw,
    device const half*      k_scales,
    device const half*      v_scales,
    device float*           part_m,
    device float*           part_l,
    device float4*          part_o,
    constant SikAttnParams& p,
    threadgroup half4       (&tg_k)[16][32],
    threadgroup half4       (&tg_v)[16][32],
    threadgroup float       (&tg_rel)[24][16],
    uint3                   tgid,
    uint                    sgid,
    uint                    lane
) {
    const uint c   = p.c_batch_first + tgid.x;
    const uint hkv = tgid.y;
    const uint m0  = tgid.z * M_SUB;
    if (hkv >= p.n_kv || m0 >= p.M_valid) return;

    const uint mb           = min(M_SUB, p.M_valid - m0);
    const uint B            = p.chunk_size;
    const uint kv_chunk_beg = c * B;
    const uint kv_chunk_end = kv_chunk_beg + B;
    const uint hq           = hkv * 3u + sgid;

    float4 q_vec[M_SUB];
    float  r_lane[M_SUB];
    float  m_val[M_SUB];
    float  l_val[M_SUB];
    float4 o_vec[M_SUB];

    uint active_mask = 0u;
    uint kv_min_beg  = kv_chunk_end;
    uint kv_max_end  = kv_chunk_beg;

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_SUB; ++m) {
        const uint gm = min(m0 + m, p.M_valid - 1u);
        const uint qp = p.seq_pos_start + gm;
        const uint ws = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
        const uint cf = ws / B;
        const uint cl = qp / B;
        if (m < mb && c >= cf && c <= cl) {
            active_mask |= (1u << m);
            kv_min_beg = min(kv_min_beg, max(kv_chunk_beg, ws));
            kv_max_end = max(kv_max_end, min(kv_chunk_end, qp + 1u));
        }
        const ulong q_row = ulong(p.m_offset + gm) * p.n_q + hq;
        q_vec[m]  = q_in[q_row * 32ul + lane] * p.attn_scale;
        r_lane[m] = (lane < p.d_rel) ? r_in[q_row * p.d_rel + lane] : 0.0f;
        m_val[m]  = -INFINITY;
        l_val[m]  = 0.0f;
        o_vec[m]  = float4(0.0f);
    }

    if (active_mask == 0u || kv_min_beg >= kv_max_end) return;

    device const half4* k_head_h = reinterpret_cast<device const half4*>(k_cache_raw) + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* v_head_h = reinterpret_cast<device const half4*>(v_cache_raw) + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const char4* k_head_q = reinterpret_cast<device const char4*>(k_cache_raw) + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const char4* v_head_q = reinterpret_cast<device const char4*>(v_cache_raw) + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half*  k_s_head = k_scales + ulong(hkv) * ulong(p.kv_stride);
    device const half*  v_s_head = v_scales + ulong(hkv) * ulong(p.kv_stride);

    const uint k_start = kv_chunk_beg + ((kv_min_beg - kv_chunk_beg) & ~15u);
    const uint q_first = p.seq_pos_start + m0;
    const uint q_last  = p.seq_pos_start + m0 + mb - 1u;

    for (uint k_base = k_start; k_base < kv_max_end; k_base += 16u) {
        const uint tile_cnt = min(16u, kv_max_end - k_base);

        for (uint ti = sgid; ti < tile_cnt; ti += 3u) {
            const uint  k_cur = k_base + ti;
            const uint  slot  = (p.is_local != 0u) ? (k_cur & (p.kv_stride - 1u)) : k_cur;
            const ulong off   = ulong(slot) * 32ul + lane;
            if (IS_Q8) {
                tg_k[ti][lane] = half4(k_head_q[off]) * k_s_head[slot];
                tg_v[ti][lane] = half4(v_head_q[off]) * v_s_head[slot];
            } else {
                tg_k[ti][lane] = k_head_h[off];
                tg_v[ti][lane] = v_head_h[off];
            }
        }

        const int  d_min_raw    = int(q_first) - int(k_base + tile_cnt - 1u);
        const int  d_max_raw    = int(q_last)  - int(k_base);
        const bool tile_has_rel = (d_max_raw >= 0) && (d_min_raw < int(p.rel_extent));
        const int  base_dist    = max(0, d_min_raw);

        if (tile_has_rel && lane < p.d_rel) {
            const uint rel_cnt = min(24u, uint(d_max_raw - base_dist + 1));
            for (uint di = sgid; di < rel_cnt; di += 3u) {
                const int d = base_dist + int(di);
                tg_rel[di][lane] = (d >= 0 && d < int(p.rel_extent))
                    ? rel_proj[ulong(d) * p.d_rel + lane] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const bool all_valid =
            (k_base + tile_cnt - 1u <= q_first) &&
            (p.is_local == 0u || (q_last - k_base) < p.swa_win);

        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_SUB; ++m) {
            if ((active_mask & (1u << m)) == 0u) continue;
            const uint qp = p.seq_pos_start + m0 + m;
            const uint ws = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;

            float sc[16];
            float tmax = -INFINITY;
            for (uint ti = 0u; ti < tile_cnt; ++ti) {
                const uint k_pos = k_base + ti;
                if (all_valid || (k_pos >= ws && k_pos <= qp)) {
                    float part = dot(q_vec[m], float4(tg_k[ti][lane]));
                    if (tile_has_rel) {
                        const uint dist = qp - k_pos;
                        if (dist < p.rel_extent && lane < p.d_rel) {
                            part = fma(r_lane[m], tg_rel[int(dist) - base_dist][lane], part);
                        }
                    }
                    const float s = simd_sum(part);
                    sc[ti] = s;
                    tmax   = max(tmax, s);
                } else {
                    sc[ti] = -INFINITY;
                }
            }
            if (tmax > -INFINITY) {
                const float m_new = max(m_val[m], tmax);
                const float alpha = (m_val[m] == -INFINITY) ? 0.0f : exp(m_val[m] - m_new);
                float  l_tile = 0.0f;
                float4 o_tile(0.0f);
                for (uint ti = 0u; ti < tile_cnt; ++ti) {
                    const uint k_pos = k_base + ti;
                    if (all_valid || (k_pos >= ws && k_pos <= qp)) {
                        const float w = exp(sc[ti] - m_new);
                        l_tile += w;
                        o_tile  = fma(float4(w), float4(tg_v[ti][lane]), o_tile);
                    }
                }
                m_val[m] = m_new;
                l_val[m] = fma(l_val[m], alpha, l_tile);
                o_vec[m] = fma(o_vec[m], alpha, o_tile);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_SUB; ++m) {
        if ((active_mask & (1u << m)) != 0u) {
            const uint  gm     = m0 + m;
            const uint  qp     = p.seq_pos_start + gm;
            const uint  ws     = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
            const uint  s_slot = c - (ws / B);
            const ulong ml_idx = (ulong(p.m_offset + gm) * p.n_q + hq) * p.max_splits + s_slot;
            if (lane == 0u) {
                part_m[ml_idx] = m_val[m];
                part_l[ml_idx] = l_val[m];
            }
            part_o[ml_idx * 32ul + lane] = o_vec[m];
        }
    }
}

kernel void sik_attn_decode_splitkv_shared(
    device const float4*    q_in     [[buffer(0)]],
    device const float*     r_in     [[buffer(1)]],
    device const float*     rel_proj [[buffer(2)]],
    device const uchar*     k_cache  [[buffer(3)]],
    device const uchar*     v_cache  [[buffer(4)]],
    device float*           part_m   [[buffer(5)]],
    device float*           part_l   [[buffer(6)]],
    device float4*          part_o   [[buffer(7)]],
    constant SikAttnParams& p        [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup half4 tg_k[16][32];
    threadgroup half4 tg_v[16][32];
    threadgroup float tg_rel[24][16];
    if (SIK_CONST_M == 1u) {
        sik_attn_decode_splitkv_sub_impl<1u, false>(q_in, r_in, rel_proj, k_cache, v_cache, nullptr, nullptr, part_m, part_l, part_o, p, tg_k, tg_v, tg_rel, tgid, sgid, lane);
    } else {
        sik_attn_decode_splitkv_sub_impl<2u, false>(q_in, r_in, rel_proj, k_cache, v_cache, nullptr, nullptr, part_m, part_l, part_o, p, tg_k, tg_v, tg_rel, tgid, sgid, lane);
    }
}

kernel void sik_attn_decode_splitkv_shared_q8(
    device const float4*    q_in     [[buffer(0)]],
    device const float*     r_in     [[buffer(1)]],
    device const float*     rel_proj [[buffer(2)]],
    device const uchar*     k_cache  [[buffer(3)]],
    device const uchar*     v_cache  [[buffer(4)]],
    device float*           part_m   [[buffer(5)]],
    device float*           part_l   [[buffer(6)]],
    device float4*          part_o   [[buffer(7)]],
    constant SikAttnParams& p        [[buffer(8)]],
    device const half*      k_scales [[buffer(9)]],
    device const half*      v_scales [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup half4 tg_k[16][32];
    threadgroup half4 tg_v[16][32];
    threadgroup float tg_rel[24][16];
    if (SIK_CONST_M == 1u) {
        sik_attn_decode_splitkv_sub_impl<1u, true>(q_in, r_in, rel_proj, k_cache, v_cache, k_scales, v_scales, part_m, part_l, part_o, p, tg_k, tg_v, tg_rel, tgid, sgid, lane);
    } else {
        sik_attn_decode_splitkv_sub_impl<2u, true>(q_in, r_in, rel_proj, k_cache, v_cache, k_scales, v_scales, part_m, part_l, part_o, p, tg_k, tg_v, tg_rel, tgid, sgid, lane);
    }
}

// High-Occupancy In-Register Q8_0 Global KV Decode Split-KV (4 SGs/TG, 2.08 KiB TG mem, 0 barriers in KV loop!)
//   - 100% coalesced 128-byte `char4` reads dequantized directly in registers `half4(char4) * scale`
//   - 100.0000% BIT-EXACT `K>0 == K=0`: fixed `k = kv_chunk_beg + sgid (stride 4)` & fixed `sg=0..3` merge!
// Grid: `(num_batch_chunks, 8 /* n_kv */, (mb + M_SUB - 1) / M_SUB)`, `128` threads/TG (`4` SGs)
template <uint M_SUB>
inline void sik_attn_decode_q8_reg_impl(
    device const float4*    q_in,
    device const float*     r_in,
    device const float*     rel_proj,
    device const char4*     k_cache,
    device const char4*     v_cache,
    device const half*      k_scales,
    device const half*      v_scales,
    device float*           part_m,
    device float*           part_l,
    device float4*          part_o,
    constant SikAttnParams& p,
    threadgroup float4      (&tg_o)[4][32],
    threadgroup float       (&tg_m)[4],
    threadgroup float       (&tg_l)[4],
    uint3                   tgid,
    uint                    sgid,
    uint                    lane
) {
    const uint c   = p.c_batch_first + tgid.x;
    const uint hkv = tgid.y;
    const uint m0  = tgid.z * M_SUB;
    if (hkv >= p.n_kv || m0 >= p.M_valid) return;

    const uint mb           = min(M_SUB, p.M_valid - m0);
    const uint B            = p.chunk_size;
    const uint kv_chunk_beg = c * B;
    const uint kv_chunk_end = kv_chunk_beg + B;

    float4 q_vec[M_SUB][3];
    float  r_lane[M_SUB][3];
    float  m_val[M_SUB][3];
    float  l_val[M_SUB][3];
    float4 o_vec[M_SUB][3];

    uint active_mask = 0u;
    uint kv_max_end  = kv_chunk_beg;

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_SUB; ++m) {
        const uint gm = min(m0 + m, p.M_valid - 1u);
        const uint qp = p.seq_pos_start + gm;
        if (m < mb && c <= (qp / B)) {
            active_mask |= (1u << m);
            kv_max_end = max(kv_max_end, min(kv_chunk_end, qp + 1u));
        }
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            const uint  hq    = hkv * 3u + h;
            const ulong q_row = ulong(p.m_offset + gm) * p.n_q + hq;
            q_vec[m][h]  = q_in[q_row * 32ul + lane] * p.attn_scale;
            r_lane[m][h] = (lane < p.d_rel) ? r_in[q_row * p.d_rel + lane] : 0.0f;
            m_val[m][h]  = -INFINITY;
            l_val[m][h]  = 0.0f;
            o_vec[m][h]  = float4(0.0f);
        }
    }
    if (active_mask == 0u || kv_chunk_beg >= kv_max_end) return;

    device const char4* k_head = k_cache  + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const char4* v_head = v_cache  + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half*  k_s    = k_scales + ulong(hkv) * ulong(p.kv_stride);
    device const half*  v_s    = v_scales + ulong(hkv) * ulong(p.kv_stride);

    const uint q_first       = p.seq_pos_start + m0;
    const bool chunk_has_rel = (int(q_first) - int(kv_max_end - 1u) < int(p.rel_extent));
    const bool all_valid     = (kv_max_end - 1u <= q_first);

    for (uint k = kv_chunk_beg + sgid; k < kv_max_end; k += 4u) {
        const ulong  off = ulong(k) * 32ul + lane;
        const float4 kv  = float4(half4(k_head[off]) * k_s[k]);
        const float4 vv  = float4(half4(v_head[off]) * v_s[k]);

        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_SUB; ++m) {
            if ((active_mask & (1u << m)) == 0u) continue;
            const uint qp = p.seq_pos_start + m0 + m;
            if (!all_valid && k > qp) continue;

            float rel_w = 0.0f;
            if (chunk_has_rel && lane < p.d_rel) {
                const uint dist = qp - k;
                if (dist < p.rel_extent) rel_w = rel_proj[ulong(dist) * p.d_rel + lane];
            }

            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 3u; ++h) {
                float part = dot(q_vec[m][h], kv);
                if (chunk_has_rel && lane < p.d_rel) part = fma(r_lane[m][h], rel_w, part);
                const float s  = simd_sum(part);
                const float mn = max(m_val[m][h], s);
                const float al = (m_val[m][h] == -INFINITY) ? 0.0f : exp(m_val[m][h] - mn);
                const float w  = exp(s - mn);
                m_val[m][h] = mn;
                l_val[m][h] = fma(l_val[m][h], al, w);
                o_vec[m][h] = fma(o_vec[m][h], al, w * vv);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_SUB; ++m) {
        if ((active_mask & (1u << m)) == 0u) continue;
        const uint gm = m0 + m;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            if (lane == 0u) {
                tg_m[sgid] = m_val[m][h];
                tg_l[sgid] = l_val[m][h];
            }
            tg_o[sgid][lane] = o_vec[m][h];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgid == 0u) {
                const float mg = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
                float  lg = 0.0f;
                float4 og(0.0f);
                if (mg > -INFINITY) {
                    #pragma clang loop unroll(full)
                    for (uint s = 0u; s < 4u; ++s) {
                        if (tg_l[s] > 0.0f) {
                            const float sc = exp(tg_m[s] - mg);
                            lg = fma(tg_l[s], sc, lg);
                            og = fma(tg_o[s][lane], sc, og);
                        }
                    }
                }
                const uint  hq     = hkv * 3u + h;
                const ulong ml_idx = (ulong(p.m_offset + gm) * p.n_q + hq) * p.max_splits + c;
                if (lane == 0u) {
                    part_m[ml_idx] = mg;
                    part_l[ml_idx] = lg;
                }
                part_o[ml_idx * 32ul + lane] = og;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void sik_attn_decode_q8_reg(
    device const float4*    q_in     [[buffer(0)]],
    device const float*     r_in     [[buffer(1)]],
    device const float*     rel_proj [[buffer(2)]],
    device const char4*     k_cache  [[buffer(3)]],
    device const char4*     v_cache  [[buffer(4)]],
    device float*           part_m   [[buffer(5)]],
    device float*           part_l   [[buffer(6)]],
    device float4*          part_o   [[buffer(7)]],
    constant SikAttnParams& p        [[buffer(8)]],
    device const half*      k_scales [[buffer(9)]],
    device const half*      v_scales [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup float4 tg_o[4][32];
    threadgroup float  tg_m[4];
    threadgroup float  tg_l[4];
    if (SIK_CONST_M == 1u) {
        sik_attn_decode_q8_reg_impl<1u>(q_in, r_in, rel_proj, k_cache, v_cache, k_scales, v_scales, part_m, part_l, part_o, p, tg_o, tg_m, tg_l, tgid, sgid, lane);
    } else {
        sik_attn_decode_q8_reg_impl<2u>(q_in, r_in, rel_proj, k_cache, v_cache, k_scales, v_scales, part_m, part_l, part_o, p, tg_o, tg_m, tg_l, tgid, sgid, lane);
    }
}

// Metal 4.1 NAX Decode Split-KV (`q_pos >= 256`, `B >= 32` aligned, `M_SUB <= 5` packed into ONE `16x32` NAX tile!):
//   - Packs up to `5` tokens x `3` Q-heads = `15 <= 16` rows into ONE single `16x32x128` QK + `16x64x32` PV NAX tile!
//   - Stages `tg_rel[37][16]` ONCE per 32-token tile!
//   - PROVEN `bit_diffs = 0, max_abs = 0` between M=1 and M=2..8 on M5 Max NAX!
// Grid: `(num_batch_chunks, 8 /* n_kv */, (mb + 4) / 5)`, `32` threads/TG (`1` SG = `execution_simdgroup`!)
kernel void sik_attn_decode_nax(
    device const float*     q_in     [[buffer(0)]],
    device const float*     r_in     [[buffer(1)]],
    device const float*     rel_proj [[buffer(2)]],
    device const half*      k_cache  [[buffer(3)]],
    device const half*      v_cache  [[buffer(4)]],
    device float*           part_m   [[buffer(5)]],
    device float*           part_l   [[buffer(6)]],
    device float*           part_o   [[buffer(7)]],
    constant SikAttnParams& p        [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    const uint c   = p.c_batch_first + tgid.x;
    const uint hkv = tgid.y;
    const uint m0  = tgid.z * 5u;
    if (hkv >= p.n_kv || m0 >= p.M_valid) return;

    const uint mb           = min(5u, p.M_valid - m0);
    const uint n_rows       = mb * 3u;
    const uint B            = p.chunk_size;
    const uint kv_chunk_beg = c * B;
    const uint kv_chunk_end = kv_chunk_beg + B;

    uint active_mask = 0u;
    uint kv_min_beg  = kv_chunk_end;
    uint kv_max_end  = kv_chunk_beg;
    for (uint m = 0u; m < mb; ++m) {
        const uint qp = p.seq_pos_start + m0 + m;
        const uint ws = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
        if (c >= (ws / B) && c <= (qp / B)) {
            active_mask |= (1u << m);
            kv_min_beg = min(kv_min_beg, max(kv_chunk_beg, ws));
            kv_max_end = max(kv_max_end, min(kv_chunk_end, qp + 1u));
        }
    }
    if (active_mask == 0u || kv_min_beg >= kv_max_end) return;

    // Union `tg_q` (4KB, init-only!) with `tg_s` (2KB) + `tg_p` (1KB) -> ONLY 5.2 KB TG mem (6 TGs/core!)
    threadgroup ulong tg_union[512]; // 4096 bytes
    threadgroup float tg_r[16][16];  // 1024 bytes
    threadgroup float tg_m[16];
    threadgroup float tg_l[16];
    threadgroup float tg_a[16];

    threadgroup half* tg_q = reinterpret_cast<threadgroup half*>(tg_union);

    for (uint idx = lane; idx < 16u * 128u; idx += 32u) {
        const uint rr = idx >> 7u;
        const uint dd = idx & 127u;
        if (rr < n_rows) {
            const uint m_loc = rr / 3u;
            const uint hq    = hkv * 3u + (rr - m_loc * 3u);
            tg_q[rr * 128u + dd] = sik_clamp_f16(q_in[(ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * 128ul + dd] * p.attn_scale);
        } else {
            tg_q[rr * 128u + dd] = 0.0h;
        }
    }
    for (uint idx = lane; idx < 16u * 16u; idx += 32u) {
        const uint rr = idx >> 4u;
        const uint dd = idx & 15u;
        if (rr < n_rows) {
            const uint m_loc = rr / 3u;
            const uint hq    = hkv * 3u + (rr - m_loc * 3u);
            tg_r[rr][dd] = r_in[(ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * p.d_rel + dd];
        } else {
            tg_r[rr][dd] = 0.0f;
        }
    }
    if (lane < 16u) { tg_m[lane] = -INFINITY; tg_l[lane] = 0.0f; tg_a[lane] = 1.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device half* kh = const_cast<device half*>(k_cache) + ulong(hkv) * ulong(p.kv_stride) * 128ul;
    device half* vh = const_cast<device half*>(v_cache) + ulong(hkv) * ulong(p.kv_stride) * 128ul;

    tensor<threadgroup half, extents<int, 128, 16>, tensor_inline> tQ(tg_q, extents<int, 128, 16>());
    constexpr auto desc_qk = matmul2d_descriptor(16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;
    auto qCT = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(tQ);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Now reuse `tg_union` as `tg_s [16, 32]` float (bytes 0..2047) + `tg_p [16, 32]` half (bytes 2048..3071)!
    threadgroup float* tg_s = reinterpret_cast<threadgroup float*>(tg_union);
    threadgroup half*  tg_p = reinterpret_cast<threadgroup half*>(tg_union + 256);

    tensor<threadgroup half, extents<int, 32, 16>, tensor_inline> tP(tg_p, extents<int, 32, 16>());
    tensor<device half, dextents<int, 2>, tensor_inline>          tK(kh, dextents<int, 2>(128, int(p.kv_stride)));
    tensor<device half, dextents<int, 2>, tensor_inline>          tV(vh, dextents<int, 2>(128, int(p.kv_stride)));

    constexpr auto desc_pv = matmul2d_descriptor(16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    auto sV_proto = tV.slice<64, 32>(0, 0);
    auto oCT0 = op_pv.get_destination_cooperative_tensor<sik_strip_as<decltype(tP)>, sik_strip_as<decltype(sV_proto)>, float>();
    auto oCT1 = op_pv.get_destination_cooperative_tensor<sik_strip_as<decltype(tP)>, sik_strip_as<decltype(sV_proto)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) { oCT0[i] = 0.0f; oCT1[i] = 0.0f; }
    }

    const uint k_start = kv_chunk_beg + ((kv_min_beg - kv_chunk_beg) & ~31u);
    const uint q_first = p.seq_pos_start + m0;
    const uint q_last  = p.seq_pos_start + m0 + mb - 1u;

    for (uint kb = k_start; kb < kv_max_end; kb += 32u) {
        const uint k_slot = (p.is_local != 0u) ? (kb & (p.kv_stride - 1u)) : kb;
        const uint tc     = min(32u, kv_max_end - kb);

        auto sK  = tK.slice<128, 32>(0, int(k_slot));
        auto sCT = op_qk.get_destination_cooperative_tensor<sik_strip_as<decltype(qCT)>, sik_strip_as<decltype(sK)>, float>();
        op_qk.run(qCT, sK, sCT);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < sCT.get_capacity(); ++i) {
            if (sCT.is_valid_element(i)) {
                auto idx = sCT.get_multidimensional_index(i);
                tg_s[uint(idx[1]) * 32u + uint(idx[0])] = sCT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const int  d_min_raw    = int(q_first) - int(kb + tc - 1u);
        const int  d_max_raw    = int(q_last)  - int(kb);
        const bool tile_has_rel = (d_max_raw >= 0) && (d_min_raw < int(p.rel_extent));
        const uint kp           = kb + lane;

        for (uint rr = 0u; rr < n_rows; ++rr) {
            const uint m_loc = rr / 3u;
            const uint qp    = p.seq_pos_start + m0 + m_loc;
            const uint ws    = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
            const bool valid = ((active_mask & (1u << m_loc)) != 0u) && (lane < tc) && (kp >= ws) && (kp <= qp);
            float sc = -INFINITY;
            if (valid) {
                sc = tg_s[rr * 32u + lane];
                if (tile_has_rel) {
                    const uint dist = qp - kp;
                    if (dist < p.rel_extent) {
                        device const float4* rp4 = reinterpret_cast<device const float4*>(rel_proj + ulong(dist) * 16ul);
                        const float4 r0(tg_r[rr][0], tg_r[rr][1], tg_r[rr][2], tg_r[rr][3]);
                        const float4 r1(tg_r[rr][4], tg_r[rr][5], tg_r[rr][6], tg_r[rr][7]);
                        const float4 r2(tg_r[rr][8], tg_r[rr][9], tg_r[rr][10], tg_r[rr][11]);
                        const float4 r3(tg_r[rr][12], tg_r[rr][13], tg_r[rr][14], tg_r[rr][15]);
                        sc += (dot(r0, rp4[0]) + dot(r1, rp4[1])) + (dot(r2, rp4[2]) + dot(r3, rp4[3]));
                    }
                }
            }
            const float tmax = simd_max(sc);
            if (tmax > -INFINITY) {
                const float mold  = tg_m[rr];
                const float mnew  = max(mold, tmax);
                const float alpha = (mold == -INFINITY) ? 0.0f : exp(mold - mnew);
                const float w     = valid ? exp(sc - mnew) : 0.0f;
                const float lsum  = simd_sum(w);
                tg_p[rr * 32u + lane] = sik_clamp_f16(w);
                if (lane == 0u) {
                    tg_m[rr] = mnew;
                    tg_l[rr] = fma(tg_l[rr], alpha, lsum);
                    tg_a[rr] = alpha;
                }
            } else {
                tg_p[rr * 32u + lane] = 0.0h;
                if (lane == 0u) tg_a[rr] = 1.0f;
            }
        }
        for (uint rr = n_rows; rr < 16u; ++rr) {
            tg_p[rr * 32u + lane] = 0.0h;
            if (lane == 0u) tg_a[rr] = 1.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
            if (oCT0.is_valid_element(i)) {
                const float a = tg_a[uint(oCT0.get_multidimensional_index(i)[1])];
                oCT0[i] *= a;
                oCT1[i] *= a;
            }
        }

        auto sV0 = tV.slice<64, 32>(0, int(k_slot));
        auto sV1 = tV.slice<64, 32>(64, int(k_slot));
        op_pv.run(tP, sV0, oCT0);
        op_pv.run(tP, sV1, oCT1);
    }

    // Coalesced float4 `part_o` store via `tg_union` (0 integer divisions! 4x wider coalesced stores!)
    const uint B_shift = ctz(B);
    uint m_slot[5];
    for (uint m = 0u; m < mb; ++m) {
        const uint qp = p.seq_pos_start + m0 + m;
        const uint ws = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
        m_slot[m] = c - (ws >> B_shift);
    }

    threadgroup float* tg_out = reinterpret_cast<threadgroup float*>(tg_union); // [16, 64] = 4096 B exact!
    device float4*     part_o4 = reinterpret_cast<device float4*>(part_o);

    // Half 0 (dims 0..63)
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            auto idx = oCT0.get_multidimensional_index(i);
            tg_out[uint(idx[1]) * 64u + uint(idx[0])] = oCT0[i];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rr = 0u; rr < n_rows; ++rr) {
        const uint m_loc = rr / 3u;
        if ((active_mask & (1u << m_loc)) != 0u && lane < 16u) {
            const uint  hq   = hkv * 3u + (rr - m_loc * 3u);
            const ulong oidx = (ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * p.max_splits + m_slot[m_loc];
            part_o4[oidx * 32ul + lane] = reinterpret_cast<threadgroup const float4*>(tg_out + rr * 64u)[lane];
        }
    }

    // Half 1 (dims 64..127)
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT1.get_capacity(); ++i) {
        if (oCT1.is_valid_element(i)) {
            auto idx = oCT1.get_multidimensional_index(i);
            tg_out[uint(idx[1]) * 64u + uint(idx[0])] = oCT1[i];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rr = 0u; rr < n_rows; ++rr) {
        const uint m_loc = rr / 3u;
        if ((active_mask & (1u << m_loc)) != 0u && lane < 16u) {
            const uint  hq   = hkv * 3u + (rr - m_loc * 3u);
            const ulong oidx = (ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * p.max_splits + m_slot[m_loc];
            part_o4[oidx * 32ul + 16u + lane] = reinterpret_cast<threadgroup const float4*>(tg_out + rr * 64u)[lane];
        }
    }

    if (lane < n_rows) {
        const uint m_loc = lane / 3u;
        if ((active_mask & (1u << m_loc)) != 0u) {
            const uint  hq   = hkv * 3u + (lane - m_loc * 3u);
            const ulong oidx = (ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * p.max_splits + m_slot[m_loc];
            part_m[oidx] = tg_m[lane];
            part_l[oidx] = tg_l[lane];
        }
    }
#endif
}

// Coalesced Head-Major 4-SG Split-KV Reduce + Virtual Sink + Output Sigmoid Gate -> FP16 `[M_valid, 3072]` + `wo_xsum`!
// Zero-Barrier 1-SG (`32` th/TG) Split-KV Reduce + Virtual Sink + Sigmoid Gate + `wo_xsum`!
// Supports `p.m_offset` for mixed single-pass + split-KV speculative batches straddling `pos=160`!
// Grid: `(24, mb, 1)`, `32` threads/TG (`1` SG, ZERO TG mem, ZERO TG barriers!)
kernel void sik_attn_decode_reduce(
    device const float*     part_m   [[buffer(0)]],
    device const float*     part_l   [[buffer(1)]],
    device const float4*    part_o   [[buffer(2)]],
    device const float*     sink_in  [[buffer(3)]],
    device const float4*    gate_in  [[buffer(4)]],
    device half4*           out_f16  [[buffer(5)]],
    constant SikAttnParams& p        [[buffer(6)]],
    device float*           wo_xsum  [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint hq = tgid.x;
    const uint m  = tgid.y;
    if (hq >= p.n_q || m >= p.M_valid) return;

    const uint  gm        = p.m_offset + m;
    const uint  q_pos     = p.seq_pos_start + m;
    const uint  B         = sik_attn_chunk_size(q_pos, p.is_local);
    const uint  win_start = (p.is_local != 0u && q_pos + 1u > p.swa_win) ? (q_pos + 1u - p.swa_win) : 0u;
    const uint  my_splits = (q_pos / B) - (win_start / B) + 1u;
    const ulong head_base = (ulong(gm) * p.n_q + hq) * p.max_splits;

    const float sink_v = sink_in[ulong(gm) * p.n_q + hq];
    const float m0     = (lane < my_splits) ? part_m[head_base + lane] : -INFINITY;
    const float m1     = (lane + 32u < my_splits) ? part_m[head_base + lane + 32u] : -INFINITY;
    const float m_max  = max(sink_v, simd_max(max(m0, m1)));

    const float w0     = (lane < my_splits && m0 > -INFINITY) ? exp(m0 - m_max) : 0.0f;
    const float w1     = (lane + 32u < my_splits && m1 > -INFINITY) ? exp(m1 - m_max) : 0.0f;
    const float l0     = (w0 > 0.0f) ? (part_l[head_base + lane] * w0) : 0.0f;
    const float l1     = (w1 > 0.0f) ? (part_l[head_base + lane + 32u] * w1) : 0.0f;
    const float l_tot  = exp(sink_v - m_max) + simd_sum(l0 + l1);
    const float inv_l  = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;

    float4 o_tot(0.0f);
    for (uint s = 0u; s < my_splits; ++s) {
        const float w = (s < 32u) ? simd_broadcast(w0, ushort(s)) : simd_broadcast(w1, ushort(s - 32u));
        if (w > 0.0f) {
            o_tot = fma(part_o[(head_base + s) * 32ul + lane], float4(w), o_tot);
        }
    }

    const ulong  out_off = (ulong(gm) * p.n_q + hq) * 32ul + lane;
    const float4 gv      = gate_in[out_off];
    const float4 sig_g   = float4(1.0f) / (float4(1.0f) + exp(-gv));
    const half4  h       = sik_clamp_f16x4((o_tot * inv_l) * sig_g);
    out_f16[out_off]     = h;
    if (gm < 8u) {
        const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
        if ((lane & 15u) == 0u) {
            wo_xsum[ulong(gm) * (p.n_q * 2u) + hq * 2u + (lane >> 4u)] = bs;
        }
    }
}

// Metal 4.1 NAX FlashAttention (`execution_simdgroup`, `Q_TILE=16, KV_TILE=32, D=128`, 4 SGs/TG -> 64 Q rows/TG!)
// Metal 4.1 NAX FlashAttention Prefill (GQA-3 Shared-KV TG: 3 SGs = 3 Q heads on SAME 16 queries -> 3x L1 KV sharing!):
//   - `qCT` pinned in registers + `16x32x128` QK & `16x64x32` PV (avoids K=64 cliff!) + `9.8 KiB` TG mem (`3` TGs/core = 9 SGs/core!)
//   - Unioned `S[16][32]` & `P[16][32]` + `sCT.store(tS)` + `constant float* rel_proj` L0 constant cache!
// Grid: `((M_valid + 15) / 16, 8 /* n_kv */, 1)`, `96` threads/TG (`3` SGs = 3 GQA Q heads!)
kernel void sik_attn_prefill_nax(
    device const half*      q_hm     [[buffer(0)]], // [24, M_pad16, 128] pre-scaled by 1/sqrt(128)
    device const float*     r_hm     [[buffer(1)]], // [24, M_pad16, 16]
    constant float*         rel_proj [[buffer(2)]], // [1024, 16] (exact 64 KiB -> pins in M5 Max constant cache!)
    device const float*     sink_in  [[buffer(3)]], // [M_valid, 24]
    device const float*     gate_in  [[buffer(4)]], // [M_valid, 24, 128]
    device const half*      k_cache  [[buffer(5)]], // [8, kv_stride, 128]
    device const half*      v_cache  [[buffer(6)]], // [8, kv_stride, 128]
    device half*            out_f16  [[buffer(7)]], // [M_valid, 24, 128]
    constant SikAttnParams& p        [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    const uint hkv = tgid.y;
    const uint hq  = hkv * 3u + sgid;
    const uint m0  = tgid.x * 16u;
    if (m0 >= p.M_valid || hq >= p.n_q) return;

    // Separate S_tg & P_tg (12.8 KiB total -> 2 TGs/core, zero union aliasing hazard!)
    threadgroup float S_tg[3][16][32];
    threadgroup half  P_tg[3][16][32];
    threadgroup float R_tg[3][16][16];
    threadgroup float M_tg[3][16];
    threadgroup float L_tg[3][16];
    threadgroup float A_tg[3][16];

    if (lane < 16u) {
        const uint mg    = m0 + lane;
        M_tg[sgid][lane] = (mg < p.M_valid) ? sink_in[ulong(mg) * p.n_q + hq] : 0.0f;
        L_tg[sgid][lane] = 1.0f; // exact virtual zero-value attention sink!
    }
    // Preload this SG's 16x16 R tile into TG memory (256 floats / 32 lanes = 8 floats/lane)
    {
        device const float* r_base = r_hm + (ulong(hq) * p.M_pad16 + m0) * 16ul;
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
            const uint lin = i * 32u + lane;
            R_tg[sgid][lin >> 4u][lin & 15u] = r_base[lin];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device half* qh = const_cast<device half*>(q_hm)    + ulong(hq)  * p.M_pad16  * 128ul;
    device half* kh = const_cast<device half*>(k_cache) + ulong(hkv) * p.kv_stride * 128ul;
    device half* vh = const_cast<device half*>(v_cache) + ulong(hkv) * p.kv_stride * 128ul;

    tensor<device half, dextents<int, 2>, tensor_inline> tQ(qh, dextents<int, 2>(128, int(p.M_pad16)));
    tensor<device half, dextents<int, 2>, tensor_inline> tK(kh, dextents<int, 2>(128, int(p.kv_stride)));
    tensor<device half, dextents<int, 2>, tensor_inline> tV(vh, dextents<int, 2>(128, int(p.kv_stride)));
    tensor<threadgroup float, extents<int, 32, 16>, tensor_inline> tS(&S_tg[sgid][0][0], extents<int, 32, 16>());
    tensor<threadgroup half,  extents<int, 32, 16>, tensor_inline> tP(&P_tg[sgid][0][0], extents<int, 32, 16>());

    constexpr auto desc_qk = matmul2d_descriptor(
        16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;
    constexpr auto desc_pv = matmul2d_descriptor(
        16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    auto sQ  = tQ.slice<128, 16>(0, int(m0));
    auto qCT = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(sQ);

    auto sV0  = tV.slice<64, 32>(0, 0);
    auto oCT0 = op_pv.get_destination_cooperative_tensor<sik_strip_as<decltype(tP)>, sik_strip_as<decltype(sV0)>, float>();
    auto oCT1 = op_pv.get_destination_cooperative_tensor<sik_strip_as<decltype(tP)>, sik_strip_as<decltype(sV0)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) { oCT0[i] = 0.0f; oCT1[i] = 0.0f; }
    }

    const uint q_first = p.seq_pos_start + m0;
    const uint q_last  = p.seq_pos_start + min(m0 + 15u, p.M_valid - 1u);
    const uint k_start = (p.is_local != 0u && q_first + 1u > p.swa_win)
        ? ((q_first + 1u - p.swa_win) & ~31u) : 0u;
    const uint k_end   = q_last + 1u;

    for (uint kb = k_start; kb < k_end; kb += 32u) {
        const uint k_slot = (p.is_local != 0u) ? (kb & (p.kv_stride - 1u)) : kb;
        auto sK  = tK.slice<128, 32>(0, int(k_slot));
        auto sCT = op_qk.get_destination_cooperative_tensor<sik_strip_as<decltype(qCT)>, sik_strip_as<decltype(sK)>, float>();
        op_qk.run(qCT, sK, sCT);
        sCT.store(tS);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint k_tile_end   = min(kb + 32u, k_end);
        const int  d_min_raw    = int(q_first) - int(k_tile_end - 1u);
        const int  d_max_raw    = int(q_last)  - int(kb);
        const bool tile_has_rel = (d_max_raw >= 0) && (d_min_raw < int(p.rel_extent));
        const uint k_pos        = kb + lane;
        const bool all_causal   = (kb + 31u <= q_first) && (m0 + 16u <= p.M_valid) &&
                                  (p.is_local == 0u || (q_last - kb) < p.swa_win);

        if (all_causal && !tile_has_rel) {
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 16u; ++r) {
                const float sc   = S_tg[sgid][r][lane];
                const float rmax = simd_max(sc);
                const float mp   = M_tg[sgid][r];
                const float mn   = max(mp, rmax);
                const float al   = exp(mp - mn);
                const float pv   = exp(sc - mn);
                const float ps   = simd_sum(pv);
                P_tg[sgid][r][lane] = half(pv);
                if (lane == 0u) {
                    M_tg[sgid][r] = mn;
                    L_tg[sgid][r] = fma(L_tg[sgid][r], al, ps);
                    A_tg[sgid][r] = al;
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 16u; ++r) {
                const uint mg    = m0 + r;
                const uint qp    = p.seq_pos_start + mg;
                const bool valid = all_causal || ((mg < p.M_valid) && (k_pos <= qp) &&
                                   (p.is_local == 0u || (qp - k_pos) < p.swa_win));
                float sc = valid ? S_tg[sgid][r][lane] : -INFINITY;
                if (tile_has_rel && valid) {
                    const uint dist = qp - k_pos;
                    if (dist < p.rel_extent) {
                        threadgroup const float4* r4 = reinterpret_cast<threadgroup const float4*>(&R_tg[sgid][r][0]);
                        constant float4*          w4 = reinterpret_cast<constant float4*>(rel_proj + ulong(dist) * 16ul);
                        sc += (dot(r4[0], w4[0]) + dot(r4[1], w4[1]))
                            + (dot(r4[2], w4[2]) + dot(r4[3], w4[3]));
                    }
                }
                const float rmax = simd_max(sc);
                const float mp   = M_tg[sgid][r];
                const float mn   = max(mp, rmax);
                const float al   = exp(mp - mn);
                const float pv   = valid ? exp(sc - mn) : 0.0f;
                const float ps   = simd_sum(pv);
                P_tg[sgid][r][lane] = half(pv);
                if (lane == 0u) {
                    M_tg[sgid][r] = mn;
                    L_tg[sgid][r] = fma(L_tg[sgid][r], al, ps);
                    A_tg[sgid][r] = al;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
            if (oCT0.is_valid_element(i)) {
                auto idx = oCT0.get_multidimensional_index(i);
                const float al = A_tg[sgid][uint(idx[1])];
                oCT0[i] *= al;
                oCT1[i] *= al;
            }
        }
        auto sV_0 = tV.slice<64, 32>(0,  int(k_slot));
        auto sV_1 = tV.slice<64, 32>(64, int(k_slot));
        op_pv.run(tP, sV_0, oCT0);
        op_pv.run(tP, sV_1, oCT1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store normalized + sigmoid-gated O directly to token-major `[M_valid, 24, 128]` FP16!
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            auto idx = oCT0.get_multidimensional_index(i);
            const uint r = uint(idx[1]);
            const uint c = uint(idx[0]);
            const uint mg = m0 + r;
            if (mg < p.M_valid) {
                const float invL    = 1.0f / L_tg[sgid][r];
                const ulong out_off = (ulong(mg) * p.n_q + hq) * 128ul + c;
                const float g0      = gate_in[out_off];
                const float g1      = gate_in[out_off + 64u];
                out_f16[out_off]       = sik_clamp_f16((oCT0[i] * invL) * sik_sigmoid(g0));
                out_f16[out_off + 64u] = sik_clamp_f16((oCT1[i] * invL) * sik_sigmoid(g1));
            }
        }
    }
#else
    (void)q_hm;(void)r_hm;(void)rel_proj;(void)sink_in;(void)gate_in;(void)k_cache;(void)v_cache;(void)out_f16;(void)p;(void)tgid;(void)sgid;(void)lane;
#endif
}

// Per-Head Q8_0 Global KV NAX Decode Split-KV (`q_pos >= 256`, 2x less global KV DRAM, 100% bit-exact K>0 == K=0!)
// Coalesced 16B `uint4` dequant into 8 KiB reusable `tg_kv[32][128]` -> 13.2 KiB total TG mem!
// Grid: `(num_batch_chunks, 8, (mb + 4) / 5)`, `32` threads/TG (`1` SG = `execution_simdgroup`)
kernel void sik_attn_decode_nax_q8(
    device const float*     q_in     [[buffer(0)]],
    device const float*     r_in     [[buffer(1)]],
    device const float*     rel_proj [[buffer(2)]],
    device const uint4*     k_cache  [[buffer(3)]], // [8, kv_stride, 8] uint4 (128 int8s/row)
    device const uint4*     v_cache  [[buffer(4)]], // [8, kv_stride, 8] uint4
    device float*           part_m   [[buffer(5)]],
    device float*           part_l   [[buffer(6)]],
    device float*           part_o   [[buffer(7)]],
    constant SikAttnParams& p        [[buffer(8)]],
    device const half*      k_scales [[buffer(9)]], // [8, kv_stride] half
    device const half*      v_scales [[buffer(10)]],// [8, kv_stride] half
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    const uint c   = p.c_batch_first + tgid.x;
    const uint hkv = tgid.y;
    const uint m0  = tgid.z * 5u;
    if (hkv >= p.n_kv || m0 >= p.M_valid) return;

    const uint mb           = min(5u, p.M_valid - m0);
    const uint n_rows       = mb * 3u;
    const uint B            = p.chunk_size;
    const uint kv_chunk_beg = c * B;
    const uint kv_chunk_end = kv_chunk_beg + B;

    uint active_mask = 0u;
    uint kv_min_beg  = kv_chunk_end;
    uint kv_max_end  = kv_chunk_beg;
    for (uint m = 0u; m < mb; ++m) {
        const uint qp = p.seq_pos_start + m0 + m;
        const uint ws = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
        if (c >= (ws / B) && c <= (qp / B)) {
            active_mask |= (1u << m);
            kv_min_beg = min(kv_min_beg, max(kv_chunk_beg, ws));
            kv_max_end = max(kv_max_end, min(kv_chunk_end, qp + 1u));
        }
    }
    if (active_mask == 0u || kv_min_beg >= kv_max_end) return;

    threadgroup ulong tg_union[512];   // 4096 bytes
    threadgroup half  tg_kv[32][128];  // 8192 bytes (reused for K then V!)
    threadgroup float tg_r[16][16];    // 1024 bytes
    threadgroup float tg_m[16];
    threadgroup float tg_l[16];
    threadgroup float tg_a[16];

    threadgroup half* tg_q = reinterpret_cast<threadgroup half*>(tg_union);
    for (uint idx = lane; idx < 16u * 128u; idx += 32u) {
        const uint rr = idx >> 7u;
        const uint dd = idx & 127u;
        if (rr < n_rows) {
            const uint m_loc = rr / 3u;
            const uint hq    = hkv * 3u + (rr - m_loc * 3u);
            tg_q[rr * 128u + dd] = sik_clamp_f16(q_in[(ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * 128ul + dd] * p.attn_scale);
        } else {
            tg_q[rr * 128u + dd] = 0.0h;
        }
    }
    for (uint idx = lane; idx < 16u * 16u; idx += 32u) {
        const uint rr = idx >> 4u;
        const uint dd = idx & 15u;
        if (rr < n_rows) {
            const uint m_loc = rr / 3u;
            const uint hq    = hkv * 3u + (rr - m_loc * 3u);
            tg_r[rr][dd] = r_in[(ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * p.d_rel + dd];
        } else {
            tg_r[rr][dd] = 0.0f;
        }
    }
    if (lane < 16u) { tg_m[lane] = -INFINITY; tg_l[lane] = 0.0f; tg_a[lane] = 1.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const uint4* kh_q = k_cache  + ulong(hkv) * ulong(p.kv_stride) * 8ul;
    device const uint4* vh_q = v_cache  + ulong(hkv) * ulong(p.kv_stride) * 8ul;
    device const half*  kh_s = k_scales + ulong(hkv) * ulong(p.kv_stride);
    device const half*  vh_s = v_scales + ulong(hkv) * ulong(p.kv_stride);

    tensor<threadgroup half, extents<int, 128, 16>, tensor_inline> tQ(tg_q, extents<int, 128, 16>());
    tensor<threadgroup half, extents<int, 128, 32>, tensor_inline> tKV(&tg_kv[0][0], extents<int, 128, 32>());
    constexpr auto desc_qk = matmul2d_descriptor(16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;
    auto qCT = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    qCT.load(tQ);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* tg_s = reinterpret_cast<threadgroup float*>(tg_union);
    threadgroup half*  tg_p = reinterpret_cast<threadgroup half*>(tg_union + 256);

    tensor<threadgroup half, extents<int, 32, 16>, tensor_inline> tP(tg_p, extents<int, 32, 16>());
    constexpr auto desc_pv = matmul2d_descriptor(16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    auto sV0  = tKV.slice<64, 32>(0, 0);
    auto sV1  = tKV.slice<64, 32>(64, 0);
    auto oCT0 = op_pv.get_destination_cooperative_tensor<sik_strip_as<decltype(tP)>, sik_strip_as<decltype(sV0)>, float>();
    auto oCT1 = op_pv.get_destination_cooperative_tensor<sik_strip_as<decltype(tP)>, sik_strip_as<decltype(sV0)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) { oCT0[i] = 0.0f; oCT1[i] = 0.0f; }
    }

    const uint k_start = kv_chunk_beg + ((kv_min_beg - kv_chunk_beg) & ~31u);
    const uint q_first = p.seq_pos_start + m0;
    const uint q_last  = p.seq_pos_start + m0 + mb - 1u;
    const uint l_tok   = lane >> 3u;
    const uint l_blk   = lane & 7u;

    for (uint kb = k_start; kb < kv_max_end; kb += 32u) {
        const uint k_slot = kb;
        const uint tc     = min(32u, kv_max_end - kb);

        // Coalesced 16B `uint4` Q8_0 K dequant -> `tg_kv[32][128]`
        #pragma clang loop unroll(full)
        for (uint it = 0u; it < 8u; ++it) {
            const uint  tr    = it * 4u + l_tok;
            const uint  pos_r = k_slot + tr;
            const half  sc    = (tr < tc) ? kh_s[pos_r] : 0.0h;
            const uint4 q16   = (tr < tc) ? kh_q[ulong(pos_r) * 8ul + l_blk] : uint4(0u);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&tg_kv[tr][l_blk * 16u]);
            dst[0] = half4(as_type<char4>(q16.x)) * sc;
            dst[1] = half4(as_type<char4>(q16.y)) * sc;
            dst[2] = half4(as_type<char4>(q16.z)) * sc;
            dst[3] = half4(as_type<char4>(q16.w)) * sc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        auto sCT = op_qk.get_destination_cooperative_tensor<sik_strip_as<decltype(qCT)>, sik_strip_as<decltype(tKV)>, float>();
        op_qk.run(qCT, tKV, sCT);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < sCT.get_capacity(); ++i) {
            if (sCT.is_valid_element(i)) {
                auto idx = sCT.get_multidimensional_index(i);
                tg_s[uint(idx[1]) * 32u + uint(idx[0])] = sCT[i];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const int  d_min_raw    = int(q_first) - int(kb + tc - 1u);
        const int  d_max_raw    = int(q_last)  - int(kb);
        const bool tile_has_rel = (d_max_raw >= 0) && (d_min_raw < int(p.rel_extent));
        const uint kp           = kb + lane;

        for (uint rr = 0u; rr < n_rows; ++rr) {
            const uint m_loc = rr / 3u;
            const uint qp    = p.seq_pos_start + m0 + m_loc;
            const bool valid = ((active_mask & (1u << m_loc)) != 0u) && (lane < tc) && (kp <= qp);
            float sc = -INFINITY;
            if (valid) {
                sc = tg_s[rr * 32u + lane];
                if (tile_has_rel) {
                    const uint dist = qp - kp;
                    if (dist < p.rel_extent) {
                        device const float4* rp4 = reinterpret_cast<device const float4*>(rel_proj + ulong(dist) * 16ul);
                        const float4 r0(tg_r[rr][0], tg_r[rr][1], tg_r[rr][2], tg_r[rr][3]);
                        const float4 r1(tg_r[rr][4], tg_r[rr][5], tg_r[rr][6], tg_r[rr][7]);
                        const float4 r2(tg_r[rr][8], tg_r[rr][9], tg_r[rr][10], tg_r[rr][11]);
                        const float4 r3(tg_r[rr][12], tg_r[rr][13], tg_r[rr][14], tg_r[rr][15]);
                        sc += (dot(r0, rp4[0]) + dot(r1, rp4[1])) + (dot(r2, rp4[2]) + dot(r3, rp4[3]));
                    }
                }
            }
            const float tmax = simd_max(sc);
            if (tmax > -INFINITY) {
                const float mold  = tg_m[rr];
                const float mnew  = max(mold, tmax);
                const float alpha = (mold == -INFINITY) ? 0.0f : exp(mold - mnew);
                const float w     = valid ? exp(sc - mnew) : 0.0f;
                const float lsum  = simd_sum(w);
                tg_p[rr * 32u + lane] = sik_clamp_f16(w);
                if (lane == 0u) {
                    tg_m[rr] = mnew;
                    tg_l[rr] = fma(tg_l[rr], alpha, lsum);
                    tg_a[rr] = alpha;
                }
            } else {
                tg_p[rr * 32u + lane] = 0.0h;
                if (lane == 0u) tg_a[rr] = 1.0f;
            }
        }
        for (uint rr = n_rows; rr < 16u; ++rr) {
            tg_p[rr * 32u + lane] = 0.0h;
            if (lane == 0u) tg_a[rr] = 1.0f;
        }

        // Coalesced 16B `uint4` Q8_0 V dequant -> reuse `tg_kv[32][128]`!
        #pragma clang loop unroll(full)
        for (uint it = 0u; it < 8u; ++it) {
            const uint  tr    = it * 4u + l_tok;
            const uint  pos_r = k_slot + tr;
            const half  sc    = (tr < tc) ? vh_s[pos_r] : 0.0h;
            const uint4 q16   = (tr < tc) ? vh_q[ulong(pos_r) * 8ul + l_blk] : uint4(0u);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&tg_kv[tr][l_blk * 16u]);
            dst[0] = half4(as_type<char4>(q16.x)) * sc;
            dst[1] = half4(as_type<char4>(q16.y)) * sc;
            dst[2] = half4(as_type<char4>(q16.z)) * sc;
            dst[3] = half4(as_type<char4>(q16.w)) * sc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
            if (oCT0.is_valid_element(i)) {
                const float a = tg_a[uint(oCT0.get_multidimensional_index(i)[1])];
                oCT0[i] *= a;
                oCT1[i] *= a;
            }
        }
        op_pv.run(tP, sV0, oCT0);
        op_pv.run(tP, sV1, oCT1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint B_shift = ctz(B);
    uint m_slot[5];
    for (uint m = 0u; m < mb; ++m) m_slot[m] = c;

    threadgroup float* tg_out  = reinterpret_cast<threadgroup float*>(tg_union);
    device float4*     part_o4 = reinterpret_cast<device float4*>(part_o);

    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) {
            auto idx = oCT0.get_multidimensional_index(i);
            tg_out[uint(idx[1]) * 64u + uint(idx[0])] = oCT0[i];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rr = 0u; rr < n_rows; ++rr) {
        const uint m_loc = rr / 3u;
        if ((active_mask & (1u << m_loc)) != 0u && lane < 16u) {
            const uint  hq   = hkv * 3u + (rr - m_loc * 3u);
            const ulong oidx = (ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * p.max_splits + m_slot[m_loc];
            part_o4[oidx * 32ul + lane] = reinterpret_cast<threadgroup const float4*>(tg_out + rr * 64u)[lane];
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT1.get_capacity(); ++i) {
        if (oCT1.is_valid_element(i)) {
            auto idx = oCT1.get_multidimensional_index(i);
            tg_out[uint(idx[1]) * 64u + uint(idx[0])] = oCT1[i];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rr = 0u; rr < n_rows; ++rr) {
        const uint m_loc = rr / 3u;
        if ((active_mask & (1u << m_loc)) != 0u && lane < 16u) {
            const uint  hq   = hkv * 3u + (rr - m_loc * 3u);
            const ulong oidx = (ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * p.max_splits + m_slot[m_loc];
            part_o4[oidx * 32ul + 16u + lane] = reinterpret_cast<threadgroup const float4*>(tg_out + rr * 64u)[lane];
        }
    }

    if (lane < n_rows) {
        const uint m_loc = lane / 3u;
        if ((active_mask & (1u << m_loc)) != 0u) {
            const uint  hq   = hkv * 3u + (lane - m_loc * 3u);
            const ulong oidx = (ulong(p.m_offset + m0 + m_loc) * p.n_q + hq) * p.max_splits + m_slot[m_loc];
            part_m[oidx] = tg_m[lane];
            part_l[oidx] = tg_l[lane];
        }
    }
    (void)B_shift;
#else
    (void)q_in;(void)r_in;(void)rel_proj;(void)k_cache;(void)v_cache;(void)part_m;(void)part_l;(void)part_o;(void)p;(void)k_scales;(void)v_scales;(void)tgid;(void)lane;
#endif
}

// Per-Head Q8_0 Global KV NAX FlashAttention Prefill (4 SGs cooperatively dequant 1 shared 8 KiB KV_tg
// unioned with S_tg -> 16.5 KiB total TG mem, 4x less KV L2/DRAM traffic vs per-SG loads!)
// Grid: `((M_valid + 63) / 64, 24, 1)`, `128` threads/TG (`4` SGs)
kernel void sik_attn_prefill_nax_q8(
    device const half*      q_hm     [[buffer(0)]],
    device const float*     r_hm     [[buffer(1)]],
    device const float*     rel_proj [[buffer(2)]],
    device const float*     sink_in  [[buffer(3)]],
    device const float*     gate_in  [[buffer(4)]],
    device const uint4*     k_cache  [[buffer(5)]], // [8, kv_stride, 8] uint4
    device const uint4*     v_cache  [[buffer(6)]], // [8, kv_stride, 8] uint4
    device half*            out_f16  [[buffer(7)]],
    constant SikAttnParams& p        [[buffer(8)]],
    device const half*      k_scales [[buffer(9)]], // [8, kv_stride] half
    device const half*      v_scales [[buffer(10)]],// [8, kv_stride] half
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
#if defined(__HAVE_TENSOR__)
    const uint hq    = tgid.y;
    const uint m_tg0 = tgid.x * 64u;
    if (m_tg0 >= p.M_valid || hq >= p.n_q) return;

    const uint hkv       = hq / 3u;
    const uint m0        = m_tg0 + sgid * 16u;
    const bool sg_active = (m0 < p.M_valid);

    // Union `KV_tg[32][128]` (8 KiB) with `S_tg[4][16][32]` (8 KiB) -> 16.5 KiB total TG mem!
    threadgroup ulong tg_skv[1024]; // 8192 B
    threadgroup half  P_tg[4][16][32];
    threadgroup float R_tg[4][16][16];
    threadgroup float M_tg[4][16];
    threadgroup float L_tg[4][16];
    threadgroup float A_tg[4][16];

    threadgroup half*  KV_tg_ptr = reinterpret_cast<threadgroup half*>(tg_skv);
    threadgroup float* S_tg_ptr  = reinterpret_cast<threadgroup float*>(tg_skv);

    if (lane < 16u) {
        const uint mg    = m0 + lane;
        M_tg[sgid][lane] = (mg < p.M_valid) ? sink_in[ulong(mg) * p.n_q + hq] : 0.0f;
        L_tg[sgid][lane] = 1.0f;
    }
    if (sg_active) {
        device const float* r_base = r_hm + (ulong(hq) * p.M_pad16 + m0) * 16ul;
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
            const uint lin = i * 32u + lane;
            R_tg[sgid][lin >> 4u][lin & 15u] = r_base[lin];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half*        qh   = const_cast<device half*>(q_hm) + ulong(hq) * p.M_pad16 * 128ul;
    device const uint4* kh_q = k_cache  + ulong(hkv) * ulong(p.kv_stride) * 8ul;
    device const uint4* vh_q = v_cache  + ulong(hkv) * ulong(p.kv_stride) * 8ul;
    device const half*  kh_s = k_scales + ulong(hkv) * ulong(p.kv_stride);
    device const half*  vh_s = v_scales + ulong(hkv) * ulong(p.kv_stride);

    tensor<device half, dextents<int, 2>, tensor_inline>            tQ(qh, dextents<int, 2>(128, int(p.M_pad16)));
    tensor<threadgroup half, extents<int, 128, 32>, tensor_inline>  tKV(KV_tg_ptr, extents<int, 128, 32>());
    tensor<threadgroup half, extents<int, 32, 16>, tensor_inline>   tP(&P_tg[sgid][0][0], extents<int, 32, 16>());

    constexpr auto desc_qk = matmul2d_descriptor(
        16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;
    constexpr auto desc_pv = matmul2d_descriptor(
        16, 64, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    auto qCT = op_qk.get_left_input_cooperative_tensor<half, half, float>();
    if (sg_active) {
        auto sQ = tQ.slice<128, 16>(0, int(m0));
        qCT.load(sQ);
    }

    auto sV0  = tKV.slice<64, 32>(0, 0);
    auto sV1  = tKV.slice<64, 32>(64, 0);
    auto oCT0 = op_pv.get_destination_cooperative_tensor<sik_strip_as<decltype(tP)>, sik_strip_as<decltype(sV0)>, float>();
    auto oCT1 = op_pv.get_destination_cooperative_tensor<sik_strip_as<decltype(tP)>, sik_strip_as<decltype(sV0)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
        if (oCT0.is_valid_element(i)) { oCT0[i] = 0.0f; oCT1[i] = 0.0f; }
    }

    const uint q_first   = p.seq_pos_start + min(m0, p.M_valid - 1u);
    const uint q_last    = p.seq_pos_start + min(m0 + 15u, p.M_valid - 1u);
    const uint k_end_tg  = p.seq_pos_start + min(m_tg0 + 63u, p.M_valid - 1u) + 1u;
    const uint t_row0    = tid >> 3u; // 0..15
    const uint t_blk     = tid & 7u;  // 0..7

    for (uint kb = 0u; kb < k_end_tg; kb += 32u) {
        const uint k_tile_end = min(kb + 32u, k_end_tg);
        const uint tc         = k_tile_end - kb;

        // 1. All 128 threads cooperatively dequant K[kb..kb+31] -> KV_tg[32][128] (2 uint4/thread!)
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 2u; ++q) {
            const uint  tr    = q * 16u + t_row0;
            const uint  pos_r = kb + tr;
            const half  sc    = (tr < tc) ? kh_s[pos_r] : 0.0h;
            const uint4 q16   = (tr < tc) ? kh_q[ulong(pos_r) * 8ul + t_blk] : uint4(0u);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(KV_tg_ptr + tr * 128u + t_blk * 16u);
            dst[0] = half4(as_type<char4>(q16.x)) * sc;
            dst[1] = half4(as_type<char4>(q16.y)) * sc;
            dst[2] = half4(as_type<char4>(q16.z)) * sc;
            dst[3] = half4(as_type<char4>(q16.w)) * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const bool sg_run = sg_active && (kb <= q_last);
        auto sCT = op_qk.get_destination_cooperative_tensor<sik_strip_as<decltype(qCT)>, sik_strip_as<decltype(tKV)>, float>();
        if (sg_run) {
            op_qk.run(qCT, tKV, sCT);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // KV_tg dead -> now safe to reuse as S_tg!

        if (sg_run) {
            threadgroup float* s_sg = S_tg_ptr + sgid * 512u;
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < sCT.get_capacity(); ++i) {
                if (sCT.is_valid_element(i)) {
                    auto idx = sCT.get_multidimensional_index(i);
                    s_sg[uint(idx[1]) * 32u + uint(idx[0])] = sCT[i];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            const int  d_min_raw    = int(q_first) - int(k_tile_end - 1u);
            const int  d_max_raw    = int(q_last)  - int(kb);
            const bool tile_has_rel = (d_max_raw >= 0) && (d_min_raw < int(p.rel_extent));
            const uint k_pos        = kb + lane;

            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 16u; ++r) {
                const uint mg    = m0 + r;
                const uint qp    = p.seq_pos_start + mg;
                const bool valid = (mg < p.M_valid) && (k_pos <= qp);
                float sc = valid ? s_sg[r * 32u + lane] : -INFINITY;
                if (tile_has_rel && valid) {
                    const uint dist = qp - k_pos;
                    if (dist < p.rel_extent) {
                        threadgroup const float4* r4 = reinterpret_cast<threadgroup const float4*>(&R_tg[sgid][r][0]);
                        device const float4*      w4 = reinterpret_cast<device const float4*>(rel_proj + ulong(dist) * 16ul);
                        sc += (dot(r4[0], w4[0]) + dot(r4[1], w4[1]))
                            + (dot(r4[2], w4[2]) + dot(r4[3], w4[3]));
                    }
                }
                const float rmax = simd_max(sc);
                const float mp   = M_tg[sgid][r];
                const float mn   = max(mp, rmax);
                const float al   = exp(mp - mn);
                const float pv   = valid ? exp(sc - mn) : 0.0f;
                const float ps   = simd_sum(pv);
                P_tg[sgid][r][lane] = half(pv);
                if (lane == 0u) {
                    M_tg[sgid][r] = mn;
                    L_tg[sgid][r] = fma(L_tg[sgid][r], al, ps);
                    A_tg[sgid][r] = al;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // S_tg dead -> now safe to load V into KV_tg!

        // 2. All 128 threads cooperatively dequant V[kb..kb+31] -> KV_tg[32][128]
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 2u; ++q) {
            const uint  tr    = q * 16u + t_row0;
            const uint  pos_r = kb + tr;
            const half  sc    = (tr < tc) ? vh_s[pos_r] : 0.0h;
            const uint4 q16   = (tr < tc) ? vh_q[ulong(pos_r) * 8ul + t_blk] : uint4(0u);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(KV_tg_ptr + tr * 128u + t_blk * 16u);
            dst[0] = half4(as_type<char4>(q16.x)) * sc;
            dst[1] = half4(as_type<char4>(q16.y)) * sc;
            dst[2] = half4(as_type<char4>(q16.z)) * sc;
            dst[3] = half4(as_type<char4>(q16.w)) * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_run) {
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
                if (oCT0.is_valid_element(i)) {
                    auto idx = oCT0.get_multidimensional_index(i);
                    const float al = A_tg[sgid][uint(idx[1])];
                    oCT0[i] *= al;
                    oCT1[i] *= al;
                }
            }
            op_pv.run(tP, sV0, oCT0);
            op_pv.run(tP, sV1, oCT1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg_active) {
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < oCT0.get_capacity(); ++i) {
            if (oCT0.is_valid_element(i)) {
                auto idx = oCT0.get_multidimensional_index(i);
                const uint r  = uint(idx[1]);
                const uint c  = uint(idx[0]);
                const uint mg = m0 + r;
                if (mg < p.M_valid) {
                    const float invL    = 1.0f / L_tg[sgid][r];
                    const ulong out_off = (ulong(mg) * p.n_q + hq) * 128ul + c;
                    const float g0      = gate_in[out_off];
                    const float g1      = gate_in[out_off + 64u];
                    out_f16[out_off]       = sik_clamp_f16((oCT0[i] * invL) * sik_sigmoid(g0));
                    out_f16[out_off + 64u] = sik_clamp_f16((oCT1[i] * invL) * sik_sigmoid(g1));
                }
            }
        }
    }
#else
    (void)q_hm;(void)r_hm;(void)rel_proj;(void)sink_in;(void)gate_in;(void)k_cache;(void)v_cache;(void)out_f16;(void)p;(void)k_scales;(void)v_scales;(void)tgid;(void)tid;(void)sgid;(void)lane;
#endif
}

/* ============================================================================
 * 7. FP32 MoE Router (Tiled simdgroup_float8x8 Prefill + Unrolled Decode) +
 *    Dual-Bin Indirect NAX Prefill (BM=32/64) + 16-Lane Coalesced Decode
 * ============================================================================ */

struct SikMoeRouterParams {
    uint  M_valid;
    uint  D;
    uint  num_experts; // 128
    uint  top_k;       // 8
    float route_scale; // 8.0f
    uint  inter_size;  // 1152 (for writing indirect dispatch args!)
    uint  pad0;
    uint  pad1;
};

// High-MLP 128-Thread Decode FP32 Router (`M <= 8`, specialized by `SIK_CONST_M = 1..8`, 100% bit-exact K>0 == K=0!)
// Grid: `(128, 1, 1)`, `128` threads/TG (`4` SGs -> 4x MLP per expert!)
template <uint M_FIX>
inline void sik_moe_router_f32_fixed(
    device const float4*         gate_w,
    device const float4*         x_f32,
    device float*                logits,
    constant SikMoeRouterParams& p,
    threadgroup float            (&tg_red)[8][4],
    uint e, uint tid, uint sgid, uint lane
) {
    const uint d4_cnt = p.D >> 2u;
    device const float4* w_row = gate_w + ulong(e) * d4_cnt;

    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = tid; i < d4_cnt; i += 128u) {
        const float4 wv = w_row[i];
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            const float4 xv = x_f32[ulong(m) * d4_cnt + i];
            acc[m] = fma(wv.x, xv.x, fma(wv.y, xv.y, fma(wv.z, xv.z, fma(wv.w, xv.w, acc[m]))));
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            const float tot = simd_sum((lane < 4u) ? tg_red[m][lane] : 0.0f);
            if (lane == 0u) logits[ulong(m) * p.num_experts + e] = tot;
        }
    }
}

kernel void sik_moe_router_f32(
    device const float4*         gate_w [[buffer(0)]],
    device const float4*         x_f32  [[buffer(1)]],
    device float*                logits [[buffer(2)]],
    constant SikMoeRouterParams& p      [[buffer(3)]],
    uint e [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    if (e >= p.num_experts) return;
    threadgroup float tg_red[8][4];
    #define SIK_RTR_CASE(MF) { sik_moe_router_f32_fixed<MF>(gate_w, x_f32, logits, p, tg_red, e, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_RTR_CASE(1u) case 2u: SIK_RTR_CASE(2u) case 3u: SIK_RTR_CASE(3u) case 4u: SIK_RTR_CASE(4u)
        case 5u: SIK_RTR_CASE(5u) case 6u: SIK_RTR_CASE(6u) case 7u: SIK_RTR_CASE(7u) case 8u: SIK_RTR_CASE(8u)
        default: break;
    }
    #undef SIK_RTR_CASE
}

// `M == 1` Fused Router + Sigmoid + Bias (writes `float2(sc, sc + bias)` -> eliminates `48x sik_moe_topk_m1`!)
kernel void sik_moe_router_f32_m1(
    device const float4*         gate_w    [[buffer(0)]],
    device const float4*         x_f32     [[buffer(1)]],
    device const float*          gate_bias [[buffer(2)]],
    device float2*               out_pairs [[buffer(3)]],
    constant SikMoeRouterParams& p         [[buffer(4)]],
    uint e [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    if (e >= p.num_experts) return;
    threadgroup float tg_red[4];
    const uint d4_cnt = p.D >> 2u;
    device const float4* w_row = gate_w + ulong(e) * d4_cnt;
    float acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 128u) {
        const float4 wv = w_row[i];
        const float4 xv = x_f32[i];
        acc = fma(wv.x, xv.x, fma(wv.y, xv.y, fma(wv.z, xv.z, fma(wv.w, xv.w, acc))));
    }
    const float s = simd_sum(acc);
    if (lane == 0u) tg_red[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot = simd_sum((lane < 4u) ? tg_red[lane] : 0.0f);
        if (lane == 0u) {
            const float sc = sik_sigmoid(tot);
            out_pairs[e] = float2(sc, sc + gate_bias[e]);
        }
    }
}

// Fused `PreMoE RMSNorm + MoE Router` (`M = 1..8`, eliminates `24x PreMoE_RMS1` AND `24x TopK_M1`!)
// `e == 0` emits `out_f16` (`b_in_f16[0]`) for MoE W13; 100.000000% bit-exact `K=7 == K=0`!
struct SikMoeRouterRmsParams {
    uint  M_valid;
    uint  D;
    uint  num_experts;
    uint  top_k;
    float eps;
    uint  n_sq_tiles;
    uint  emit_m1_pairs;
    uint  pad0;
};

template <uint M_FIX>
inline void sik_moe_router_f32_rms_fixed(
    device const float4*            gate_w,
    device const float4*            inp_f32,
    device const float4*            mlp_norm,
    device const float*             rms_sq,
    device const float*             gate_bias,
    device float*                   logits,
    device half4*                   out_f16,
    constant SikMoeRouterRmsParams& p,
    threadgroup float               (&tg_inv)[8],
    threadgroup float               (&tg_red)[8][4],
    uint e, uint tid, uint sgid, uint lane
) {
    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            float part = 0.0f;
            for (uint t = lane; t < p.n_sq_tiles; t += 32u) {
                part += rms_sq[ulong(m) * p.n_sq_tiles + t];
            }
            const float tot = simd_sum(part);
            if (lane == 0u) tg_inv[m] = rsqrt(tot / float(p.D) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint d4_cnt = p.D >> 2u;
    device const float4* w_row = gate_w + ulong(e) * d4_cnt;
    float acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = 0.0f;

    for (uint i = tid; i < d4_cnt; i += 128u) {
        const float4 wv = w_row[i];
        const float4 nw = mlp_norm[i];
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            const ulong  off = ulong(m) * d4_cnt + i;
            const float4 xv  = (inp_f32[off] * tg_inv[m]) * nw;
            if (e == 0u) out_f16[off] = sik_clamp_f16x4(xv);
            acc[m] = fma(wv.x, xv.x, fma(wv.y, xv.y, fma(wv.z, xv.z, fma(wv.w, xv.w, acc[m]))));
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float s = simd_sum(acc[m]);
        if (lane == 0u) tg_red[m][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            const float tot = simd_sum((lane < 4u) ? tg_red[m][lane] : 0.0f);
            if (lane == 0u) {
                if (M_FIX == 1u && p.emit_m1_pairs != 0u) {
                    const float sc = sik_sigmoid(tot);
                    reinterpret_cast<device float2*>(logits)[e] = float2(sc, sc + gate_bias[e]);
                } else {
                    logits[ulong(m) * p.num_experts + e] = tot;
                }
            }
        }
    }
}

kernel void sik_moe_router_f32_rms(
    device const float4*            gate_w    [[buffer(0)]],
    device const float4*            inp_f32   [[buffer(1)]],
    device const float4*            mlp_norm  [[buffer(2)]],
    device const float*             rms_sq    [[buffer(3)]],
    device const float*             gate_bias [[buffer(4)]],
    device float*                   logits    [[buffer(5)]],
    device half4*                   out_f16   [[buffer(6)]],
    constant SikMoeRouterRmsParams& p         [[buffer(7)]],
    uint e [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    if (e >= p.num_experts) return;
    threadgroup float tg_inv[8];
    threadgroup float tg_red[8][4];
    #define SIK_RTR_RMS_CASE(MF) { sik_moe_router_f32_rms_fixed<MF>(gate_w, inp_f32, mlp_norm, rms_sq, gate_bias, logits, out_f16, p, tg_inv, tg_red, e, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_RTR_RMS_CASE(1u) case 2u: SIK_RTR_RMS_CASE(2u) case 3u: SIK_RTR_RMS_CASE(3u) case 4u: SIK_RTR_RMS_CASE(4u)
        case 5u: SIK_RTR_RMS_CASE(5u) case 6u: SIK_RTR_RMS_CASE(6u) case 7u: SIK_RTR_RMS_CASE(7u) case 8u: SIK_RTR_RMS_CASE(8u)
        default: break;
    }
    #undef SIK_RTR_RMS_CASE
}

// 100% Pure FP32 Tiled `simdgroup_float8x8` Prefill MoE Router (`BM=16, BN=32, BK=16`, 4 SGs/TG -> 128 TGs on pp512!)
// Cuts router from `12.2 ms -> ~0.8 ms` with ZERO FP16 routing drift!
// Grid: `(num_experts / 32, (M_valid + 15) / 16, 1)`, `128` threads/TG (`4` SGs)
kernel void sik_moe_router_f32_prefill(
    device const float*          gate_w [[buffer(0)]], // [128, D]
    device const float*          x_f32  [[buffer(1)]], // [M_valid, D]
    device float*                logits [[buffer(2)]], // [M_valid, 128]
    constant SikMoeRouterParams& p      [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint n0 = tgid.x * 32u;
    const uint m0 = tgid.y * 16u;
    if (n0 >= p.num_experts || m0 >= p.M_valid) return;

    threadgroup float X_tg[16][16];
    threadgroup float W_tg[16][32]; // transposed [k][n]
    threadgroup float C_tg[16][32];

    const uint sg_m = sgid >> 1u; // 0..1 (8 rows each)
    const uint sg_n = sgid & 1u;  // 0..1 (16 cols each = two 8x8 tiles)

    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint xr = tid >> 3u;  // 0..15 (row in X_tg)
    const uint xc = (tid & 7u) * 2u; // 0,2..14 (2 floats/thread)
    const uint wr_base = tid >> 2u;  // 0..31 (row in gate_w = col in W_tg)
    const uint wk_base = (tid & 3u) * 4u; // 0,4,8,12 (4 floats/thread)

    for (uint k0 = 0u; k0 < p.D; k0 += 16u) {
        // Load X_tg[16][16] (256 floats / 128 threads = 2 floats/thread)
        const uint mg = m0 + xr;
        if (mg < p.M_valid) {
            X_tg[xr][xc + 0u] = x_f32[ulong(mg) * p.D + k0 + xc + 0u];
            X_tg[xr][xc + 1u] = x_f32[ulong(mg) * p.D + k0 + xc + 1u];
        } else {
            X_tg[xr][xc + 0u] = 0.0f;
            X_tg[xr][xc + 1u] = 0.0f;
        }
        // Load W_tg[16][32] transposed (512 floats / 128 threads = 4 floats/thread)
        const uint ng = n0 + wr_base;
        if (ng < p.num_experts) {
            device const float* wp = gate_w + ulong(ng) * p.D + k0 + wk_base;
            W_tg[wk_base + 0u][wr_base] = wp[0];
            W_tg[wk_base + 1u][wr_base] = wp[1];
            W_tg[wk_base + 2u][wr_base] = wp[2];
            W_tg[wk_base + 3u][wr_base] = wp[3];
        } else {
            W_tg[wk_base + 0u][wr_base] = 0.0f;
            W_tg[wk_base + 1u][wr_base] = 0.0f;
            W_tg[wk_base + 2u][wr_base] = 0.0f;
            W_tg[wk_base + 3u][wr_base] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a0, a1, b00, b10, b01, b11;
        simdgroup_load(a0,  &X_tg[sg_m * 8u][0],  16);
        simdgroup_load(a1,  &X_tg[sg_m * 8u][8],  16);
        simdgroup_load(b00, &W_tg[0][sg_n * 16u + 0u], 32);
        simdgroup_load(b10, &W_tg[8][sg_n * 16u + 0u], 32);
        simdgroup_load(b01, &W_tg[0][sg_n * 16u + 8u], 32);
        simdgroup_load(b11, &W_tg[8][sg_n * 16u + 8u], 32);

        simdgroup_multiply_accumulate(acc0, a0, b00, acc0);
        simdgroup_multiply_accumulate(acc0, a1, b10, acc0);
        simdgroup_multiply_accumulate(acc1, a0, b01, acc1);
        simdgroup_multiply_accumulate(acc1, a1, b11, acc1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc0, &C_tg[sg_m * 8u][sg_n * 16u + 0u], 32);
    simdgroup_store(acc1, &C_tg[sg_m * 8u][sg_n * 16u + 8u], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 512u; idx += 128u) {
        const uint r = idx >> 5u;
        const uint c = idx & 31u;
        if (m0 + r < p.M_valid && n0 + c < p.num_experts) {
            logits[ulong(m0 + r) * p.num_experts + (n0 + c)] = C_tg[r][c];
        }
    }
}

kernel void sik_moe_topk_sigmoid(
    device const float*          logits    [[buffer(0)]],
    device const float*          gate_bias [[buffer(1)]],
    device uint*                 topk_idx  [[buffer(2)]],
    device float*                topk_w    [[buffer(3)]],
    constant SikMoeRouterParams& p         [[buffer(4)]],
    uint m    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_valid) return;
    device const float* row = logits + ulong(m) * p.num_experts;

    float scores[4], sel[4];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
        const uint e = j * 32u + lane;
        if (e < p.num_experts) {
            const float sc = sik_sigmoid(row[e]);
            scores[j] = sc;
            sel[j]    = sc + gate_bias[e];
        } else {
            scores[j] = 0.0f;
            sel[j]    = -INFINITY;
        }
    }

    uint  chosen_id[8];
    float chosen_sc[8];
    const uint K = min(p.top_k, 8u);

    for (uint k = 0u; k < K; ++k) {
        float best_sel = -INFINITY, best_sc = 0.0f;
        uint  best_e   = 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            const uint  e = j * 32u + lane;
            const float v = sel[j];
            if (v > best_sel || (v == best_sel && e < best_e)) {
                best_sel = v; best_sc = scores[j]; best_e = e;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_sel, off), osc = simd_shuffle_xor(best_sc, off);
            const uint  oe = simd_shuffle_xor(best_e, off);
            if (ov > best_sel || (ov == best_sel && oe < best_e)) {
                best_sel = ov; best_sc = osc; best_e = oe;
            }
        }
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            if (j * 32u + lane == best_e) sel[j] = -INFINITY;
        }
        chosen_id[k] = (best_e < p.num_experts) ? best_e : 0u;
        chosen_sc[k] = best_sc;
    }

    if (lane == 0u) {
        float sum_sc = 0.0f;
        for (uint k = 0u; k < K; ++k) sum_sc += chosen_sc[k];
        const float inv_s = p.route_scale / (sum_sc + 1e-20f);
        for (uint k = 0u; k < K; ++k) {
            topk_idx[ulong(m) * p.top_k + k] = chosen_id[k];
            topk_w[ulong(m) * p.top_k + k]   = chosen_sc[k] * inv_s;
        }
    }
}

struct SikMoeWorkTile {
    uint expert_id;
    uint pair_start;
    uint pair_count;
    uint pad;
};

struct SikMoeExecParams {
    uint  M_valid;
    uint  top_k;             // 8
    uint  D;                 // 2304
    uint  intermediate_size; // 1152
    uint  num_experts;       // 128
    uint  tile_cap;          // 8 (decode) or 64 (prefill dual-bin)
    float gate_softcap;      // 4.0f
    float up_softcap;        // 25.0f
};

// Quad-Bin Prefill Token Binner (`BM = 16 / 32 / 64 / 128` into 1 strided `work_tiles[2048]` buffer!):
//   - `bin 0` (`work_tiles +    0`): `1..16`  toks -> `BM=16` (`SG=2`) NAX
//   - `bin 1` (`work_tiles +  512`): `17..32` toks -> `BM=32` (`SG=4`) NAX
//   - `bin 2` (`work_tiles + 1024`): `33..64` toks -> `BM=64` (`SG=4`) NAX
//   - `bin 3` (`work_tiles + 1536`): `65..128` toks (and full 128-tok chunks) -> `BM=128` (`SG=4`) NAX (1x weight read!)
//   - Writes 8 `MTLDispatchThreadgroupsIndirectArguments` (16B-strided!) directly into `indirect_args`!
constant uint SIK_MOE_BIN_STRIDE = 512u;

kernel void sik_moe_bin_tokens_quad(
    device const uint*         topk_idx      [[buffer(0)]],
    device uint*               sorted_pairs  [[buffer(1)]],
    device SikMoeWorkTile*     work_tiles    [[buffer(2)]], // 4 bins x 512 tiles = 2048 tiles
    device uint*               num_tiles_io  [[buffer(3)]], // [0]=n16, [1]=n32, [2]=n64, [3]=n128
    device uint*               indirect_args [[buffer(4)]], // 8 x uint4 (w13/w2 for m16, m32, m64, m128)
    constant SikMoeExecParams& p             [[buffer(5)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup atomic_uint counts[128];
    threadgroup atomic_uint cursors[128];
    threadgroup uint        offs[128];
    threadgroup uint        b_offs[4][128];
    threadgroup uint        tg_sg[5][4];

    atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pairs = p.M_valid * p.top_k;
    for (uint i = tid; i < total_pairs; i += 128u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c_val   = (tid < p.num_experts) ? atomic_load_explicit(&counts[tid], memory_order_relaxed) : 0u;
    const bool cap64   = (p.tile_cap == 64u);
    const uint full128 = cap64 ? 0u : (c_val >> 7u);
    const uint full64  = cap64 ? (c_val >> 6u) : 0u;
    const uint rem     = cap64 ? (c_val & 63u) : (c_val & 127u);
    const uint n128    = cap64 ? 0u : (full128 + ((rem > 80u) ? 1u : 0u));
    const uint n64     = cap64 ? (full64 + ((rem > 32u) ? 1u : 0u)) : ((rem > 32u && rem <= 80u) ? 1u : 0u);
    const uint n32     = (rem > 16u && rem <= 32u) ? 1u : 0u;
    const uint n16     = cap64 ? ((rem > 0u && rem <= 16u) ? 1u : 0u)
                               : (((rem > 0u && rem <= 16u) || (rem > 64u && rem <= 80u)) ? 1u : 0u);

    // Fused 5-way prefix sum (`offs`, `n16`, `n32`, `n64`, `n128`) in ONE barrier pair!
    const uint c_pre   = simd_prefix_exclusive_sum(c_val);
    const uint n16_pre = simd_prefix_exclusive_sum(n16);
    const uint n32_pre = simd_prefix_exclusive_sum(n32);
    const uint n64_pre = simd_prefix_exclusive_sum(n64);
    const uint n128_pre = simd_prefix_exclusive_sum(n128);

    const uint c_sum    = simd_sum(c_val);
    const uint n16_sum  = simd_sum(n16);
    const uint n32_sum  = simd_sum(n32);
    const uint n64_sum  = simd_sum(n64);
    const uint n128_sum = simd_sum(n128);
    if (lane == 0u) {
        tg_sg[0][sgid] = c_sum;
        tg_sg[1][sgid] = n16_sum;
        tg_sg[2][sgid] = n32_sum;
        tg_sg[3][sgid] = n64_sum;
        tg_sg[4][sgid] = n128_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const uint inter_g = p.intermediate_size >> 6u; // BN=64 1-loop gate + up_swiglu (`43+ TF`!)
        const uint d_g     = p.D >> 6u;
        #pragma clang loop unroll(full)
        for (uint b = 0u; b < 5u; ++b) {
            const uint v   = (lane < 4u) ? tg_sg[b][lane] : 0u;
            const uint pre = simd_prefix_exclusive_sum(v);
            const uint tot = simd_sum(v);
            if (lane < 4u) tg_sg[b][lane] = pre;
            if (lane == 0u && b > 0u) {
                const uint bi = b - 1u;
                num_tiles_io[bi] = tot;
                const uint base  = bi * 8u;
                indirect_args[base + 0u] = (tot > 0u) ? inter_g : 0u;
                indirect_args[base + 1u] = tot;
                indirect_args[base + 2u] = (tot > 0u) ? 1u : 0u;
                indirect_args[base + 3u] = 0u;
                indirect_args[base + 4u] = (tot > 0u) ? d_g : 0u;
                indirect_args[base + 5u] = tot;
                indirect_args[base + 6u] = (tot > 0u) ? 1u : 0u;
                indirect_args[base + 7u] = 0u;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint exp_off = tg_sg[0][sgid] + c_pre;
    offs[tid]          = exp_off;
    b_offs[0][tid]     = tg_sg[1][sgid] + n16_pre;
    b_offs[1][tid]     = tg_sg[2][sgid] + n32_pre;
    b_offs[2][tid]     = tg_sg[3][sgid] + n64_pre;
    b_offs[3][tid]     = tg_sg[4][sgid] + n128_pre;
    atomic_store_explicit(&cursors[tid], exp_off, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter pairs into sorted order
    for (uint i = tid; i < total_pairs; i += 128u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) {
            const uint dst = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
            sorted_pairs[dst] = i;
        }
    }

    // Emit work tiles into quad bins (`0=16, 1=32, 2=64, 3=128`)
    if (tid < p.num_experts && c_val > 0u) {
        const uint base_pair = offs[tid];
        device SikMoeWorkTile* tiles_16  = work_tiles + 0u * SIK_MOE_BIN_STRIDE;
        device SikMoeWorkTile* tiles_32  = work_tiles + 1u * SIK_MOE_BIN_STRIDE;
        device SikMoeWorkTile* tiles_64  = work_tiles + 2u * SIK_MOE_BIN_STRIDE;
        device SikMoeWorkTile* tiles_128 = work_tiles + 3u * SIK_MOE_BIN_STRIDE;
        if (cap64) {
            uint dst64 = b_offs[2][tid];
            for (uint k = 0u; k < full64; ++k) {
                tiles_64[dst64++] = SikMoeWorkTile{tid, base_pair + k * 64u, 64u, 0u};
            }
            const uint rem_pair = base_pair + full64 * 64u;
            if (rem > 32u) {
                tiles_64[dst64++] = SikMoeWorkTile{tid, rem_pair, rem, 0u};
            } else if (rem > 16u) {
                tiles_32[b_offs[1][tid]] = SikMoeWorkTile{tid, rem_pair, rem, 0u};
            } else if (rem > 0u) {
                tiles_16[b_offs[0][tid]] = SikMoeWorkTile{tid, rem_pair, rem, 0u};
            }
        } else {
            uint dst128 = b_offs[3][tid];
            for (uint k = 0u; k < full128; ++k) {
                tiles_128[dst128++] = SikMoeWorkTile{tid, base_pair + k * 128u, 128u, 0u};
            }
            const uint rem_pair = base_pair + full128 * 128u;
            if (rem > 80u) {
                tiles_128[dst128++] = SikMoeWorkTile{tid, rem_pair, rem, 0u};
            } else if (rem > 64u) {
                tiles_64[b_offs[2][tid]] = SikMoeWorkTile{tid, rem_pair, 64u, 0u};
                tiles_16[b_offs[0][tid]] = SikMoeWorkTile{tid, rem_pair + 64u, rem - 64u, 0u};
            } else if (rem > 32u) {
                tiles_64[b_offs[2][tid]] = SikMoeWorkTile{tid, rem_pair, rem, 0u};
            } else if (rem > 16u) {
                tiles_32[b_offs[1][tid]] = SikMoeWorkTile{tid, rem_pair, rem, 0u};
            } else if (rem > 0u) {
                tiles_16[b_offs[0][tid]] = SikMoeWorkTile{tid, rem_pair, rem, 0u};
            }
        }
    }
}

// Single-Dispatch High-Occupancy Grouped-4 Token Binner (`M = 9..64`, tile_cap=4 -> `b[1]` 2x occupancy + 0 extra indirect barriers!)
kernel void sik_moe_bin_tokens_grouped(
    device const uint*         topk_idx      [[buffer(0)]],
    device uint*               sorted_pairs  [[buffer(1)]],
    device SikMoeWorkTile*     work_tiles    [[buffer(2)]],
    device uint*               num_tiles     [[buffer(3)]],
    device uint*               indirect_args [[buffer(4)]], // 2 x uint4 ([0]=w13, [1]=w2)
    constant SikMoeExecParams& p             [[buffer(5)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup atomic_uint counts[128];
    threadgroup atomic_uint cursors[128];
    threadgroup uint        offs[128];
    threadgroup uint        tile_offs[128];
    threadgroup uint        tg_c[4];
    threadgroup uint        tg_t[4];

    atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pairs = p.M_valid * p.top_k;
    for (uint i = tid; i < total_pairs; i += 128u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint tcap  = (p.tile_cap >= 1u && p.tile_cap <= 8u) ? p.tile_cap : 4u;
    const uint c_val = (tid < p.num_experts) ? atomic_load_explicit(&counts[tid], memory_order_relaxed) : 0u;
    const uint t_val = (c_val + tcap - 1u) / tcap;
    const uint c_pre = simd_prefix_exclusive_sum(c_val);
    const uint c_sum = simd_sum(c_val);
    const uint t_pre = simd_prefix_exclusive_sum(t_val);
    const uint t_sum = simd_sum(t_val);
    if (lane == 0u) { tg_c[sgid] = c_sum; tg_t[sgid] = t_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        const uint vc   = (lane < 4u) ? tg_c[lane] : 0u;
        const uint vt   = (lane < 4u) ? tg_t[lane] : 0u;
        const uint prec = simd_prefix_exclusive_sum(vc);
        const uint pret = simd_prefix_exclusive_sum(vt);
        const uint tot  = simd_sum(vt);
        if (lane < 4u) { tg_c[lane] = prec; tg_t[lane] = pret; }
        if (lane == 0u) {
            num_tiles[0]     = tot;
            indirect_args[0] = (tot > 0u) ? ((p.intermediate_size + 7u) / 8u) : 0u;
            indirect_args[1] = tot;
            indirect_args[2] = (tot > 0u) ? 1u : 0u;
            indirect_args[3] = 0u;
            indirect_args[4] = (tot > 0u) ? ((p.D + 7u) / 8u) : 0u;
            indirect_args[5] = tot;
            indirect_args[6] = (tot > 0u) ? 1u : 0u;
            indirect_args[7] = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint exp_off = tg_c[sgid] + c_pre;
    offs[tid]          = exp_off;
    tile_offs[tid]     = tg_t[sgid] + t_pre;
    atomic_store_explicit(&cursors[tid], exp_off, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < total_pairs; i += 128u) {
        const uint e = topk_idx[i];
        if (e < p.num_experts) {
            const uint dst = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
            sorted_pairs[dst] = i;
        }
    }
    if (tid < p.num_experts && t_val > 0u) {
        const uint base_pair = offs[tid];
        const uint base_tile = tile_offs[tid];
        for (uint ti = 0u; ti < t_val; ++ti) {
            const uint st  = ti * tcap;
            const uint cnt = min(tcap, c_val - st);
            work_tiles[base_tile + ti] = SikMoeWorkTile{tid, base_pair + st, cnt, 0u};
        }
    }
}

// Single-Dispatch 256-Thread (`8` SGs = 1 SG/token in parallel!) Fused `Top-8 + Grouped-4 Binning + Indirect Args` (`M <= 8`)!
// Zero local arrays, 1x TG-cached `gate_bias[128]`, 2x faster @ M=8 (`0.497 -> ~0.24 ms`), 100% bit-exact with M=1!
kernel void sik_moe_topk_bin_small(
    device const float*          logits        [[buffer(0)]],
    device const float*          gate_bias     [[buffer(1)]],
    device uint*                 topk_idx      [[buffer(2)]],
    device float*                topk_w        [[buffer(3)]],
    device uint*                 sorted_pairs  [[buffer(4)]],
    device SikMoeWorkTile*       work_tiles    [[buffer(5)]],
    device uint*                 num_tiles     [[buffer(6)]],
    device uint*                 indirect_args [[buffer(7)]], // 2 x uint4 ([0]=w13, [1]=w2)
    constant SikMoeRouterParams& p             [[buffer(8)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float       tg_bias[128];
    threadgroup uint        tg_topk[64];
    threadgroup atomic_uint counts[128];
    threadgroup atomic_uint cursors[128];
    threadgroup uint        offs[128];
    threadgroup uint        tile_offs[128];
    threadgroup uint        tg_c[4];
    threadgroup uint        tg_t[4];

    const uint K = min(p.top_k, 8u);

    if (tid < 128u) {
        atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
        tg_bias[tid] = (tid < p.num_experts) ? gate_bias[tid] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1 SG per token `m = sgid` in 100% parallel (up to 8 tokens simultaneously, ZERO local arrays!)
    if (sgid < p.M_valid) {
        device const float* row = logits + ulong(sgid) * p.num_experts;
        float4 scores, sel;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            const uint e = j * 32u + lane;
            if (e < p.num_experts) {
                const float sc = sik_sigmoid(row[e]);
                scores[j] = sc;
                sel[j]    = sc + tg_bias[e];
            } else {
                scores[j] = 0.0f; sel[j] = -INFINITY;
            }
        }
        float sum_sc = 0.0f;
        uint  my_id  = 0u;
        float my_sc  = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float best_sel = sel[0], best_sc = scores[0];
            uint  best_e   = lane;
            if (sel[1] > best_sel) { best_sel = sel[1]; best_sc = scores[1]; best_e = 32u + lane; }
            if (sel[2] > best_sel) { best_sel = sel[2]; best_sc = scores[2]; best_e = 64u + lane; }
            if (sel[3] > best_sel) { best_sel = sel[3]; best_sc = scores[3]; best_e = 96u + lane; }
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(best_sel, off), osc = simd_shuffle_xor(best_sc, off);
                const uint  oe = simd_shuffle_xor(best_e, off);
                if (ov > best_sel || (ov == best_sel && oe < best_e)) {
                    best_sel = ov; best_sc = osc; best_e = oe;
                }
            }
            if      (best_e == lane)       sel[0] = -INFINITY;
            else if (best_e == 32u + lane) sel[1] = -INFINITY;
            else if (best_e == 64u + lane) sel[2] = -INFINITY;
            else if (best_e == 96u + lane) sel[3] = -INFINITY;
            const uint eid = (best_e < p.num_experts) ? best_e : 0u;
            sum_sc += best_sc;
            if (lane == k) { my_id = eid; my_sc = best_sc; }
        }
        if (lane < K) {
            const float inv_s = p.route_scale / (sum_sc + 1e-20f);
            const uint  pid   = sgid * p.top_k + lane;
            topk_idx[pid] = my_id;
            topk_w[pid]   = my_sc * inv_s;
            tg_topk[pid]  = my_id;
            if (my_id < p.num_experts) atomic_fetch_add_explicit(&counts[my_id], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combined dual prefix sum (`offs` + `tile_offs`, `tid < 128` only!)
    const uint tcap  = (p.pad0 >= 1u && p.pad0 <= 8u) ? p.pad0 : 8u;
    const uint c_val = (tid < p.num_experts) ? atomic_load_explicit(&counts[tid], memory_order_relaxed) : 0u;
    const uint t_val = (c_val + tcap - 1u) / tcap;
    if (sgid < 4u) {
        const uint c_pre = simd_prefix_exclusive_sum(c_val);
        const uint c_sum = simd_sum(c_val);
        const uint t_pre = simd_prefix_exclusive_sum(t_val);
        const uint t_sum = simd_sum(t_val);
        if (lane == 0u) { tg_c[sgid] = c_sum; tg_t[sgid] = t_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const uint vc   = (lane < 4u) ? tg_c[lane] : 0u;
            const uint vt   = (lane < 4u) ? tg_t[lane] : 0u;
            const uint prec = simd_prefix_exclusive_sum(vc);
            const uint pret = simd_prefix_exclusive_sum(vt);
            const uint tot  = simd_sum(vt);
            if (lane < 4u) { tg_c[lane] = prec; tg_t[lane] = pret; }
            if (lane == 0u) {
                num_tiles[0]     = tot;
                indirect_args[0] = (p.inter_size + 7u) / 8u;
                indirect_args[1] = tot;
                indirect_args[2] = 1u;
                indirect_args[3] = 0u;
                indirect_args[4] = (p.D + 7u) / 8u;
                indirect_args[5] = tot;
                indirect_args[6] = 1u;
                indirect_args[7] = 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint exp_off = tg_c[sgid] + c_pre;
        offs[tid]          = exp_off;
        tile_offs[tid]     = tg_t[sgid] + t_pre;
        atomic_store_explicit(&cursors[tid], exp_off, memory_order_relaxed);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_pairs = p.M_valid * p.top_k;
    if (tid < total_pairs) {
        const uint e = tg_topk[tid];
        if (e < p.num_experts) {
            const uint dst = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
            sorted_pairs[dst] = tid;
        }
    }
    if (tid < p.num_experts && t_val > 0u) {
        const uint base_pair = offs[tid];
        const uint base_tile = tile_offs[tid];
        for (uint ti = 0u; ti < t_val; ++ti) {
            const uint st  = ti * tcap;
            const uint cnt = min(tcap, c_val - st);
            work_tiles[base_tile + ti] = SikMoeWorkTile{tid, base_pair + st, cnt, 0u};
        }
    }
}

// Ultra-Lean Zero-TG-Mem 1-SG (`32` th) `M == 1` Top-8 Selection + Direct Work Tile Emit (2.3x faster!)
kernel void sik_moe_topk_m1(
    device const float*          logits       [[buffer(0)]],
    device const float*          gate_bias    [[buffer(1)]],
    device uint*                 topk_idx     [[buffer(2)]],
    device float*                topk_w       [[buffer(3)]],
    device uint*                 sorted_pairs [[buffer(4)]],
    device SikMoeWorkTile*       work_tiles   [[buffer(5)]],
    device uint*                 num_tiles    [[buffer(6)]],
    constant SikMoeRouterParams& p            [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint K = min(p.top_k, 8u);
    float scores[4], sel[4];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
        const uint e = j * 32u + lane;
        if (e < p.num_experts) {
            const float sc = sik_sigmoid(logits[e]);
            scores[j] = sc;
            sel[j]    = sc + gate_bias[e];
        } else {
            scores[j] = 0.0f;
            sel[j]    = -INFINITY;
        }
    }
    uint  chosen_id[8];
    float chosen_sc[8];
    for (uint k = 0u; k < K; ++k) {
        float best_sel = -INFINITY, best_sc = 0.0f;
        uint  best_e   = 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            const uint  e = j * 32u + lane;
            const float v = sel[j];
            if (v > best_sel || (v == best_sel && e < best_e)) {
                best_sel = v; best_sc = scores[j]; best_e = e;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_sel, off), osc = simd_shuffle_xor(best_sc, off);
            const uint  oe = simd_shuffle_xor(best_e, off);
            if (ov > best_sel || (ov == best_sel && oe < best_e)) {
                best_sel = ov; best_sc = osc; best_e = oe;
            }
        }
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            if (j * 32u + lane == best_e) sel[j] = -INFINITY;
        }
        chosen_id[k] = (best_e < p.num_experts) ? best_e : 0u;
        chosen_sc[k] = best_sc;
    }
    if (lane < K) {
        float sum_sc = 0.0f;
        for (uint k = 0u; k < K; ++k) sum_sc += chosen_sc[k];
        const float inv_s = p.route_scale / (sum_sc + 1e-20f);
        const uint  eid   = chosen_id[lane];
        topk_idx[lane]     = eid;
        topk_w[lane]       = chosen_sc[lane] * inv_s;
        sorted_pairs[lane] = lane;
        work_tiles[lane]   = SikMoeWorkTile{eid, lane, 1u, 0u};
        if (lane == 0u) num_tiles[0] = K;
    }
}

// ---- Zero-Barrier 16-Lane Native NVFP4 & HW4.1 MXFP8 MoE Grouped Decode (Packed `tok_m` + Epilogue `pair_id`!) ----

template <uint B_FIX>
inline void sik_moe_nvfp4_w13_fixed(
    device const uint2* w_u2, device const uchar* w_scales, device const float* exp_scales,
    device const half* x_in, device const uint* sorted_pairs, device half* mid_out,
    threadgroup const half2* tg_lut,
    uint e, uint pair_start, uint row, uint D, uint inter, uint top_k,
    float gate_cap, float up_cap, uint lane16
) {
    // Pack `tok_m[b]` tightly (`<= 2047`) and reload `pair_id` only in epilogue (saves 10-14 regs/thread!)
    ushort tok_m[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        tok_m[b] = ushort(sorted_pairs[pair_start + b] / top_k);
    }

    const uint  blk_per_row = D >> 4u;
    const ulong rg          = ulong(e) * (2ul * inter) + row;
    const ulong ru          = rg + inter;
    const float esc         = exp_scales[e];

    device const uint2* wg_row = w_u2     + rg * blk_per_row;
    device const uint2* wu_row = w_u2     + ru * blk_per_row;
    device const uchar* sg_row = w_scales + rg * blk_per_row;
    device const uchar* su_row = w_scales + ru * blk_per_row;

    float g_acc[B_FIX], u_acc[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) { g_acc[b] = 0.0f; u_acc[b] = 0.0f; }

    // 16B `uint4` coalesced waves (`256B` bursts = 532 GB/s!) + exact tail (`100%` bit-exact M=1 == M=2..8!)
    const uint n_full = ((blk_per_row & 1u) == 0u) ? (blk_per_row >> 5u) : 0u;
    if (n_full > 0u) {
        device const uint4*  wg4 = reinterpret_cast<device const uint4*>(wg_row);
        device const uint4*  wu4 = reinterpret_cast<device const uint4*>(wu_row);
        device const ushort* sg2 = reinterpret_cast<device const ushort*>(sg_row);
        device const ushort* su2 = reinterpret_cast<device const ushort*>(su_row);
        for (uint kw = 0u; kw < n_full; ++kw) {
            const uint   bi   = kw * 16u + lane16;
            const uint4  qg   = wg4[bi];
            const uint4  qu   = wu4[bi];
            const ushort g2   = sg2[bi];
            const ushort u2   = su2[bi];
            const float  gsc0 = float(sik_e4m3_to_f16(uchar(g2 & 255u)));
            const float  gsc1 = float(sik_e4m3_to_f16(uchar(g2 >> 8u)));
            const float  usc0 = float(sik_e4m3_to_f16(uchar(u2 & 255u)));
            const float  usc1 = float(sik_e4m3_to_f16(uchar(u2 >> 8u)));

            half4 w0, w1, w2, w3;
            if (B_FIX == 1u) {
                device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(tok_m[0]) * D + bi * 32u);
                {
                    const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                    sik_unpack_nvfp4_u32_tg(qg.x, tg_lut, w0, w1);
                    sik_unpack_nvfp4_u32_tg(qg.y, tg_lut, w2, w3);
                    g_acc[0] = fma(gsc0, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), g_acc[0]);
                    sik_unpack_nvfp4_u32_tg(qu.x, tg_lut, w0, w1);
                    sik_unpack_nvfp4_u32_tg(qu.y, tg_lut, w2, w3);
                    u_acc[0] = fma(usc0, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), u_acc[0]);
                }
                {
                    const half4 x0 = xp[4], x1 = xp[5], x2 = xp[6], x3 = xp[7];
                    sik_unpack_nvfp4_u32_tg(qg.z, tg_lut, w0, w1);
                    sik_unpack_nvfp4_u32_tg(qg.w, tg_lut, w2, w3);
                    g_acc[0] = fma(gsc1, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), g_acc[0]);
                    sik_unpack_nvfp4_u32_tg(qu.z, tg_lut, w0, w1);
                    sik_unpack_nvfp4_u32_tg(qu.w, tg_lut, w2, w3);
                    u_acc[0] = fma(usc1, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), u_acc[0]);
                }
            } else {
                sik_unpack_nvfp4_u32_tg(qg.x, tg_lut, w0, w1);
                sik_unpack_nvfp4_u32_tg(qg.y, tg_lut, w2, w3);
                #pragma clang loop unroll(full)
                for (uint b = 0u; b < B_FIX; ++b) {
                    device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(tok_m[b]) * D + bi * 32u);
                    g_acc[b] = fma(gsc0, sik_dot_h4x4_fast(w0, w1, w2, w3, xp[0], xp[1], xp[2], xp[3]), g_acc[b]);
                }
                sik_unpack_nvfp4_u32_tg(qu.x, tg_lut, w0, w1);
                sik_unpack_nvfp4_u32_tg(qu.y, tg_lut, w2, w3);
                #pragma clang loop unroll(full)
                for (uint b = 0u; b < B_FIX; ++b) {
                    device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(tok_m[b]) * D + bi * 32u);
                    u_acc[b] = fma(usc0, sik_dot_h4x4_fast(w0, w1, w2, w3, xp[0], xp[1], xp[2], xp[3]), u_acc[b]);
                }
                sik_unpack_nvfp4_u32_tg(qg.z, tg_lut, w0, w1);
                sik_unpack_nvfp4_u32_tg(qg.w, tg_lut, w2, w3);
                #pragma clang loop unroll(full)
                for (uint b = 0u; b < B_FIX; ++b) {
                    device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(tok_m[b]) * D + bi * 32u + 16u);
                    g_acc[b] = fma(gsc1, sik_dot_h4x4_fast(w0, w1, w2, w3, xp[0], xp[1], xp[2], xp[3]), g_acc[b]);
                }
                sik_unpack_nvfp4_u32_tg(qu.z, tg_lut, w0, w1);
                sik_unpack_nvfp4_u32_tg(qu.w, tg_lut, w2, w3);
                #pragma clang loop unroll(full)
                for (uint b = 0u; b < B_FIX; ++b) {
                    device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(tok_m[b]) * D + bi * 32u + 16u);
                    u_acc[b] = fma(usc1, sik_dot_h4x4_fast(w0, w1, w2, w3, xp[0], xp[1], xp[2], xp[3]), u_acc[b]);
                }
            }
        }
    }
    for (uint blk = n_full * 32u + lane16; blk < blk_per_row; blk += 16u) {
        const uint2 qg  = wg_row[blk];
        const uint2 qu  = wu_row[blk];
        const float gsc = float(sik_e4m3_to_f16(sg_row[blk]));
        const float usc = float(sik_e4m3_to_f16(su_row[blk]));

        half4 w0, w1, w2, w3;
        if (B_FIX == 1u) {
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(tok_m[0]) * D + blk * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            sik_unpack_nvfp4_u32_tg(qg.x, tg_lut, w0, w1);
            sik_unpack_nvfp4_u32_tg(qg.y, tg_lut, w2, w3);
            g_acc[0] = fma(gsc, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), g_acc[0]);
            sik_unpack_nvfp4_u32_tg(qu.x, tg_lut, w0, w1);
            sik_unpack_nvfp4_u32_tg(qu.y, tg_lut, w2, w3);
            u_acc[0] = fma(usc, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), u_acc[0]);
        } else {
            sik_unpack_nvfp4_u32_tg(qg.x, tg_lut, w0, w1);
            sik_unpack_nvfp4_u32_tg(qg.y, tg_lut, w2, w3);
            #pragma clang loop unroll(full)
            for (uint b = 0u; b < B_FIX; ++b) {
                device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(tok_m[b]) * D + blk * 16u);
                g_acc[b] = fma(gsc, sik_dot_h4x4_fast(w0, w1, w2, w3, xp[0], xp[1], xp[2], xp[3]), g_acc[b]);
            }
            sik_unpack_nvfp4_u32_tg(qu.x, tg_lut, w0, w1);
            sik_unpack_nvfp4_u32_tg(qu.y, tg_lut, w2, w3);
            #pragma clang loop unroll(full)
            for (uint b = 0u; b < B_FIX; ++b) {
                device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(tok_m[b]) * D + blk * 16u);
                u_acc[b] = fma(usc, sik_dot_h4x4_fast(w0, w1, w2, w3, xp[0], xp[1], xp[2], xp[3]), u_acc[b]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        const float g = sik_reduce16(g_acc[b]);
        const float u = sik_reduce16(u_acc[b]);
        if (lane16 == 0u) {
            const uint pid = sorted_pairs[pair_start + b];
            mid_out[ulong(pid) * inter + row] =
                sik_clamp_f16(sik_situ_swiglu(g * esc, u * esc, gate_cap, up_cap));
        }
    }
}

kernel void sik_moe_grouped_w13_nvfp4(
    device const uint2*          w_u2         [[buffer(0)]],
    device const uchar*          w_scales     [[buffer(1)]],
    device const float*          exp_scales   [[buffer(2)]],
    device const half*           x_in         [[buffer(3)]],
    device const uint*           sorted_pairs [[buffer(4)]],
    device const SikMoeWorkTile* work_tiles   [[buffer(5)]],
    device const uint*           num_tiles    [[buffer(6)]],
    device half*                 mid_out      [[buffer(7)]],
    constant SikMoeExecParams&   p            [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint tile_idx = tgid.y;
    if (tile_idx >= num_tiles[0]) return;
    threadgroup half2 tg_lut[256];
    tg_lut[tid]        = half2(SIK_E2M1_H[tid & 15u], SIK_E2M1_H[tid >> 4u]);
    tg_lut[tid + 128u] = half2(SIK_E2M1_H[(tid + 128u) & 15u], SIK_E2M1_H[(tid + 128u) >> 4u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = (tgid.x * 4u + sgid) * 2u + (lane >> 4u);
    if (row >= p.intermediate_size) return;
    const uint lane16 = lane & 15u;
    const SikMoeWorkTile tile = work_tiles[tile_idx];
    #define SIK_NV13_CASE(BF) { sik_moe_nvfp4_w13_fixed<BF>(w_u2, w_scales, exp_scales, x_in, sorted_pairs, mid_out, tg_lut, tile.expert_id, tile.pair_start, row, p.D, p.intermediate_size, p.top_k, p.gate_softcap, p.up_softcap, lane16); break; }
    switch (tile.pair_count) {
        case 1u: SIK_NV13_CASE(1u)
        case 2u: if (SIK_CONST_M >= 2u) { SIK_NV13_CASE(2u) } break;
        case 3u: if (SIK_CONST_M >= 3u) { SIK_NV13_CASE(3u) } break;
        case 4u: if (SIK_CONST_M >= 4u) { SIK_NV13_CASE(4u) } break;
        case 5u: if (SIK_CONST_M >= 5u) { SIK_NV13_CASE(5u) } break;
        case 6u: if (SIK_CONST_M >= 6u) { SIK_NV13_CASE(6u) } break;
        case 7u: if (SIK_CONST_M >= 7u) { SIK_NV13_CASE(7u) } break;
        case 8u: if (SIK_CONST_M >= 8u) { SIK_NV13_CASE(8u) } break;
        default: break;
    }
    #undef SIK_NV13_CASE
}

// `M == 1` Direct Fused `Top-8 Selection + 64-Thread 16-Lane NVFP4 W13`!
// Eliminates `48x sik_moe_topk_m1` dispatches (`-0.28 ms`), hits `493.3 GB/s` (vs `471 GB/s`), 100% bit-exact with `M=2..8`!
// Grid: `((inter + 3) / 4, top_k, 1)`, `64` threads/TG (`2` SGs, `4` rows/TG = `16` lanes/row)
struct SikMoeW13M1Params {
    uint  D;
    uint  inter;
    uint  num_experts;
    uint  top_k;
    float route_scale;
    float gate_softcap;
    float up_softcap;
    uint  pad0;
};

kernel void sik_moe_w13_nvfp4_m1_topk(
    device const uint2*        w_u2         [[buffer(0)]],
    device const uchar*        w_scales     [[buffer(1)]],
    device const float*        exp_scales   [[buffer(2)]],
    device const half*         x_in         [[buffer(3)]],
    device const float2*       router_pairs [[buffer(4)]], // [E] float2(sigmoid(logit), sigmoid(logit) + bias)
    device uint*               topk_idx     [[buffer(5)]],
    device float*              topk_w       [[buffer(6)]],
    device half*               mid_out      [[buffer(7)]],
    constant SikMoeW13M1Params& p           [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgs  [[simdgroups_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup half2 tg_lut[256];
    threadgroup uint  tg_e;
    const uint n_th = sgs * 32u;
    for (uint i = tid; i < 256u; i += n_th) {
        tg_lut[i] = half2(SIK_E2M1_H[i & 15u], SIK_E2M1_H[i >> 4u]);
    }
    if (sgid == 0u) {
        const uint K         = min(p.top_k, 8u);
        const bool write_out = (tgid.x == 0u && tgid.y == 0u);
        const uint k_lim     = write_out ? K : (tgid.y + 1u);
        float4 sel;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            const uint e = j * 32u + lane;
            sel[j] = (e < p.num_experts) ? router_pairs[e].y : -INFINITY;
        }
        float sum_sc = 0.0f;
        uint  my_id  = 0u;
        float my_sc  = 0.0f;
        for (uint k = 0u; k < k_lim; ++k) {
            float best_sel = sel[0];
            uint  best_e   = lane;
            if (sel[1] > best_sel) { best_sel = sel[1]; best_e = 32u + lane; }
            if (sel[2] > best_sel) { best_sel = sel[2]; best_e = 64u + lane; }
            if (sel[3] > best_sel) { best_sel = sel[3]; best_e = 96u + lane; }
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(best_sel, off);
                const uint  oe = simd_shuffle_xor(best_e, off);
                if (ov > best_sel || (ov == best_sel && oe < best_e)) {
                    best_sel = ov; best_e = oe;
                }
            }
            if      (best_e == lane)       sel[0] = -INFINITY;
            else if (best_e == 32u + lane) sel[1] = -INFINITY;
            else if (best_e == 64u + lane) sel[2] = -INFINITY;
            else if (best_e == 96u + lane) sel[3] = -INFINITY;

            const uint eid = (best_e < p.num_experts) ? best_e : 0u;
            if (k == tgid.y && lane == 0u) tg_e = eid;
            if (write_out) {
                const float sc = (best_e < p.num_experts) ? router_pairs[best_e].x : 0.0f;
                sum_sc += sc;
                if (lane == k) { my_id = eid; my_sc = sc; }
            }
        }
        if (write_out && lane < K) {
            const float inv_s = p.route_scale / (sum_sc + 1e-20f);
            topk_idx[lane] = my_id;
            topk_w[lane]   = my_sc * inv_s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = (tgid.x * sgs + sgid) * 2u + (lane >> 4u);
    if (row >= p.inter) return;
    const uint  lane16      = lane & 15u;
    const uint  e           = tg_e;
    const uint  blk_per_row = p.D >> 4u;
    const ulong rg          = ulong(e) * (2ul * p.inter) + row;
    const ulong ru          = rg + p.inter;
    const float esc         = exp_scales[e];

    device const uint2* wg_row = w_u2     + rg * blk_per_row;
    device const uint2* wu_row = w_u2     + ru * blk_per_row;
    device const uchar* sg_row = w_scales + rg * blk_per_row;
    device const uchar* su_row = w_scales + ru * blk_per_row;

    float g_acc = 0.0f, u_acc = 0.0f;
    const uint n_full = ((blk_per_row & 1u) == 0u) ? (blk_per_row >> 5u) : 0u;
    if (n_full > 0u) {
        device const uint4*  wg4 = reinterpret_cast<device const uint4*>(wg_row);
        device const uint4*  wu4 = reinterpret_cast<device const uint4*>(wu_row);
        device const ushort* sg2 = reinterpret_cast<device const ushort*>(sg_row);
        device const ushort* su2 = reinterpret_cast<device const ushort*>(su_row);
        for (uint kw = 0u; kw < n_full; ++kw) {
            const uint   bi   = kw * 16u + lane16;
            const uint4  qg   = wg4[bi];
            const uint4  qu   = wu4[bi];
            const ushort g2   = sg2[bi];
            const ushort u2   = su2[bi];
            const float  gsc0 = float(sik_e4m3_to_f16(uchar(g2 & 255u)));
            const float  gsc1 = float(sik_e4m3_to_f16(uchar(g2 >> 8u)));
            const float  usc0 = float(sik_e4m3_to_f16(uchar(u2 & 255u)));
            const float  usc1 = float(sik_e4m3_to_f16(uchar(u2 >> 8u)));
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + bi * 32u);
            half4 w0, w1, w2, w3;
            {
                const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                sik_unpack_nvfp4_u32_tg(qg.x, tg_lut, w0, w1);
                sik_unpack_nvfp4_u32_tg(qg.y, tg_lut, w2, w3);
                g_acc = fma(gsc0, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), g_acc);
                sik_unpack_nvfp4_u32_tg(qu.x, tg_lut, w0, w1);
                sik_unpack_nvfp4_u32_tg(qu.y, tg_lut, w2, w3);
                u_acc = fma(usc0, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), u_acc);
            }
            {
                const half4 x0 = xp[4], x1 = xp[5], x2 = xp[6], x3 = xp[7];
                sik_unpack_nvfp4_u32_tg(qg.z, tg_lut, w0, w1);
                sik_unpack_nvfp4_u32_tg(qg.w, tg_lut, w2, w3);
                g_acc = fma(gsc1, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), g_acc);
                sik_unpack_nvfp4_u32_tg(qu.z, tg_lut, w0, w1);
                sik_unpack_nvfp4_u32_tg(qu.w, tg_lut, w2, w3);
                u_acc = fma(usc1, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), u_acc);
            }
        }
    }
    for (uint blk = n_full * 32u + lane16; blk < blk_per_row; blk += 16u) {
        const uint2 qg  = wg_row[blk];
        const uint2 qu  = wu_row[blk];
        const float gsc = float(sik_e4m3_to_f16(sg_row[blk]));
        const float usc = float(sik_e4m3_to_f16(su_row[blk]));
        device const half4* xp = reinterpret_cast<device const half4*>(x_in + blk * 16u);
        const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        half4 w0, w1, w2, w3;
        sik_unpack_nvfp4_u32_tg(qg.x, tg_lut, w0, w1);
        sik_unpack_nvfp4_u32_tg(qg.y, tg_lut, w2, w3);
        g_acc = fma(gsc, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), g_acc);
        sik_unpack_nvfp4_u32_tg(qu.x, tg_lut, w0, w1);
        sik_unpack_nvfp4_u32_tg(qu.y, tg_lut, w2, w3);
        u_acc = fma(usc, sik_dot_h4x4_fast(w0, w1, w2, w3, x0, x1, x2, x3), u_acc);
    }
    const float g = sik_reduce16(g_acc);
    const float u = sik_reduce16(u_acc);
    if (lane16 == 0u) {
        mid_out[ulong(tgid.y) * p.inter + row] =
            sik_clamp_f16(sik_situ_swiglu(g * esc, u * esc, p.gate_softcap, p.up_softcap));
    }
}

// UPGRADED 16-lane Coalesced NVFP4 `w2` (full 128B cache-line bursts + 512B TG SRAM LUT256 -> 503 GB/s!)
template <uint B_FIX>
inline void sik_moe_nvfp4_w2_fixed(
    device const uint2* w_u2, device const uchar* w_scales, device const float* exp_scales,
    device const half* mid_in, device const uint* sorted_pairs, device float* down_part,
    threadgroup const half2* tg_lut,
    uint e, uint pair_start, uint row, uint D, uint inter, uint lane16
) {
    ushort pair_id[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) pair_id[b] = ushort(sorted_pairs[pair_start + b]);

    const uint  blk_per_row = inter >> 4u; // 72
    const ulong r           = ulong(e) * D + row;
    const float esc         = exp_scales[e];

    device const uint2* rw = w_u2     + r * blk_per_row;
    device const uchar* rs = w_scales + r * blk_per_row;

    float acc[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) acc[b] = 0.0f;

    for (uint blk = lane16; blk < blk_per_row; blk += 16u) {
        const uint2 q0 = rw[blk];
        const float sc = float(sik_e4m3_to_f16(rs[blk]));
        half4 n0, n1, n2, n3;
        sik_unpack_nvfp4_u32_tg(q0.x, tg_lut, n0, n1);
        sik_unpack_nvfp4_u32_tg(q0.y, tg_lut, n2, n3);

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) {
            device const half4* xp = reinterpret_cast<device const half4*>(mid_in + ulong(pair_id[b]) * inter + blk * 16u);
            acc[b] = fma(sc, sik_dot_h4x4_fast(n0, n1, n2, n3, xp[0], xp[1], xp[2], xp[3]), acc[b]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        const float s = sik_reduce16(acc[b]);
        if (lane16 == 0u) down_part[ulong(pair_id[b]) * D + row] = s * esc;
    }
}

// Grid: `((D + 7) / 8, num_tiles, 1)`, `128` threads/TG (`4` SGs, `2` rows/SG = `16` lanes/row!)
kernel void sik_moe_grouped_w2_nvfp4(
    device const uint2*          w_u2         [[buffer(0)]],
    device const uchar*          w_scales     [[buffer(1)]],
    device const float*          exp_scales   [[buffer(2)]],
    device const half*           mid_in       [[buffer(3)]],
    device const uint*           sorted_pairs [[buffer(4)]],
    device const SikMoeWorkTile* work_tiles   [[buffer(5)]],
    device const uint*           num_tiles    [[buffer(6)]],
    device float*                down_part    [[buffer(7)]],
    constant SikMoeExecParams&   p            [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint tile_idx = tgid.y;
    if (tile_idx >= num_tiles[0]) return;
    threadgroup half2 tg_lut[256];
    tg_lut[tid]        = half2(SIK_E2M1_H[tid & 15u], SIK_E2M1_H[tid >> 4u]);
    tg_lut[tid + 128u] = half2(SIK_E2M1_H[(tid + 128u) & 15u], SIK_E2M1_H[(tid + 128u) >> 4u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = (tgid.x * 4u + sgid) * 2u + (lane >> 4u);
    if (row >= p.D) return;
    const uint lane16 = lane & 15u;
    const SikMoeWorkTile tile = work_tiles[tile_idx];
    #define SIK_NV2_CASE(BF) { sik_moe_nvfp4_w2_fixed<BF>(w_u2, w_scales, exp_scales, mid_in, sorted_pairs, down_part, tg_lut, tile.expert_id, tile.pair_start, row, p.D, p.intermediate_size, lane16); break; }
    switch (tile.pair_count) {
        case 1u: SIK_NV2_CASE(1u)
        case 2u: if (SIK_CONST_M >= 2u) { SIK_NV2_CASE(2u) } break;
        case 3u: if (SIK_CONST_M >= 3u) { SIK_NV2_CASE(3u) } break;
        case 4u: if (SIK_CONST_M >= 4u) { SIK_NV2_CASE(4u) } break;
        case 5u: if (SIK_CONST_M >= 5u) { SIK_NV2_CASE(5u) } break;
        case 6u: if (SIK_CONST_M >= 6u) { SIK_NV2_CASE(6u) } break;
        case 7u: if (SIK_CONST_M >= 7u) { SIK_NV2_CASE(7u) } break;
        case 8u: if (SIK_CONST_M >= 8u) { SIK_NV2_CASE(8u) } break;
        default: break;
    }
    #undef SIK_NV2_CASE
}

// M == 1 Fused 16-Lane NVFP4 `W2` (All Top-8 Experts) + Weighted Combine + `mlp_sconv` + Residual Add!
// Uses 512B TG SRAM LUT256 @ 64th (`576` TGs = `470 GB/s` cold DRAM!), 100.0000% bit-exact with `M=2..8`!
// Grid: `((D + 3) / 4, 1, 1)`, `64` threads/TG (`2` SGs, `2` rows/SG = `16` lanes/row)
kernel void sik_moe_w2_nvfp4_m1_fused_sconv(
    device const uint2*        w_u2       [[buffer(0)]],
    device const uchar*        w_scales   [[buffer(1)]],
    device const float*        exp_scales [[buffer(2)]],
    device const half*         mid_in     [[buffer(3)]],
    device const uint*         topk_idx   [[buffer(4)]],
    device const float*        topk_w     [[buffer(5)]],
    device const float4*       sconv_w    [[buffer(6)]],
    device float4*             conv_state [[buffer(7)]],
    device float*              cur_io     [[buffer(8)]],
    constant SikMoeExecParams& p          [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgs  [[simdgroups_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup half2 tg_lut[256];
    threadgroup uint  tg_e[8];
    threadgroup float tg_es[8];
    threadgroup float tg_tw[8];
    const uint n_th = sgs * 32u;
    for (uint i = tid; i < 256u; i += n_th) {
        tg_lut[i] = half2(SIK_E2M1_H[i & 15u], SIK_E2M1_H[i >> 4u]);
    }
    const uint K = min(p.top_k, 8u);
    if (tid < K) {
        const uint e = topk_idx[tid];
        tg_e[tid]  = e;
        tg_es[tid] = exp_scales[e];
        tg_tw[tid] = topk_w[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = (tgid * sgs + sgid) * 2u + (lane >> 4u);
    if (row >= p.D) return;
    const uint lane16      = lane & 15u;
    const uint blk_per_row = p.intermediate_size >> 4u;

    float cur = 0.0f;
    for (uint k = 0u; k < K; ++k) {
        const uint  e   = tg_e[k];
        const ulong r   = ulong(e) * p.D + row;
        const float esc = tg_es[k];
        device const uint2* rw  = w_u2     + r * blk_per_row;
        device const uchar* rs  = w_scales + r * blk_per_row;
        device const half*  mid = mid_in   + ulong(k) * p.intermediate_size;

        float acc = 0.0f;
        for (uint blk = lane16; blk < blk_per_row; blk += 16u) {
            const uint2 q0 = rw[blk];
            const float sc = float(sik_e4m3_to_f16(rs[blk]));
            half4 n0, n1, n2, n3;
            sik_unpack_nvfp4_u32_tg(q0.x, tg_lut, n0, n1);
            sik_unpack_nvfp4_u32_tg(q0.y, tg_lut, n2, n3);
            device const half4* xp = reinterpret_cast<device const half4*>(mid + blk * 16u);
            acc = fma(sc, sik_dot_h4x4_fast(n0, n1, n2, n3, xp[0], xp[1], xp[2], xp[3]), acc);
        }
        const float s = sik_reduce16(acc);
        if (lane16 == 0u) {
            cur = fma(tg_tw[k], s * esc, cur);
        }
    }
    if (lane16 == 0u) {
        const float4 w     = sconv_w[row];
        const float  inv_s = 0.4472135955f;
        const float4 st    = conv_state[row];
        const float  sum   = fma(st.x, w.x, fma(st.y, w.y, fma(st.z, w.z, cur * w.w)));
        const float  y     = (sum + cur) * inv_s;
        conv_state[row]    = float4(st.y, st.z, cur, 0.0f);
        cur_io[row]        = cur_io[row] + y;
    }
}

// Native MXFP8 MoE Grouped Decode w/ Metal 4.1 HW `packed_metal_fp8_e4m3<8>` + Packed `tok_m` + Epilogue `pair_id`!
template <uint B_FIX>
inline void sik_moe_mxfp8_w13_fixed(
    device const uint4* w_u4, device const uchar* w_scales, device const half* x_in,
    device const uint* sorted_pairs, device half* mid_out,
    uint e, uint pair_start, uint row, uint D, uint inter, uint top_k,
    float gate_cap, float up_cap, uint lane16
) {
    ushort tok_m[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        tok_m[b] = ushort(sorted_pairs[pair_start + b] / top_k);
    }

    const uint  c16_per_row = D >> 4u;
    const uint  blk_per_row = D >> 5u;
    const ulong rg          = ulong(e) * (2ul * inter) + row;
    const ulong ru          = rg + inter;

    device const uint4* wg_row = w_u4     + rg * c16_per_row;
    device const uint4* wu_row = w_u4     + ru * c16_per_row;
    device const uchar* sg_row = w_scales + rg * blk_per_row;
    device const uchar* su_row = w_scales + ru * blk_per_row;

    float g_acc[B_FIX], u_acc[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) { g_acc[b] = 0.0f; u_acc[b] = 0.0f; }

    for (uint c16 = lane16; c16 < c16_per_row; c16 += 16u) {
        const uint  blk = c16 >> 1u;
        const uint4 qg  = wg_row[c16];
        const uint4 qu  = wu_row[c16];
        const float gsc = sik_ue8m0_to_f32(sg_row[blk]);
        const float usc = sik_ue8m0_to_f32(su_row[blk]);

        half4 g0, g1, g2, g3, u0, u1, u2, u3;
        sik_e4m3x8_to_f16x4x2(qg.xy, g0, g1); sik_e4m3x8_to_f16x4x2(qg.zw, g2, g3);
        sik_e4m3x8_to_f16x4x2(qu.xy, u0, u1); sik_e4m3x8_to_f16x4x2(qu.zw, u2, u3);

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) {
            device const half4* xp = reinterpret_cast<device const half4*>(x_in + ulong(tok_m[b]) * D + c16 * 16u);
            const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            g_acc[b] = fma(gsc, sik_dot_f32_h4x4(g0, g1, g2, g3, x0, x1, x2, x3), g_acc[b]);
            u_acc[b] = fma(usc, sik_dot_f32_h4x4(u0, u1, u2, u3, x0, x1, x2, x3), u_acc[b]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        const float g = sik_reduce16(g_acc[b]);
        const float u = sik_reduce16(u_acc[b]);
        if (lane16 == 0u) {
            const uint pid = sorted_pairs[pair_start + b];
            mid_out[ulong(pid) * inter + row] =
                sik_clamp_f16(sik_situ_swiglu(g, u, gate_cap, up_cap));
        }
    }
}

kernel void sik_moe_grouped_w13_mxfp8(
    device const uint4*          w_u4         [[buffer(0)]],
    device const uchar*          w_scales     [[buffer(1)]],
    device const half*           x_in         [[buffer(2)]],
    device const uint*           sorted_pairs [[buffer(3)]],
    device const SikMoeWorkTile* work_tiles   [[buffer(4)]],
    device const uint*           num_tiles    [[buffer(5)]],
    device half*                 mid_out      [[buffer(6)]],
    constant SikMoeExecParams&   p            [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint tile_idx = tgid.y;
    if (tile_idx >= num_tiles[0]) return;
    const uint row = (tgid.x * 4u + sgid) * 2u + (lane >> 4u);
    if (row >= p.intermediate_size) return;
    const uint lane16 = lane & 15u;
    const SikMoeWorkTile tile = work_tiles[tile_idx];
    #define SIK_MX13_CASE(BF) { sik_moe_mxfp8_w13_fixed<BF>(w_u4, w_scales, x_in, sorted_pairs, mid_out, tile.expert_id, tile.pair_start, row, p.D, p.intermediate_size, p.top_k, p.gate_softcap, p.up_softcap, lane16); break; }
    switch (tile.pair_count) {
        case 1u: SIK_MX13_CASE(1u)
        case 2u: if (SIK_CONST_M >= 2u) { SIK_MX13_CASE(2u) } break;
        case 3u: if (SIK_CONST_M >= 3u) { SIK_MX13_CASE(3u) } break;
        case 4u: if (SIK_CONST_M >= 4u) { SIK_MX13_CASE(4u) } break;
        case 5u: if (SIK_CONST_M >= 5u) { SIK_MX13_CASE(5u) } break;
        case 6u: if (SIK_CONST_M >= 6u) { SIK_MX13_CASE(6u) } break;
        case 7u: if (SIK_CONST_M >= 7u) { SIK_MX13_CASE(7u) } break;
        case 8u: if (SIK_CONST_M >= 8u) { SIK_MX13_CASE(8u) } break;
        default: break;
    }
    #undef SIK_MX13_CASE
}

template <uint B_FIX>
inline void sik_moe_mxfp8_w2_fixed(
    device const uint4* w_u4, device const uchar* w_scales, device const half* mid_in,
    device const uint* sorted_pairs, device float* down_part,
    uint e, uint pair_start, uint row, uint D, uint inter, uint lane16
) {
    ushort pair_id[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) pair_id[b] = ushort(sorted_pairs[pair_start + b]);

    const uint  c16_per_row = inter >> 4u; // 72
    const uint  blk_per_row = inter >> 5u;
    const ulong r           = ulong(e) * D + row;

    device const uint4* rw = w_u4     + r * c16_per_row;
    device const uchar* rs = w_scales + r * blk_per_row;

    float acc[B_FIX];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) acc[b] = 0.0f;

    for (uint c16 = lane16; c16 < c16_per_row; c16 += 16u) {
        const uint4 q0 = rw[c16];
        const float sc = sik_ue8m0_to_f32(rs[c16 >> 1u]);
        half4 n0, n1, n2, n3;
        sik_e4m3x8_to_f16x4x2(q0.xy, n0, n1);
        sik_e4m3x8_to_f16x4x2(q0.zw, n2, n3);

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < B_FIX; ++b) {
            device const half4* xp = reinterpret_cast<device const half4*>(mid_in + ulong(pair_id[b]) * inter + c16 * 16u);
            acc[b] = fma(sc, sik_dot_f32_h4x4(n0, n1, n2, n3, xp[0], xp[1], xp[2], xp[3]), acc[b]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < B_FIX; ++b) {
        const float s = sik_reduce16(acc[b]);
        if (lane16 == 0u) down_part[ulong(pair_id[b]) * D + row] = s;
    }
}

kernel void sik_moe_grouped_w2_mxfp8(
    device const uint4*          w_u4         [[buffer(0)]],
    device const uchar*          w_scales     [[buffer(1)]],
    device const half*           mid_in       [[buffer(2)]],
    device const uint*           sorted_pairs [[buffer(3)]],
    device const SikMoeWorkTile* work_tiles   [[buffer(4)]],
    device const uint*           num_tiles    [[buffer(5)]],
    device float*                down_part    [[buffer(6)]],
    constant SikMoeExecParams&   p            [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint tile_idx = tgid.y;
    if (tile_idx >= num_tiles[0]) return;
    const uint row = (tgid.x * 4u + sgid) * 2u + (lane >> 4u);
    if (row >= p.D) return;
    const uint lane16 = lane & 15u;
    const SikMoeWorkTile tile = work_tiles[tile_idx];
    #define SIK_MX2_CASE(BF) { sik_moe_mxfp8_w2_fixed<BF>(w_u4, w_scales, mid_in, sorted_pairs, down_part, tile.expert_id, tile.pair_start, row, p.D, p.intermediate_size, lane16); break; }
    switch (tile.pair_count) {
        case 1u: SIK_MX2_CASE(1u)
        case 2u: if (SIK_CONST_M >= 2u) { SIK_MX2_CASE(2u) } break;
        case 3u: if (SIK_CONST_M >= 3u) { SIK_MX2_CASE(3u) } break;
        case 4u: if (SIK_CONST_M >= 4u) { SIK_MX2_CASE(4u) } break;
        case 5u: if (SIK_CONST_M >= 5u) { SIK_MX2_CASE(5u) } break;
        case 6u: if (SIK_CONST_M >= 6u) { SIK_MX2_CASE(6u) } break;
        case 7u: if (SIK_CONST_M >= 7u) { SIK_MX2_CASE(7u) } break;
        case 8u: if (SIK_CONST_M >= 8u) { SIK_MX2_CASE(8u) } break;
        default: break;
    }
    #undef SIK_MX2_CASE
}

// ---- Pre-Gather Contiguous Sorted Activations (`uint4` 16B) + `+128` Tail Zeroing for Quad-Bin NAX ----
// Zeros `+128` tail rows on BOTH `x_sorted` (aliased to `down_part`) AND `mid_sorted` so `BM=128` NAX never reads OOB!
// Grid: `(M_valid * top_k + 128, 1, 1)`, `256` threads/TG
kernel void sik_moe_gather_x_f16(
    device const uint4*        x_in         [[buffer(0)]],
    device const uint*         sorted_pairs [[buffer(1)]],
    device uint4*              x_sorted     [[buffer(2)]],
    device uint4*              mid_sorted   [[buffer(3)]],
    constant SikMoeExecParams& p            [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint total_pairs = p.M_valid * p.top_k;
    const uint d8_cnt      = p.D >> 3u;
    device uint4* dst      = x_sorted + ulong(row) * d8_cnt;
    if (row >= total_pairs) {
        for (uint i = tid; i < d8_cnt; i += 256u) dst[i] = uint4(0u);
        const uint m8_cnt = p.intermediate_size >> 3u;
        device uint4* mdst = mid_sorted + ulong(row) * m8_cnt;
        for (uint i = tid; i < m8_cnt; i += 256u) mdst[i] = uint4(0u);
        return;
    }
    const uint tok = sorted_pairs[row] / p.top_k;
    device const uint4* src = x_in + ulong(tok) * d8_cnt;
    for (uint i = tid; i < d8_cnt; i += 256u) dst[i] = src[i];
}

// ---- Quad-Bin (`BM = 16[SG=2] / 32 / 64 / 128`) `BK = 128` Direct-tX Metal 4.1 NAX MoE Prefill ----
// Eliminates the M5 Max `K = 64` 2.17x NAX cliff AND keeps exact `16.0 KiB` TG mem (`2` TGs/core occupancy)!

union SikMoeW2Tg128 {
    half W[64][128];
    uint pid[128];
};

#if defined(__HAVE_TENSOR__)
// 1-Loop `BN = 64, BK = 128` (`W_tile[64][128] = 16.0 KiB` exact!) Gate/Up Pass (`SIK_CONST_M == 0: gate`, `1: up+SiTU`):
// - Matches `w2`'s `43+ TF` single-loop register footprint (`2` TGs/core, 0 spills, `mid_sorted` stays in 48 MiB SLC!)
template <int BM, int SG>
inline void sik_moe_prefill_w13_nvfp4_pass_impl(
    device const uint2*          w_u2,
    device const uchar*          w_scales,
    device const float*          exp_scales,
    device const half*           x_sorted,
    device const SikMoeWorkTile* work_tiles,
    device half*                 mid_sorted,
    constant SikMoeExecParams&   p,
    threadgroup half             (&W_tile)[64][128],
    uint2                        tgid,
    uint                         tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.intermediate_size) return;

    const SikMoeWorkTile tile = work_tiles[tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    tensor<device half, extents<int, dynamic_extent, dynamic_extent>, tensor_inline> tX_dev(
        const_cast<device half*>(x_sorted + ulong(tile.pair_start) * p.D),
        extents<int, dynamic_extent, dynamic_extent>(int(p.D), BM));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto tX0 = tX_dev.template slice<128, BM>(0, 0);
    auto cT  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(tX0)>, sik_strip_as<decltype(tW)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    constexpr uint ROWS_PER_STEP = (SG * 32u) / 8u;
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;
    const uint  blk_per_row = p.D >> 4u;
    const uint  k_tiles     = p.D >> 7u;
    const uint  r_grp       = tid >> 3u;
    const uint  blk8        = tid & 7u;
    const ulong r_base      = ulong(tile.expert_id) * (2ul * p.intermediate_size)
                            + ((SIK_CONST_M != 0u) ? ulong(p.intermediate_size) : 0ul) + n0;
    const float esc         = exp_scales[tile.expert_id];

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const ulong bi = ulong(kt) * 8ul + blk8;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const ulong r_glob = r_base + r_loc;
            const uint2 v      = w_u2[r_glob * blk_per_row + bi];
            const half  sc     = sik_e4m3_to_f16(w_scales[r_glob * blk_per_row + bi]);
            half4 n0h, n1h, n2h, n3h;
            sik_unpack_nvfp4_u32_h4(v.x, n0h, n1h); sik_unpack_nvfp4_u32_h4(v.y, n2h, n3h);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][blk8 * 16u]);
            dst[0] = n0h * sc; dst[1] = n1h * sc; dst[2] = n2h * sc; dst[3] = n3h * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX_dev.template slice<128, BM>(int(kt * 128u), 0);
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < cnt) {
                const ulong dst_idx = ulong(tile.pair_start + r) * p.intermediate_size + (n0 + c);
                if (SIK_CONST_M == 0u) {
                    const float g_raw = cT[i] * esc;
                    const float cg    = sik_softcap(g_raw, p.gate_softcap);
                    mid_sorted[dst_idx] = half(cg * sik_sigmoid(g_raw));
                } else {
                    const float gs = float(mid_sorted[dst_idx]);
                    const float u  = sik_softcap(cT[i] * esc, p.up_softcap);
                    mid_sorted[dst_idx] = sik_clamp_f16(gs * u);
                }
            }
        }
    }
}

template <int BM, int SG>
inline void sik_moe_prefill_w2_nvfp4_impl(
    device const uint2*          w_u2,
    device const uchar*          w_scales,
    device const float*          exp_scales,
    device const half*           mid_sorted,
    device const uint*           sorted_pairs,
    device const SikMoeWorkTile* work_tiles,
    device half*                 down_part_f16,
    constant SikMoeExecParams&   p,
    threadgroup SikMoeW2Tg128&   tg,
    uint2                        tgid,
    uint                         tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.D) return;
    const SikMoeWorkTile tile = work_tiles[tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    tensor<device half, extents<int, dynamic_extent, dynamic_extent>, tensor_inline> tX_dev(
        const_cast<device half*>(mid_sorted + ulong(tile.pair_start) * p.intermediate_size),
        extents<int, dynamic_extent, dynamic_extent>(int(p.intermediate_size), BM));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&tg.W[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto tX0 = tX_dev.template slice<128, BM>(0, 0);
    auto cT  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(tX0)>, sik_strip_as<decltype(tW)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    constexpr uint ROWS_PER_STEP = (SG * 32u) / 8u;
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;
    const uint  blk_per_row = p.intermediate_size >> 4u;
    const uint  k_tiles     = p.intermediate_size >> 7u;
    const uint  r_grp       = tid >> 3u;
    const uint  blk8        = tid & 7u;
    const ulong r_base      = ulong(tile.expert_id) * p.D + n0;
    const float esc         = exp_scales[tile.expert_id];

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const ulong bi = ulong(kt) * 8ul + blk8;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const ulong r_glob = r_base + r_loc;
            const uint2 v      = w_u2[r_glob * blk_per_row + bi];
            const half  sc     = sik_e4m3_to_f16(w_scales[r_glob * blk_per_row + bi]);
            half4 n0h, n1h, n2h, n3h;
            sik_unpack_nvfp4_u32_h4(v.x, n0h, n1h); sik_unpack_nvfp4_u32_h4(v.y, n2h, n3h);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&tg.W[r_loc][blk8 * 16u]);
            dst[0] = n0h * sc; dst[1] = n1h * sc; dst[2] = n2h * sc; dst[3] = n3h * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX_dev.template slice<128, BM>(int(kt * 128u), 0);
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reuse `tg.pid` (unioned with `tg.W` -> exact `16.0 KiB` TG mem -> 2 TGs/core!)
    if (tid < cnt) tg.pid[tid] = sorted_pairs[tile.pair_start + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < cnt) down_part_f16[ulong(tg.pid[r]) * p.D + (n0 + c)] = sik_clamp_f16(cT[i] * esc);
        }
    }
}

template <int BM, int SG>
inline void sik_moe_prefill_w13_mxfp8_pass_impl(
    device const uint4*          w_u4,
    device const uchar*          w_scales,
    device const half*           x_sorted,
    device const SikMoeWorkTile* work_tiles,
    device half*                 mid_sorted,
    constant SikMoeExecParams&   p,
    threadgroup half             (&W_tile)[64][128],
    uint2                        tgid,
    uint                         tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.intermediate_size) return;
    const SikMoeWorkTile tile = work_tiles[tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    tensor<device half, extents<int, dynamic_extent, dynamic_extent>, tensor_inline> tX_dev(
        const_cast<device half*>(x_sorted + ulong(tile.pair_start) * p.D),
        extents<int, dynamic_extent, dynamic_extent>(int(p.D), BM));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&W_tile[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto tX0 = tX_dev.template slice<128, BM>(0, 0);
    auto cT  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(tX0)>, sik_strip_as<decltype(tW)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    constexpr uint ROWS_PER_STEP = (SG * 32u) / 8u;
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;
    const uint  u4_per_row  = p.D >> 4u;
    const uint  blk_per_row = p.D >> 5u;
    const uint  k_tiles     = p.D >> 7u;
    const uint  r_grp       = tid >> 3u;
    const uint  blk8        = tid & 7u;
    const ulong r_base      = ulong(tile.expert_id) * (2ul * p.intermediate_size)
                            + ((SIK_CONST_M != 0u) ? ulong(p.intermediate_size) : 0ul) + n0;

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const ulong bi = ulong(kt) * 8ul + blk8;
        const ulong si = ulong(kt) * 4ul + (blk8 >> 1u);
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const ulong r_glob = r_base + r_loc;
            const uint4 w16    = w_u4[r_glob * u4_per_row + bi];
            const float sc     = sik_ue8m0_to_f32(w_scales[r_glob * blk_per_row + si]);
            half4 n0h, n1h, n2h, n3h;
            sik_e4m3x8_to_f16x4x2(w16.xy, n0h, n1h); sik_e4m3x8_to_f16x4x2(w16.zw, n2h, n3h);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&W_tile[r_loc][blk8 * 16u]);
            dst[0] = sik_clamp_f16x4(float4(n0h) * sc); dst[1] = sik_clamp_f16x4(float4(n1h) * sc);
            dst[2] = sik_clamp_f16x4(float4(n2h) * sc); dst[3] = sik_clamp_f16x4(float4(n3h) * sc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX_dev.template slice<128, BM>(int(kt * 128u), 0);
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < cnt) {
                const ulong dst_idx = ulong(tile.pair_start + r) * p.intermediate_size + (n0 + c);
                if (SIK_CONST_M == 0u) {
                    const float g_raw = cT[i];
                    const float cg    = sik_softcap(g_raw, p.gate_softcap);
                    mid_sorted[dst_idx] = half(cg * sik_sigmoid(g_raw));
                } else {
                    const float gs = float(mid_sorted[dst_idx]);
                    const float u  = sik_softcap(cT[i], p.up_softcap);
                    mid_sorted[dst_idx] = sik_clamp_f16(gs * u);
                }
            }
        }
    }
}

template <int BM, int SG>
inline void sik_moe_prefill_w2_mxfp8_impl(
    device const uint4*          w_u4,
    device const uchar*          w_scales,
    device const half*           mid_sorted,
    device const uint*           sorted_pairs,
    device const SikMoeWorkTile* work_tiles,
    device half*                 down_part_f16,
    constant SikMoeExecParams&   p,
    threadgroup SikMoeW2Tg128&   tg,
    uint2                        tgid,
    uint                         tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.D) return;
    const SikMoeWorkTile tile = work_tiles[tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    tensor<device half, extents<int, dynamic_extent, dynamic_extent>, tensor_inline> tX_dev(
        const_cast<device half*>(mid_sorted + ulong(tile.pair_start) * p.intermediate_size),
        extents<int, dynamic_extent, dynamic_extent>(int(p.intermediate_size), BM));
    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tW(&tg.W[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto tX0 = tX_dev.template slice<128, BM>(0, 0);
    auto cT  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(tX0)>, sik_strip_as<decltype(tW)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    constexpr uint ROWS_PER_STEP = (SG * 32u) / 8u;
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;
    const uint  u4_per_row  = p.intermediate_size >> 4u;
    const uint  blk_per_row = p.intermediate_size >> 5u;
    const uint  k_tiles     = p.intermediate_size >> 7u;
    const uint  r_grp       = tid >> 3u;
    const uint  blk8        = tid & 7u;
    const ulong r_base      = ulong(tile.expert_id) * p.D + n0;

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const ulong bi = ulong(kt) * 8ul + blk8;
        const ulong si = ulong(kt) * 4ul + (blk8 >> 1u);
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const ulong r_glob = r_base + r_loc;
            const uint4 w16    = w_u4[r_glob * u4_per_row + bi];
            const float sc     = sik_ue8m0_to_f32(w_scales[r_glob * blk_per_row + si]);
            half4 n0h, n1h, n2h, n3h;
            sik_e4m3x8_to_f16x4x2(w16.xy, n0h, n1h); sik_e4m3x8_to_f16x4x2(w16.zw, n2h, n3h);
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&tg.W[r_loc][blk8 * 16u]);
            dst[0] = sik_clamp_f16x4(float4(n0h) * sc); dst[1] = sik_clamp_f16x4(float4(n1h) * sc);
            dst[2] = sik_clamp_f16x4(float4(n2h) * sc); dst[3] = sik_clamp_f16x4(float4(n3h) * sc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX_dev.template slice<128, BM>(int(kt * 128u), 0);
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < cnt) tg.pid[tid] = sorted_pairs[tile.pair_start + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < cnt) down_part_f16[ulong(tg.pid[r]) * p.D + (n0 + c)] = sik_clamp_f16(cT[i]);
        }
    }
}
#endif

// ---- Quad-Bin NVFP4 Prefill Entrypoints (`m16_sg2`, `m32`, `m64`, `m128`; `SIK_CONST_M=0: gate`, `1: up+SiTU`) ----
kernel void sik_moe_prefill_w13_nvfp4_nax_m16_sg2(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const half* x [[buffer(3)]], device const uint* sp [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* mo [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    (void)sp;
    threadgroup half Wt[64][128];
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_nvfp4_pass_impl<16, 2>(w, s, es, x, wt, mo, p, Wt, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)x;(void)wt;(void)mo;(void)p;(void)Wt;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_nvfp4_nax_m32(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const half* x [[buffer(3)]], device const uint* sp [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* mo [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    (void)sp;
    threadgroup half Wt[64][128];
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_nvfp4_pass_impl<32, 4>(w, s, es, x, wt, mo, p, Wt, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)x;(void)wt;(void)mo;(void)p;(void)Wt;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_nvfp4_nax_m64(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const half* x [[buffer(3)]], device const uint* sp [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* mo [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    (void)sp;
    threadgroup half Wt[64][128];
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_nvfp4_pass_impl<64, 4>(w, s, es, x, wt, mo, p, Wt, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)x;(void)wt;(void)mo;(void)p;(void)Wt;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_nvfp4_nax_m128(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const half* x [[buffer(3)]], device const uint* sp [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* mo [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    (void)sp;
    threadgroup half Wt[64][128];
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_nvfp4_pass_impl<128, 4>(w, s, es, x, wt, mo, p, Wt, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)x;(void)wt;(void)mo;(void)p;(void)Wt;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_nvfp4_nax_m16_sg2(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const half* mi [[buffer(3)]], device const uint* sp [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* dp [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeW2Tg128 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_nvfp4_impl<16, 2>(w, s, es, mi, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)mi;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_nvfp4_nax_m32(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const half* mi [[buffer(3)]], device const uint* sp [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* dp [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeW2Tg128 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_nvfp4_impl<32, 4>(w, s, es, mi, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)mi;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_nvfp4_nax_m64(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const half* mi [[buffer(3)]], device const uint* sp [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* dp [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeW2Tg128 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_nvfp4_impl<64, 4>(w, s, es, mi, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)mi;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_nvfp4_nax_m128(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const half* mi [[buffer(3)]], device const uint* sp [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* dp [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeW2Tg128 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_nvfp4_impl<128, 4>(w, s, es, mi, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)mi;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

// ---- Quad-Bin MXFP8 Prefill Entrypoints (`m16_sg2`, `m32`, `m64`, `m128`; `SIK_CONST_M=0: gate`, `1: up+SiTU`) ----
kernel void sik_moe_prefill_w13_mxfp8_nax_m16_sg2(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const half* x [[buffer(2)]], device const uint* sp [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* mo [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    (void)sp;
    threadgroup half Wt[64][128];
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_mxfp8_pass_impl<16, 2>(w, s, x, wt, mo, p, Wt, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)wt;(void)mo;(void)p;(void)Wt;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_mxfp8_nax_m32(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const half* x [[buffer(2)]], device const uint* sp [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* mo [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    (void)sp;
    threadgroup half Wt[64][128];
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_mxfp8_pass_impl<32, 4>(w, s, x, wt, mo, p, Wt, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)wt;(void)mo;(void)p;(void)Wt;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_mxfp8_nax_m64(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const half* x [[buffer(2)]], device const uint* sp [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* mo [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    (void)sp;
    threadgroup half Wt[64][128];
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_mxfp8_pass_impl<64, 4>(w, s, x, wt, mo, p, Wt, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)wt;(void)mo;(void)p;(void)Wt;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_mxfp8_nax_m128(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const half* x [[buffer(2)]], device const uint* sp [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* mo [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    (void)sp;
    threadgroup half Wt[64][128];
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_mxfp8_pass_impl<128, 4>(w, s, x, wt, mo, p, Wt, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)wt;(void)mo;(void)p;(void)Wt;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_mxfp8_nax_m16_sg2(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const half* mi [[buffer(2)]], device const uint* sp [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* dp [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeW2Tg128 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_mxfp8_impl<16, 2>(w, s, mi, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)mi;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_mxfp8_nax_m32(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const half* mi [[buffer(2)]], device const uint* sp [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* dp [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeW2Tg128 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_mxfp8_impl<32, 4>(w, s, mi, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)mi;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_mxfp8_nax_m64(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const half* mi [[buffer(2)]], device const uint* sp [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* dp [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeW2Tg128 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_mxfp8_impl<64, 4>(w, s, mi, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)mi;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_mxfp8_nax_m128(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const half* mi [[buffer(2)]], device const uint* sp [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* dp [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeW2Tg128 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_mxfp8_impl<128, 4>(w, s, mi, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)mi;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

// ---- Opt-In `--int8-prefill` Double-Buffered INT8xINT8->INT32 MoE Prefill (NVFP4 + MXFP8, `BM=64/32/16`) ----

// Gather INT8 sorted activations (`uint4` = 16 int8s) + scales + zero `+128` tail rows!
// Grid: `(M_valid * top_k + 128, 1, 1)`, `256` threads/TG
kernel void sik_moe_gather_x_i8(
    device const uint4*        x_i8_in     [[buffer(0)]],
    device const half*         x_sc_in     [[buffer(1)]],
    device const uint*         sorted_pairs [[buffer(2)]],
    device uint4*              x_i8_sorted [[buffer(3)]],
    device half*               x_sc_sorted [[buffer(4)]],
    device uint4*              mid_i8_out  [[buffer(5)]],
    device half*               mid_sc_out  [[buffer(6)]],
    constant SikMoeExecParams& p           [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint total_pairs = p.M_valid * p.top_k;
    const uint d16_cnt     = p.D >> 4u;
    device uint4* dst      = x_i8_sorted + ulong(row) * d16_cnt;
    if (row >= total_pairs) {
        for (uint i = tid; i < d16_cnt; i += 256u) dst[i] = uint4(0u);
        const uint m16_cnt = p.intermediate_size >> 4u;
        device uint4* mdst = mid_i8_out + ulong(row) * m16_cnt;
        for (uint i = tid; i < m16_cnt; i += 256u) mdst[i] = uint4(0u);
        if (tid == 0u) {
            x_sc_sorted[row] = 0.0h;
            mid_sc_out[row]  = 0.0h;
        }
        return;
    }
    const uint tok = sorted_pairs[row] / p.top_k;
    device const uint4* src = x_i8_in + ulong(tok) * d16_cnt;
    for (uint i = tid; i < d16_cnt; i += 256u) dst[i] = src[i];
    if (tid == 0u) x_sc_sorted[row] = x_sc_in[tok];
}

// 8.0 KiB Unioned TG Workspace (fits 4 TGs per 32 KiB M5 core — HALF the TG memory of FP16!)
union SikMoeTgI8 {
    int8_t W[64][128]; // 8,192 B (K-loop)
    struct {
        float ws[64];  //   256 B (epilogue only!)
        uint  pid[128];//   512 B (W2 epilogue only!)
    } epi;
};

#if defined(__HAVE_TENSOR__)
// High-Occupancy 8KiB Whole-Row Zero-Acc NVFP4 -> INT8xINT8->INT32 MoE W13 (`84.5 TOPS`, 4 TGs/core!)
template <int BM, int SG, int CAP>
inline void sik_moe_prefill_w13_nvfp4_i8_pass_impl(
    device const uint2*          w_u2,
    device const uchar*          w_scales,
    device const float*          exp_scales,
    device const int8_t*         x_sorted_i8,
    device const half*           x_sorted_sc,
    device const SikMoeWorkTile* work_tiles,
    device half*                 mid_sorted_f16,
    constant SikMoeExecParams&   p,
    threadgroup SikMoeTgI8&      tg,
    uint2                        tgid,
    uint                         tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.intermediate_size) return;
    const SikMoeWorkTile tile = work_tiles[tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    tensor<device int8_t, extents<int, dynamic_extent, dynamic_extent>, tensor_inline> tX_dev(
        const_cast<device int8_t*>(x_sorted_i8 + ulong(tile.pair_start) * p.D),
        extents<int, dynamic_extent, dynamic_extent>(int(p.D), BM));
    tensor<threadgroup int8_t, extents<int, 128, 64>, tensor_inline> tW(&tg.W[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto tX0 = tX_dev.template slice<128, BM>(0, 0);
    auto cI  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(tX0)>, sik_strip_as<decltype(tW)>, int32_t>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) if (cI.is_valid_element(i)) cI[i] = 0;

    constexpr uint ROWS_PER_STEP = (uint(SG) * 32u) >> 3u; // 16 (SG=4) or 8 (SG=2)
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;
    const uint  blk_per_row = p.D >> 4u;
    const uint  k_tiles     = p.D >> 7u;
    const uint  r_grp       = tid >> 3u;
    const uint  blk8        = tid & 7u;
    const uint  lane_base   = (tid & 31u) & ~7u;
    const ulong r_base      = ulong(tile.expert_id) * (2ul * p.intermediate_size)
                            + ((SIK_CONST_M != 0u) ? ulong(p.intermediate_size) : 0ul) + n0;
    const float esc         = exp_scales[tile.expert_id];

    float inv_arr[Q_STEPS];
    float wsc_arr[Q_STEPS];
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < Q_STEPS; ++q) {
        const ulong r_glob = r_base + (q * ROWS_PER_STEP + r_grp);
        uchar bmax = 0u;
        for (uint b = blk8; b < blk_per_row; b += 8u) {
            bmax = max(bmax, uchar(w_scales[r_glob * blk_per_row + b] & 0x7Fu));
        }
        uint um = uint(bmax);
        um = max(um, simd_shuffle_xor(um, 1u));
        um = max(um, simd_shuffle_xor(um, 2u));
        um = max(um, simd_shuffle_xor(um, 4u));
        const float smax = float(sik_e4m3_to_f16(uchar(um)));
        wsc_arr[q] = smax * (esc * (1.0f / 21.0f));
        inv_arr[q] = simd_shuffle((smax > 1e-12f) ? (21.0f / smax) : 0.0f, lane_base);
    }

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const ulong bi = ulong(kt) * 8ul + blk8;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const ulong r_glob = r_base + r_loc;
            const half  mult   = half(float(sik_e4m3_to_f16(w_scales[r_glob * blk_per_row + bi])) * inv_arr[q]);
            const uint2 v      = w_u2[r_glob * blk_per_row + bi];
            half4 n0h, n1h, n2h, n3h;
            sik_unpack_nvfp4_u32_h4(v.x, n0h, n1h);
            sik_unpack_nvfp4_u32_h4(v.y, n2h, n3h);
            threadgroup char4* dst = reinterpret_cast<threadgroup char4*>(&tg.W[r_loc][blk8 * 16u]);
            dst[0] = char4(clamp(rint(n0h * mult), half4(-127.0h), half4(127.0h)));
            dst[1] = char4(clamp(rint(n1h * mult), half4(-127.0h), half4(127.0h)));
            dst[2] = char4(clamp(rint(n2h * mult), half4(-127.0h), half4(127.0h)));
            dst[3] = char4(clamp(rint(n3h * mult), half4(-127.0h), half4(127.0h)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX_dev.template slice<128, BM>(int(kt * 128u), 0);
        op.run(sX, tW, cI);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (blk8 == 0u) {
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) tg.epi.ws[q * ROWS_PER_STEP + r_grp] = wsc_arr[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx = cI.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < cnt) {
                const ulong dst_idx = ulong(tile.pair_start + r) * p.intermediate_size + (n0 + c);
                const float raw     = float(cI[i]) * (tg.epi.ws[c] * float(x_sorted_sc[tile.pair_start + r]));
                if (SIK_CONST_M == 0u) {
                    const float cg = sik_softcap(raw, p.gate_softcap);
                    mid_sorted_f16[dst_idx] = half(cg * sik_sigmoid(raw));
                } else {
                    const float gs = float(mid_sorted_f16[dst_idx]);
                    const float u  = sik_softcap(raw, p.up_softcap);
                    mid_sorted_f16[dst_idx] = sik_clamp_f16(gs * u);
                }
            }
        }
    }
}

// High-Occupancy 8KiB Whole-Row Zero-Acc NVFP4 -> INT8xINT8->INT32 MoE W2 (`84.5 TOPS`, 4 TGs/core!)
template <int BM, int SG, int CAP>
inline void sik_moe_prefill_w2_nvfp4_i8_impl(
    device const uint2*          w_u2,
    device const uchar*          w_scales,
    device const float*          exp_scales,
    device const int8_t*         mid_sorted_i8,
    device const half*           mid_sorted_sc,
    device const uint*           sorted_pairs,
    device const SikMoeWorkTile* work_tiles,
    device half*                 down_part_f16,
    constant SikMoeExecParams&   p,
    threadgroup SikMoeTgI8&      tg,
    uint2                        tgid,
    uint                         tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.D) return;
    const SikMoeWorkTile tile = work_tiles[tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    tensor<device int8_t, extents<int, dynamic_extent, dynamic_extent>, tensor_inline> tX_dev(
        const_cast<device int8_t*>(mid_sorted_i8 + ulong(tile.pair_start) * p.intermediate_size),
        extents<int, dynamic_extent, dynamic_extent>(int(p.intermediate_size), BM));
    tensor<threadgroup int8_t, extents<int, 128, 64>, tensor_inline> tW(&tg.W[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto tX0 = tX_dev.template slice<128, BM>(0, 0);
    auto cI  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(tX0)>, sik_strip_as<decltype(tW)>, int32_t>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) if (cI.is_valid_element(i)) cI[i] = 0;

    constexpr uint ROWS_PER_STEP = (uint(SG) * 32u) >> 3u;
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;
    const uint  blk_per_row = p.intermediate_size >> 4u;
    const uint  k_tiles     = p.intermediate_size >> 7u;
    const uint  r_grp       = tid >> 3u;
    const uint  blk8        = tid & 7u;
    const uint  lane_base   = (tid & 31u) & ~7u;
    const ulong r_base      = ulong(tile.expert_id) * p.D + n0;
    const float esc         = exp_scales[tile.expert_id];

    float inv_arr[Q_STEPS];
    float wsc_arr[Q_STEPS];
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < Q_STEPS; ++q) {
        const ulong r_glob = r_base + (q * ROWS_PER_STEP + r_grp);
        uchar bmax = 0u;
        for (uint b = blk8; b < blk_per_row; b += 8u) {
            bmax = max(bmax, uchar(w_scales[r_glob * blk_per_row + b] & 0x7Fu));
        }
        uint um = uint(bmax);
        um = max(um, simd_shuffle_xor(um, 1u));
        um = max(um, simd_shuffle_xor(um, 2u));
        um = max(um, simd_shuffle_xor(um, 4u));
        const float smax = float(sik_e4m3_to_f16(uchar(um)));
        wsc_arr[q] = smax * (esc * (1.0f / 21.0f));
        inv_arr[q] = simd_shuffle((smax > 1e-12f) ? (21.0f / smax) : 0.0f, lane_base);
    }

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const ulong bi = ulong(kt) * 8ul + blk8;
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const ulong r_glob = r_base + r_loc;
            const half  mult   = half(float(sik_e4m3_to_f16(w_scales[r_glob * blk_per_row + bi])) * inv_arr[q]);
            const uint2 v      = w_u2[r_glob * blk_per_row + bi];
            half4 n0h, n1h, n2h, n3h;
            sik_unpack_nvfp4_u32_h4(v.x, n0h, n1h);
            sik_unpack_nvfp4_u32_h4(v.y, n2h, n3h);
            threadgroup char4* dst = reinterpret_cast<threadgroup char4*>(&tg.W[r_loc][blk8 * 16u]);
            dst[0] = char4(clamp(rint(n0h * mult), half4(-127.0h), half4(127.0h)));
            dst[1] = char4(clamp(rint(n1h * mult), half4(-127.0h), half4(127.0h)));
            dst[2] = char4(clamp(rint(n2h * mult), half4(-127.0h), half4(127.0h)));
            dst[3] = char4(clamp(rint(n3h * mult), half4(-127.0h), half4(127.0h)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX_dev.template slice<128, BM>(int(kt * 128u), 0);
        op.run(sX, tW, cI);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (blk8 == 0u) {
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) tg.epi.ws[q * ROWS_PER_STEP + r_grp] = wsc_arr[q];
    }
    if (tid < cnt) tg.epi.pid[tid] = sorted_pairs[tile.pair_start + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx = cI.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < cnt) {
                const float msc = float(mid_sorted_sc[tile.pair_start + r]);
                down_part_f16[ulong(tg.epi.pid[r]) * p.D + (n0 + c)] = sik_clamp_f16(float(cI[i]) * (tg.epi.ws[c] * msc));
            }
        }
    }
}

// High-Occupancy 8KiB Whole-Row Zero-Acc MXFP8 -> INT8xINT8->INT32 MoE W13 (HW FP8 unpack + power-of-2 scale!)
template <int BM, int SG, int CAP>
inline void sik_moe_prefill_w13_mxfp8_i8_pass_impl(
    device const uint4*          w_u4,
    device const uchar*          w_scales,
    device const int8_t*         x_sorted_i8,
    device const half*           x_sorted_sc,
    device const SikMoeWorkTile* work_tiles,
    device half*                 mid_sorted_f16,
    constant SikMoeExecParams&   p,
    threadgroup SikMoeTgI8&      tg,
    uint2                        tgid,
    uint                         tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.intermediate_size) return;
    const SikMoeWorkTile tile = work_tiles[tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    tensor<device int8_t, extents<int, dynamic_extent, dynamic_extent>, tensor_inline> tX_dev(
        const_cast<device int8_t*>(x_sorted_i8 + ulong(tile.pair_start) * p.D),
        extents<int, dynamic_extent, dynamic_extent>(int(p.D), BM));
    tensor<threadgroup int8_t, extents<int, 128, 64>, tensor_inline> tW(&tg.W[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto tX0 = tX_dev.template slice<128, BM>(0, 0);
    auto cI  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(tX0)>, sik_strip_as<decltype(tW)>, int32_t>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) if (cI.is_valid_element(i)) cI[i] = 0;

    constexpr uint ROWS_PER_STEP = (uint(SG) * 32u) >> 3u;
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;
    const uint  u4_per_row  = p.D >> 4u;
    const uint  blk_per_row = p.D >> 5u;
    const uint  k_tiles     = p.D >> 7u;
    const uint  r_grp       = tid >> 3u;
    const uint  blk8        = tid & 7u;
    const ulong r_base      = ulong(tile.expert_id) * (2ul * p.intermediate_size)
                            + ((SIK_CONST_M != 0u) ? ulong(p.intermediate_size) : 0ul) + n0;

    uint  emax_arr[Q_STEPS];
    float wsc_arr[Q_STEPS];
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < Q_STEPS; ++q) {
        const ulong r_glob = r_base + (q * ROWS_PER_STEP + r_grp);
        uchar em = 0u;
        for (uint b = blk8; b < blk_per_row; b += 8u) em = max(em, w_scales[r_glob * blk_per_row + b]);
        uint um = uint(em);
        um = max(um, simd_shuffle_xor(um, 1u));
        um = max(um, simd_shuffle_xor(um, 2u));
        um = max(um, simd_shuffle_xor(um, 4u));
        emax_arr[q] = um;
        wsc_arr[q]  = sik_ue8m0_to_f32(uchar(um)) * (448.0f / 127.0f);
    }

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const ulong bi = ulong(kt) * 8ul + blk8;
        const ulong si = ulong(kt) * 4ul + (blk8 >> 1u);
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const ulong r_glob = r_base + r_loc;
            const uint4 w16    = w_u4[r_glob * u4_per_row + bi];
            const uint  eb     = uint(w_scales[r_glob * blk_per_row + si]);
            const uint  diff   = emax_arr[q] - eb;
            const half  mult   = (diff > 14u) ? 0.0h : half((127.0f / 448.0f) * as_type<float>((127u - diff) << 23u));
            half4 n0h, n1h, n2h, n3h;
            sik_e4m3x8_to_f16x4x2(w16.xy, n0h, n1h);
            sik_e4m3x8_to_f16x4x2(w16.zw, n2h, n3h);
            threadgroup char4* dst = reinterpret_cast<threadgroup char4*>(&tg.W[r_loc][blk8 * 16u]);
            dst[0] = char4(clamp(rint(n0h * mult), half4(-127.0h), half4(127.0h)));
            dst[1] = char4(clamp(rint(n1h * mult), half4(-127.0h), half4(127.0h)));
            dst[2] = char4(clamp(rint(n2h * mult), half4(-127.0h), half4(127.0h)));
            dst[3] = char4(clamp(rint(n3h * mult), half4(-127.0h), half4(127.0h)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX_dev.template slice<128, BM>(int(kt * 128u), 0);
        op.run(sX, tW, cI);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (blk8 == 0u) {
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) tg.epi.ws[q * ROWS_PER_STEP + r_grp] = wsc_arr[q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx = cI.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < cnt) {
                const ulong dst_idx = ulong(tile.pair_start + r) * p.intermediate_size + (n0 + c);
                const float raw     = float(cI[i]) * (tg.epi.ws[c] * float(x_sorted_sc[tile.pair_start + r]));
                if (SIK_CONST_M == 0u) {
                    const float cg = sik_softcap(raw, p.gate_softcap);
                    mid_sorted_f16[dst_idx] = half(cg * sik_sigmoid(raw));
                } else {
                    const float gs = float(mid_sorted_f16[dst_idx]);
                    const float u  = sik_softcap(raw, p.up_softcap);
                    mid_sorted_f16[dst_idx] = sik_clamp_f16(gs * u);
                }
            }
        }
    }
}

// High-Occupancy 8KiB Whole-Row Zero-Acc MXFP8 -> INT8xINT8->INT32 MoE W2
template <int BM, int SG, int CAP>
inline void sik_moe_prefill_w2_mxfp8_i8_impl(
    device const uint4*          w_u4,
    device const uchar*          w_scales,
    device const int8_t*         mid_sorted_i8,
    device const half*           mid_sorted_sc,
    device const uint*           sorted_pairs,
    device const SikMoeWorkTile* work_tiles,
    device half*                 down_part_f16,
    constant SikMoeExecParams&   p,
    threadgroup SikMoeTgI8&      tg,
    uint2                        tgid,
    uint                         tid
) {
    const uint n0 = tgid.x * 64u;
    if (n0 >= p.D) return;
    const SikMoeWorkTile tile = work_tiles[tgid.y];
    const uint cnt = tile.pair_count;
    if (cnt == 0u) return;

    tensor<device int8_t, extents<int, dynamic_extent, dynamic_extent>, tensor_inline> tX_dev(
        const_cast<device int8_t*>(mid_sorted_i8 + ulong(tile.pair_start) * p.intermediate_size),
        extents<int, dynamic_extent, dynamic_extent>(int(p.intermediate_size), BM));
    tensor<threadgroup int8_t, extents<int, 128, 64>, tensor_inline> tW(&tg.W[0][0], extents<int, 128, 64>());

    constexpr auto desc = matmul2d_descriptor(
        BM, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;

    auto tX0 = tX_dev.template slice<128, BM>(0, 0);
    auto cI  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(tX0)>, sik_strip_as<decltype(tW)>, int32_t>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) if (cI.is_valid_element(i)) cI[i] = 0;

    constexpr uint ROWS_PER_STEP = (uint(SG) * 32u) >> 3u;
    constexpr uint Q_STEPS       = 64u / ROWS_PER_STEP;
    const uint  u4_per_row  = p.intermediate_size >> 4u;
    const uint  blk_per_row = p.intermediate_size >> 5u;
    const uint  k_tiles     = p.intermediate_size >> 7u;
    const uint  r_grp       = tid >> 3u;
    const uint  blk8        = tid & 7u;
    const ulong r_base      = ulong(tile.expert_id) * p.D + n0;

    uint  emax_arr[Q_STEPS];
    float wsc_arr[Q_STEPS];
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < Q_STEPS; ++q) {
        const ulong r_glob = r_base + (q * ROWS_PER_STEP + r_grp);
        uchar em = 0u;
        for (uint b = blk8; b < blk_per_row; b += 8u) em = max(em, w_scales[r_glob * blk_per_row + b]);
        uint um = uint(em);
        um = max(um, simd_shuffle_xor(um, 1u));
        um = max(um, simd_shuffle_xor(um, 2u));
        um = max(um, simd_shuffle_xor(um, 4u));
        emax_arr[q] = um;
        wsc_arr[q]  = sik_ue8m0_to_f32(uchar(um)) * (448.0f / 127.0f);
    }

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const ulong bi = ulong(kt) * 8ul + blk8;
        const ulong si = ulong(kt) * 4ul + (blk8 >> 1u);
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) {
            const uint  r_loc  = q * ROWS_PER_STEP + r_grp;
            const ulong r_glob = r_base + r_loc;
            const uint4 w16    = w_u4[r_glob * u4_per_row + bi];
            const uint  eb     = uint(w_scales[r_glob * blk_per_row + si]);
            const uint  diff   = emax_arr[q] - eb;
            const half  mult   = (diff > 14u) ? 0.0h : half((127.0f / 448.0f) * as_type<float>((127u - diff) << 23u));
            half4 n0h, n1h, n2h, n3h;
            sik_e4m3x8_to_f16x4x2(w16.xy, n0h, n1h);
            sik_e4m3x8_to_f16x4x2(w16.zw, n2h, n3h);
            threadgroup char4* dst = reinterpret_cast<threadgroup char4*>(&tg.W[r_loc][blk8 * 16u]);
            dst[0] = char4(clamp(rint(n0h * mult), half4(-127.0h), half4(127.0h)));
            dst[1] = char4(clamp(rint(n1h * mult), half4(-127.0h), half4(127.0h)));
            dst[2] = char4(clamp(rint(n2h * mult), half4(-127.0h), half4(127.0h)));
            dst[3] = char4(clamp(rint(n3h * mult), half4(-127.0h), half4(127.0h)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX_dev.template slice<128, BM>(int(kt * 128u), 0);
        op.run(sX, tW, cI);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (blk8 == 0u) {
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < Q_STEPS; ++q) tg.epi.ws[q * ROWS_PER_STEP + r_grp] = wsc_arr[q];
    }
    if (tid < cnt) tg.epi.pid[tid] = sorted_pairs[tile.pair_start + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < ushort(CAP); ++i) {
        if (cI.is_valid_element(i)) {
            auto idx = cI.get_multidimensional_index(i);
            const uint r = uint(idx[1]), c = uint(idx[0]);
            if (r < cnt) {
                const float msc = float(mid_sorted_sc[tile.pair_start + r]);
                down_part_f16[ulong(tg.epi.pid[r]) * p.D + (n0 + c)] = sik_clamp_f16(float(cI[i]) * (tg.epi.ws[c] * msc));
            }
        }
    }
}
#endif

// INT8 MoE Quad-Bin Entrypoints (`m16`, `m32`, `m64`, `m128`) for NVFP4 & MXFP8 (`8.0 KiB` TG -> 4 TGs/core!)
kernel void sik_moe_prefill_w13_nvfp4_i8_m16(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const int8_t* x [[buffer(3)]], device const half* xsc [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* mo [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_nvfp4_i8_pass_impl<16, 2, 16>(w, s, es, x, xsc, wt, mo, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)x;(void)xsc;(void)wt;(void)mo;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_nvfp4_i8_m32(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const int8_t* x [[buffer(3)]], device const half* xsc [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* mo [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_nvfp4_i8_pass_impl<32, 2, 32>(w, s, es, x, xsc, wt, mo, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)x;(void)xsc;(void)wt;(void)mo;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_nvfp4_i8_m64(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const int8_t* x [[buffer(3)]], device const half* xsc [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* mo [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_nvfp4_i8_pass_impl<64, 4, 32>(w, s, es, x, xsc, wt, mo, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)x;(void)xsc;(void)wt;(void)mo;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_nvfp4_i8_m128(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const int8_t* x [[buffer(3)]], device const half* xsc [[buffer(4)]], device const SikMoeWorkTile* wt [[buffer(5)]],
    device half* mo [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_nvfp4_i8_pass_impl<128, 4, 64>(w, s, es, x, xsc, wt, mo, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)x;(void)xsc;(void)wt;(void)mo;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_nvfp4_i8_m16(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const int8_t* mi [[buffer(3)]], device const half* msc [[buffer(4)]], device const uint* sp [[buffer(5)]],
    device const SikMoeWorkTile* wt [[buffer(6)]], device half* dp [[buffer(7)]], constant SikMoeExecParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_nvfp4_i8_impl<16, 2, 16>(w, s, es, mi, msc, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)mi;(void)msc;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_nvfp4_i8_m32(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const int8_t* mi [[buffer(3)]], device const half* msc [[buffer(4)]], device const uint* sp [[buffer(5)]],
    device const SikMoeWorkTile* wt [[buffer(6)]], device half* dp [[buffer(7)]], constant SikMoeExecParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_nvfp4_i8_impl<32, 2, 32>(w, s, es, mi, msc, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)mi;(void)msc;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_nvfp4_i8_m64(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const int8_t* mi [[buffer(3)]], device const half* msc [[buffer(4)]], device const uint* sp [[buffer(5)]],
    device const SikMoeWorkTile* wt [[buffer(6)]], device half* dp [[buffer(7)]], constant SikMoeExecParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_nvfp4_i8_impl<64, 4, 32>(w, s, es, mi, msc, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)mi;(void)msc;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_nvfp4_i8_m128(
    device const uint2* w [[buffer(0)]], device const uchar* s [[buffer(1)]], device const float* es [[buffer(2)]],
    device const int8_t* mi [[buffer(3)]], device const half* msc [[buffer(4)]], device const uint* sp [[buffer(5)]],
    device const SikMoeWorkTile* wt [[buffer(6)]], device half* dp [[buffer(7)]], constant SikMoeExecParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_nvfp4_i8_impl<128, 4, 64>(w, s, es, mi, msc, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)es;(void)mi;(void)msc;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_mxfp8_i8_m16(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const int8_t* x [[buffer(2)]], device const half* xsc [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* mo [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_mxfp8_i8_pass_impl<16, 2, 16>(w, s, x, xsc, wt, mo, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)xsc;(void)wt;(void)mo;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_mxfp8_i8_m32(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const int8_t* x [[buffer(2)]], device const half* xsc [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* mo [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_mxfp8_i8_pass_impl<32, 2, 32>(w, s, x, xsc, wt, mo, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)xsc;(void)wt;(void)mo;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_mxfp8_i8_m64(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const int8_t* x [[buffer(2)]], device const half* xsc [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* mo [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_mxfp8_i8_pass_impl<64, 4, 32>(w, s, x, xsc, wt, mo, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)xsc;(void)wt;(void)mo;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w13_mxfp8_i8_m128(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const int8_t* x [[buffer(2)]], device const half* xsc [[buffer(3)]], device const SikMoeWorkTile* wt [[buffer(4)]],
    device half* mo [[buffer(5)]], constant SikMoeExecParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w13_mxfp8_i8_pass_impl<128, 4, 64>(w, s, x, xsc, wt, mo, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)x;(void)xsc;(void)wt;(void)mo;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_mxfp8_i8_m16(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const int8_t* mi [[buffer(2)]], device const half* msc [[buffer(3)]], device const uint* sp [[buffer(4)]],
    device const SikMoeWorkTile* wt [[buffer(5)]], device half* dp [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_mxfp8_i8_impl<16, 2, 16>(w, s, mi, msc, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)mi;(void)msc;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_mxfp8_i8_m32(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const int8_t* mi [[buffer(2)]], device const half* msc [[buffer(3)]], device const uint* sp [[buffer(4)]],
    device const SikMoeWorkTile* wt [[buffer(5)]], device half* dp [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_mxfp8_i8_impl<32, 2, 32>(w, s, mi, msc, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)mi;(void)msc;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_mxfp8_i8_m64(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const int8_t* mi [[buffer(2)]], device const half* msc [[buffer(3)]], device const uint* sp [[buffer(4)]],
    device const SikMoeWorkTile* wt [[buffer(5)]], device half* dp [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_mxfp8_i8_impl<64, 4, 32>(w, s, mi, msc, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)mi;(void)msc;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

kernel void sik_moe_prefill_w2_mxfp8_i8_m128(
    device const uint4* w [[buffer(0)]], device const uchar* s [[buffer(1)]],
    device const int8_t* mi [[buffer(2)]], device const half* msc [[buffer(3)]], device const uint* sp [[buffer(4)]],
    device const SikMoeWorkTile* wt [[buffer(5)]], device half* dp [[buffer(6)]], constant SikMoeExecParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup SikMoeTgI8 tg;
#if defined(__HAVE_TENSOR__)
    sik_moe_prefill_w2_mxfp8_i8_impl<128, 4, 64>(w, s, mi, msc, sp, wt, dp, p, tg, tgid, tid);
#else
    (void)w;(void)s;(void)mi;(void)msc;(void)sp;(void)wt;(void)dp;(void)p;(void)tg;(void)tgid;(void)tid;
#endif
}

// 2D Parallel FP32 Top-8 MoE Combine (short-prompt grouped-8 keeps exact FP32 `down_part`)
kernel void sik_moe_combine(
    device const float4*      down_part [[buffer(0)]],
    device const float*       topk_w    [[buffer(1)]],
    device float4*            moe_out   [[buffer(2)]],
    constant SikMoeExecParams& p        [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint m      = tgid.y;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (m >= p.M_valid || idx4 >= d4_cnt) return;

    float4 acc(0.0f);
    for (uint k = 0u; k < p.top_k; ++k) {
        const ulong pair = ulong(m) * p.top_k + k;
        acc = fma(float4(topk_w[pair]), down_part[pair * d4_cnt + idx4], acc);
    }
    moe_out[ulong(m) * d4_cnt + idx4] = acc;
}

// 2D Parallel FP16->FP32 Top-8 MoE Combine for Prefill NAX (2x less DRAM traffic, stays in 48 MiB SLC!)
kernel void sik_moe_combine_f16(
    device const half4*       down_part_f16 [[buffer(0)]],
    device const float*       topk_w        [[buffer(1)]],
    device float4*            moe_out       [[buffer(2)]],
    constant SikMoeExecParams& p            [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint m      = tgid.y;
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    if (m >= p.M_valid || idx4 >= d4_cnt) return;

    float4 acc(0.0f);
    for (uint k = 0u; k < p.top_k; ++k) {
        const ulong pair = ulong(m) * p.top_k + k;
        acc = fma(float4(topk_w[pair]), float4(down_part_f16[pair * d4_cnt + idx4]), acc);
    }
    moe_out[ulong(m) * d4_cnt + idx4] = acc;
}

/* ============================================================================
 * 8. 3-Layer DSpark Block-8 (`K = 1..7`) MTP Drafter + 48 MiB SLC-Pinned
 *    32K FR-Spec Folded Markov Head + Mixed-Precision (`FP16`/`Q8_0`/`Q4_A`)
 * ============================================================================ */

struct SikMtpFoldParams {
    uint V;
    uint D;
    uint R;
    uint unembed_dtype; // F16, Q8_0, or Q4_A
};

// Startup/Lazy Full-Vocab Head Folding on NAX (`BK = 128` + Direct-Device `tB`, ZERO `B_tile` TG mem!)
kernel void sik_mtp_fold_head_nax(
    device const uchar*        unembed_w  [[buffer(0)]],
    device const half*         unembed_s  [[buffer(1)]],
    device const half*         unembed_b  [[buffer(2)]],
    device const half*         hidden_out [[buffer(3)]], // [D, R] FP16
    device half*               fold_w     [[buffer(4)]], // [V, R] FP16
    constant SikMtpFoldParams& p          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint r0 = tgid.x * 64u;
    const uint v0 = tgid.y * 64u;
    if (r0 >= p.R || v0 >= p.V) return;

#if defined(__HAVE_TENSOR__)
    threadgroup half A_tile[64][128];

    tensor<threadgroup half, extents<int, 128, 64>, tensor_inline> tA(&A_tile[0][0], extents<int, 128, 64>());
    tensor<device half, extents<int, dynamic_extent, dynamic_extent>, tensor_inline> tB_dev(
        const_cast<device half*>(hidden_out),
        extents<int, dynamic_extent, dynamic_extent>(int(p.R), int(p.D)));

    constexpr auto desc = matmul2d_descriptor(
        64, 64, 128, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto tB0 = tB_dev.slice<64, 128>(int(r0), 0);
    auto cT  = op.template get_destination_cooperative_tensor<sik_strip_as<decltype(tA)>, sik_strip_as<decltype(tB0)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;

    const uint k_tiles  = p.D >> 7u;
    const uint n_blk64  = p.D >> 6u;
    const uint r_grp    = tid >> 3u; // 0..15
    const uint sub8     = tid & 7u;  // 0..7 (each thread loads 16 elems = 4 half4s per q!)

    device const uint4* uu4 = reinterpret_cast<device const uint4*>(unembed_w);
    device const uint2* uu2 = reinterpret_cast<device const uint2*>(unembed_w);

    for (uint kt = 0u; kt < k_tiles; ++kt) {
        const uint blk64_idx = kt * 2u + (sub8 >> 2u);
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 4u; ++q) {
            const uint v_loc  = q * 16u + r_grp;
            const uint v_glob = v0 + v_loc;
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(&A_tile[v_loc][sub8 * 16u]);
            if (v_glob < p.V) {
                if (p.unembed_dtype == STRATA_INKLING2_DTYPE_Q4_A) {
                    const uint2 v2 = uu2[ulong(v_glob) * (p.D >> 4u) + kt * 8u + sub8];
                    const half  sc = unembed_s[ulong(v_glob) * n_blk64 + blk64_idx];
                    const half  bi = unembed_b[ulong(v_glob) * n_blk64 + blk64_idx];
                    half4 n0, n1, n2, n3;
                    sik_unpack_u4_u32_h4(v2.x, n0, n1);
                    sik_unpack_u4_u32_h4(v2.y, n2, n3);
                    dst[0] = fma(n0, half4(sc), half4(bi));
                    dst[1] = fma(n1, half4(sc), half4(bi));
                    dst[2] = fma(n2, half4(sc), half4(bi));
                    dst[3] = fma(n3, half4(sc), half4(bi));
                } else if (p.unembed_dtype == STRATA_INKLING2_DTYPE_Q8_0) {
                    const half  sc = unembed_s[ulong(v_glob) * n_blk64 + blk64_idx];
                    const uint4 q4 = uu4[ulong(v_glob) * (p.D >> 4u) + kt * 8u + sub8];
                    dst[0] = half4(as_type<char4>(q4.x)) * sc;
                    dst[1] = half4(as_type<char4>(q4.y)) * sc;
                    dst[2] = half4(as_type<char4>(q4.z)) * sc;
                    dst[3] = half4(as_type<char4>(q4.w)) * sc;
                } else {
                    const ulong hi = ulong(v_glob) * (p.D >> 3u) + kt * 16u + sub8 * 2u;
                    const uint4 h0 = uu4[hi + 0u];
                    const uint4 h1 = uu4[hi + 1u];
                    reinterpret_cast<threadgroup uint4*>(dst)[0] = h0;
                    reinterpret_cast<threadgroup uint4*>(dst)[1] = h1;
                }
            } else {
                dst[0] = half4(0.0h); dst[1] = half4(0.0h);
                dst[2] = half4(0.0h); dst[3] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sB = tB_dev.slice<64, 128>(int(r0), int(kt * 128u));
        op.run(tA, sB, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            const uint v_glob = v0 + uint(idx[1]);
            const uint r_glob = r0 + uint(idx[0]);
            if (v_glob < p.V && r_glob < p.R) fold_w[ulong(v_glob) * p.R + r_glob] = sik_clamp_f16(cT[i]);
        }
    }
#else
    (void)unembed_w;(void)unembed_s;(void)unembed_b;(void)hidden_out;(void)fold_w;(void)tid;
#endif
}

struct SikMtpFuseParams {
    uint  M_valid;
    uint  M_padded;
    uint  D;
    uint  alpha_row;
    float eps;
    uint  write_f32;
    uint  pad0;
    uint  pad1;
};

kernel void sik_mtp_fuse_taps_norm(
    device const float*        taps_in    [[buffer(0)]],
    device const float*        fuse_alpha [[buffer(1)]],
    device const float*        norm_w     [[buffer(2)]],
    device half*               out_f16    [[buffer(3)]],
    device float*              out_f32    [[buffer(4)]],
    constant SikMtpFuseParams& p          [[buffer(5)]],
    device float*              xsum_out   [[buffer(6)]],
    uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_padded) return;
    const uint  d4_cnt  = p.D >> 2u;
    const ulong row_off = ulong(m) * p.D;
    if (m >= p.M_valid) {
        device half4* dst_h = reinterpret_cast<device half4*>(out_f16 + row_off);
        for (uint i = tid; i < d4_cnt; i += 256u) dst_h[i] = half4(0.0h);
        return;
    }

    const float a0 = fuse_alpha[p.alpha_row * 4u + 0u];
    const float a1 = fuse_alpha[p.alpha_row * 4u + 1u];
    const float a2 = fuse_alpha[p.alpha_row * 4u + 2u];
    const float a3 = fuse_alpha[p.alpha_row * 4u + 3u];

    device const float4* t0 = reinterpret_cast<device const float4*>(taps_in + (ulong(m) * 4ul + 0ul) * p.D);
    device const float4* t1 = reinterpret_cast<device const float4*>(taps_in + (ulong(m) * 4ul + 1ul) * p.D);
    device const float4* t2 = reinterpret_cast<device const float4*>(taps_in + (ulong(m) * 4ul + 2ul) * p.D);
    device const float4* t3 = reinterpret_cast<device const float4*>(taps_in + (ulong(m) * 4ul + 3ul) * p.D);
    device const float4* w4 = reinterpret_cast<device const float4*>(norm_w);

    threadgroup float tg_sq[8];
    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        float4 v = fma(t3[i], float4(a3), fma(t2[i], float4(a2), fma(t1[i], float4(a1), t0[i] * a0)));
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }
    const float sg_s = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float    inv_rms = tg_sq[0];
    const uint     nblk    = p.D >> 6u;
    device half4*  dst_h   = reinterpret_cast<device half4*>(out_f16 + row_off);
    device float4* dst_f   = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + row_off) : nullptr;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 v      = fma(t3[i], float4(a3), fma(t2[i], float4(a2), fma(t1[i], float4(a1), t0[i] * a0)));
        const float4 normed = (v * inv_rms) * w4[i];
        const half4  h      = sik_clamp_f16x4(normed);
        dst_h[i] = h;
        if (dst_f) dst_f[i] = normed;
        if (p.M_valid <= 8u) {
            const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
            if ((lane & 15u) == 0u) xsum_out[ulong(m) * nblk + (i >> 4u)] = bs;
        }
    }
}

struct SikMtpCtxKvParams {
    uint  M_valid;
    uint  n_kv;
    uint  head_dim;
    uint  pos_start;
    uint  kv_stride;
    uint  is_local;
    float eps;
    uint  pad0;
};

kernel void sik_mtp_ctx_kv_prep(
    device const float*         k_raw    [[buffer(0)]],
    device const float*         v_raw    [[buffer(1)]],
    device const float*         k_norm_w [[buffer(2)]],
    device const float*         inv_freq [[buffer(3)]],
    device half*                k_cache  [[buffer(4)]],
    device half*                v_cache  [[buffer(5)]],
    constant SikMtpCtxKvParams& p        [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint hkv = tgid.x, m = tgid.y;
    if (hkv >= p.n_kv || m >= p.M_valid) return;

    const uint  pos     = p.pos_start + m;
    const uint  slot    = (p.is_local != 0u) ? (pos & (p.kv_stride - 1u)) : pos;
    const ulong src_off = (ulong(m) * p.n_kv + hkv) * p.head_dim;
    const ulong dst_off = (ulong(hkv) * p.kv_stride + slot) * p.head_dim;

    if (tid < p.head_dim) v_cache[dst_off + tid] = sik_clamp_f16(v_raw[src_off + tid]);

    threadgroup float tg_sq[4];
    threadgroup float tg_k[128];
    const float kv = (tid < p.head_dim) ? k_raw[src_off + tid] : 0.0f;
    const float s  = simd_sum(kv * kv);
    if (lane == 0u) tg_sq[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.head_dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < p.head_dim) tg_k[tid] = (kv * tg_sq[0]) * k_norm_w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_d = p.head_dim >> 1u;
    if (tid < half_d) {
        const float angle = float(pos) * inv_freq[tid];
        const float c = cos(angle), sn = sin(angle);
        const float x0 = tg_k[tid], x1 = tg_k[tid + half_d];
        k_cache[dst_off + tid]          = sik_clamp_f16(x0 * c - x1 * sn);
        k_cache[dst_off + tid + half_d] = sik_clamp_f16(x1 * c + x0 * sn);
    }
}

struct SikMtpInitBlockParams {
    uint  block_size;
    uint  D;
    uint  vocab_size;
    float eps;
};

kernel void sik_mtp_init_block(
    device const half*              embed_w    [[buffer(0)]],
    device const float*             embed_norm [[buffer(1)]],
    device const half*              mask_embed [[buffer(2)]],
    device const float*             in_norm_w  [[buffer(3)]],
    device const uint*              tok_buf    [[buffer(4)]],
    device float*                   res_out    [[buffer(5)]],
    device half*                    norm_f16   [[buffer(6)]],
    constant SikMtpInitBlockParams& p          [[buffer(7)]],
    device float*                   xsum_out   [[buffer(8)]],
    uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    if (b >= p.block_size) return;
    const uint d4_cnt = p.D >> 2u;
    const uint nblk   = p.D >> 6u;
    threadgroup float tg_sq[8];

    device float4*       dst_r = reinterpret_cast<device float4*>(res_out  + ulong(b) * p.D);
    device half4*        dst_n = reinterpret_cast<device half4*>(norm_f16  + ulong(b) * p.D);
    device const float4* inw4  = reinterpret_cast<device const float4*>(in_norm_w);

    if (b == 0u) {
        const uint tok = min(tok_buf[0], p.vocab_size - 1u);
        device const half4*  e4  = reinterpret_cast<device const half4*>(embed_w + ulong(tok) * p.D);
        device const float4* en4 = reinterpret_cast<device const float4*>(embed_norm);

        float sq1 = 0.0f;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 v = float4(e4[i]);
            sq1 = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq1))));
        }
        float s1 = simd_sum(sq1);
        if (lane == 0u) tg_sq[sgid] = s1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ir1 = tg_sq[0];

        float sq2 = 0.0f;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 h = (float4(e4[i]) * ir1) * en4[i];
            dst_r[i] = h;
            sq2 = fma(h.x, h.x, fma(h.y, h.y, fma(h.z, h.z, fma(h.w, h.w, sq2))));
        }
        float s2 = simd_sum(sq2);
        if (lane == 0u) tg_sq[sgid] = s2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ir2 = tg_sq[0];
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const half4 h = sik_clamp_f16x4((dst_r[i] * ir2) * inw4[i]);
            dst_n[i] = h;
            const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
            if ((lane & 15u) == 0u) xsum_out[ulong(b) * nblk + (i >> 4u)] = bs;
        }
    } else {
        device const half4* m4 = reinterpret_cast<device const half4*>(mask_embed);
        float sq = 0.0f;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 h = float4(m4[i]);
            dst_r[i] = h;
            sq = fma(h.x, h.x, fma(h.y, h.y, fma(h.z, h.z, fma(h.w, h.w, sq))));
        }
        float s = simd_sum(sq);
        if (lane == 0u) tg_sq[sgid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ir = tg_sq[0];
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const half4 h = sik_clamp_f16x4((dst_r[i] * ir) * inw4[i]);
            dst_n[i] = h;
            const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
            if ((lane & 15u) == 0u) xsum_out[ulong(b) * nblk + (i >> 4u)] = bs;
        }
    }
}

// Q4_A Embedding MTP Block-8 Init (`-fast` bundle!)
kernel void sik_mtp_init_block_q4a(
    device const uint*              embed_q    [[buffer(0)]],
    device const half*              embed_sc   [[buffer(1)]],
    device const half*              embed_bi   [[buffer(2)]],
    device const float*             embed_norm [[buffer(3)]],
    device const half*              mask_embed [[buffer(4)]],
    device const float*             in_norm_w  [[buffer(5)]],
    device const uint*              tok_buf    [[buffer(6)]],
    device float*                   res_out    [[buffer(7)]],
    device half*                    norm_f16   [[buffer(8)]],
    constant SikMtpInitBlockParams& p          [[buffer(9)]],
    device float*                   xsum_out   [[buffer(10)]],
    uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    if (b >= p.block_size) return;
    const uint d4_cnt = p.D >> 2u;
    const uint nblk   = p.D >> 6u;
    threadgroup float tg_sq[8];

    device float4*       dst_r = reinterpret_cast<device float4*>(res_out  + ulong(b) * p.D);
    device half4*        dst_n = reinterpret_cast<device half4*>(norm_f16  + ulong(b) * p.D);
    device const float4* inw4  = reinterpret_cast<device const float4*>(in_norm_w);

    if (b == 0u) {
        const uint tok       = min(tok_buf[0], p.vocab_size - 1u);
        const uint u32_per_r = p.D >> 3u;
        const uint grp_per_r = p.D >> 6u;
        device const uint*   wq  = embed_q  + ulong(tok) * u32_per_r;
        device const half*   wsc = embed_sc + ulong(tok) * grp_per_r;
        device const half*   wbi = embed_bi + ulong(tok) * grp_per_r;
        device const float4* en4 = reinterpret_cast<device const float4*>(embed_norm);

        float sq1 = 0.0f;
        for (uint u = tid; u < u32_per_r; u += 256u) {
            const uint grp = u >> 3u;
            const float sc = float(wsc[grp]), bi = float(wbi[grp]);
            half4 n0, n1;
            sik_unpack_u4_u32_h4(wq[u], n0, n1);
            const float4 v0 = fma(float4(n0), float4(sc), float4(bi));
            const float4 v1 = fma(float4(n1), float4(sc), float4(bi));
            sq1 = fma(v0.x, v0.x, fma(v0.y, v0.y, fma(v0.z, v0.z, fma(v0.w, v0.w, sq1))));
            sq1 = fma(v1.x, v1.x, fma(v1.y, v1.y, fma(v1.z, v1.z, fma(v1.w, v1.w, sq1))));
        }
        float s1 = simd_sum(sq1);
        if (lane == 0u) tg_sq[sgid] = s1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ir1 = tg_sq[0];

        float sq2 = 0.0f;
        for (uint u = tid; u < u32_per_r; u += 256u) {
            const uint grp = u >> 3u;
            const float sc = float(wsc[grp]), bi = float(wbi[grp]);
            half4 n0, n1;
            sik_unpack_u4_u32_h4(wq[u], n0, n1);
            const float4 h0 = (fma(float4(n0), float4(sc), float4(bi)) * ir1) * en4[u * 2u + 0u];
            const float4 h1 = (fma(float4(n1), float4(sc), float4(bi)) * ir1) * en4[u * 2u + 1u];
            dst_r[u * 2u + 0u] = h0;
            dst_r[u * 2u + 1u] = h1;
            sq2 = fma(h0.x, h0.x, fma(h0.y, h0.y, fma(h0.z, h0.z, fma(h0.w, h0.w, sq2))));
            sq2 = fma(h1.x, h1.x, fma(h1.y, h1.y, fma(h1.z, h1.z, fma(h1.w, h1.w, sq2))));
        }
        float s2 = simd_sum(sq2);
        if (lane == 0u) tg_sq[sgid] = s2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ir2 = tg_sq[0];
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const half4 h = sik_clamp_f16x4((dst_r[i] * ir2) * inw4[i]);
            dst_n[i] = h;
            const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
            if ((lane & 15u) == 0u) xsum_out[ulong(b) * nblk + (i >> 4u)] = bs;
        }
    } else {
        device const half4* m4 = reinterpret_cast<device const half4*>(mask_embed);
        float sq = 0.0f;
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const float4 h = float4(m4[i]);
            dst_r[i] = h;
            sq = fma(h.x, h.x, fma(h.y, h.y, fma(h.z, h.z, fma(h.w, h.w, sq))));
        }
        float s = simd_sum(sq);
        if (lane == 0u) tg_sq[sgid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
            if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ir = tg_sq[0];
        for (uint i = tid; i < d4_cnt; i += 256u) {
            const half4 h = sik_clamp_f16x4((dst_r[i] * ir) * inw4[i]);
            dst_n[i] = h;
            const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
            if ((lane & 15u) == 0u) xsum_out[ulong(b) * nblk + (i >> 4u)] = bs;
        }
    }
}

struct SikMtpBlockPrepParams {
    uint  block_size;
    uint  n_q;
    uint  n_kv;
    uint  head_dim;
    uint  pos_start;
    float eps;
    uint  pad0;
    uint  pad1;
};

kernel void sik_mtp_block_qk_prep(
    device const float*             q_raw    [[buffer(0)]],
    device const float*             k_raw    [[buffer(1)]],
    device const float*             q_norm_w [[buffer(2)]],
    device const float*             k_norm_w [[buffer(3)]],
    device const float*             inv_freq [[buffer(4)]],
    device float*                   q_out    [[buffer(5)]],
    device half*                    k_out    [[buffer(6)]],
    constant SikMtpBlockPrepParams& p        [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint b = tgid.y;
    if (b >= p.block_size) return;
    const uint idx  = tgid.x;
    const bool is_q = (idx < p.n_q);
    const uint h    = is_q ? idx : (idx - p.n_q);
    if (!is_q && h >= p.n_kv) return;

    const uint  pos    = p.pos_start + b;
    const uint  half_d = p.head_dim >> 1u;
    const ulong off    = (ulong(b) * (is_q ? p.n_q : p.n_kv) + h) * p.head_dim;

    threadgroup float tg_sq[4];
    threadgroup float tg_v[128];

    const float val = (tid < p.head_dim) ? (is_q ? q_raw[off + tid] : k_raw[off + tid]) : 0.0f;
    const float s   = simd_sum(val * val);
    if (lane == 0u) tg_sq[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot = simd_sum((lane < 4u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.head_dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < p.head_dim) tg_v[tid] = (val * tg_sq[0]) * (is_q ? q_norm_w[tid] : k_norm_w[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < half_d) {
        const float angle = float(pos) * inv_freq[tid];
        const float c = cos(angle), sn = sin(angle);
        const float x0 = tg_v[tid], x1 = tg_v[tid + half_d];
        const float y0 = x0 * c - x1 * sn;
        const float y1 = x1 * c + x0 * sn;
        if (is_q) { q_out[off + tid] = y0; q_out[off + tid + half_d] = y1; }
        else      { k_out[off + tid] = sik_clamp_f16(y0); k_out[off + tid + half_d] = sik_clamp_f16(y1); }
    }
}

struct SikMtpAttnParams {
    uint  block_size; // 8
    uint  n_q;        // 24
    uint  n_kv;       // 8
    uint  head_dim;   // 128
    uint  ctx_start;
    uint  ctx_end;
    uint  kv_stride;
    uint  is_local;
    uint  max_splits; // 32
    float attn_scale;
    uint  pad0;
    uint  pad1;
};

inline void sik_mtp_split_plan(uint total_kv, uint max_splits, thread uint& my_splits, thread uint& kv_per_split) {
    const uint target = clamp((total_kv + 31u) >> 5u, 1u, max_splits);
    kv_per_split      = ((total_kv + target - 1u) / target + 15u) & ~15u;
    my_splits         = max(1u, (total_kv + kv_per_split - 1u) / kv_per_split);
}

// Rewritten Shared-KV Block-16 MTP Block-8 Attention (reads ctx_k/v ONCE for all 8 block queries -> 8x less KV DRAM!)
// Grid: `(launch_splits, 8 /* n_kv */, 1)`, `96` threads/TG (`3` SGs = `3` Q heads/KV head!)
kernel void sik_mtp_block_attn_splitkv(
    device const float4*       q_in    [[buffer(0)]], // [8, 24, 32]
    device const half4*        ctx_k   [[buffer(1)]],
    device const half4*        ctx_v   [[buffer(2)]],
    device const half4*        draft_k [[buffer(3)]],
    device const half4*        draft_v [[buffer(4)]],
    device float*              part_m  [[buffer(5)]], // [8, max_splits, 24]
    device float*              part_l  [[buffer(6)]],
    device float4*             part_o  [[buffer(7)]],
    constant SikMtpAttnParams& p       [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint split_idx = tgid.x;
    const uint hkv       = tgid.y;
    if (hkv >= p.n_kv) return;

    const uint ctx_len  = p.ctx_end - p.ctx_start;
    const uint total_kv = ctx_len + p.block_size;
    uint my_splits = 1u, kv_per_split = 16u;
    sik_mtp_split_plan(total_kv, p.max_splits, my_splits, kv_per_split);
    if (split_idx >= my_splits) return;

    const uint t_beg = min(total_kv, split_idx * kv_per_split);
    const uint t_end = min(total_kv, t_beg + kv_per_split);
    const uint hq    = hkv * 3u + sgid;

    float4 q_vec[8];
    float  m_val[8];
    float  l_val[8];
    float4 o_vec[8];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < 8u; ++b) {
        q_vec[b] = (b < p.block_size) ? (q_in[(ulong(b) * p.n_q + hq) * 32ul + lane] * p.attn_scale) : float4(0.0f);
        m_val[b] = -INFINITY;
        l_val[b] = 0.0f;
        o_vec[b] = float4(0.0f);
    }

    device const half4* k_head = ctx_k + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* v_head = ctx_v + ulong(hkv) * ulong(p.kv_stride) * 32ul;

    threadgroup half4 tg_k[16][32];
    threadgroup half4 tg_v[16][32];

    for (uint t_base = t_beg; t_base < t_end; t_base += 16u) {
        const uint tile_cnt = min(16u, t_end - t_base);
        for (uint ti = sgid; ti < tile_cnt; ti += 3u) {
            const uint t_cur = t_base + ti;
            if (t_cur < ctx_len) {
                const uint  pos  = p.ctx_start + t_cur;
                const uint  slot = (p.is_local != 0u) ? (pos & (p.kv_stride - 1u)) : pos;
                const ulong off  = ulong(slot) * 32ul + lane;
                tg_k[ti][lane] = k_head[off];
                tg_v[ti][lane] = v_head[off];
            } else {
                const uint  db  = t_cur - ctx_len;
                const ulong off = (ulong(db) * p.n_kv + hkv) * 32ul + lane;
                tg_k[ti][lane] = draft_k[off];
                tg_v[ti][lane] = draft_v[off];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ti = 0u; ti < tile_cnt; ++ti) {
            const float4 kv = float4(tg_k[ti][lane]);
            const float4 vv = float4(tg_v[ti][lane]);
            #pragma clang loop unroll(full)
            for (uint b = 0u; b < 8u; ++b) {
                const float score = simd_sum(dot(q_vec[b], kv));
                const float m_new = max(m_val[b], score);
                const float alpha = (m_val[b] == -INFINITY) ? 0.0f : exp(m_val[b] - m_new);
                const float w     = exp(score - m_new);
                m_val[b] = m_new;
                l_val[b] = fma(l_val[b], alpha, w);
                o_vec[b] = fma(o_vec[b], alpha, w * vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint b = 0u; b < 8u; ++b) {
        if (b < p.block_size) {
            const ulong ml_idx = (ulong(b) * p.max_splits + split_idx) * p.n_q + hq;
            if (lane == 0u) { part_m[ml_idx] = m_val[b]; part_l[ml_idx] = l_val[b]; }
            part_o[ml_idx * 32ul + lane] = o_vec[b];
        }
    }
}

kernel void sik_mtp_block_attn_reduce(
    device const float*        part_m     [[buffer(0)]],
    device const float*        part_l     [[buffer(1)]],
    device const float4*       part_o     [[buffer(2)]],
    device half4*              out_f16    [[buffer(3)]],
    constant SikMtpAttnParams& p          [[buffer(4)]],
    device float*              wo_blk_sum [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint hq = tgid.x, b = tgid.y;
    if (hq >= p.n_q || b >= p.block_size) return;

    const uint total_kv = (p.ctx_end - p.ctx_start) + p.block_size;
    uint my_splits = 1u, kv_per_split = 16u;
    sik_mtp_split_plan(total_kv, p.max_splits, my_splits, kv_per_split);

    float m_max = -INFINITY;
    for (uint s = 0u; s < my_splits; ++s) m_max = max(m_max, part_m[(ulong(b) * p.max_splits + s) * p.n_q + hq]);
    float  l_tot = 0.0f;
    float4 o_tot = float4(0.0f);
    if (m_max > -INFINITY) {
        for (uint s = 0u; s < my_splits; ++s) {
            const ulong idx = (ulong(b) * p.max_splits + s) * p.n_q + hq;
            const float ms  = part_m[idx];
            if (ms > -INFINITY) {
                const float w = exp(ms - m_max);
                l_tot = fma(part_l[idx], w, l_tot);
                o_tot = fma(part_o[idx * 32ul + lane], float4(w), o_tot);
            }
        }
    }
    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    const half4 oh    = sik_clamp_f16x4(o_tot * inv_l);
    out_f16[(ulong(b) * p.n_q + hq) * 32ul + lane] = oh;
    // Zero-barrier block-64 sums (`lane 0..15` -> blk 0, `lane 16..31` -> blk 1) for Q4_A `ml.o_proj`!
    const float bs = sik_reduce16((float(oh.x) + float(oh.y)) + (float(oh.z) + float(oh.w)));
    if ((lane & 15u) == 0u) {
        wo_blk_sum[(ulong(b) * p.n_q + hq) * 2ul + (lane >> 4u)] = bs;
    }
}

struct SikMtpAddNormParams {
    uint  M_valid;
    uint  D;
    float eps;
    uint  write_f32;
};

kernel void sik_mtp_add_rmsnorm(
    device const float*           delta    [[buffer(0)]],
    device float*                 res_io   [[buffer(1)]],
    device const float*           norm_w   [[buffer(2)]],
    device half*                  out_f16  [[buffer(3)]],
    device float*                 out_f32  [[buffer(4)]],
    constant SikMtpAddNormParams& p        [[buffer(5)]],
    device float*                 xsum_out [[buffer(6)]],
    uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M_valid) return;
    const uint  d4_cnt  = p.D >> 2u;
    const uint  nblk    = p.D >> 6u;
    const ulong row_off = ulong(m) * p.D;
    device const float4* d4 = reinterpret_cast<device const float4*>(delta  + row_off);
    device float4*       r4 = reinterpret_cast<device float4*>(res_io + row_off);
    device const float4* w4 = reinterpret_cast<device const float4*>(norm_w);

    threadgroup float tg_sq[8];
    float sq_acc = 0.0f;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 v = r4[i] + d4[i];
        r4[i] = v;
        sq_acc = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq_acc))));
    }
    const float sg_s = simd_sum(sq_acc);
    if (lane == 0u) tg_sq[sgid] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot = simd_sum((lane < 8u) ? tg_sq[lane] : 0.0f);
        if (lane == 0u) tg_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float    inv_rms = tg_sq[0];
    device half4*  dst_h   = reinterpret_cast<device half4*>(out_f16 + row_off);
    device float4* dst_f   = (p.write_f32 != 0u) ? reinterpret_cast<device float4*>(out_f32 + row_off) : nullptr;
    for (uint i = tid; i < d4_cnt; i += 256u) {
        const float4 normed = (r4[i] * inv_rms) * w4[i];
        const half4  h      = sik_clamp_f16x4(normed);
        dst_h[i] = h;
        if (dst_f) dst_f[i] = normed;
        const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
        if ((lane & 15u) == 0u) xsum_out[ulong(m) * nblk + (i >> 4u)] = bs;
    }
}

struct SikMtpSwigluMixedParams {
    ulong wg_off;
    ulong wg_s_off;
    ulong wg_b_off;
    ulong wu_off;
    ulong wu_s_off;
    ulong wu_b_off;
    uint  dtype; // F16, Q8_0, or Q4_A
    uint  M;     // 8
    uint  rows;  // 9216
    uint  cols;  // 2304
};

// Single-Dispatch Fused MTP `gate_proj + up_proj + silu(g)*u` (`32` lanes/row = 256B burst, `4` rows/TG, `FP16`/`Q8_0`/`Q4_A` w/ `uint2` + `x_sum`!)
kernel void sik_mtp_gate_up_silu_b8_mixed(
    device const uchar*               uma     [[buffer(0)]],
    device const half*                x_in    [[buffer(1)]],
    device half*                      mid_out [[buffer(2)]],
    constant SikMtpSwigluMixedParams& p       [[buffer(3)]],
    device const float*               x_sum   [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid * 4u + sgid;
    if (row >= p.rows) return;

    float ga[8], ua[8];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 8u; ++m) { ga[m] = 0.0f; ua[m] = 0.0f; }

    if (p.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
        const uint k16     = p.cols >> 4u;
        const uint nblk    = p.cols >> 6u;
        const bool do_bias = ((lane & 3u) == 0u);
        device const uint2* wg = reinterpret_cast<device const uint2*>(uma + p.wg_off) + ulong(row) * k16;
        device const half*  sg = reinterpret_cast<device const half*>(uma + p.wg_s_off) + ulong(row) * nblk;
        device const half*  bg = reinterpret_cast<device const half*>(uma + p.wg_b_off) + ulong(row) * nblk;
        device const uint2* wu = reinterpret_cast<device const uint2*>(uma + p.wu_off) + ulong(row) * k16;
        device const half*  su = reinterpret_cast<device const half*>(uma + p.wu_s_off) + ulong(row) * nblk;
        device const half*  bu = reinterpret_cast<device const half*>(uma + p.wu_b_off) + ulong(row) * nblk;

        for (uint i = lane; i < k16; i += 32u) {
            const uint  grp = i >> 2u;
            half4 w0, w1, w2, w3;

            const uint2 qg  = wg[i];
            const float scg = float(sg[grp]);
            sik_unpack_u4_u32_h4(qg.x, w0, w1);
            sik_unpack_u4_u32_h4(qg.y, w2, w3);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < 8u; ++m) {
                device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + i * 16u);
                const half4 dh = fma(w0, xm[0], w1 * xm[1]) + fma(w2, xm[2], w3 * xm[3]);
                ga[m] = fma(scg, (float(dh.x) + float(dh.y)) + (float(dh.z) + float(dh.w)), ga[m]);
            }

            const uint2 qu  = wu[i];
            const float scu = float(su[grp]);
            sik_unpack_u4_u32_h4(qu.x, w0, w1);
            sik_unpack_u4_u32_h4(qu.y, w2, w3);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < 8u; ++m) {
                device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + i * 16u);
                const half4 dh = fma(w0, xm[0], w1 * xm[1]) + fma(w2, xm[2], w3 * xm[3]);
                ua[m] = fma(scu, (float(dh.x) + float(dh.y)) + (float(dh.z) + float(dh.w)), ua[m]);
            }

            if (do_bias) {
                const float big = float(bg[grp]);
                const float biu = float(bu[grp]);
                #pragma clang loop unroll(full)
                for (uint m = 0u; m < 8u; ++m) {
                    const float xs = x_sum[ulong(m) * nblk + grp];
                    ga[m] = fma(big, xs, ga[m]);
                    ua[m] = fma(biu, xs, ua[m]);
                }
            }
        }
    } else if (p.dtype == STRATA_INKLING2_DTYPE_Q8_0) {
        const uint k16  = p.cols >> 4u;
        const uint nblk = p.cols >> 6u;
        device const char4* wg = reinterpret_cast<device const char4*>(uma + p.wg_off + ulong(row) * p.cols);
        device const half*  sg = reinterpret_cast<device const half*>(uma + p.wg_s_off) + ulong(row) * nblk;
        device const char4* wu = reinterpret_cast<device const char4*>(uma + p.wu_off + ulong(row) * p.cols);
        device const half*  su = reinterpret_cast<device const half*>(uma + p.wu_s_off) + ulong(row) * nblk;

        for (uint i = lane; i < k16; i += 32u) {
            const uint   grp = i >> 2u;
            const float  scg = float(sg[grp]), scu = float(su[grp]);
            const float4 g0 = float4(wg[i*4u+0u]), g1 = float4(wg[i*4u+1u]), g2 = float4(wg[i*4u+2u]), g3 = float4(wg[i*4u+3u]);
            const float4 u0 = float4(wu[i*4u+0u]), u1 = float4(wu[i*4u+1u]), u2 = float4(wu[i*4u+2u]), u3 = float4(wu[i*4u+3u]);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < 8u; ++m) {
                device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + i * 16u);
                const float4 x0 = float4(xm[0]), x1 = float4(xm[1]), x2 = float4(xm[2]), x3 = float4(xm[3]);
                const float dg = (dot(g0, x0) + dot(g1, x1)) + (dot(g2, x2) + dot(g3, x3));
                const float du = (dot(u0, x0) + dot(u1, x1)) + (dot(u2, x2) + dot(u3, x3));
                ga[m] = fma(scg, dg, ga[m]);
                ua[m] = fma(scu, du, ua[m]);
            }
        }
    } else {
        const uint c8_cnt = p.cols >> 3u;
        device const half4* wg4 = reinterpret_cast<device const half4*>(uma + p.wg_off + ulong(row) * p.cols * 2ul);
        device const half4* wu4 = reinterpret_cast<device const half4*>(uma + p.wu_off + ulong(row) * p.cols * 2ul);
        for (uint c8 = lane; c8 < c8_cnt; c8 += 32u) {
            const float4 fg0 = float4(wg4[c8 * 2u + 0u]), fg1 = float4(wg4[c8 * 2u + 1u]);
            const float4 fu0 = float4(wu4[c8 * 2u + 0u]), fu1 = float4(wu4[c8 * 2u + 1u]);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < 8u; ++m) {
                device const half4* x4 = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + c8 * 8u);
                const float4 fx0 = float4(x4[0]), fx1 = float4(x4[1]);
                ga[m] += dot(fg0, fx0) + dot(fg1, fx1);
                ua[m] += dot(fu0, fx0) + dot(fu1, fx1);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 8u; ++m) {
        const float gv = simd_sum(ga[m]);
        const float uv = simd_sum(ua[m]);
        if (lane == 0u && m < p.M) mid_out[ulong(m) * p.rows + row] = sik_clamp_f16(sik_silu(gv) * uv);
    }
}

// Pure-Q4_A MTP Block-8 `gate_proj + up_proj + silu(g)*u` (32-lane 1-row, ZERO Q8/FP16 register bloat!)
kernel void sik_mtp_gate_up_silu_b8_q4a(
    device const uchar*               uma     [[buffer(0)]],
    device const half*                x_in    [[buffer(1)]],
    device half*                      mid_out [[buffer(2)]],
    constant SikMtpSwigluMixedParams& p       [[buffer(3)]],
    device const float*               x_sum   [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid * 4u + sgid;
    if (row >= p.rows) return;

    float ga[8], ua[8];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 8u; ++m) { ga[m] = 0.0f; ua[m] = 0.0f; }

    const uint k16     = p.cols >> 4u;
    const uint nblk    = p.cols >> 6u;
    const bool do_bias = ((lane & 3u) == 0u);
    device const uint2* wg = reinterpret_cast<device const uint2*>(uma + p.wg_off) + ulong(row) * k16;
    device const half*  sg = reinterpret_cast<device const half*>(uma + p.wg_s_off) + ulong(row) * nblk;
    device const half*  bg = reinterpret_cast<device const half*>(uma + p.wg_b_off) + ulong(row) * nblk;
    device const uint2* wu = reinterpret_cast<device const uint2*>(uma + p.wu_off) + ulong(row) * k16;
    device const half*  su = reinterpret_cast<device const half*>(uma + p.wu_s_off) + ulong(row) * nblk;
    device const half*  bu = reinterpret_cast<device const half*>(uma + p.wu_b_off) + ulong(row) * nblk;

    for (uint i = lane; i < k16; i += 32u) {
        const uint  grp = i >> 2u;
        half4 w0, w1, w2, w3;

        const uint2 qg  = wg[i];
        const float scg = float(sg[grp]);
        sik_unpack_u4_u32_h4(qg.x, w0, w1);
        sik_unpack_u4_u32_h4(qg.y, w2, w3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < 8u; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + i * 16u);
            const half4 dh = fma(w0, xm[0], w1 * xm[1]) + fma(w2, xm[2], w3 * xm[3]);
            ga[m] = fma(scg, (float(dh.x) + float(dh.y)) + (float(dh.z) + float(dh.w)), ga[m]);
        }

        const uint2 qu  = wu[i];
        const float scu = float(su[grp]);
        sik_unpack_u4_u32_h4(qu.x, w0, w1);
        sik_unpack_u4_u32_h4(qu.y, w2, w3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < 8u; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + i * 16u);
            const half4 dh = fma(w0, xm[0], w1 * xm[1]) + fma(w2, xm[2], w3 * xm[3]);
            ua[m] = fma(scu, (float(dh.x) + float(dh.y)) + (float(dh.z) + float(dh.w)), ua[m]);
        }

        if (do_bias) {
            const float big = float(bg[grp]);
            const float biu = float(bu[grp]);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < 8u; ++m) {
                const float xs = x_sum[ulong(m) * nblk + grp];
                ga[m] = fma(big, xs, ga[m]);
                ua[m] = fma(biu, xs, ua[m]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < 8u; ++m) {
        const float gv = simd_sum(ga[m]);
        const float uv = simd_sum(ua[m]);
        if (lane == 0u && m < p.M) mid_out[ulong(m) * p.rows + row] = sik_clamp_f16(sik_silu(gv) * uv);
    }
}

struct SikMtpRnnPrepParams {
    uint rank;
    uint D;
    uint step_k;
    uint vocab_pad;
};

kernel void sik_mtp_rnn_prep(
    device const float*           state_f32 [[buffer(0)]],
    device const half*            markov_w1 [[buffer(1)]],
    device const float*           norm_f32  [[buffer(2)]],
    device const float*           f_rnn     [[buffer(3)]],
    device const uint*            tok_buf   [[buffer(4)]],
    device half*                  rnn_in    [[buffer(5)]],
    constant SikMtpRnnPrepParams& p         [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint b1 = p.rank, b2 = b1 + p.rank, b3 = b2 + p.D, b4 = b3 + p.D;
    if (gid >= b4) return;
    if (gid < b1) {
        rnn_in[gid] = sik_clamp_f16(state_f32[gid]);
    } else if (gid < b2) {
        const uint prev = min(tok_buf[p.step_k], p.vocab_pad - 1u);
        rnn_in[gid] = markov_w1[ulong(prev) * p.rank + (gid - b1)];
    } else if (gid < b3) {
        rnn_in[gid] = sik_clamp_f16(norm_f32[ulong(p.step_k + 1u) * p.D + (gid - b2)]);
    } else {
        rnn_in[gid] = sik_clamp_f16(f_rnn[gid - b3]);
    }
}

kernel void sik_mtp_rnn_gate(
    device const float* raw_in     [[buffer(0)]],
    device const float* rnn_bias   [[buffer(1)]],
    device float*       state_io   [[buffer(2)]],
    device half*        latent_f16 [[buffer(3)]],
    constant uint&      rank       [[buffer(4)]],
    uint r [[thread_position_in_grid]]
) {
    if (r >= rank) return;
    const float g_raw = raw_in[r]             + rnn_bias[r];
    const float c_raw = raw_in[rank + r]      + rnn_bias[rank + r];
    const float o_raw = raw_in[2u * rank + r] + rnn_bias[2u * rank + r];
    const float gate  = sik_sigmoid(g_raw);
    state_io[r]       = gate * state_io[r] + (1.0f - gate) * sik_safe_tanh(c_raw);
    latent_f16[r]     = sik_clamp_f16(sik_safe_tanh(o_raw));
}

struct SikMtpFoldAddParams {
    uint  V;            // 200058 (full) or 32768 (FR-Spec SLC shortlist!)
    uint  rank;         // 256
    float softcap;      // 30.0f
    uint  compute_pmax; // 1 => 1-pass online safe-softmax LSE into stage_lse!
};

// Full-Vocab FP16 Folded Markov Head (`16` lanes/row, `8` rows/TG)
kernel void sik_mtp_fold_add_softcap(
    device const half*            fold_w      [[buffer(0)]],
    device const float*           fold_bias   [[buffer(1)]],
    device const float*           base_logits [[buffer(2)]],
    device const half*            latent_f16  [[buffer(3)]],
    device float*                 logits_out  [[buffer(4)]],
    constant SikMtpFoldAddParams& p           [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row    = (tgid * 4u + sgid) * 2u + (lane >> 4u);
    const uint lane16 = lane & 15u;
    if (row >= p.V) return;

    const uint c16_cnt = p.rank >> 4u;
    device const half4* w4 = reinterpret_cast<device const half4*>(fold_w + ulong(row) * p.rank);
    device const half4* x4 = reinterpret_cast<device const half4*>(latent_f16);

    float acc = 0.0f;
    for (uint c16 = lane16; c16 < c16_cnt; c16 += 16u) {
        const uint k4 = c16 * 4u;
        acc += sik_dot_f32_h4x4(w4[k4 + 0u], w4[k4 + 1u], w4[k4 + 2u], w4[k4 + 3u],
                                x4[k4 + 0u], x4[k4 + 1u], x4[k4 + 2u], x4[k4 + 3u]);
    }
    acc = sik_reduce16(acc);
    if (lane16 == 0u) {
        logits_out[row] = sik_softcap(base_logits[row] + acc + fold_bias[row], p.softcap);
    }
}

// 48 MiB SLC-Pinned 32K FR-Spec `Q8_0` Folded Markov Head (`8.7 MB` -> `uint4` 16B loads!)
// Grid: `((V + 7) / 8, 1, 1)`, `128` threads/TG (`4` SGs)
kernel void sik_mtp_fold_add_softcap_q8(
    device const char*            fold_w_q    [[buffer(0)]], // [V, 256] int8
    device const half*            fold_w_s    [[buffer(1)]], // [V, 4]   half scales (gs=64)
    device const float*           fold_bias   [[buffer(2)]], // [V]
    device const float*           base_logits [[buffer(3)]], // [V]
    device const half*            latent_f16  [[buffer(4)]], // [256]
    device float*                 logits_out  [[buffer(5)]], // [V]
    constant SikMtpFoldAddParams& p           [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row    = (tgid * 4u + sgid) * 2u + (lane >> 4u);
    const uint lane16 = lane & 15u;
    if (row >= p.V) return;

    const uint k16  = p.rank >> 4u; // 16 -> 1 exact wave!
    const uint nblk = p.rank >> 6u; // 4
    device const uint4* w16 = reinterpret_cast<device const uint4*>(fold_w_q + ulong(row) * p.rank);
    device const half*  sc  = fold_w_s + ulong(row) * nblk;
    device const half4* x4  = reinterpret_cast<device const half4*>(latent_f16);

    float acc = 0.0f;
    for (uint i = lane16; i < k16; i += 16u) {
        const float s  = float(sc[i >> 2u]);
        const uint4 q  = w16[i];
        const uint  k4 = i * 4u;
        const float d  = (dot(float4(as_type<char4>(q.x)), float4(x4[k4 + 0u])) +
                          dot(float4(as_type<char4>(q.y)), float4(x4[k4 + 1u]))) +
                         (dot(float4(as_type<char4>(q.z)), float4(x4[k4 + 2u])) +
                          dot(float4(as_type<char4>(q.w)), float4(x4[k4 + 3u])));
        acc = fma(s, d, acc);
    }
    acc = sik_reduce16(acc);
    if (lane16 == 0u) {
        logits_out[row] = sik_softcap(base_logits[row] + acc + fold_bias[row], p.softcap);
    }
}

// FUSED 1-PASS FR-Spec `Q8_0` Folded Markov Head + Softcap + 128-Block Greedy Argmax + Optional Online LSE (`p_max`)!
// Eliminates 7x `b_logits` DRAM roundtrips & 7x `sample_greedy_stage1` dispatches inside `cb_draft` for BOTH
// `want_pmax == false` (State B / Fixed-K) AND `want_pmax == true` (State C Prose/Probe)! (100% bit-exact argmax!)
// Grid: `(128, 1, 1)`, `128` threads/TG (`4` SGs = `8` rows/wave via 16 lanes/row)
kernel void sik_mtp_fold_sample_greedy_q8(
    device const char*            fold_w_q    [[buffer(0)]],
    device const half*            fold_w_s    [[buffer(1)]],
    device const float*           fold_bias   [[buffer(2)]],
    device const float*           base_logits [[buffer(3)]],
    device const half*            latent_f16  [[buffer(4)]],
    device float*                 stage_v     [[buffer(5)]], // [128]
    device uint*                  stage_i     [[buffer(6)]], // [128]
    device float*                 stage_lse   [[buffer(7)]], // [128] (written when p.compute_pmax != 0)
    constant SikMtpFoldAddParams& p           [[buffer(8)]],
    uint blk  [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (blk >= 128u) return;
    const uint lane16 = lane & 15u;
    const uint row_in = sgid * 2u + (lane >> 4u);
    const uint k16    = p.rank >> 4u;
    const uint nblk   = p.rank >> 6u;
    const uint chunk  = max(1u, (p.V + 127u) >> 7u);
    const uint beg    = blk * chunk;
    const uint end    = min(p.V, beg + chunk);

    device const half4* x4 = reinterpret_cast<device const half4*>(latent_f16);
    float4 fx0 = float4(0.0f), fx1 = float4(0.0f), fx2 = float4(0.0f), fx3 = float4(0.0f);
    if (k16 == 16u) {
        const uint k4 = lane16 * 4u;
        fx0 = float4(x4[k4 + 0u]); fx1 = float4(x4[k4 + 1u]);
        fx2 = float4(x4[k4 + 2u]); fx3 = float4(x4[k4 + 3u]);
    }

    float best_v = -INFINITY;
    uint  best_i = 0xFFFFFFFFu;
    float sum_e  = 0.0f;
    const bool do_pmax = (p.compute_pmax != 0u);
    for (uint r0 = beg; r0 < end; r0 += 8u) {
        const uint row = r0 + row_in;
        if (row < end) {
            device const uint4* w16 = reinterpret_cast<device const uint4*>(fold_w_q + ulong(row) * p.rank);
            device const half*  sc  = fold_w_s + ulong(row) * nblk;
            float acc = 0.0f;
            if (k16 == 16u) {
                const float s = float(sc[lane16 >> 2u]);
                const uint4 q = w16[lane16];
                const float d = (dot(float4(as_type<char4>(q.x)), fx0) + dot(float4(as_type<char4>(q.y)), fx1))
                              + (dot(float4(as_type<char4>(q.z)), fx2) + dot(float4(as_type<char4>(q.w)), fx3));
                acc = fma(s, d, 0.0f);
            } else {
                for (uint i = lane16; i < k16; i += 16u) {
                    const float s  = float(sc[i >> 2u]);
                    const uint4 q  = w16[i];
                    const uint  k4 = i * 4u;
                    const float d  = (dot(float4(as_type<char4>(q.x)), float4(x4[k4 + 0u])) +
                                      dot(float4(as_type<char4>(q.y)), float4(x4[k4 + 1u]))) +
                                     (dot(float4(as_type<char4>(q.z)), float4(x4[k4 + 2u])) +
                                      dot(float4(as_type<char4>(q.w)), float4(x4[k4 + 3u])));
                    acc = fma(s, d, acc);
                }
            }
            acc = sik_reduce16(acc);
            if (lane16 == 0u) {
                const float v = sik_softcap(base_logits[row] + acc + fold_bias[row], p.softcap);
                if (do_pmax) {
                    if (v > best_v) {
                        sum_e  = (best_v > -INFINITY) ? fma(sum_e, exp(best_v - v), 1.0f) : 1.0f;
                        best_v = v;
                        best_i = row;
                    } else {
                        sum_e += exp(v - best_v);
                        if (v == best_v && row < best_i) best_i = row;
                    }
                } else {
                    if (v > best_v || (v == best_v && row < best_i)) { best_v = v; best_i = row; }
                }
            }
        }
    }

    const float my_max = best_v;
    #pragma clang loop unroll(full)
    for (ushort off = 16u; off > 0u; off >>= 1u) {
        const float ov = simd_shuffle_xor(best_v, off);
        const uint  oi = simd_shuffle_xor(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    threadgroup float tg_v[4];
    threadgroup uint  tg_i[4];
    threadgroup float tg_bmax[1];
    if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v   = (lane < 4u) ? tg_v[lane] : -INFINITY;
        uint  idx = (lane < 4u) ? tg_i[lane] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort off = 2u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(v, off);
            const uint  oi = simd_shuffle_xor(idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0u) {
            tg_bmax[0]   = v;
            stage_v[blk] = v;
            stage_i[blk] = idx;
        }
    }
    if (do_pmax) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float bmax = tg_bmax[0];
        const float my_e = (lane16 == 0u && my_max > -INFINITY) ? (sum_e * exp(my_max - bmax)) : 0.0f;
        const float s_e  = simd_sum(my_e);
        if (lane == 0u) tg_v[sgid] = s_e;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float tot_e = simd_sum((lane < 4u) ? tg_v[lane] : 0.0f);
            if (lane == 0u) stage_lse[blk] = tot_e;
        }
    }
}

/* ============================================================================
 * 9. 100% Bit-Exact Sliding-Window Logit Penalties (Full & FR-Spec Shortlist) +
 *    2-Stage Parallel Vocab Sampler w/ `p_max` (Bayes Adaptive-K) & `token_map`
 * ============================================================================ */

struct SikLogitPenaltyParams {
    uint  M;
    uint  vocab_size;         // full vocab (200058)
    uint  logits_stride;      // 200058 (full) or 32768 (FR-Spec shortlist)
    uint  num_hist_ids;
    uint  extra_mode;         // 0 => cb_ver; 1 => cb_draft
    uint  extra_count;
    float presence_penalty;
    float frequency_penalty;
    float repetition_penalty;
    uint  use_shortlist;      // 1 => map full tok `t` -> `full_to_short[t]` before indexing `logits`!
    uint  pad1;
    uint  pad2;
    uint  drop_toks[8];
};

inline float sik_apply_pen_val(float v, uint c, float pres, float freq, float rep) {
    if (c == 0u) return v;
    if (rep > 0.0f && rep != 1.0f) v = (v > 0.0f) ? (v / rep) : (v * rep);
    return v - (pres + freq * float(c));
}

kernel void sik_apply_logit_penalties(
    device float*                   logits        [[buffer(0)]],
    device const uint*              pen_ids       [[buffer(1)]],
    device const ushort*            pen_counts    [[buffer(2)]],
    device const uint*              extra_toks    [[buffer(3)]],
    device const uint*              full_to_short [[buffer(4)]], // [vocab_size] -> short_idx or 0xFFFFFFFF
    constant SikLogitPenaltyParams& p             [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_items = p.num_hist_ids + p.extra_count;
    if (gid >= total_items) return;

    uint t = 0u, base_c = 0u;
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

    const uint logit_col = (p.use_shortlist != 0u) ? full_to_short[t] : t;
    if (logit_col >= p.logits_stride) return;

    if (p.extra_mode == 1u) {
        uint c = base_c;
        for (uint j = 1u; j <= p.extra_count; ++j) {
            if (extra_toks[j] == t) ++c;
            if ((j - 1u) < 8u && p.drop_toks[j - 1u] == t && c > 0u) --c;
        }
        if (c > 0u) {
            logits[logit_col] = sik_apply_pen_val(logits[logit_col], c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
        }
    } else {
        uint c = base_c;
        for (uint m = 0u; m < p.M; ++m) {
            if (m > 0u && m <= p.extra_count) {
                if (extra_toks[m] == t) ++c;
                if ((m - 1u) < 8u && p.drop_toks[m - 1u] == t && c > 0u) --c;
            }
            if (c > 0u) {
                const ulong idx = ulong(m) * p.logits_stride + logit_col;
                logits[idx] = sik_apply_pen_val(logits[idx], c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
            }
        }
    }
}

struct SikSampleParams {
    uint  M;
    uint  vocab_size;    // 200058 (full) or 32768 (FR-Spec shortlist)
    uint  greedy;        // 1 => argmax, 0 => top-k/top-p stochastic
    float inv_temp;
    uint  top_k;
    float top_p;
    uint  rng_lo;
    uint  rng_hi;
    uint  use_token_map; // 1 => map sampled short_idx -> `token_map[short_idx]`
    uint  compute_pmax;  // 1 => output softmax prob of drafted token into `out_pmax[m]`
    uint  pad0;
    uint  pad1;
};

inline float sik_rand_u01(uint lo, uint hi, uint salt) {
    ulong z = ((ulong(hi) << 32ul) | ulong(lo)) + ulong(salt) * 0x9E3779B97F4A7C15ul;
    z += 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30ul)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27ul)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31ul);
    const uint u24 = uint(z >> 40ul) & 0xFFFFFFu;
    return (float(u24) + 0.5f) * (1.0f / 16777216.0f);
}

// Ultra-Lean 128-TG x 128-th Vectorized Greedy Vocab Argmax + Optional Exact Block-LSE (`p_max`)!
// Zero `rv[16]/ri[16]` non-greedy register bloat -> `6 us` @ M=1, `16 us` @ M=8 (14x-16x faster!), 100% bit-exact!
kernel void sik_sample_greedy_stage1(
    device const float*       logits    [[buffer(0)]],
    device float*             stage_v   [[buffer(1)]],
    device uint*              stage_i   [[buffer(2)]],
    device float*             stage_lse [[buffer(3)]], // [M, 128]
    constant SikSampleParams& p         [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint blk = tgid.x;
    const uint m   = tgid.y;
    if (blk >= 128u || m >= p.M) return;

    device const float* row = logits + ulong(m) * p.vocab_size;
    float best_v = -INFINITY;
    uint  best_i = 0xFFFFFFFFu;

    const bool even_v = ((p.vocab_size & 1u) == 0u);
    const uint V2     = p.vocab_size >> 1u;
    const uint ch2    = max(1u, (V2 + 127u) >> 7u);
    const uint beg2   = blk * ch2;
    const uint end2   = min(V2, beg2 + ch2);
    const uint ch1    = max(1u, (p.vocab_size + 127u) >> 7u);
    const uint beg1   = blk * ch1;
    const uint end1   = min(p.vocab_size, beg1 + ch1);

    if (even_v) {
        device const float2* row2 = reinterpret_cast<device const float2*>(row);
        for (uint i2 = beg2 + tid; i2 < end2; i2 += 128u) {
            const float2 v = row2[i2];
            const uint   b = i2 << 1u;
            if (v.x > best_v || (v.x == best_v && b < best_i))      { best_v = v.x; best_i = b; }
            if (v.y > best_v || (v.y == best_v && b + 1u < best_i)) { best_v = v.y; best_i = b + 1u; }
        }
    } else {
        for (uint i = beg1 + tid; i < end1; i += 128u) {
            const float v = row[i];
            if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
        }
    }

    #pragma clang loop unroll(full)
    for (ushort off = 16u; off > 0u; off >>= 1u) {
        const float ov = simd_shuffle_xor(best_v, off);
        const uint  oi = simd_shuffle_xor(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    threadgroup float tg_v[4];
    threadgroup uint  tg_i[4];
    threadgroup float tg_bmax[1];
    if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v   = (lane < 4u) ? tg_v[lane] : -INFINITY;
        uint  idx = (lane < 4u) ? tg_i[lane] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort off = 2u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(v, off);
            const uint  oi = simd_shuffle_xor(idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0u) {
            tg_bmax[0] = v;
            stage_v[ulong(m) * 128ul + blk] = v;
            stage_i[ulong(m) * 128ul + blk] = idx;
        }
    }
    if (p.compute_pmax != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float bmax = tg_bmax[0];
        float se = 0.0f;
        if (even_v) {
            device const float2* row2 = reinterpret_cast<device const float2*>(row);
            for (uint i2 = beg2 + tid; i2 < end2; i2 += 128u) {
                const float2 v = row2[i2];
                se += exp(v.x - bmax) + exp(v.y - bmax);
            }
        } else {
            for (uint i = beg1 + tid; i < end1; i += 128u) {
                se += exp(row[i] - bmax);
            }
        }
        const float s_se = simd_sum(se);
        if (lane == 0u) tg_v[sgid] = s_se;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float tot_se = simd_sum((lane < 4u) ? tg_v[lane] : 0.0f);
            if (lane == 0u) stage_lse[ulong(m) * 128ul + blk] = tot_se;
        }
    }
}

kernel void sik_sample_greedy_stage2(
    device const float*       stage_v   [[buffer(0)]],
    device const uint*        stage_i   [[buffer(1)]],
    device const float*       stage_lse [[buffer(2)]],
    device const uint*        token_map [[buffer(3)]],
    device uint*              out_tok   [[buffer(4)]],
    device float*             out_pmax  [[buffer(5)]],
    constant SikSampleParams& p         [[buffer(6)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M || tid >= 128u) return;
    const float my_v = stage_v[ulong(m) * 128ul + tid];
    const uint  my_i = stage_i[ulong(m) * 128ul + tid];
    float best_v = my_v;
    uint  best_i = my_i;
    #pragma clang loop unroll(full)
    for (ushort off = 16u; off > 0u; off >>= 1u) {
        const float ov = simd_shuffle_xor(best_v, off);
        const uint  oi = simd_shuffle_xor(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    threadgroup float tg_v[4];
    threadgroup uint  tg_i[4];
    threadgroup float tg_gmax[1];
    if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v   = (lane < 4u) ? tg_v[lane] : -INFINITY;
        uint  idx = (lane < 4u) ? tg_i[lane] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort off = 2u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(v, off);
            const uint  oi = simd_shuffle_xor(idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0u) {
            tg_gmax[0] = v;
            const uint raw_id = (idx < p.vocab_size) ? idx : 0u;
            out_tok[m] = (p.use_token_map != 0u) ? token_map[raw_id] : raw_id;
        }
    }
    if (p.compute_pmax != 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float gmax    = tg_gmax[0];
        const float contrib = (my_v > -INFINITY) ? (stage_lse[ulong(m) * 128ul + tid] * exp(my_v - gmax)) : 0.0f;
        const float s_c     = simd_sum(contrib);
        if (lane == 0u) tg_v[sgid] = s_c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float tot = simd_sum((lane < 4u) ? tg_v[lane] : 0.0f);
            if (lane == 0u && out_pmax) out_pmax[m] = (tot > 0.0f) ? clamp(1.0f / tot, 0.0f, 1.0f) : 1.0f;
        }
    }
}

// 2-Stage Parallel Vocab Sampler w/ exact block-LSE for greedy `p_max`!
kernel void sik_sample_vocab_stage1(
    device const float*       logits    [[buffer(0)]],
    device float*             stage_v   [[buffer(1)]],
    device uint*              stage_i   [[buffer(2)]],
    device float*             stage_lse [[buffer(3)]], // [M, 64] block sum_exp(v - block_max)
    constant SikSampleParams& p         [[buffer(4)]],
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
    threadgroup float tg_bmax;

    if (p.greedy != 0u) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;
        for (uint i = beg + tid; i < end; i += 256u) {
            const float v = row[i];
            if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_bmax = v;
                stage_v[ulong(m) * 64ul + blk] = v;
                stage_i[ulong(m) * 64ul + blk] = idx;
            }
        }
        if (p.compute_pmax != 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float bmax = tg_bmax;
            float se = 0.0f;
            for (uint i = beg + tid; i < end; i += 256u) {
                se += exp(row[i] - bmax);
            }
            const float s_se = simd_sum(se);
            if (lane == 0u) tg_v[sgid] = s_se;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                const float tot_se = simd_sum((lane < 8u) ? tg_v[lane] : 0.0f);
                if (lane == 0u) stage_lse[ulong(m) * 64ul + blk] = tot_se;
            }
        }
        return;
    }

    const uint  K        = min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
    const ulong base_out = (ulong(m) * 64ul + blk) * K;

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
            const float v = rv[j]; const uint idx = ri[j];
            if (v > best_v || (v == best_v && idx < best_i)) { best_v = v; best_i = idx; best_j = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_i[0]               = idx;
                stage_v[base_out + k] = v;
                stage_i[base_out + k] = idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint win_i = tg_i[0];
        if (best_j < 16u && ri[best_j] == win_i) {
            rv[best_j] = -INFINITY;
            ri[best_j] = 0xFFFFFFFFu;
        }
        if (k + 1u < K) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void sik_sample_vocab_stage2(
    device const float*       stage_v       [[buffer(0)]],
    device const uint*        stage_i       [[buffer(1)]],
    device const float*       stage_lse     [[buffer(2)]],
    device const uint*        token_map     [[buffer(3)]], // [vocab_size] -> full vocab ID (when use_token_map != 0)
    device uint*              out_tok       [[buffer(4)]],
    device float*             out_pmax      [[buffer(5)]], // optional [M]
    device uint*              out_top_ids   [[buffer(6)]], // optional [M, 64]
    device float*             out_top_probs [[buffer(7)]], // optional [M, 64]
    device uint*              out_keep_k    [[buffer(8)]], // optional [M]
    constant SikSampleParams& p             [[buffer(9)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.M) return;

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    threadgroup float tg_gmax;

    if (p.greedy != 0u) {
        const float my_v = (tid < 64u) ? stage_v[ulong(m) * 64ul + tid] : -INFINITY;
        const uint  my_i = (tid < 64u) ? stage_i[ulong(m) * 64ul + tid] : 0xFFFFFFFFu;
        float best_v = my_v;
        uint  best_i = my_i;
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_gmax = v;
                const uint raw_id = (idx < p.vocab_size) ? idx : 0u;
                out_tok[m] = (p.use_token_map != 0u) ? token_map[raw_id] : raw_id;
            }
        }
        if (p.compute_pmax != 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float gmax = tg_gmax;
            float contrib = 0.0f;
            if (tid < 64u && my_v > -INFINITY) {
                contrib = stage_lse[ulong(m) * 64ul + tid] * exp(my_v - gmax);
            }
            const float s_c = simd_sum(contrib);
            if (lane == 0u) tg_v[sgid] = s_c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                const float tot = simd_sum((lane < 8u) ? tg_v[lane] : 0.0f);
                if (lane == 0u && out_pmax) out_pmax[m] = (tot > 0.0f) ? clamp(1.0f / tot, 0.0f, 1.0f) : 1.0f;
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
            const float v = rv[j]; const uint idx = ri[j];
            if (v > best_v || (v == best_v && idx < best_i)) { best_v = v; best_i = idx; best_j = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_i[0]     = idx;
                top_vals[k] = v;
                top_ids[k]  = (idx < p.vocab_size) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint win_i = tg_i[0];
        if (best_j < 16u && ri[best_j] == win_i) {
            rv[best_j] = -INFINITY;
            ri[best_j] = 0xFFFFFFFFu;
        }
        if (k + 1u < K) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float max_v = top_vals[0];
        float probs[64];
        float sum_p = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            const float w = exp((top_vals[k] - max_v) * p.inv_temp);
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
            const float norm_p = (k < keep_k) ? (probs[k] * inv_kept) : 0.0f;
            const uint  mapped = (k < K) ? ((p.use_token_map != 0u) ? token_map[top_ids[k]] : top_ids[k]) : 0u;
            if (out_top_ids)   out_top_ids[ulong(m) * 64ul + k]   = mapped;
            if (out_top_probs) out_top_probs[ulong(m) * 64ul + k] = norm_p;
        }
        if (out_keep_k) out_keep_k[m] = keep_k;

        const float u        = sik_rand_u01(p.rng_lo, p.rng_hi, m);
        float       cdf      = 0.0f;
        uint        chosen_k = 0u;
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += probs[k] * inv_kept;
            if (u <= cdf || k + 1u == keep_k) {
                chosen_k = k;
                break;
            }
        }
        const uint raw_chosen = top_ids[chosen_k];
        out_tok[m] = (p.use_token_map != 0u) ? token_map[raw_chosen] : raw_chosen;
        if (p.compute_pmax != 0u && out_pmax) {
            out_pmax[m] = clamp(probs[chosen_k] * inv_kept, 0.0f, 1.0f);
        }
    }
}

/* ============================================================================
 * 10. 2-Stage Bitplane Exact Q8_0 LM Head (`vhead_bitplane`, 100% Bit-Exact!)
 * ============================================================================ */

#define SIK_BP_MAX_CAND    32768u
#define SIK_BP_STAGE1_BLKS 512u

struct SikBpMeta {
    atomic_uint cand_count;
    atomic_uint overflow;
    float       T_raw;
    float       T_pen;
    // MTLDispatchThreadgroupsIndirectArguments (3 x uint32 + 1 pad = 16 B each)
    uint ind_stage2[4];
    uint ind_cand_s1[4];
    uint ind_cand_s2[4];
    uint ind_fb_gemv[4];
    uint ind_fb_pen[4];
    uint ind_fb_s1[4];
    uint ind_fb_s2[4];
};

struct SikBpParams {
    uint  V;
    uint  D;
    float softcap;
    uint  top_k;        // 1 => greedy; >1 => stochastic top-k (<=64)
    uint  num_seeds;    // Stage 1.5 exact seeds (<=64)
    uint  has_pen;
    float pres_pen;
    float freq_pen;
    float rep_pen;
    uint  pen_tgs;      // fallback penalty TGs
    uint  is_greedy;
    float inv_temp;
    float top_p;
    uint  rng_lo;
    uint  rng_hi;
    uint  pad0;
};

// One-time GPU repack of `Q8_0 (gs=64)` unembed -> Plane 0 (`hi4` + grouped-4 exact L2 residual norms)!
// Grid: `((V + 3) / 4, 1, 1)`, `128` threads/TG (`4` SGs = `4` rows/TG, `32` lanes/row)
kernel void sik_bp_repack_q8_to_hi4(
    device const char*       w_q       [[buffer(0)]], // [V, D] int8
    device const half*       w_s       [[buffer(1)]], // [V, D/64] half
    device uint2*            hi4_out   [[buffer(2)]], // [V, D/16] uint2 (16 4-bit unsigned codes `uhi = (w+128)>>4`)
    device half4*            rnorm4_o  [[buffer(3)]], // [V] half4 exact grouped-4 L2 norms of `s * (ulo - 7.5)`
    constant SikBpParams&    p         [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tgid * 4u + sgid;
    if (row >= p.V) return;

    const uint k16    = p.D >> 4u;
    const uint nblk   = p.D >> 6u;
    const uint q_size = max(1u, nblk >> 2u);
    device const uint4* rw = reinterpret_cast<device const uint4*>(w_q + ulong(row) * p.D);
    device const half*  rs = w_s + ulong(row) * nblk;
    device uint2*       dh = hi4_out + ulong(row) * k16;

    float rsq[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = lane; i < k16; i += 32u) {
        const uint  grp = i >> 2u;
        const uint  qtr = min(3u, grp / q_size);
        const float s   = float(rs[grp]);
        const uint4 q   = rw[i];
        const char4 c0  = as_type<char4>(q.x);
        const char4 c1  = as_type<char4>(q.y);
        const char4 c2  = as_type<char4>(q.z);
        const char4 c3  = as_type<char4>(q.w);

        const uint4 u0 = uint4(int4(c0) + 128);
        const uint4 u1 = uint4(int4(c1) + 128);
        const uint4 u2 = uint4(int4(c2) + 128);
        const uint4 u3 = uint4(int4(c3) + 128);

        const uint4 h0 = u0 >> 4u, l0 = u0 & 15u;
        const uint4 h1 = u1 >> 4u, l1 = u1 & 15u;
        const uint4 h2 = u2 >> 4u, l2 = u2 & 15u;
        const uint4 h3 = u3 >> 4u, l3 = u3 & 15u;

        const uint p0 = h0.x | (h0.y << 4u) | (h0.z << 8u)  | (h0.w << 12u)
                      | (h1.x << 16u) | (h1.y << 20u) | (h1.z << 24u) | (h1.w << 28u);
        const uint p1 = h2.x | (h2.y << 4u) | (h2.z << 8u)  | (h2.w << 12u)
                      | (h3.x << 16u) | (h3.y << 20u) | (h3.z << 24u) | (h3.w << 28u);
        dh[i] = uint2(p0, p1);

        const float4 r0 = float4(l0) - 7.5f;
        const float4 r1 = float4(l1) - 7.5f;
        const float4 r2 = float4(l2) - 7.5f;
        const float4 r3 = float4(l3) - 7.5f;
        const float  ss = (dot(r0, r0) + dot(r1, r1)) + (dot(r2, r2) + dot(r3, r3));
        rsq[qtr] = fma(s * s, ss, rsq[qtr]);
    }

    const float t0 = simd_sum(rsq[0]);
    const float t1 = simd_sum(rsq[1]);
    const float t2 = simd_sum(rsq[2]);
    const float t3 = simd_sum(rsq[3]);
    if (lane == 0u) {
        // Multiply by 1.004f + 1e-4f before half cast so float(half(rn)) is ALWAYS >= exact L2 norm!
        const half rn0 = sik_clamp_f16(sqrt(t0) * 1.004f + 1e-4f);
        const half rn1 = sik_clamp_f16(sqrt(t1) * 1.004f + 1e-4f);
        const half rn2 = sik_clamp_f16(sqrt(t2) * 1.004f + 1e-4f);
        const half rn3 = sik_clamp_f16(sqrt(t3) * 1.004f + 1e-4f);
        rnorm4_o[row] = half4(rn0, rn1, rn2, rn3);
    }
}

// Prep `x` quarter L2 norms + `-120.5 * x_sum[g]` + reset `SikBpMeta` counters (`Grid(1,1,1)`, `256` th)
kernel void sik_bp_prep_x(
    device const half*    x_in   [[buffer(0)]],
    device float*         xprep  [[buffer(1)]], // [0..nblk-1]: -120.5*sum_g; [128..131]: ||x_q||_2
    device SikBpMeta*     meta   [[buffer(2)]],
    constant SikBpParams& p      [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint c4_cnt = p.D >> 2u;
    const uint nblk   = p.D >> 6u;
    const uint q_size = max(1u, nblk >> 2u);
    device const half4* x4 = reinterpret_cast<device const half4*>(x_in);

    threadgroup float tg_qsq[4][8];
    float qsq[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = tid; i < c4_cnt; i += 256u) {
        const float4 fx = float4(x4[i]);
        const float  bs = sik_reduce16((fx.x + fx.y) + (fx.z + fx.w));
        const uint   g  = i >> 4u;
        if ((lane & 15u) == 0u && g < nblk) {
            xprep[g] = -120.5f * bs;
        }
        const uint qtr = min(3u, g / q_size);
        qsq[qtr] += dot(fx, fx);
    }

    #pragma clang loop unroll(full)
    for (uint q = 0u; q < 4u; ++q) {
        const float s = simd_sum(qsq[q]);
        if (lane == 0u) tg_qsq[q][sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u && lane < 32u) {
        #pragma clang loop unroll(full)
        for (uint q = 0u; q < 4u; ++q) {
            const float tot = simd_sum((lane < 8u) ? tg_qsq[q][lane] : 0.0f);
            if (lane == 0u) xprep[128u + q] = sqrt(tot);
        }
    }
    if (tid == 0u) {
        atomic_store_explicit(&meta->cand_count, 0u, memory_order_relaxed);
        atomic_store_explicit(&meta->overflow,   0u, memory_order_relaxed);
        meta->T_raw = -INFINITY;
        meta->T_pen = -INFINITY;
    }
}

// Stage 1: Reads ONLY Plane 0 (`hi4` + `w_s` + `rnorm4` = 51.9% of Q8_0!), 16-lane `uint4` (256B/half-SG
// coalesced loads, 64B TG mem -> 100% occupancy, L1-cached `x_in`), emits `ub[v]` + 512 block seeds!
// Grid: `(512, 1, 1)`, `128` threads/TG (`4` SGs = `8` rows/wave via 16 lanes/row)
kernel void sik_bp_stage1_hi4(
    device const uint4*   hi4_in   [[buffer(0)]], // [V, D/32] uint4 (32 4-bit unsigned codes per uint4!)
    device const half*    w_s      [[buffer(1)]], // [V, D/64]
    device const half4*   rnorm4   [[buffer(2)]], // [V]
    device const half*    x_in     [[buffer(3)]], // [D]
    device const float*   xprep    [[buffer(4)]], // [0..nblk-1], [128..131]
    device float*         ub_out   [[buffer(5)]], // [V]
    device float*         stage_v  [[buffer(6)]], // [512]
    device uint*          stage_i  [[buffer(7)]], // [512]
    constant SikBpParams& p        [[buffer(8)]],
    uint blk  [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (blk >= SIK_BP_STAGE1_BLKS) return;

    const uint k32    = p.D >> 5u;
    const uint nblk   = p.D >> 6u;
    const uint lane16 = lane & 15u;
    const uint row_in = sgid * 2u + (lane >> 4u);

    device const half4* x4_in = reinterpret_cast<device const half4*>(x_in);
    const float4 x_qnorm(xprep[128u], xprep[129u], xprep[130u], xprep[131u]);

    const uint chunk = (p.V + SIK_BP_STAGE1_BLKS - 1u) / SIK_BP_STAGE1_BLKS;
    const uint beg   = blk * chunk;
    const uint end   = min(p.V, beg + chunk);
    const bool do_bias = ((lane16 & 1u) == 0u);

    float best_v = -INFINITY;
    uint  best_i = 0xFFFFFFFFu;

    for (uint r0 = beg; r0 < end; r0 += 8u) {
        const uint row = r0 + row_in;
        if (row < end) {
            device const uint4* rw = hi4_in + ulong(row) * k32;
            device const half*  rs = w_s    + ulong(row) * nblk;
            float acc = 0.0f;
            for (uint i = lane16; i < k32; i += 16u) {
                const uint4 q   = rw[i];
                const uint  grp = i >> 1u;
                const float sc  = float(rs[grp]);
                half4 n0, n1;
                const uint k4 = i * 8u;
                sik_unpack_u4_u32_h4(q.x, n0, n1);
                float d = dot(float4(n0), float4(x4_in[k4 + 0u])) + dot(float4(n1), float4(x4_in[k4 + 1u]));
                sik_unpack_u4_u32_h4(q.y, n0, n1);
                d += dot(float4(n0), float4(x4_in[k4 + 2u])) + dot(float4(n1), float4(x4_in[k4 + 3u]));
                sik_unpack_u4_u32_h4(q.z, n0, n1);
                d += dot(float4(n0), float4(x4_in[k4 + 4u])) + dot(float4(n1), float4(x4_in[k4 + 5u]));
                sik_unpack_u4_u32_h4(q.w, n0, n1);
                d += dot(float4(n0), float4(x4_in[k4 + 6u])) + dot(float4(n1), float4(x4_in[k4 + 7u]));
                const float dg  = fma(16.0f, d, do_bias ? xprep[grp] : 0.0f);
                acc = fma(sc, dg, acc);
            }
            const float est = sik_reduce16(acc);
            if (lane16 == 0u) {
                const float err = dot(float4(rnorm4[row]), x_qnorm) * 1.003f + 0.50f;
                ub_out[row] = est + err;
                if (est > best_v || (est == best_v && row < best_i)) {
                    best_v = est;
                    best_i = row;
                }
            }
        }
    }

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    if (lane16 == 0u) { tg_v[row_in] = best_v; tg_i[row_in] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
        uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort off = 4u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(v, off);
            const uint  oi = simd_shuffle_xor(idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0u) {
            stage_v[blk] = v;
            stage_i[blk] = idx;
        }
    }
}

// Stage 1.5: Evaluates exact `Q8_0` dot product on top `num_seeds` (<=64) rows from Stage 1's 512 block winners
// to produce a 100% mathematically sound tight threshold `T_raw`/`T_pen` (halves pruning radius from 2*err to 1*err!).
// Grid: `(1, 1, 1)`, `256` threads/TG (`8` SGs = `16` half-SGs)
kernel void sik_bp_stage15_seed(
    device const float*   stage_v    [[buffer(0)]], // [512]
    device const uint*    stage_i    [[buffer(1)]], // [512]
    device const char*    w_q        [[buffer(2)]], // [V, D]
    device const half*    w_s        [[buffer(3)]], // [V, D/64]
    device const half*    x_in       [[buffer(4)]], // [D]
    device const ushort*  pen_counts [[buffer(5)]], // [V] (used iff has_pen != 0)
    device SikBpMeta*     meta       [[buffer(6)]],
    constant SikBpParams& p          [[buffer(7)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    threadgroup uint  tg_seeds[64];
    threadgroup float tg_raw[64];
    threadgroup float tg_pen[64];

    // 512 items / 256 threads = 2 items/thread
    float rv0 = stage_v[tid],        rv1 = stage_v[tid + 256u];
    uint  ri0 = stage_i[tid],        ri1 = stage_i[tid + 256u];
    const uint n_seeds = min(64u, max(1u, p.num_seeds));

    for (uint s = 0u; s < n_seeds; ++s) {
        float bv = rv0; uint bi = ri0; uint bs = 0u;
        if (rv1 > bv || (rv1 == bv && ri1 < bi)) { bv = rv1; bi = ri1; bs = 1u; }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(bv, off);
            const uint  oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = bv; tg_i[sgid] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_i[0]     = idx;
                tg_seeds[s] = idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint win = tg_i[0];
        if (bs == 0u && ri0 == win) { rv0 = -INFINITY; ri0 = 0xFFFFFFFFu; }
        else if (bs == 1u && ri1 == win) { rv1 = -INFINITY; ri1 = 0xFFFFFFFFu; }
        if (s + 1u < n_seeds) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Evaluate EXACT Q8_0 dot product (identical accumulation order to `sik_q8_gemv_l16_fixed`) for all valid seeds!
    const uint hsg    = sgid * 2u + (lane >> 4u); // 0..15
    const uint lane16 = lane & 15u;
    const uint k16    = p.D >> 4u;
    const uint nblk   = p.D >> 6u;

    for (uint s = hsg; s < n_seeds; s += 16u) {
        const uint row = tg_seeds[s];
        if (row < p.V) {
            device const uint4* w16 = reinterpret_cast<device const uint4*>(w_q + ulong(row) * p.D);
            device const half*  sc  = w_s + ulong(row) * nblk;
            float acc = 0.0f;
            for (uint i = lane16; i < k16; i += 16u) {
                const float  sv = float(sc[i >> 2u]);
                const uint4  q  = w16[i];
                const float4 f0 = float4(as_type<char4>(q.x)), f1 = float4(as_type<char4>(q.y));
                const float4 f2 = float4(as_type<char4>(q.z)), f3 = float4(as_type<char4>(q.w));
                device const half4* xm = reinterpret_cast<device const half4*>(x_in + i * 16u);
                const float d = (dot(f0, float4(xm[0])) + dot(f1, float4(xm[1])))
                              + (dot(f2, float4(xm[2])) + dot(f3, float4(xm[3])));
                acc = fma(sv, d, acc);
            }
            acc = sik_reduce16(acc);
            if (lane16 == 0u) {
                tg_raw[s] = acc;
                float val = sik_softcap(acc, p.softcap);
                if (p.has_pen != 0u) {
                    const uint c = uint(pen_counts[row]);
                    val = sik_apply_pen_val(val, c, p.pres_pen, p.freq_pen, p.rep_pen);
                }
                tg_pen[s] = val;
            }
        } else if (lane16 == 0u) {
            tg_raw[s] = -INFINITY;
            tg_pen[s] = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint valid_cnt = 0u;
        for (uint s = 0u; s < n_seeds; ++s) {
            if (tg_seeds[s] < p.V) ++valid_cnt;
        }
        const uint K_need = max(1u, p.top_k);
        if (valid_cnt < K_need) {
            meta->T_raw = -1e30f;
            meta->T_pen = -1e30f;
        } else if (K_need == 1u) {
            float best_p = -INFINITY;
            float best_r = -INFINITY;
            for (uint s = 0u; s < n_seeds; ++s) {
                if (tg_seeds[s] < p.V && tg_pen[s] > best_p) {
                    best_p = tg_pen[s];
                    best_r = tg_raw[s];
                }
            }
            // Clamp raw threshold below softcap saturation (~250 for cap=30) so saturated ties stay exact!
            const float raw_cap = (p.softcap > 0.0f) ? (p.softcap * 8.0f) : 1e30f;
            meta->T_raw = min(best_r, raw_cap) - 1e-3f;
            meta->T_pen = best_p - 1e-4f;
        } else {
            // Find the exact K_need-th largest score among ALL valid seeds (100% sound lower bound on true K_need-th!)
            float sorted_p[64];
            float sorted_r[64];
            uint  cnt = 0u;
            for (uint s = 0u; s < n_seeds; ++s) {
                if (tg_seeds[s] < p.V) {
                    const float vp = tg_pen[s];
                    const float vr = tg_raw[s];
                    uint pos_p = cnt;
                    while (pos_p > 0u && vp > sorted_p[pos_p - 1u]) {
                        sorted_p[pos_p] = sorted_p[pos_p - 1u];
                        --pos_p;
                    }
                    sorted_p[pos_p] = vp;
                    uint pos_r = cnt;
                    while (pos_r > 0u && vr > sorted_r[pos_r - 1u]) {
                        sorted_r[pos_r] = sorted_r[pos_r - 1u];
                        --pos_r;
                    }
                    sorted_r[pos_r] = vr;
                    ++cnt;
                }
            }
            const float kth_p = sorted_p[K_need - 1u];
            const float kth_r = sorted_r[K_need - 1u];
            const float raw_cap = (p.softcap > 0.0f) ? (p.softcap * 8.0f) : 1e30f;
            meta->T_raw = min(kth_r, raw_cap) - 1e-3f;
            meta->T_pen = kth_p - 1e-4f;
        }
    }
}

// Stage 1.75: Compacts ALL rows `v` with `ub[v] >= T` into `cand_ids` (`Grid(128, 1, 1)`, `256` th/TG, `< 5 us`!)
kernel void sik_bp_filter_compact(
    device const float*   ub_in    [[buffer(0)]],
    device uint*          cand_ids [[buffer(1)]],
    device SikBpMeta*     meta     [[buffer(2)]],
    constant SikBpParams& p        [[buffer(3)]],
    uint blk [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint chunk = max(1u, (p.V + 127u) >> 7u);
    const uint beg   = blk * chunk;
    const uint end   = min(p.V, beg + chunk);
    const float T_r  = meta->T_raw;
    const float T_p  = meta->T_pen;
    const bool  pen  = (p.has_pen != 0u);

    threadgroup atomic_uint tg_cnt;
    threadgroup uint        tg_buf[256];
    threadgroup uint        tg_gbase;

    for (uint base = beg; base < end; base += 256u) {
        if (tid == 0u) atomic_store_explicit(&tg_cnt, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint row = base + tid;
        if (row < end) {
            const float u = ub_in[row];
            const bool keep = pen ? (sik_softcap(u, p.softcap) >= T_p) : (u >= T_r);
            if (keep) {
                const uint slot = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
                tg_buf[slot] = row;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint cnt = atomic_load_explicit(&tg_cnt, memory_order_relaxed);
        if (cnt > 0u) {
            if (tid == 0u) {
                const uint gb = atomic_fetch_add_explicit(&meta->cand_count, cnt, memory_order_relaxed);
                tg_gbase = gb;
                if (gb + cnt > SIK_BP_MAX_CAND) {
                    atomic_store_explicit(&meta->overflow, 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < cnt && (tg_gbase + tid) < SIK_BP_MAX_CAND) {
                cand_ids[tg_gbase + tid] = tg_buf[tid];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Prep GPU Indirect Dispatch Args: either sparse Stage 2 (`cand_count <= 32768`) OR full `Q8_0` fallback (`0` CPU syncs!)
kernel void sik_bp_prep_indirect(
    device SikBpMeta*     meta [[buffer(0)]],
    constant SikBpParams& p    [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tid != 0u) return;
    const uint cnt = atomic_load_explicit(&meta->cand_count, memory_order_relaxed);
    const uint ovf = atomic_load_explicit(&meta->overflow,   memory_order_relaxed);
    const bool ok  = (ovf == 0u && cnt > 0u && cnt <= SIK_BP_MAX_CAND);
    if (ok) {
        meta->ind_stage2[0]  = (cnt + 7u) >> 3u; meta->ind_stage2[1]  = 1u; meta->ind_stage2[2]  = 1u;
        meta->ind_cand_s1[0] = (p.is_greedy != 0u) ? 1u : 64u;
        meta->ind_cand_s1[1] = 1u; meta->ind_cand_s1[2] = 1u;
        meta->ind_cand_s2[0] = (p.is_greedy != 0u) ? 0u : 1u;
        meta->ind_cand_s2[1] = 1u; meta->ind_cand_s2[2] = 1u;
        meta->ind_fb_gemv[0] = 0u; meta->ind_fb_gemv[1] = 0u; meta->ind_fb_gemv[2] = 0u;
        meta->ind_fb_pen[0]  = 0u; meta->ind_fb_pen[1]  = 0u; meta->ind_fb_pen[2]  = 0u;
        meta->ind_fb_s1[0]   = 0u; meta->ind_fb_s1[1]   = 0u; meta->ind_fb_s1[2]   = 0u;
        meta->ind_fb_s2[0]   = 0u; meta->ind_fb_s2[1]   = 0u; meta->ind_fb_s2[2]   = 0u;
    } else {
        meta->ind_stage2[0]  = 0u; meta->ind_stage2[1]  = 0u; meta->ind_stage2[2]  = 0u;
        meta->ind_cand_s1[0] = 0u; meta->ind_cand_s1[1] = 0u; meta->ind_cand_s1[2] = 0u;
        meta->ind_cand_s2[0] = 0u; meta->ind_cand_s2[1] = 0u; meta->ind_cand_s2[2] = 0u;
        meta->ind_fb_gemv[0] = (p.V + 7u) >> 3u; meta->ind_fb_gemv[1] = 1u; meta->ind_fb_gemv[2] = 1u;
        meta->ind_fb_pen[0]  = p.pen_tgs;        meta->ind_fb_pen[1]  = (p.pen_tgs > 0u) ? 1u : 0u; meta->ind_fb_pen[2] = (p.pen_tgs > 0u) ? 1u : 0u;
        meta->ind_fb_s1[0]   = (p.is_greedy != 0u) ? 128u : 64u; meta->ind_fb_s1[1] = 1u; meta->ind_fb_s1[2] = 1u;
        meta->ind_fb_s2[0]   = 1u; meta->ind_fb_s2[1] = 1u; meta->ind_fb_s2[2] = 1u;
    }
}

// Stage 2: Exact `Q8_0` dot product ONLY for surviving candidates (`100.0000%` bit-identical to `sik_q8_gemv_l16_fixed`!)
kernel void sik_bp_stage2_exact_cand(
    device const char*    w_q         [[buffer(0)]],
    device const half*    w_s         [[buffer(1)]],
    device const half*    x_in        [[buffer(2)]],
    device const uint*    cand_ids    [[buffer(3)]],
    device const ushort*  pen_counts  [[buffer(4)]],
    device float*         cand_logits [[buffer(5)]],
    device SikBpMeta*     meta        [[buffer(6)]],
    constant SikBpParams& p           [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint cidx = (tgid * 4u + sgid) * 2u + (lane >> 4u);
    const uint cnt  = atomic_load_explicit(&meta->cand_count, memory_order_relaxed);
    if (cidx >= cnt || cidx >= SIK_BP_MAX_CAND) return;

    const uint row    = cand_ids[cidx];
    const uint lane16 = lane & 15u;
    const uint k16    = p.D >> 4u;
    const uint nblk   = p.D >> 6u;
    device const uint4* w16 = reinterpret_cast<device const uint4*>(w_q + ulong(row) * p.D);
    device const half*  sc  = w_s + ulong(row) * nblk;

    float acc = 0.0f;
    for (uint i = lane16; i < k16; i += 16u) {
        const float  sv = float(sc[i >> 2u]);
        const uint4  q  = w16[i];
        const float4 f0 = float4(as_type<char4>(q.x)), f1 = float4(as_type<char4>(q.y));
        const float4 f2 = float4(as_type<char4>(q.z)), f3 = float4(as_type<char4>(q.w));
        device const half4* xm = reinterpret_cast<device const half4*>(x_in + i * 16u);
        const float d = (dot(f0, float4(xm[0])) + dot(f1, float4(xm[1])))
                      + (dot(f2, float4(xm[2])) + dot(f3, float4(xm[3])));
        acc = fma(sv, d, acc);
    }
    acc = sik_reduce16(acc);
    if (lane16 == 0u) {
        float val = sik_softcap(acc, p.softcap);
        if (p.has_pen != 0u) {
            val = sik_apply_pen_val(val, uint(pen_counts[row]), p.pres_pen, p.freq_pen, p.rep_pen);
        }
        cand_logits[cidx] = val;
    }
}

// Sparse Candidate Greedy Argmax (1 TG x 256 th, exact vocab-ID tie-breaking, `< 2 us`!)
kernel void sik_bp_cand_sample_greedy(
    device const uint*  cand_ids    [[buffer(0)]],
    device const float* cand_logits [[buffer(1)]],
    device uint*        out_tok     [[buffer(2)]],
    device SikBpMeta*   meta        [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint cnt = min(SIK_BP_MAX_CAND, atomic_load_explicit(&meta->cand_count, memory_order_relaxed));
    float best_v = -INFINITY;
    uint  best_r = 0xFFFFFFFFu;
    for (uint i = tid; i < cnt; i += 256u) {
        const float v = cand_logits[i];
        const uint  r = cand_ids[i];
        if (v > best_v || (v == best_v && r < best_r)) { best_v = v; best_r = r; }
    }
    #pragma clang loop unroll(full)
    for (ushort off = 16u; off > 0u; off >>= 1u) {
        const float ov = simd_shuffle_xor(best_v, off);
        const uint  or_ = simd_shuffle_xor(best_r, off);
        if (ov > best_v || (ov == best_v && or_ < best_r)) { best_v = ov; best_r = or_; }
    }
    threadgroup float tg_v[8];
    threadgroup uint  tg_r[8];
    if (lane == 0u) { tg_v[sgid] = best_v; tg_r[sgid] = best_r; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v = (lane < 8u) ? tg_v[lane] : -INFINITY;
        uint  r = (lane < 8u) ? tg_r[lane] : 0xFFFFFFFFu;
        #pragma clang loop unroll(full)
        for (ushort off = 4u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(v, off);
            const uint  or_ = simd_shuffle_xor(r, off);
            if (ov > v || (ov == v && or_ < r)) { v = ov; r = or_; }
        }
        if (lane == 0u) out_tok[0] = (r != 0xFFFFFFFFu) ? r : 0u;
    }
}

// Sparse Candidate Stochastic Top-K Stage 1 (`64` TGs x `256` th, exact vocab-ID tie-breaking!)
kernel void sik_bp_cand_sample_stoch_s1(
    device const uint*    cand_ids    [[buffer(0)]],
    device const float*   cand_logits [[buffer(1)]],
    device float*         stage_v     [[buffer(2)]],
    device uint*          stage_i     [[buffer(3)]],
    device SikBpMeta*     meta        [[buffer(4)]],
    constant SikBpParams& p           [[buffer(5)]],
    uint blk  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (blk >= 64u) return;
    const uint cnt   = min(SIK_BP_MAX_CAND, atomic_load_explicit(&meta->cand_count, memory_order_relaxed));
    const uint chunk = max(1u, (cnt + 63u) >> 6u);
    const uint beg   = blk * chunk;
    const uint end   = min(cnt, beg + chunk);
    const uint K     = min(64u, max(1u, p.top_k));

    float rv[4];
    uint  ri[4];
    uint  n_loc = 0u;
    for (uint i = beg + tid; i < end && n_loc < 4u; i += 256u) {
        rv[n_loc] = cand_logits[i];
        ri[n_loc] = cand_ids[i];
        ++n_loc;
    }

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    const ulong base_out = ulong(blk) * K;
    for (uint k = 0u; k < K; ++k) {
        float bv = -INFINITY; uint bi = 0xFFFFFFFFu; uint bj = 4u;
        for (uint j = 0u; j < n_loc; ++j) {
            if (rv[j] > bv || (rv[j] == bv && ri[j] < bi)) { bv = rv[j]; bi = ri[j]; bj = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(bv, off);
            const uint  oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = bv; tg_i[sgid] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_i[0] = idx;
                stage_v[base_out + k] = v;
                stage_i[base_out + k] = idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint win = tg_i[0];
        if (bj < 4u && ri[bj] == win) { rv[bj] = -INFINITY; ri[bj] = 0xFFFFFFFFu; }
        if (k + 1u < K) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Sparse Candidate Stochastic Top-K/Top-P Stage 2 (`1` TG x `256` th, exact RNG parity w/ `sik_sample_vocab_stage2`!)
kernel void sik_bp_cand_sample_stoch_s2(
    device const float*   stage_v [[buffer(0)]],
    device const uint*    stage_i [[buffer(1)]],
    device uint*          out_tok [[buffer(2)]],
    constant SikBpParams& p       [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint K   = min(64u, max(1u, p.top_k));
    const uint tot = 64u * K;

    float rv[16];
    uint  ri[16];
    uint  cnt = 0u;
    for (uint i = tid; i < tot && cnt < 16u; i += 256u) {
        rv[cnt] = stage_v[i];
        ri[cnt] = stage_i[i];
        ++cnt;
    }

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    threadgroup float top_vals[64];
    threadgroup uint  top_ids[64];

    for (uint k = 0u; k < K; ++k) {
        float bv = -INFINITY; uint bi = 0xFFFFFFFFu; uint bj = 16u;
        for (uint j = 0u; j < cnt; ++j) {
            if (rv[j] > bv || (rv[j] == bv && ri[j] < bi)) { bv = rv[j]; bi = ri[j]; bj = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(bv, off);
            const uint  oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = bv; tg_i[sgid] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_i[0] = idx;
                top_vals[k] = v;
                top_ids[k]  = (idx < p.V) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint win = tg_i[0];
        if (bj < 16u && ri[bj] == win) { rv[bj] = -INFINITY; ri[bj] = 0xFFFFFFFFu; }
        if (k + 1u < K) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float max_v = top_vals[0];
        float probs[64];
        float sum_p = 0.0f;
        uint valid_k = 0u;
        for (uint k = 0u; k < K; ++k) {
            if (top_vals[k] > -INFINITY) {
                const float w = exp((top_vals[k] - max_v) * p.inv_temp);
                probs[k] = w;
                sum_p   += w;
                valid_k  = k + 1u;
            } else {
                probs[k] = 0.0f;
            }
        }
        const uint eff_K = max(1u, valid_k);
        const float inv_sum = (sum_p > 0.0f) ? (1.0f / sum_p) : 1.0f;
        for (uint k = 0u; k < eff_K; ++k) probs[k] *= inv_sum;

        uint  keep_k    = eff_K;
        float kept_mass = 1.0f;
        if (p.top_p > 0.0f && p.top_p < 1.0f) {
            float cum = 0.0f;
            for (uint k = 0u; k < eff_K; ++k) {
                cum += probs[k];
                if (cum >= p.top_p) { keep_k = k + 1u; kept_mass = cum; break; }
            }
        }
        const float inv_kept = (kept_mass > 0.0f) ? (1.0f / kept_mass) : 1.0f;
        const float u        = sik_rand_u01(p.rng_lo, p.rng_hi, 0u);
        float       cdf      = 0.0f;
        uint        chosen_k = 0u;
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += probs[k] * inv_kept;
            if (u <= cdf || k + 1u == keep_k) { chosen_k = k; break; }
        }
        out_tok[0] = top_ids[chosen_k];
    }
}

/* ============================================================================
 * 11. Dynamic W=4 Multi-Branch On-GPU Markov Beam Tree DSpark (`0.00000` KL!)
 * ============================================================================ */

struct SikTreeMeta {
    uint  M;                // Current nodes in tree (1..8)
    uint  max_nodes;        // Target M = 1 + eff_k (2..8)
    uint  is_linear;        // 1 iff linear chain 0->1->...->M-1
    uint  active_parent;    // Node u currently expanding (0..6, or 0xFFFFFFFF if done)
    uint  parent[8];        // parent[0]=0; parent[i] < i
    uint  depth[8];         // depth[0]=0; depth[i] = depth[parent[i]] + 1
    uint  anc_mask[8];      // bit j set iff j is ancestor of i (incl. i)
    uint  path_packed[8];   // 4 bits per depth d: node ID of ancestor of i at depth d
    uint  p2[8];            // grandparent of i (0xFFFFFFFF if depth < 2)
    uint  p3[8];            // great-grandparent of i (0xFFFFFFFF if depth < 3)
    uint  tok_id[8];        // tok_id[0]=pending; tok_id[1..M-1]=drafted tokens
    float node_prob[8];     // P(tok_id[i] | parent[i])
    float cum_prob[8];      // cumulative path prob from root
    uint  sampled_tok[8];   // Stochastic T>0 primary sampled token at expanded node u (or 0xFFFFFFFF)
    uint  pool_count;
    uint  beam_width;       // Max W (default 4)
    uint  pool_parent[32];
    uint  pool_tok[32];
    float pool_cond_p[32];
    float pool_cum_p[32];
    uint  pool_used[32];
};

struct SikTreeInitParams {
    uint max_nodes;
    uint beam_width;
    uint rank;
    uint pad0;
};

// Initialize root node 0 & zero 8 x rank tree RNN state buffer (`Grid(1,1,1)`, `256` th)
kernel void sik_tree_init(
    device const uint*          tok_ids     [[buffer(0)]],
    device SikTreeMeta*         tree        [[buffer(1)]],
    device float*               tree_states [[buffer(2)]],
    constant SikTreeInitParams& p           [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint tot_s = 8u * p.rank;
    for (uint i = tid; i < tot_s; i += 256u) tree_states[i] = 0.0f;
    if (tid == 0u) {
        const uint root_tok = tok_ids[0];
        tree->M             = 1u;
        tree->max_nodes     = clamp(p.max_nodes, 1u, 8u);
        tree->is_linear     = 1u;
        tree->active_parent = (tree->max_nodes > 1u) ? 0u : 0xFFFFFFFFu;
        tree->pool_count    = 0u;
        tree->beam_width    = clamp(p.beam_width, 1u, 4u);
        for (uint i = 0u; i < 8u; ++i) {
            tree->parent[i]      = 0u;
            tree->depth[i]       = 0u;
            tree->anc_mask[i]    = (i == 0u) ? 1u : 0u;
            tree->path_packed[i] = 0u;
            tree->p2[i]          = 0xFFFFFFFFu;
            tree->p3[i]          = 0xFFFFFFFFu;
            tree->tok_id[i]      = (i == 0u) ? root_tok : 0u;
            tree->node_prob[i]   = (i == 0u) ? 1.0f : 0.0f;
            tree->cum_prob[i]    = (i == 0u) ? 1.0f : 0.0f;
            tree->sampled_tok[i] = 0xFFFFFFFFu;
        }
        for (uint i = 0u; i < 32u; ++i) {
            tree->pool_parent[i] = 0u;
            tree->pool_tok[i]    = 0u;
            tree->pool_cond_p[i] = 0.0f;
            tree->pool_cum_p[i]  = 0.0f;
            tree->pool_used[i]   = 1u;
        }
    }
}

// Tree-indexed Markov RNN input prep: reads `u = tree.active_parent`, `prev_state = (u==0) ? 0 : tree_states[parent[u]]`!
kernel void sik_mtp_rnn_prep_tree(
    device const float*           tree_states [[buffer(0)]], // [8, rank]
    device const half*            markov_w1   [[buffer(1)]],
    device const float*           norm_f32    [[buffer(2)]],
    device const float*           f_rnn       [[buffer(3)]],
    device const SikTreeMeta*     tree        [[buffer(4)]],
    device half*                  rnn_in      [[buffer(5)]],
    constant SikMtpRnnPrepParams& p           [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint u = tree->active_parent;
    if (u >= 8u) return;
    const uint b1 = p.rank, b2 = b1 + p.rank, b3 = b2 + p.D, b4 = b3 + p.D;
    if (gid >= b4) return;
    if (gid < b1) {
        const float sv = (u == 0u) ? 0.0f : tree_states[ulong(tree->parent[u]) * p.rank + gid];
        rnn_in[gid] = sik_clamp_f16(sv);
    } else if (gid < b2) {
        const uint prev = min(tree->tok_id[u], p.vocab_pad - 1u);
        rnn_in[gid] = markov_w1[ulong(prev) * p.rank + (gid - b1)];
    } else if (gid < b3) {
        const uint d_child = min(7u, tree->depth[u] + 1u);
        rnn_in[gid] = sik_clamp_f16(norm_f32[ulong(d_child) * p.D + (gid - b2)]);
    } else {
        rnn_in[gid] = sik_clamp_f16(f_rnn[gid - b3]);
    }
}

// Tree-indexed Markov RNN gate: writes post-step state into `tree_states[u]` + `latent_f16`!
kernel void sik_mtp_rnn_gate_tree(
    device const float*       raw_in      [[buffer(0)]],
    device const float*       rnn_bias    [[buffer(1)]],
    device float*             tree_states [[buffer(2)]], // [8, rank]
    device half*              latent_f16  [[buffer(3)]],
    device const SikTreeMeta* tree        [[buffer(4)]],
    constant uint&            rank        [[buffer(5)]],
    uint r [[thread_position_in_grid]]
) {
    const uint u = tree->active_parent;
    if (u >= 8u || r >= rank) return;
    const float prev_s = (u == 0u) ? 0.0f : tree_states[ulong(tree->parent[u]) * rank + r];
    const float g_raw  = raw_in[r]             + rnn_bias[r];
    const float c_raw  = raw_in[rank + r]      + rnn_bias[rank + r];
    const float o_raw  = raw_in[2u * rank + r] + rnn_bias[2u * rank + r];
    const float gate   = sik_sigmoid(g_raw);
    tree_states[ulong(u) * rank + r] = gate * prev_s + (1.0f - gate) * sik_safe_tanh(c_raw);
    latent_f16[r] = sik_clamp_f16(sik_safe_tanh(o_raw));
}

// Fused 1-Pass FR-Spec Q8_0 Folded Markov Head + Softcap + Top-4 + Online LSE for Tree Expansion!
// Grid: `(128, 1, 1)`, `128` threads/TG (`4` SGs = `8` rows/wave via 16 lanes/row)
kernel void sik_mtp_fold_top4_q8(
    device const char*            fold_w_q    [[buffer(0)]],
    device const half*            fold_w_s    [[buffer(1)]],
    device const float*           fold_bias   [[buffer(2)]],
    device const float*           base_logits [[buffer(3)]], // [7, V]
    device const half*            latent_f16  [[buffer(4)]],
    device const SikTreeMeta*     tree        [[buffer(5)]],
    device float*                 stage_v     [[buffer(6)]], // [128 * 4]
    device uint*                  stage_i     [[buffer(7)]], // [128 * 4]
    device float*                 stage_lse   [[buffer(8)]], // [128]
    constant SikMtpFoldAddParams& p           [[buffer(9)]],
    uint blk  [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint u = tree->active_parent;
    if (u >= 8u || blk >= 128u) return;

    const uint d_idx  = min(6u, tree->depth[u]);
    device const float* base_row = base_logits + ulong(d_idx) * p.V;

    const uint lane16 = lane & 15u;
    const uint row_in = sgid * 2u + (lane >> 4u);
    const uint k16    = p.rank >> 4u;
    const uint nblk   = p.rank >> 6u;
    const uint chunk  = max(1u, (p.V + 127u) >> 7u);
    const uint beg    = blk * chunk;
    const uint end    = min(p.V, beg + chunk);

    device const half4* x4 = reinterpret_cast<device const half4*>(latent_f16);
    float4 fx0 = float4(0.0f), fx1 = float4(0.0f), fx2 = float4(0.0f), fx3 = float4(0.0f);
    if (k16 == 16u) {
        const uint k4 = lane16 * 4u;
        fx0 = float4(x4[k4 + 0u]); fx1 = float4(x4[k4 + 1u]);
        fx2 = float4(x4[k4 + 2u]); fx3 = float4(x4[k4 + 3u]);
    }

    // Per-row-leader top-4 + online safe-softmax LSE (only 8 extra scalars, 0 spill!)
    float bv0 = -INFINITY, bv1 = -INFINITY, bv2 = -INFINITY, bv3 = -INFINITY;
    uint  bi0 = 0xFFFFFFFFu, bi1 = 0xFFFFFFFFu, bi2 = 0xFFFFFFFFu, bi3 = 0xFFFFFFFFu;
    float sum_e = 0.0f;

    for (uint r0 = beg; r0 < end; r0 += 8u) {
        const uint row = r0 + row_in;
        if (row < end) {
            device const uint4* w16 = reinterpret_cast<device const uint4*>(fold_w_q + ulong(row) * p.rank);
            device const half*  sc  = fold_w_s + ulong(row) * nblk;
            float acc = 0.0f;
            if (k16 == 16u) {
                const float s = float(sc[lane16 >> 2u]);
                const uint4 q = w16[lane16];
                const float d = (dot(float4(as_type<char4>(q.x)), fx0) + dot(float4(as_type<char4>(q.y)), fx1))
                              + (dot(float4(as_type<char4>(q.z)), fx2) + dot(float4(as_type<char4>(q.w)), fx3));
                acc = fma(s, d, 0.0f);
            } else {
                for (uint i = lane16; i < k16; i += 16u) {
                    const float s  = float(sc[i >> 2u]);
                    const uint4 q  = w16[i];
                    const uint  k4 = i * 4u;
                    const float d  = (dot(float4(as_type<char4>(q.x)), float4(x4[k4 + 0u])) +
                                      dot(float4(as_type<char4>(q.y)), float4(x4[k4 + 1u]))) +
                                     (dot(float4(as_type<char4>(q.z)), float4(x4[k4 + 2u])) +
                                      dot(float4(as_type<char4>(q.w)), float4(x4[k4 + 3u])));
                    acc = fma(s, d, acc);
                }
            }
            acc = sik_reduce16(acc);
            if (lane16 == 0u) {
                const float v = sik_softcap(base_row[row] + acc + fold_bias[row], p.softcap);
                if (v > bv0) {
                    sum_e = (bv0 > -INFINITY) ? fma(sum_e, exp(bv0 - v), 1.0f) : 1.0f;
                    bv3 = bv2; bi3 = bi2; bv2 = bv1; bi2 = bi1; bv1 = bv0; bi1 = bi0;
                    bv0 = v;   bi0 = row;
                } else {
                    if (v > bv0 - 14.0f) sum_e += exp(v - bv0);
                    if (v > bv1 || (v == bv1 && row < bi1)) {
                        bv3 = bv2; bi3 = bi2; bv2 = bv1; bi2 = bi1; bv1 = v; bi1 = row;
                    } else if (v > bv2 || (v == bv2 && row < bi2)) {
                        bv3 = bv2; bi3 = bi2; bv2 = v; bi2 = row;
                    } else if (v > bv3 || (v == bv3 && row < bi3)) {
                        bv3 = v; bi3 = row;
                    }
                }
            }
        }
    }

    // Store all 8 row-leaders' top-4 (32 items) + LSE into TG mem and merge cleanly in SG0!
    threadgroup float tg_tv[32];
    threadgroup uint  tg_ti[32];
    threadgroup float tg_max[8];
    threadgroup float tg_se[8];
    if (lane16 == 0u) {
        tg_tv[row_in * 4u + 0u] = bv0; tg_ti[row_in * 4u + 0u] = bi0;
        tg_tv[row_in * 4u + 1u] = bv1; tg_ti[row_in * 4u + 1u] = bi1;
        tg_tv[row_in * 4u + 2u] = bv2; tg_ti[row_in * 4u + 2u] = bi2;
        tg_tv[row_in * 4u + 3u] = bv3; tg_ti[row_in * 4u + 3u] = bi3;
        tg_max[row_in] = bv0;
        tg_se[row_in]  = sum_e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float lv = tg_tv[lane];
        uint  li = tg_ti[lane];
        float bmax = -INFINITY;
        for (uint k = 0u; k < 4u; ++k) {
            float bv = lv; uint bi = li;
            #pragma clang loop unroll(full)
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(bv, off);
                const uint  oi = simd_shuffle_xor(bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (k == 0u) bmax = bv;
            if (lane == 0u) {
                stage_v[blk * 4u + k] = bv;
                stage_i[blk * 4u + k] = bi;
            }
            if (li == bi) { lv = -INFINITY; li = 0xFFFFFFFFu; }
        }
        const float me = (lane < 8u && tg_max[lane] > -INFINITY) ? (tg_se[lane] * exp(tg_max[lane] - bmax)) : 0.0f;
        const float te = simd_sum(me);
        if (lane == 0u) stage_lse[blk] = te;
    }
}

// Tree-aware Folded Markov Head (`Q8_0` FR-Spec & `FP16` Full-Vocab) for penalties / stochastic / non-FR-Spec!
kernel void sik_mtp_fold_add_softcap_tree_q8(
    device const char*            fold_w_q    [[buffer(0)]],
    device const half*            fold_w_s    [[buffer(1)]],
    device const float*           fold_bias   [[buffer(2)]],
    device const float*           base_logits [[buffer(3)]],
    device const half*            latent_f16  [[buffer(4)]],
    device const SikTreeMeta*     tree        [[buffer(5)]],
    device float*                 logits_out  [[buffer(6)]],
    constant SikMtpFoldAddParams& p           [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint u = tree->active_parent;
    if (u >= 8u) return;
    const uint row    = (tgid * 4u + sgid) * 2u + (lane >> 4u);
    const uint lane16 = lane & 15u;
    if (row >= p.V) return;

    const uint d_idx = min(6u, tree->depth[u]);
    const uint k16   = p.rank >> 4u;
    const uint nblk  = p.rank >> 6u;
    device const uint4* w16 = reinterpret_cast<device const uint4*>(fold_w_q + ulong(row) * p.rank);
    device const half*  sc  = fold_w_s + ulong(row) * nblk;
    device const half4* x4  = reinterpret_cast<device const half4*>(latent_f16);

    float acc = 0.0f;
    for (uint i = lane16; i < k16; i += 16u) {
        const float s  = float(sc[i >> 2u]);
        const uint4 q  = w16[i];
        const uint  k4 = i * 4u;
        const float d  = (dot(float4(as_type<char4>(q.x)), float4(x4[k4 + 0u])) +
                          dot(float4(as_type<char4>(q.y)), float4(x4[k4 + 1u]))) +
                         (dot(float4(as_type<char4>(q.z)), float4(x4[k4 + 2u])) +
                          dot(float4(as_type<char4>(q.w)), float4(x4[k4 + 3u])));
        acc = fma(s, d, acc);
    }
    acc = sik_reduce16(acc);
    if (lane16 == 0u) {
        logits_out[row] = sik_softcap(base_logits[ulong(d_idx) * p.V + row] + acc + fold_bias[row], p.softcap);
    }
}

kernel void sik_mtp_fold_add_softcap_tree_f16(
    device const half*            fold_w      [[buffer(0)]],
    device const float*           fold_bias   [[buffer(1)]],
    device const float*           base_logits [[buffer(2)]],
    device const half*            latent_f16  [[buffer(3)]],
    device const SikTreeMeta*     tree        [[buffer(4)]],
    device float*                 logits_out  [[buffer(5)]],
    constant SikMtpFoldAddParams& p           [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint u = tree->active_parent;
    if (u >= 8u) return;
    const uint row    = (tgid * 4u + sgid) * 2u + (lane >> 4u);
    const uint lane16 = lane & 15u;
    if (row >= p.V) return;

    const uint d_idx   = min(6u, tree->depth[u]);
    const uint c16_cnt = p.rank >> 4u;
    device const half4* w4 = reinterpret_cast<device const half4*>(fold_w + ulong(row) * p.rank);
    device const half4* x4 = reinterpret_cast<device const half4*>(latent_f16);

    float acc = 0.0f;
    for (uint c16 = lane16; c16 < c16_cnt; c16 += 16u) {
        const uint k4 = c16 * 4u;
        acc += sik_dot_f32_h4x4(w4[k4 + 0u], w4[k4 + 1u], w4[k4 + 2u], w4[k4 + 3u],
                                x4[k4 + 0u], x4[k4 + 1u], x4[k4 + 2u], x4[k4 + 3u]);
    }
    acc = sik_reduce16(acc);
    if (lane16 == 0u) {
        logits_out[row] = sik_softcap(base_logits[ulong(d_idx) * p.V + row] + acc + fold_bias[row], p.softcap);
    }
}

// General Top-4 + Online LSE extractor from `b_logits` for Tree Greedy (penalties / full-vocab)
// Grid: `(128, 1, 1)`, `128` threads/TG
kernel void sik_sample_top4_stage1_tree(
    device const float*       logits    [[buffer(0)]],
    device const SikTreeMeta* tree      [[buffer(1)]],
    device float*             stage_v   [[buffer(2)]], // [128 * 4]
    device uint*              stage_i   [[buffer(3)]], // [128 * 4]
    device float*             stage_lse [[buffer(4)]], // [128]
    constant SikSampleParams& p         [[buffer(5)]],
    uint blk  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (tree->active_parent >= 8u || blk >= 128u) return;
    const uint chunk = max(1u, (p.vocab_size + 127u) >> 7u);
    const uint beg   = blk * chunk;
    const uint end   = min(p.vocab_size, beg + chunk);

    float bv0 = -INFINITY, bv1 = -INFINITY, bv2 = -INFINITY, bv3 = -INFINITY;
    uint  bi0 = 0xFFFFFFFFu, bi1 = 0xFFFFFFFFu, bi2 = 0xFFFFFFFFu, bi3 = 0xFFFFFFFFu;
    float sum_e = 0.0f;

    for (uint i = beg + tid; i < end; i += 128u) {
        const float v = logits[i];
        if (v > bv0) {
            sum_e = (bv0 > -INFINITY) ? fma(sum_e, exp(bv0 - v), 1.0f) : 1.0f;
            bv3 = bv2; bi3 = bi2; bv2 = bv1; bi2 = bi1; bv1 = bv0; bi1 = bi0;
            bv0 = v;   bi0 = i;
        } else {
            if (v > bv0 - 14.0f) sum_e += exp(v - bv0);
            if (v > bv1 || (v == bv1 && i < bi1)) {
                bv3 = bv2; bi3 = bi2; bv2 = bv1; bi2 = bi1; bv1 = v; bi1 = i;
            } else if (v > bv2 || (v == bv2 && i < bi2)) {
                bv3 = bv2; bi3 = bi2; bv2 = v; bi2 = i;
            } else if (v > bv3 || (v == bv3 && i < bi3)) {
                bv3 = v; bi3 = i;
            }
        }
    }

    // Reduce 4 SGs' top-4 sequentially (4 rounds x 128 -> 4 -> 1)
    threadgroup float tg_v[4];
    threadgroup uint  tg_i[4];
    threadgroup float tg_bmax[1];
    float cv[4] = {bv0, bv1, bv2, bv3};
    uint  ci[4] = {bi0, bi1, bi2, bi3};

    for (uint k = 0u; k < 4u; ++k) {
        float lv = cv[0]; uint li = ci[0]; uint lj = 0u;
        for (uint j = 1u; j < 4u; ++j) {
            if (cv[j] > lv || (cv[j] == lv && ci[j] < li)) { lv = cv[j]; li = ci[j]; lj = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(lv, off);
            const uint  oi = simd_shuffle_xor(li, off);
            if (ov > lv || (ov == lv && oi < li)) { lv = ov; li = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = lv; tg_i[sgid] = li; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v = (lane < 4u) ? tg_v[lane] : -INFINITY;
            uint  ix = (lane < 4u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 2u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(ix, off);
                if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
            }
            if (lane == 0u) {
                tg_i[0] = ix;
                if (k == 0u) tg_bmax[0] = v;
                stage_v[blk * 4u + k] = v;
                stage_i[blk * 4u + k] = ix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ci[lj] == tg_i[0]) { cv[lj] = -INFINITY; ci[lj] = 0xFFFFFFFFu; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float bmax = tg_bmax[0];
    const float me   = (bv0 > -INFINITY) ? (sum_e * exp(bv0 - bmax)) : 0.0f;
    const float sg_e = simd_sum(me);
    if (lane == 0u) tg_v[sgid] = sg_e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot_e = simd_sum((lane < 4u) ? tg_v[lane] : 0.0f);
        if (lane == 0u) stage_lse[blk] = tot_e;
    }
}

// Helper: Materialize highest-cum-prob frontier candidate into `tree` & pick next `active_parent`!
inline void sik_tree_materialize_next(device SikTreeMeta* tree, device uint* mtp_tok_ids) {
    if (tree->M >= tree->max_nodes) {
        tree->active_parent = 0xFFFFFFFFu;
        return;
    }
    uint  best_e = 0xFFFFFFFFu;
    float best_p = -1.0f;
    const uint pcnt = min(tree->pool_count, 32u);
    for (uint e = 0u; e < pcnt; ++e) {
        if (tree->pool_used[e] != 0u) continue;
        const uint par = tree->pool_parent[e];
        if (tree->depth[par] + 1u >= tree->max_nodes) continue; // safety: depth < max_nodes
        const float cp = tree->pool_cum_p[e];
        if (cp > best_p) {
            best_p = cp;
            best_e = e;
        }
    }
    if (best_e == 0xFFFFFFFFu) {
        tree->active_parent = 0xFFFFFFFFu;
        return;
    }
    tree->pool_used[best_e] = 1u;
    const uint par   = tree->pool_parent[best_e];
    const uint tok   = tree->pool_tok[best_e];
    const float cond = tree->pool_cond_p[best_e];
    const float cump = tree->pool_cum_p[best_e];
    const uint nid   = tree->M;
    const uint d     = tree->depth[par] + 1u;

    tree->M                = nid + 1u;
    tree->parent[nid]      = par;
    tree->depth[nid]       = d;
    tree->anc_mask[nid]    = tree->anc_mask[par] | (1u << nid);
    tree->path_packed[nid] = tree->path_packed[par] | (nid << (d * 4u));
    tree->p2[nid]          = (d >= 2u) ? tree->parent[par] : 0xFFFFFFFFu;
    tree->p3[nid]          = (d >= 3u) ? tree->p2[par]     : 0xFFFFFFFFu;
    tree->tok_id[nid]      = tok;
    tree->node_prob[nid]   = cond;
    tree->cum_prob[nid]    = cump;
    mtp_tok_ids[nid]       = tok;
    if (par != nid - 1u) tree->is_linear = 0u;

    if (tree->M >= tree->max_nodes) {
        tree->active_parent = 0xFFFFFFFFu;
    } else if (d + 1u < tree->max_nodes) {
        tree->active_parent = nid;
    } else {
        tree->active_parent = 0xFFFFFFFFu;
    }
}

// Stage 2 of Greedy Tree Expansion: merges 128 x top-4 + LSE -> exact softmax probs -> expands beam tree!
// Grid: `(1, 1, 1)`, `128` threads/TG
kernel void sik_tree_expand_top4(
    device const float*       stage_v     [[buffer(0)]], // [128 * 4]
    device const uint*        stage_i     [[buffer(1)]], // [128 * 4]
    device const float*       stage_lse   [[buffer(2)]], // [128]
    device const uint*        token_map   [[buffer(3)]],
    device SikTreeMeta*       tree        [[buffer(4)]],
    device uint*              mtp_tok_ids [[buffer(5)]],
    constant SikSampleParams& p           [[buffer(6)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint u = tree->active_parent;
    if (u >= 8u || tid >= 128u) return;

    float cv[4];
    uint  ci[4];
    for (uint j = 0u; j < 4u; ++j) {
        cv[j] = stage_v[tid * 4u + j];
        ci[j] = stage_i[tid * 4u + j];
    }
    const float blk_max = cv[0];

    threadgroup float tg_v[4];
    threadgroup uint  tg_i[4];
    threadgroup float top_v[4];
    threadgroup uint  top_i[4];

    for (uint k = 0u; k < 4u; ++k) {
        float lv = cv[0]; uint li = ci[0]; uint lj = 0u;
        for (uint j = 1u; j < 4u; ++j) {
            if (cv[j] > lv || (cv[j] == lv && ci[j] < li)) { lv = cv[j]; li = ci[j]; lj = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(lv, off);
            const uint  oi = simd_shuffle_xor(li, off);
            if (ov > lv || (ov == lv && oi < li)) { lv = ov; li = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = lv; tg_i[sgid] = li; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v = (lane < 4u) ? tg_v[lane] : -INFINITY;
            uint  ix = (lane < 4u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 2u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(ix, off);
                if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
            }
            if (lane == 0u) {
                tg_i[0]  = ix;
                top_v[k] = v;
                top_i[k] = ix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ci[lj] == tg_i[0]) { cv[lj] = -INFINITY; ci[lj] = 0xFFFFFFFFu; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float gmax    = top_v[0];
    const float contrib = (blk_max > -INFINITY) ? (stage_lse[tid] * exp(blk_max - gmax)) : 0.0f;
    const float sg_c    = simd_sum(contrib);
    if (lane == 0u) tg_v[sgid] = sg_c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        const float tot_e = simd_sum((lane < 4u) ? tg_v[lane] : 0.0f);
        if (lane == 0u) {
            const float inv_e  = (tot_e > 0.0f) ? (1.0f / tot_e) : 1.0f;
            const float par_cp = tree->cum_prob[u];
            const uint  W      = min(4u, max(1u, tree->beam_width));
            const float p0     = clamp(inv_e, 0.0f, 1.0f);
            // Exact softmax probabilities (100% consistent w/ linear pmax -> provable exp_toks_tree >= exp_toks_lin!):
            // if top-1 >= 0.85 stay 100% linear; else expand up to W=4 siblings above MoE break-even floor!
            const uint  n_add  = (p0 >= 0.85f) ? 1u : W;
            const float pr_min = (u == 0u) ? 0.15f : 0.18f;
            for (uint r = 0u; r < n_add; ++r) {
                const uint raw_id = top_i[r];
                if (raw_id >= p.vocab_size) continue;
                const float pr = (r == 0u) ? p0 : clamp(exp(top_v[r] - gmax) * inv_e, 0.0f, 1.0f);
                if (r > 0u && pr < pr_min) break;
                const uint idx = tree->pool_count;
                if (idx < 32u) {
                    const uint full_id = (p.use_token_map != 0u) ? token_map[raw_id] : raw_id;
                    tree->pool_parent[idx] = u;
                    tree->pool_tok[idx]    = full_id;
                    tree->pool_cond_p[idx] = pr;
                    tree->pool_cum_p[idx]  = par_cp * pr;
                    tree->pool_used[idx]   = 0u;
                    tree->pool_count       = idx + 1u;
                }
            }
            sik_tree_materialize_next(tree, mtp_tok_ids);
        }
    }
}

// Stochastic (`T > 0`) Tree Sampler Stage 1 (`Grid(64, 1, 1)`, `256` th)
kernel void sik_sample_vocab_stage1_tree(
    device const float*       logits  [[buffer(0)]],
    device const SikTreeMeta* tree    [[buffer(1)]],
    device float*             stage_v [[buffer(2)]],
    device uint*              stage_i [[buffer(3)]],
    constant SikSampleParams& p       [[buffer(4)]],
    uint blk  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (tree->active_parent >= 8u || blk >= 64u) return;
    const uint chunk    = max(1u, (p.vocab_size + 63u) >> 6u);
    const uint beg      = blk * chunk;
    const uint end      = min(p.vocab_size, beg + chunk);
    const uint K        = min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
    const ulong base_out = ulong(blk) * K;

    float rv[16];
    uint  ri[16];
    uint  cnt = 0u;
    for (uint i = beg + tid; i < end && cnt < 16u; i += 256u) {
        rv[cnt] = logits[i];
        ri[cnt] = i;
        ++cnt;
    }

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    for (uint k = 0u; k < K; ++k) {
        float best_v = -INFINITY; uint best_i = 0xFFFFFFFFu; uint best_j = 16u;
        for (uint j = 0u; j < cnt; ++j) {
            if (rv[j] > best_v || (rv[j] == best_v && ri[j] < best_i)) { best_v = rv[j]; best_i = ri[j]; best_j = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
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
        if (k + 1u < K) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Stochastic (`T > 0`) Tree Sampler Stage 2: saves `q_u` at slot `u` & expands beam tree (`0.00000` KL!)
// Grid: `(1, 1, 1)`, `256` threads/TG
kernel void sik_sample_vocab_stage2_tree(
    device const float*       stage_v       [[buffer(0)]],
    device const uint*        stage_i       [[buffer(1)]],
    device const uint*        token_map     [[buffer(2)]],
    device SikTreeMeta*       tree          [[buffer(3)]],
    device uint*              mtp_tok_ids   [[buffer(4)]],
    device uint*              out_top_ids   [[buffer(5)]], // [8, 64] indexed by u!
    device float*             out_top_probs [[buffer(6)]], // [8, 64] indexed by u!
    device uint*              out_keep_k    [[buffer(7)]], // [8] indexed by u!
    constant SikSampleParams& p             [[buffer(8)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint u = tree->active_parent;
    if (u >= 8u) return;

    const uint K   = min(p.vocab_size, (p.top_k > 0u && p.top_k <= 64u) ? p.top_k : 64u);
    const uint tot = 64u * K;

    float rv[16];
    uint  ri[16];
    uint  cnt = 0u;
    for (uint i = tid; i < tot && cnt < 16u; i += 256u) {
        rv[cnt] = stage_v[i];
        ri[cnt] = stage_i[i];
        ++cnt;
    }

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];
    threadgroup float top_vals[64];
    threadgroup uint  top_ids[64];

    for (uint k = 0u; k < K; ++k) {
        float best_v = -INFINITY; uint best_i = 0xFFFFFFFFu; uint best_j = 16u;
        for (uint j = 0u; j < cnt; ++j) {
            if (rv[j] > best_v || (rv[j] == best_v && ri[j] < best_i)) { best_v = rv[j]; best_i = ri[j]; best_j = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
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
        if (k + 1u < K) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float max_v = top_vals[0];
        float probs[64];
        float sum_p = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            const float w = exp((top_vals[k] - max_v) * p.inv_temp);
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
                if (cum >= p.top_p) { keep_k = k + 1u; kept_mass = cum; break; }
            }
        }
        const float inv_kept = (kept_mass > 0.0f) ? (1.0f / kept_mass) : 1.0f;
        uint  mapped_ids[64];
        float norm_ps[64];
        for (uint k = 0u; k < 64u; ++k) {
            const float np  = (k < keep_k) ? (probs[k] * inv_kept) : 0.0f;
            const uint  mid = (k < K) ? ((p.use_token_map != 0u) ? token_map[top_ids[k]] : top_ids[k]) : 0u;
            mapped_ids[k] = mid;
            norm_ps[k]    = np;
            out_top_ids[ulong(u) * 64ul + k]   = mid;
            out_top_probs[ulong(u) * 64ul + k] = np;
        }
        out_keep_k[u] = keep_k;

        const float ur       = sik_rand_u01(p.rng_lo, p.rng_hi, u);
        float       cdf      = 0.0f;
        uint        chosen_k = 0u;
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += norm_ps[k];
            if (ur <= cdf || k + 1u == keep_k) { chosen_k = k; break; }
        }
        const uint  sampled_id = mapped_ids[chosen_k];
        const float sampled_p  = norm_ps[chosen_k];
        tree->sampled_tok[u]   = sampled_id;

        const float par_cp = tree->cum_prob[u];
        const uint  W      = min(4u, max(1u, tree->beam_width));
        // Calibrated Leviathan expected-acceptance priority: smoothly tracks draft sharpness from
        // uncertain prose (~0.30) to sharp code (~0.95) without inflating flat distributions!
        const float p_prio = clamp(0.25f + 0.72f * norm_ps[0], 0.25f, 0.96f);
        if (tree->pool_count < 32u) {
            const uint idx = tree->pool_count++;
            tree->pool_parent[idx] = u;
            tree->pool_tok[idx]    = sampled_id;
            tree->pool_cond_p[idx] = p_prio;
            tree->pool_cum_p[idx]  = par_cp * p_prio;
            tree->pool_used[idx]   = 0u;
        }
        if (norm_ps[0] < 0.78f && W > 1u) {
            uint added = 1u;
            for (uint k = 0u; k < keep_k && added < W; ++k) {
                if (mapped_ids[k] == sampled_id) continue;
                if (norm_ps[k] < 0.15f) break;
                const float p_res = clamp((1.0f - p_prio) * norm_ps[k], 0.0f, 0.35f);
                if (tree->pool_count < 32u) {
                    const uint idx = tree->pool_count++;
                    tree->pool_parent[idx] = u;
                    tree->pool_tok[idx]    = mapped_ids[k];
                    tree->pool_cond_p[idx] = p_res;
                    tree->pool_cum_p[idx]  = par_cp * p_res;
                    tree->pool_used[idx]   = 0u;
                    ++added;
                }
            }
        }
        sik_tree_materialize_next(tree, mtp_tok_ids);
    }
}

// Tree-Aware Exact Sliding-Window Logit Penalties (`extra_mode == 1`: active draft parent u; `extra_mode == 2`: all M verify nodes!)
kernel void sik_apply_logit_penalties_tree(
    device float*                   logits        [[buffer(0)]],
    device const uint*              pen_ids       [[buffer(1)]],
    device const ushort*            pen_counts    [[buffer(2)]],
    constant SikTreeMeta&           tree          [[buffer(3)]],
    device const uint*              full_to_short [[buffer(4)]],
    constant SikLogitPenaltyParams& p             [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (p.extra_mode == 1u && tree.active_parent >= 8u) return;
    const uint extra_n     = (p.extra_mode == 1u) ? min(7u, tree.M > 0u ? (tree.M - 1u) : 0u) : min(7u, p.extra_count);
    const uint total_items = p.num_hist_ids + extra_n;
    if (gid >= total_items) return;

    uint t = 0u, base_c = 0u;
    if (gid < p.num_hist_ids) {
        t = pen_ids[gid];
        if (t >= p.vocab_size) return;
        base_c = uint(pen_counts[t]);
    } else {
        const uint j = 1u + (gid - p.num_hist_ids);
        t = tree.tok_id[j];
        if (t >= p.vocab_size || pen_counts[t] != 0u) return;
        for (uint pj = 1u; pj < j; ++pj) {
            if (tree.tok_id[pj] == t) return;
        }
        base_c = 0u;
    }

    const uint logit_col = (p.use_shortlist != 0u) ? full_to_short[t] : t;
    if (logit_col >= p.logits_stride) return;

    if (p.extra_mode == 1u) {
        const uint u    = tree.active_parent;
        const uint mask = tree.anc_mask[u];
        const uint d    = tree.depth[u];
        uint c = base_c;
        for (uint j = 1u; j <= u; ++j) {
            if (((mask >> j) & 1u) != 0u && tree.tok_id[j] == t) ++c;
        }
        for (uint e = 0u; e < d; ++e) {
            if (e < 8u && p.drop_toks[e] == t && c > 0u) --c;
        }
        if (c > 0u) {
            logits[logit_col] = sik_apply_pen_val(logits[logit_col], c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
        }
    } else {
        for (uint m = 0u; m < p.M; ++m) {
            const uint mask = tree.anc_mask[m];
            const uint d    = tree.depth[m];
            uint c = base_c;
            for (uint j = 1u; j <= m; ++j) {
                if (((mask >> j) & 1u) != 0u && tree.tok_id[j] == t) ++c;
            }
            for (uint e = 0u; e < d; ++e) {
                if (e < 8u && p.drop_toks[e] == t && c > 0u) --c;
            }
            if (c > 0u) {
                const ulong idx = ulong(m) * p.logits_stride + logit_col;
                logits[idx] = sik_apply_pen_val(logits[idx], c, p.presence_penalty, p.frequency_penalty, p.repetition_penalty);
            }
        }
    }
}

// Tree-Aware Fused `SConvKV + QKVR-Prep`: ZERO TG memory (max occupancy = linear!) + constant-cached `tree`!
kernel void sik_sconv_kv_qkvr_prep_tree(
    device const float*         q_raw      [[buffer(0)]],
    device const float*         k_raw      [[buffer(1)]],
    device const float*         v_raw      [[buffer(2)]],
    device const float*         r_raw      [[buffer(3)]],
    device const float4*        k_sconv_w  [[buffer(4)]],
    device const float4*        v_sconv_w  [[buffer(5)]],
    device const float4*        k_dyn      [[buffer(6)]],
    device const float4*        v_dyn      [[buffer(7)]],
    device float4*              conv_state [[buffer(8)]],
    device const float*         q_scale    [[buffer(9)]],
    device const float*         r_scale    [[buffer(10)]],
    device float*               q_out      [[buffer(11)]],
    device float*               r_out      [[buffer(12)]],
    device half*                k_cache    [[buffer(13)]],
    device half*                v_cache    [[buffer(14)]],
    device float4*              snaps      [[buffer(15)]],
    constant SikAttnPrepParams& p          [[buffer(16)]],
    device half*                k_scales   [[buffer(17)]],
    device half*                v_scales   [[buffer(18)]],
    constant SikTreeMeta&       tree       [[buffer(19)]],
    uint idx  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint  nq_m = p.n_q * p.M_valid;
    const uint  b1   = nq_m;
    const uint  b2   = b1 + p.n_kv;
    const uint  b3   = b2 + p.n_kv;
    const ulong sstr = ulong(p.snap_stride_f4);

    if (idx < b1) {
        const uint  m    = idx / p.n_q;
        const uint  hq   = idx - m * p.n_q;
        const uint  pos  = p.seq_pos_start + tree.depth[m];
        const float tau  = (p.is_local == 0u && p.log_n_floor > 0.0f)
            ? (1.0f + p.log_alpha * log(max(1.0f, float(pos + 1u) / p.log_n_floor))) : 1.0f;
        const ulong off4 = (ulong(m) * p.n_q + hq) * 32ul + lane;
        const float4 qv  = reinterpret_cast<device const float4*>(q_raw)[off4]
                         * reinterpret_cast<device const float4*>(q_scale)[ulong(hq) * 32ul + lane];
        const float tot  = simd_sum(dot(qv, qv));
        const float fac  = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps)) * tau;
        reinterpret_cast<device float4*>(q_out)[off4] = qv * fac;
    } else if (idx < b2) {
        const uint   hkv = idx - b1;
        const uint   c0  = hkv * 128u + lane * 4u;
        const float4 wt0 = k_sconv_w[c0 + 0u], wt1 = k_sconv_w[c0 + 1u];
        const float4 wt2 = k_sconv_w[c0 + 2u], wt3 = k_sconv_w[c0 + 3u];
        const float4 is0 = conv_state[c0 + 0u], is1 = conv_state[c0 + 1u];
        const float4 is2 = conv_state[c0 + 2u], is3 = conv_state[c0 + 3u];

        device const float4* kraw4 = reinterpret_cast<device const float4*>(k_raw);
        device half4*        kc4   = reinterpret_cast<device half4*>(k_cache);
        device char4*        kcc4  = reinterpret_cast<device char4*>(k_cache);
        float4 last0 = is0, last1 = is1, last2 = is2, last3 = is3;

        for (uint m = 0u; m < p.M_valid; ++m) {
            const uint   pos  = p.seq_pos_start + m; // physical KV slot = seq_pos_start + node_id!
            const uint   slot = (p.is_local != 0u) ? (pos & (p.kv_stride - 1u)) : pos;
            const float4 cur  = kraw4[(ulong(m) * p.n_kv + hkv) * 32ul + lane];

            const uint d  = tree.depth[m];
            const uint p1 = tree.parent[m] & 7u;
            const uint p2 = tree.p2[m] & 7u;
            const uint p3 = tree.p3[m] & 7u;
            const float4 r1 = (d >= 1u) ? kraw4[(ulong(p1) * p.n_kv + hkv) * 32ul + lane] : float4(0.0f);
            const float4 r2 = (d >= 2u) ? kraw4[(ulong(p2) * p.n_kv + hkv) * 32ul + lane] : float4(0.0f);
            const float4 r3 = (d >= 3u) ? kraw4[(ulong(p3) * p.n_kv + hkv) * 32ul + lane] : float4(0.0f);

            const float4 h2 = (d == 0u) ? float4(is0.z, is1.z, is2.z, is3.z) : r1;
            const float4 h1 = (d == 0u) ? float4(is0.y, is1.y, is2.y, is3.y)
                            : ((d == 1u) ? float4(is0.z, is1.z, is2.z, is3.z) : r2);
            const float4 h0 = (d == 0u) ? float4(is0.x, is1.x, is2.x, is3.x)
                            : ((d == 1u) ? float4(is0.y, is1.y, is2.y, is3.y)
                            : ((d == 2u) ? float4(is0.z, is1.z, is2.z, is3.z) : r3));

            float4 v4;
            if (p.is_local != 0u) {
                const float inv_s = 0.4472135955f;
                v4.x = (fma(h0.x, wt0.x, fma(h1.x, wt0.y, fma(h2.x, wt0.z, cur.x * wt0.w))) + cur.x) * inv_s;
                v4.y = (fma(h0.y, wt1.x, fma(h1.y, wt1.y, fma(h2.y, wt1.z, cur.y * wt1.w))) + cur.y) * inv_s;
                v4.z = (fma(h0.z, wt2.x, fma(h1.z, wt2.y, fma(h2.z, wt2.z, cur.z * wt2.w))) + cur.z) * inv_s;
                v4.w = (fma(h0.w, wt3.x, fma(h1.w, wt3.y, fma(h2.w, wt3.z, cur.w * wt3.w))) + cur.w) * inv_s;
            } else {
                const float4 dyn  = k_dyn[ulong(m) * p.n_kv + hkv];
                const float4 t0   = wt0 + dyn, t1 = wt1 + dyn, t2 = wt2 + dyn, t3 = wt3 + dyn;
                const float  inv3 = 1.0f / 3.0f;
                v4.x = (fma(cur.x, t0.x, fma(h2.x, t0.y, fma(h1.x, t0.z, h0.x * t0.w))) + cur.x) * inv3;
                v4.y = (fma(cur.y, t1.x, fma(h2.y, t1.y, fma(h1.y, t1.z, h0.y * t1.w))) + cur.y) * inv3;
                v4.z = (fma(cur.z, t2.x, fma(h2.z, t2.y, fma(h1.z, t2.z, h0.z * t2.w))) + cur.z) * inv3;
                v4.w = (fma(cur.w, t3.x, fma(h2.w, t3.y, fma(h1.w, t3.z, h0.w * t3.w))) + cur.w) * inv3;
            }
            last0 = float4(h1.x, h2.x, cur.x, 0.0f);
            last1 = float4(h1.y, h2.y, cur.y, 0.0f);
            last2 = float4(h1.z, h2.z, cur.z, 0.0f);
            last3 = float4(h1.w, h2.w, cur.w, 0.0f);
            if ((p.save_snap_mask & (1u << m)) != 0u && m < 8u) {
                const ulong sb = ulong(m) * sstr + c0;
                snaps[sb + 0u] = last0; snaps[sb + 1u] = last1;
                snaps[sb + 2u] = last2; snaps[sb + 3u] = last3;
            }
            const float tot   = simd_sum(dot(v4, v4));
            const float k_fac = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps));
            const ulong hs    = ulong(hkv) * p.kv_stride + slot;
            if (p.kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0 && p.is_local == 0u) {
                sik_store_kv_q8_head_sg32(v4 * k_fac, kcc4, k_scales, hs, lane);
            } else {
                kc4[hs * 32ul + lane] = sik_clamp_f16x4(v4 * k_fac);
            }
        }
        conv_state[c0 + 0u] = last0; conv_state[c0 + 1u] = last1;
        conv_state[c0 + 2u] = last2; conv_state[c0 + 3u] = last3;
    } else if (idx < b3) {
        const uint   hkv   = idx - b2;
        const uint   c0    = hkv * 128u + lane * 4u;
        const ulong  s_off = ulong(p.v_state_off_f4);
        const float4 wt0 = v_sconv_w[c0 + 0u], wt1 = v_sconv_w[c0 + 1u];
        const float4 wt2 = v_sconv_w[c0 + 2u], wt3 = v_sconv_w[c0 + 3u];
        const float4 is0 = conv_state[s_off + c0 + 0u], is1 = conv_state[s_off + c0 + 1u];
        const float4 is2 = conv_state[s_off + c0 + 2u], is3 = conv_state[s_off + c0 + 3u];

        device const float4* vraw4 = reinterpret_cast<device const float4*>(v_raw);
        device half4*        vc4   = reinterpret_cast<device half4*>(v_cache);
        device char4*        vcc4  = reinterpret_cast<device char4*>(v_cache);
        float4 last0 = is0, last1 = is1, last2 = is2, last3 = is3;

        for (uint m = 0u; m < p.M_valid; ++m) {
            const uint   pos  = p.seq_pos_start + m;
            const uint   slot = (p.is_local != 0u) ? (pos & (p.kv_stride - 1u)) : pos;
            const float4 cur  = vraw4[(ulong(m) * p.n_kv + hkv) * 32ul + lane];

            const uint d  = tree.depth[m];
            const uint p1 = tree.parent[m] & 7u;
            const uint p2 = tree.p2[m] & 7u;
            const uint p3 = tree.p3[m] & 7u;
            const float4 r1 = (d >= 1u) ? vraw4[(ulong(p1) * p.n_kv + hkv) * 32ul + lane] : float4(0.0f);
            const float4 r2 = (d >= 2u) ? vraw4[(ulong(p2) * p.n_kv + hkv) * 32ul + lane] : float4(0.0f);
            const float4 r3 = (d >= 3u) ? vraw4[(ulong(p3) * p.n_kv + hkv) * 32ul + lane] : float4(0.0f);

            const float4 h2 = (d == 0u) ? float4(is0.z, is1.z, is2.z, is3.z) : r1;
            const float4 h1 = (d == 0u) ? float4(is0.y, is1.y, is2.y, is3.y)
                            : ((d == 1u) ? float4(is0.z, is1.z, is2.z, is3.z) : r2);
            const float4 h0 = (d == 0u) ? float4(is0.x, is1.x, is2.x, is3.x)
                            : ((d == 1u) ? float4(is0.y, is1.y, is2.y, is3.y)
                            : ((d == 2u) ? float4(is0.z, is1.z, is2.z, is3.z) : r3));

            float4 v4;
            if (p.is_local != 0u) {
                const float inv_s = 0.4472135955f;
                v4.x = (fma(h0.x, wt0.x, fma(h1.x, wt0.y, fma(h2.x, wt0.z, cur.x * wt0.w))) + cur.x) * inv_s;
                v4.y = (fma(h0.y, wt1.x, fma(h1.y, wt1.y, fma(h2.y, wt1.z, cur.y * wt1.w))) + cur.y) * inv_s;
                v4.z = (fma(h0.z, wt2.x, fma(h1.z, wt2.y, fma(h2.z, wt2.z, cur.z * wt2.w))) + cur.z) * inv_s;
                v4.w = (fma(h0.w, wt3.x, fma(h1.w, wt3.y, fma(h2.w, wt3.z, cur.w * wt3.w))) + cur.w) * inv_s;
            } else {
                const float4 dyn  = v_dyn[ulong(m) * p.n_kv + hkv];
                const float4 t0   = wt0 + dyn, t1 = wt1 + dyn, t2 = wt2 + dyn, t3 = wt3 + dyn;
                const float  inv3 = 1.0f / 3.0f;
                v4.x = (fma(cur.x, t0.x, fma(h2.x, t0.y, fma(h1.x, t0.z, h0.x * t0.w))) + cur.x) * inv3;
                v4.y = (fma(cur.y, t1.x, fma(h2.y, t1.y, fma(h1.y, t1.z, h0.y * t1.w))) + cur.y) * inv3;
                v4.z = (fma(cur.z, t2.x, fma(h2.z, t2.y, fma(h1.z, t2.z, h0.z * t2.w))) + cur.z) * inv3;
                v4.w = (fma(cur.w, t3.x, fma(h2.w, t3.y, fma(h1.w, t3.z, h0.w * t3.w))) + cur.w) * inv3;
            }
            last0 = float4(h1.x, h2.x, cur.x, 0.0f);
            last1 = float4(h1.y, h2.y, cur.y, 0.0f);
            last2 = float4(h1.z, h2.z, cur.z, 0.0f);
            last3 = float4(h1.w, h2.w, cur.w, 0.0f);
            if ((p.save_snap_mask & (1u << m)) != 0u && m < 8u) {
                const ulong sb = ulong(m) * sstr + s_off + c0;
                snaps[sb + 0u] = last0; snaps[sb + 1u] = last1;
                snaps[sb + 2u] = last2; snaps[sb + 3u] = last3;
            }
            const float tot   = simd_sum(dot(v4, v4));
            const float v_fac = rsqrt(tot / float(p.head_dim) + p.eps);
            const ulong hs    = ulong(hkv) * p.kv_stride + slot;
            if (p.kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0 && p.is_local == 0u) {
                sik_store_kv_q8_head_sg32(v4 * v_fac, vcc4, v_scales, hs, lane);
            } else {
                vc4[hs * 32ul + lane] = sik_clamp_f16x4(v4 * v_fac);
            }
        }
        conv_state[s_off + c0 + 0u] = last0; conv_state[s_off + c0 + 1u] = last1;
        conv_state[s_off + c0 + 2u] = last2; conv_state[s_off + c0 + 3u] = last3;
    } else if (idx < b3 + nq_m) {
        const uint  ri  = idx - b3;
        const uint  m   = ri / p.n_q;
        const uint  hq  = ri - m * p.n_q;
        const uint  pos = p.seq_pos_start + tree.depth[m];
        const float tau = (p.is_local == 0u && p.log_n_floor > 0.0f)
            ? (1.0f + p.log_alpha * log(max(1.0f, float(pos + 1u) / p.log_n_floor))) : 1.0f;
        const ulong off = (ulong(m) * p.n_q + hq) * p.d_rel + lane;
        const float val = (lane < p.d_rel) ? (r_raw[off] * r_scale[ulong(hq) * p.d_rel + lane]) : 0.0f;
        const float tot = simd_sum(val * val);
        const float fac = min(1.0f, p.rel_clip / (sqrt(tot) + p.eps)) * tau;
        if (lane < p.d_rel) r_out[off] = val * fac;
    }
}

// Tree-Aware Single-Pass GQA-3 Attention (`qp = seq_pos_start + depth[m] < 256`, F16 KV, 100.0000% BIT-EXACT w/ M=1!)
kernel void sik_attn_decode_single_gqa3_sg8_tree(
    device const float4*      q_in     [[buffer(0)]],
    device const float*       r_in     [[buffer(1)]],
    device const float*       rel_proj [[buffer(2)]],
    device const half4*       k_cache  [[buffer(3)]],
    device const half4*       v_cache  [[buffer(4)]],
    device const float*       sink_in  [[buffer(5)]],
    device const float4*      gate_in  [[buffer(6)]],
    device half4*             out_f16  [[buffer(7)]],
    device float*             wo_xsum  [[buffer(8)]],
    constant SikAttnParams&   p        [[buffer(9)]],
    constant SikTreeMeta&     tree     [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint hkv = tgid.x;
    const uint m   = tgid.y;
    if (hkv >= p.n_kv || m >= p.M_valid) return;

    const uint P  = p.seq_pos_start;
    const uint qp = P + tree.depth[m];
    if (qp >= 256u) return; // handled by split-KV bucket when straddling pos=256!

    const uint path_pk = tree.path_packed[m];
    const uint ws      = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;

    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        const ulong qrow = ulong(m) * p.n_q + (hkv * 3u + h);
        qv[h] = q_in[qrow * 32ul + lane] * p.attn_scale;
        rl[h] = (lane < p.d_rel) ? r_in[qrow * p.d_rel + lane] : 0.0f;
    }

    device const half4* kh = k_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* vh = v_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    const uint mask = p.kv_stride - 1u;

    for (uint k = ws + sgid; k <= qp; k += 8u) {
        const uint   phys_k  = (k < P) ? k : (P + ((path_pk >> ((k - P) * 4u)) & 0xFu));
        const uint   slot    = (p.is_local != 0u) ? (phys_k & mask) : phys_k;
        const ulong  off     = ulong(slot) * 32ul + lane;
        const float4 kv      = float4(kh[off]);
        const float4 vv      = float4(vh[off]);
        const uint   dist    = qp - k;
        const bool   has_rel = (dist < p.rel_extent);
        const float  relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    threadgroup float  tg_m[3][8], tg_l[3][8];
    threadgroup float4 tg_o[3][8][32];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        if (lane == 0u) { tg_m[h][sgid] = mv[h]; tg_l[h][sgid] = lv[h]; }
        tg_o[h][sgid][lane] = ov[h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid < 3u) {
        const uint  h      = sgid;
        const uint  hq     = hkv * 3u + h;
        const float sink_v = sink_in[ulong(m) * p.n_q + hq];
        float m_max = sink_v;
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) m_max = max(m_max, tg_m[h][s]);
        float  l_tot = exp(sink_v - m_max);
        float4 o_tot = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) {
            if (tg_m[h][s] > -INFINITY) {
                const float w = exp(tg_m[h][s] - m_max);
                l_tot = fma(tg_l[h][s], w, l_tot);
                o_tot = fma(tg_o[h][s][lane], float4(w), o_tot);
            }
        }
        const float  inv_l   = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
        const ulong  out_off = (ulong(m) * p.n_q + hq) * 32ul + lane;
        const float4 gv      = gate_in[out_off];
        const float4 sig_g   = float4(1.0f) / (float4(1.0f) + exp(-gv));
        const half4  oh      = sik_clamp_f16x4((o_tot * inv_l) * sig_g);
        out_f16[out_off]     = oh;
        if (m < 8u) {
            const float bs = sik_reduce16((float(oh.x) + float(oh.y)) + (float(oh.z) + float(oh.w)));
            if ((lane & 15u) == 0u) {
                wo_xsum[ulong(m) * (p.n_q * 2u) + hq * 2u + (lane >> 4u)] = bs;
            }
        }
    }
}

// Tree-Aware Register-Direct GQA-3 Split-KV Attention (`qp >= 256`, F16 KV, 100.0000% BIT-EXACT w/ M=1!)
kernel void sik_attn_decode_split_gqa3_reg_tree(
    device const float4*      q_in     [[buffer(0)]],
    device const float*       r_in     [[buffer(1)]],
    device const float*       rel_proj [[buffer(2)]],
    device const half4*       k_cache  [[buffer(3)]],
    device const half4*       v_cache  [[buffer(4)]],
    device float*             part_m   [[buffer(5)]],
    device float*             part_l   [[buffer(6)]],
    device float4*            part_o   [[buffer(7)]],
    constant SikAttnParams&   p        [[buffer(8)]],
    constant SikTreeMeta&     tree     [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint c   = p.c_batch_first + tgid.x;
    const uint hkv = tgid.y;
    const uint m   = tgid.z;
    if (hkv >= p.n_kv || m >= p.M_valid) return;

    const uint P  = p.seq_pos_start;
    const uint qp = P + tree.depth[m];
    const uint B  = p.chunk_size;
    if (qp < 256u || sik_attn_chunk_size(qp, p.is_local) != B) return;

    const uint ws     = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
    const uint c_last = qp / B;
    if (c < (ws / B) || c > c_last) return;

    const uint kv_beg  = c * B;
    const uint kv_end  = min(kv_beg + B, qp + 1u);
    const uint kv_lo   = max(kv_beg, ws);
    const uint path_pk = tree.path_packed[m];

    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        const ulong qrow = ulong(m) * p.n_q + (hkv * 3u + h);
        qv[h] = q_in[qrow * 32ul + lane] * p.attn_scale;
        rl[h] = (lane < p.d_rel) ? r_in[qrow * p.d_rel + lane] : 0.0f;
    }

    device const half4* kh = k_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* vh = v_cache + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    const uint mask = p.kv_stride - 1u;

    uint k = kv_beg + sgid;
    // Fast 2-token prefetched prefix loop whenever chunk lies strictly below P (99% of chunks -> ZERO overhead!)
    if (kv_beg >= kv_lo && kv_end <= P) {
        for (; k + 4u < kv_end; k += 8u) {
            const uint   s0   = (p.is_local != 0u) ? (k & mask) : k;
            const uint   s1   = (p.is_local != 0u) ? ((k + 4u) & mask) : (k + 4u);
            const ulong  o0   = ulong(s0) * 32ul + lane;
            const ulong  o1   = ulong(s1) * 32ul + lane;
            const float4 kv0  = float4(kh[o0]), vv0 = float4(vh[o0]);
            const float4 kv1  = float4(kh[o1]), vv1 = float4(vh[o1]);
            const uint   d0   = qp - k, d1 = d0 - 4u;
            const bool   hr0  = (d0 < p.rel_extent), hr1 = (d1 < p.rel_extent);
            const float  rw0  = (hr0 && lane < p.d_rel) ? rel_proj[ulong(d0) * p.d_rel + lane] : 0.0f;
            const float  rw1  = (hr1 && lane < p.d_rel) ? rel_proj[ulong(d1) * p.d_rel + lane] : 0.0f;
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 3u; ++h) {
                float p0 = dot(qv[h], kv0);
                if (hr0 && lane < p.d_rel) p0 = fma(rl[h], rw0, p0);
                const float sc0 = simd_sum(p0);
                const float mn0 = max(mv[h], sc0);
                const float al0 = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn0);
                const float w0  = exp(sc0 - mn0);
                mv[h] = mn0; lv[h] = fma(lv[h], al0, w0); ov[h] = fma(ov[h], al0, w0 * vv0);

                float p1 = dot(qv[h], kv1);
                if (hr1 && lane < p.d_rel) p1 = fma(rl[h], rw1, p1);
                const float sc1 = simd_sum(p1);
                const float mn1 = max(mv[h], sc1);
                const float al1 = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn1);
                const float w1  = exp(sc1 - mn1);
                mv[h] = mn1; lv[h] = fma(lv[h], al1, w1); ov[h] = fma(ov[h], al1, w1 * vv1);
            }
        }
    }
    for (; k < kv_end; k += 4u) {
        if (k < kv_lo) continue;
        const uint   phys_k  = (k < P) ? k : (P + ((path_pk >> ((k - P) * 4u)) & 0xFu));
        const uint   slot    = (p.is_local != 0u) ? (phys_k & mask) : phys_k;
        const ulong  off     = ulong(slot) * 32ul + lane;
        const float4 kv      = float4(kh[off]);
        const float4 vv      = float4(vh[off]);
        const uint   dist    = qp - k;
        const bool   has_rel = (dist < p.rel_extent);
        const float  relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    threadgroup float  tg_m[4], tg_l[4];
    threadgroup float4 tg_o[4][32];
    const uint s_slot = c - (ws / B);
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        if (h > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) { tg_m[sgid] = mv[h]; tg_l[sgid] = lv[h]; }
        tg_o[sgid][lane] = ov[h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float mm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
            float  ll = 0.0f;
            float4 oo(0.0f);
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 4u; ++s) {
                if (tg_m[s] > -INFINITY) {
                    const float w = exp(tg_m[s] - mm);
                    ll = fma(tg_l[s], w, ll);
                    oo = fma(tg_o[s][lane], float4(w), oo);
                }
            }
            const uint  hq  = hkv * 3u + h;
            const ulong idx = (ulong(m) * p.n_q + hq) * p.max_splits + s_slot;
            if (lane == 0u) { part_m[idx] = mm; part_l[idx] = ll; }
            part_o[idx * 32ul + lane] = oo;
        }
    }
}

// Tree-Aware Q8_0 Global KV Decode Split-KV (`100.0000%` BIT-EXACT w/ M=1!)
kernel void sik_attn_decode_q8_reg_tree(
    device const float4*      q_in     [[buffer(0)]],
    device const float*       r_in     [[buffer(1)]],
    device const float*       rel_proj [[buffer(2)]],
    device const char4*       k_cache  [[buffer(3)]],
    device const char4*       v_cache  [[buffer(4)]],
    device float*             part_m   [[buffer(5)]],
    device float*             part_l   [[buffer(6)]],
    device float4*            part_o   [[buffer(7)]],
    constant SikAttnParams&   p        [[buffer(8)]],
    device const half*        k_scales [[buffer(9)]],
    device const half*        v_scales [[buffer(10)]],
    constant SikTreeMeta&     tree     [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint c   = p.c_batch_first + tgid.x;
    const uint hkv = tgid.y;
    const uint m   = tgid.z;
    if (hkv >= p.n_kv || m >= p.M_valid) return;

    const uint P  = p.seq_pos_start;
    const uint qp = P + tree.depth[m];
    const uint B  = p.chunk_size;
    if (sik_attn_chunk_size(qp, 0u) != B || c > (qp / B)) return;

    const uint kv_beg  = c * B;
    const uint kv_end  = min(kv_beg + B, qp + 1u);
    const uint path_pk = tree.path_packed[m];

    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        const ulong qrow = ulong(m) * p.n_q + (hkv * 3u + h);
        qv[h] = q_in[qrow * 32ul + lane] * p.attn_scale;
        rl[h] = (lane < p.d_rel) ? r_in[qrow * p.d_rel + lane] : 0.0f;
    }

    device const char4* k_head = k_cache  + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const char4* v_head = v_cache  + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half*  k_s    = k_scales + ulong(hkv) * ulong(p.kv_stride);
    device const half*  v_s    = v_scales + ulong(hkv) * ulong(p.kv_stride);

    for (uint k = kv_beg + sgid; k < kv_end; k += 4u) {
        const uint   phys_k  = (k < P) ? k : (P + ((path_pk >> ((k - P) * 4u)) & 0xFu));
        const ulong  off     = ulong(phys_k) * 32ul + lane;
        const float4 kv      = float4(half4(k_head[off]) * k_s[phys_k]);
        const float4 vv      = float4(half4(v_head[off]) * v_s[phys_k]);
        const uint   dist    = qp - k;
        const bool   has_rel = (dist < p.rel_extent);
        const float  relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    threadgroup float  tg_m[4], tg_l[4];
    threadgroup float4 tg_o[4][32];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        if (h > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) { tg_m[sgid] = mv[h]; tg_l[sgid] = lv[h]; }
        tg_o[sgid][lane] = ov[h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float mm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
            float  ll = 0.0f;
            float4 oo(0.0f);
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 4u; ++s) {
                if (tg_m[s] > -INFINITY) {
                    const float w = exp(tg_m[s] - mm);
                    ll = fma(tg_l[s], w, ll);
                    oo = fma(tg_o[s][lane], float4(w), oo);
                }
            }
            const uint  hq  = hkv * 3u + h;
            const ulong idx = (ulong(m) * p.n_q + hq) * p.max_splits + c;
            if (lane == 0u) { part_m[idx] = mm; part_l[idx] = ll; }
            part_o[idx * 32ul + lane] = oo;
        }
    }
}

// Tree-Aware Split-KV Reduce + Virtual Sink + Sigmoid Gate + `wo_xsum` (`p.c_batch_first != 0` => skip `qp < 256`!)
kernel void sik_attn_decode_reduce_tree(
    device const float*       part_m   [[buffer(0)]],
    device const float*       part_l   [[buffer(1)]],
    device const float4*      part_o   [[buffer(2)]],
    device const float*       sink_in  [[buffer(3)]],
    device const float4*      gate_in  [[buffer(4)]],
    device half4*             out_f16  [[buffer(5)]],
    constant SikAttnParams&   p        [[buffer(6)]],
    device float*             wo_xsum  [[buffer(7)]],
    constant SikTreeMeta&     tree     [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint hq = tgid.x;
    const uint m  = tgid.y;
    if (hq >= p.n_q || m >= p.M_valid) return;

    const uint q_pos = p.seq_pos_start + tree.depth[m];
    if (p.c_batch_first != 0u && q_pos < 256u) return; // handled by single_gqa3_sg8_tree!

    const uint  B         = sik_attn_chunk_size(q_pos, p.is_local);
    const uint  win_start = (p.is_local != 0u && q_pos + 1u > p.swa_win) ? (q_pos + 1u - p.swa_win) : 0u;
    const uint  my_splits = (q_pos / B) - (win_start / B) + 1u;
    const ulong head_base = (ulong(m) * p.n_q + hq) * p.max_splits;

    const float sink_v = sink_in[ulong(m) * p.n_q + hq];
    const float m0     = (lane < my_splits) ? part_m[head_base + lane] : -INFINITY;
    const float m1     = (lane + 32u < my_splits) ? part_m[head_base + lane + 32u] : -INFINITY;
    const float m_max  = max(sink_v, simd_max(max(m0, m1)));

    const float w0     = (lane < my_splits && m0 > -INFINITY) ? exp(m0 - m_max) : 0.0f;
    const float w1     = (lane + 32u < my_splits && m1 > -INFINITY) ? exp(m1 - m_max) : 0.0f;
    const float l0     = (w0 > 0.0f) ? (part_l[head_base + lane] * w0) : 0.0f;
    const float l1     = (w1 > 0.0f) ? (part_l[head_base + lane + 32u] * w1) : 0.0f;
    const float l_tot  = exp(sink_v - m_max) + simd_sum(l0 + l1);
    const float inv_l  = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;

    float4 o_tot(0.0f);
    for (uint s = 0u; s < my_splits; ++s) {
        const float w = (s < 32u) ? simd_broadcast(w0, ushort(s)) : simd_broadcast(w1, ushort(s - 32u));
        if (w > 0.0f) {
            o_tot = fma(part_o[(head_base + s) * 32ul + lane], float4(w), o_tot);
        }
    }

    const ulong  out_off = (ulong(m) * p.n_q + hq) * 32ul + lane;
    const float4 gv      = gate_in[out_off];
    const float4 sig_g   = float4(1.0f) / (float4(1.0f) + exp(-gv));
    const half4  h       = sik_clamp_f16x4((o_tot * inv_l) * sig_g);
    out_f16[out_off]     = h;
    if (m < 8u) {
        const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
        if ((lane & 15u) == 0u) {
            wo_xsum[ulong(m) * (p.n_q * 2u) + hq * 2u + (lane >> 4u)] = bs;
        }
    }
}

// Tree-Aware Fused Q4_A `Wo + Parent-Indexed attn_sconv + 8-Slot Snapshots + Residual` (works w/ `--no-refusal` alt_wo!)
template <uint M_FIX>
inline void sik_q4a_wo_sconv_tree_fixed(
    device const uint2*        w_u2,
    device const half*         w_s,
    device const half*         w_b,
    device const half*         x_in,
    device const float*        x_sum,
    device const float4*       sconv_w,
    device float4*             conv_state,
    device float*              out_or_cur,
    device float*              block_dst,
    device float4*             snaps,
    constant SikWoSconvParams& p,
    constant SikTreeMeta&      tree,
    threadgroup float* tg_row,
    uint r0,
    uint lane
) {
    const uint r1      = min(r0 + 1u, p.rows - 1u);
    const uint k16     = p.cols >> 4u;
    const uint nblk    = p.cols >> 6u;
    const bool do_bias = ((lane & 3u) == 0u);
    device const uint2* rw0 = w_u2 + ulong(r0) * k16;
    device const uint2* rw1 = w_u2 + ulong(r1) * k16;
    device const half*  rs0 = w_s  + ulong(r0) * nblk;
    device const half*  rs1 = w_s  + ulong(r1) * nblk;
    device const half*  rb0 = w_b  + ulong(r0) * nblk;
    device const half*  rb1 = w_b  + ulong(r1) * nblk;

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint i = lane; i < k16; i += 32u) {
        const uint2  q0  = rw0[i], q1 = rw1[i];
        const uint   grp = i >> 2u;
        const float2 sc  = float2(float(rs0[grp]), float(rs1[grp]));
        half4 a0, a1, a2, a3, b0, b1, b2, b3;
        sik_unpack_u4_u32_h4(q0.x, a0, a1); sik_unpack_u4_u32_h4(q0.y, a2, a3);
        sik_unpack_u4_u32_h4(q1.x, b0, b1); sik_unpack_u4_u32_h4(q1.y, b2, b3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + i * 16u);
            const half4 x0 = xm[0], x1 = xm[1], x2 = xm[2], x3 = xm[3];
            const half4 dh0 = fma(a0, x0, a1 * x1) + fma(a2, x2, a3 * x3);
            const half4 dh1 = fma(b0, x0, b1 * x1) + fma(b2, x2, b3 * x3);
            acc[m].x = fma(sc.x, (float(dh0.x) + float(dh0.y)) + (float(dh0.z) + float(dh0.w)), acc[m].x);
            acc[m].y = fma(sc.y, (float(dh1.x) + float(dh1.y)) + (float(dh1.z) + float(dh1.w)), acc[m].y);
        }
        if (do_bias) {
            const float2 bi = float2(float(rb0[grp]), float(rb1[grp]));
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                acc[m] = fma(bi, float2(x_sum[ulong(m) * nblk + grp]), acc[m]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float vx = simd_sum(acc[m].x);
        const float vy = simd_sum(acc[m].y);
        if (lane < 2u) tg_row[lane * 8u + m] = (lane == 0u) ? vx : vy;
    }

    if (lane < 2u) {
        const uint c = r0 + lane;
        if (c < p.rows) {
            threadgroup const float* tv = tg_row + lane * 8u;
            const uint   C     = p.rows;
            const ulong  sstr  = ulong(p.snap_stride_f4);
            const float4 w     = sconv_w[c];
            const float  inv_s = 0.4472135955f;
            const float4 ist   = conv_state[c];
            float4 last_st     = ist;
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                const float cur = tv[m];
                const uint  d   = tree.depth[m];
                const uint  p1  = tree.parent[m] & 7u;
                const uint  p2  = tree.p2[m] & 7u;
                const uint  p3  = tree.p3[m] & 7u;
                const float h2  = (d == 0u) ? ist.z : tv[p1];
                const float h1  = (d == 0u) ? ist.y : ((d == 1u) ? ist.z : tv[p2]);
                const float h0  = (d == 0u) ? ist.x : ((d == 1u) ? ist.y : ((d == 2u) ? ist.z : tv[p3]));
                const float sum = fma(h0, w.x, fma(h1, w.y, fma(h2, w.z, cur * w.w)));
                const float y   = (sum + cur) * inv_s;
                last_st = float4(h1, h2, cur, 0.0f);
                if ((p.save_snap_mask & (1u << m)) != 0u && m < 8u) {
                    snaps[ulong(m) * sstr + c] = last_st;
                }
                const ulong off = ulong(m) * C + c;
                if (p.res_mode == 2u) {
                    block_dst[off]  = out_or_cur[off];
                    out_or_cur[off] = y;
                } else if (p.res_mode == 1u) {
                    out_or_cur[off] = out_or_cur[off] + y;
                } else {
                    out_or_cur[off] = y;
                }
            }
            conv_state[c] = last_st;
        }
    }
}

kernel void sik_q4a_gemv_wo_sconv_tree(
    device const uint2*        w_u2       [[buffer(0)]],
    device const half*         w_s        [[buffer(1)]],
    device const half*         w_b        [[buffer(2)]],
    device const half*         x_in       [[buffer(3)]],
    device const float*        x_sum      [[buffer(4)]],
    device const float4*       sconv_w    [[buffer(5)]],
    device float4*             conv_state [[buffer(6)]],
    device float*              out_or_cur [[buffer(7)]],
    device float*              block_dst  [[buffer(8)]],
    device float4*             snaps      [[buffer(9)]],
    constant SikWoSconvParams& p          [[buffer(10)]],
    constant SikTreeMeta&      tree       [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_val[4][16]; // 4 SGs x 2 rows x 8 nodes = 256 B (0 stack spills!)
    const uint r0 = (tgid * sgs + sgid) * 2u;
    if (r0 >= p.rows) return;
    #define SIK_WOSCT_CASE(MF) { sik_q4a_wo_sconv_tree_fixed<MF>(w_u2, w_s, w_b, x_in, x_sum, sconv_w, conv_state, out_or_cur, block_dst, snaps, p, tree, tg_val[sgid], r0, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_WOSCT_CASE(1u) case 2u: SIK_WOSCT_CASE(2u) case 3u: SIK_WOSCT_CASE(3u) case 4u: SIK_WOSCT_CASE(4u)
        case 5u: SIK_WOSCT_CASE(5u) case 6u: SIK_WOSCT_CASE(6u) case 7u: SIK_WOSCT_CASE(7u) case 8u: SIK_WOSCT_CASE(8u)
        default: break;
    }
    #undef SIK_WOSCT_CASE
}

// Tree-Aware Unfused Static SConv (v1 FP16 / Q8_0 `wo` compatibility!)
kernel void sik_sconv_static_tree(
    device const float*       x_in       [[buffer(0)]],
    device const float4*      weight     [[buffer(1)]],
    device float4*            conv_state [[buffer(2)]],
    device float*             out_or_cur [[buffer(3)]],
    device float*             block_dst  [[buffer(4)]],
    device float4*            snaps      [[buffer(5)]],
    constant SikSconvParams&  p          [[buffer(6)]],
    constant SikTreeMeta&     tree       [[buffer(7)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= p.channels) return;
    const uint   C     = p.channels;
    const float4 w     = weight[c];
    const float  inv_s = 0.4472135955f;
    const ulong  sstr  = ulong(p.v_state_off_f4);
    const float4 ist   = conv_state[c];
    float4 last_st     = ist;

    for (uint m = 0u; m < p.M; ++m) {
        const ulong off = ulong(m) * C + c;
        const float cur = x_in[off];
        const uint  d   = tree.depth[m];
        const float h2  = (d == 0u) ? ist.z : x_in[ulong(tree.parent[m] & 7u) * C + c];
        const float h1  = (d == 0u) ? ist.y : ((d == 1u) ? ist.z : x_in[ulong(tree.p2[m] & 7u) * C + c]);
        const float h0  = (d == 0u) ? ist.x : ((d == 1u) ? ist.y : ((d == 2u) ? ist.z : x_in[ulong(tree.p3[m] & 7u) * C + c]));
        const float sum = fma(h0, w.x, fma(h1, w.y, fma(h2, w.z, cur * w.w)));
        const float y   = (sum + cur) * inv_s;
        last_st = float4(h1, h2, cur, 0.0f);
        if ((p.save_snap_mask & (1u << m)) != 0u && m < 8u) {
            snaps[ulong(m) * sstr + c] = last_st;
        }
        if (p.res_mode == 2u) {
            block_dst[off]  = out_or_cur[off];
            out_or_cur[off] = y;
        } else if (p.res_mode == 1u) {
            out_or_cur[off] = out_or_cur[off] + y;
        } else {
            out_or_cur[off] = y;
        }
    }
    conv_state[c] = last_st;
}

// Tree-Aware Fused `MoE Top-8 Combine + Parent-Indexed mlp_sconv + 8-Slot Snapshots + Residual`!
// Uses `threadgroup float tg_cur[8][64]` (2 KiB) for zero VGPR bloat (`1024` th/TG, 0 spills)!
kernel void sik_moe_combine_sconv_tree(
    device const float*       down_part  [[buffer(0)]],
    device const float*       topk_w     [[buffer(1)]],
    device const float4*      weight     [[buffer(2)]],
    device float4*            conv_state [[buffer(3)]],
    device float*             cur_io     [[buffer(4)]],
    device float4*            snaps      [[buffer(5)]],
    constant SikSconvParams&  p          [[buffer(6)]],
    constant SikTreeMeta&     tree       [[buffer(7)]],
    uint c   [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float tg_tw[64];
    threadgroup float tg_cur[8][64];
    const uint n_tw = min(p.M * p.top_k, 64u);
    if (tid < n_tw) tg_tw[tid] = topk_w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (c >= p.channels) return;
    const uint   C     = p.channels;
    const uint   K     = p.top_k;
    const ulong  sstr  = ulong(p.v_state_off_f4);
    const float4 w     = weight[c];
    const float  inv_s = 0.4472135955f;
    const float4 ist   = conv_state[c];
    float4 last_st     = ist;

    for (uint m = 0u; m < p.M; ++m) {
        float cur = 0.0f;
        const ulong base_pair = ulong(m) * K;
        if (K == 8u && m < 8u) {
            const uint b = m * 8u;
            cur = fma(tg_tw[b + 0u], down_part[(base_pair + 0u) * C + c], cur);
            cur = fma(tg_tw[b + 1u], down_part[(base_pair + 1u) * C + c], cur);
            cur = fma(tg_tw[b + 2u], down_part[(base_pair + 2u) * C + c], cur);
            cur = fma(tg_tw[b + 3u], down_part[(base_pair + 3u) * C + c], cur);
            cur = fma(tg_tw[b + 4u], down_part[(base_pair + 4u) * C + c], cur);
            cur = fma(tg_tw[b + 5u], down_part[(base_pair + 5u) * C + c], cur);
            cur = fma(tg_tw[b + 6u], down_part[(base_pair + 6u) * C + c], cur);
            cur = fma(tg_tw[b + 7u], down_part[(base_pair + 7u) * C + c], cur);
        } else {
            for (uint k = 0u; k < K; ++k) {
                cur = fma(topk_w[base_pair + k], down_part[(base_pair + k) * C + c], cur);
            }
        }
        tg_cur[m][tid] = cur;
        const uint  d  = tree.depth[m];
        const float h2 = (d == 0u) ? ist.z : tg_cur[tree.parent[m] & 7u][tid];
        const float h1 = (d == 0u) ? ist.y : ((d == 1u) ? ist.z : tg_cur[tree.p2[m] & 7u][tid]);
        const float h0 = (d == 0u) ? ist.x : ((d == 1u) ? ist.y : ((d == 2u) ? ist.z : tg_cur[tree.p3[m] & 7u][tid]));
        const float sum = fma(h0, w.x, fma(h1, w.y, fma(h2, w.z, cur * w.w)));
        const float y   = (sum + cur) * inv_s;

        last_st = float4(h1, h2, cur, 0.0f);
        if ((p.save_snap_mask & (1u << m)) != 0u && m < 8u) {
            snaps[ulong(m) * sstr + c] = last_st;
        }
        const ulong off = ulong(m) * C + c;
        cur_io[off] = cur_io[off] + y;
    }
    conv_state[c] = last_st;
}

// Async `cb_fix` Tree Winning-Path KV Cache & MTP Tap Compaction (runs ONLY when `!path_is_identity`, `< 5 us`!)
struct SikTreeCompactParams {
    uint  num_local_visits;
    uint  num_global_visits;
    uint  n_kv;
    uint  head_dim;
    uint  old_pos;
    uint  swa_stride;
    uint  glob_stride;
    uint  kv_dtype;
    ulong glob_layer_stride_b;
    ulong glob_q8_data_b;
    uint  path_len;    // n_commit = 1 + accepted (<= 8)
    uint  D;
    uint  path[8];     // strictly increasing: 0 = path[0] < path[1] < ... < path[path_len-1]
};

// Grid: `(num_local_visits + num_global_visits, n_kv, 1)`, `32` threads/TG (`1` SG)
kernel void sik_compact_kv_tree(
    device half4*                 swa_k  [[buffer(0)]],
    device half4*                 swa_v  [[buffer(1)]],
    device uchar*                 glob_k [[buffer(2)]],
    device uchar*                 glob_v [[buffer(3)]],
    constant SikTreeCompactParams& p     [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint l   = tgid.x;
    const uint hkv = tgid.y;
    if (hkv >= p.n_kv) return;

    if (l < p.num_local_visits) {
        const ulong lh_base = (ulong(l) * p.n_kv + hkv) * p.swa_stride;
        const uint  mask    = p.swa_stride - 1u;
        for (uint k = 1u; k < p.path_len; ++k) {
            const uint src_node = p.path[k];
            if (src_node == k) continue;
            const uint src_s = (p.old_pos + src_node) & mask;
            const uint dst_s = (p.old_pos + k)        & mask;
            swa_k[(lh_base + dst_s) * 32ul + lane] = swa_k[(lh_base + src_s) * 32ul + lane];
            swa_v[(lh_base + dst_s) * 32ul + lane] = swa_v[(lh_base + src_s) * 32ul + lane];
        }
    } else {
        const uint gl = l - p.num_local_visits;
        if (gl >= p.num_global_visits) return;
        const ulong l_off = ulong(gl) * p.glob_layer_stride_b;
        if (p.kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0) {
            device char4* kq = reinterpret_cast<device char4*>(glob_k + l_off);
            device char4* vq = reinterpret_cast<device char4*>(glob_v + l_off);
            device half*  ks = reinterpret_cast<device half*>(glob_k + l_off + p.glob_q8_data_b);
            device half*  vs = reinterpret_cast<device half*>(glob_v + l_off + p.glob_q8_data_b);
            const ulong h_base = ulong(hkv) * p.glob_stride;
            for (uint k = 1u; k < p.path_len; ++k) {
                const uint src_node = p.path[k];
                if (src_node == k) continue;
                const uint src_s = p.old_pos + src_node;
                const uint dst_s = p.old_pos + k;
                kq[(h_base + dst_s) * 32ul + lane] = kq[(h_base + src_s) * 32ul + lane];
                vq[(h_base + dst_s) * 32ul + lane] = vq[(h_base + src_s) * 32ul + lane];
                if (lane == 0u) {
                    ks[h_base + dst_s] = ks[h_base + src_s];
                    vs[h_base + dst_s] = vs[h_base + src_s];
                }
            }
        } else {
            device half4* kh = reinterpret_cast<device half4*>(glob_k + l_off);
            device half4* vh = reinterpret_cast<device half4*>(glob_v + l_off);
            const ulong h_base = ulong(hkv) * p.glob_stride;
            for (uint k = 1u; k < p.path_len; ++k) {
                const uint src_node = p.path[k];
                if (src_node == k) continue;
                const uint src_s = p.old_pos + src_node;
                const uint dst_s = p.old_pos + k;
                kh[(h_base + dst_s) * 32ul + lane] = kh[(h_base + src_s) * 32ul + lane];
                vh[(h_base + dst_s) * 32ul + lane] = vh[(h_base + src_s) * 32ul + lane];
            }
        }
    }
}

// Grid: `(((D / 4) + 255) / 256, 4, 1)`, `256` threads/TG
kernel void sik_compact_taps_tree(
    device float4*                 taps_chunk [[buffer(0)]], // [M, 4, D/4]
    constant SikTreeCompactParams& p          [[buffer(1)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint d4_cnt = p.D >> 2u;
    const uint idx4   = tgid.x * 256u + tid;
    const uint tap    = tgid.y;
    if (idx4 >= d4_cnt || tap >= 4u) return;

    for (uint k = 1u; k < p.path_len; ++k) {
        const uint src_node = p.path[k];
        if (src_node == k) continue;
        taps_chunk[(ulong(k) * 4ul + tap) * d4_cnt + idx4] =
            taps_chunk[(ulong(src_node) * 4ul + tap) * d4_cnt + idx4];
    }
}

/* ============================================================================
 * 12. True GPU-Batched Continuous-Batching Multi-Stream Decode (`B = 1..64`)
 *     Independent per-stream KV caches (FP16 & Q8_0), SConv rings, seq_pos,
 *     sliding-window penalties, and heterogeneous sampling via GPU pointer
 *     tables — 100.0000% bit-exact with isolated solo `K=0` decode!
 * ============================================================================ */

struct SikBatchSlotMeta {
    device uchar*  swa_k;      // [slot->b_swa_k_cache gpuAddress]
    device uchar*  swa_v;      // [slot->b_swa_v_cache gpuAddress]
    device uchar*  glob_k;     // [slot->b_glob_k_cache gpuAddress]
    device uchar*  glob_v;     // [slot->b_glob_v_cache gpuAddress]
    device float4* sconv;      // [slot->b_sconv_state gpuAddress]
    uint           seq_pos;    // stream b's independent seq_pos
    uint           pad0;
};

struct SikBatchAttnPrepParams {
    uint  B;
    uint  n_q;
    uint  n_kv;
    uint  head_dim;
    uint  d_rel;
    uint  kv_stride;
    uint  is_local;
    uint  kv_dtype;
    uint  sconv_k_off_f4;
    uint  v_state_off_f4;
    uint  pad0;
    uint  pad1;
    ulong kv_lyr_off;
    ulong kv_q8_data_b;
    float qk_clip;
    float rel_clip;
    float eps;
    float log_alpha;
    float log_n_floor;
    float pad2;
};

struct SikBatchAttnParams {
    uint  B;
    uint  n_q;
    uint  n_kv;
    uint  head_dim;
    uint  d_rel;
    uint  rel_extent;
    uint  swa_win;
    uint  is_local;
    uint  kv_stride;
    uint  max_splits;
    uint  skip_under_256;
    uint  pad0;
    ulong kv_lyr_off;
    ulong kv_q8_data_b;
    float attn_scale;
    uint  pad1;
};

struct SikBatchSconvParams {
    uint B;
    uint channels;
    uint res_mode;
    uint sconv_off_f4;
    uint top_k;
    uint cols;
    uint pad0;
    uint pad1;
};

struct SikBatchPenItem {
    uint  pen_off;
    uint  pen_len;
    float pres_pen;
    float freq_pen;
    float rep_pen;
    uint  pad0;
};

struct SikBatchPenParams {
    uint B_chunk;
    uint vocab_size;
    uint item_offset;
    uint pad0;
};

struct SikBatchSampleItem {
    uint  greedy;
    float inv_temp;
    uint  top_k;
    float top_p;
    uint  rng_lo;
    uint  rng_hi;
    uint  pad0;
    uint  pad1;
};

struct SikBatchSampleParams {
    uint B_chunk;
    uint vocab_size;
    uint item_offset;
    uint pad0;
};

// Per-Stream Independent Fused `SConvKV + QKVR-Prep + KV Store` (FP16 SWA/Global & Q8_0 Global!)
// Grid: `(2 * n_q + 2 * n_kv, B, 1)` = `(64, B, 1)`, `32` threads/TG (`1` SG, ZERO TG mem, ZERO barriers!)
kernel void sik_sconv_kv_qkvr_prep_batched(
    device const float*              q_raw     [[buffer(0)]],
    device const float*              k_raw     [[buffer(1)]],
    device const float*              v_raw     [[buffer(2)]],
    device const float*              r_raw     [[buffer(3)]],
    device const float4*             k_sconv_w [[buffer(4)]],
    device const float4*             v_sconv_w [[buffer(5)]],
    device const float4*             k_dyn     [[buffer(6)]],
    device const float4*             v_dyn     [[buffer(7)]],
    device const float*              q_scale   [[buffer(8)]],
    device const float*              r_scale   [[buffer(9)]],
    device float*                    q_out     [[buffer(10)]],
    device float*                    r_out     [[buffer(11)]],
    constant SikBatchSlotMeta*       slots     [[buffer(12)]],
    constant SikBatchAttnPrepParams& p         [[buffer(13)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint idx = tgid.x;
    const uint b   = tgid.y;
    if (b >= p.B) return;

    const SikBatchSlotMeta sm  = slots[b];
    const uint             pos = sm.seq_pos;
    const uint             b1  = p.n_q;
    const uint             b2  = b1 + p.n_kv;
    const uint             b3  = b2 + p.n_kv;

    if (idx < b1) {
        const uint  hq   = idx;
        const float tau  = (p.is_local == 0u && p.log_n_floor > 0.0f)
            ? (1.0f + p.log_alpha * log(max(1.0f, float(pos + 1u) / p.log_n_floor))) : 1.0f;
        const ulong off4 = (ulong(b) * p.n_q + hq) * 32ul + lane;
        const float4 qv  = reinterpret_cast<device const float4*>(q_raw)[off4]
                         * reinterpret_cast<device const float4*>(q_scale)[ulong(hq) * 32ul + lane];
        const float tot  = simd_sum(dot(qv, qv));
        const float fac  = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps)) * tau;
        reinterpret_cast<device float4*>(q_out)[off4] = qv * fac;
    } else if (idx < b2) {
        const uint    hkv   = idx - b1;
        const uint    c0    = hkv * 128u + lane * 4u;
        device float4* st_p = sm.sconv + ulong(p.sconv_k_off_f4);
        const float4  wt0   = k_sconv_w[c0 + 0u], wt1 = k_sconv_w[c0 + 1u];
        const float4  wt2   = k_sconv_w[c0 + 2u], wt3 = k_sconv_w[c0 + 3u];
        const float4  st0   = st_p[c0 + 0u],      st1 = st_p[c0 + 1u];
        const float4  st2   = st_p[c0 + 2u],      st3 = st_p[c0 + 3u];
        const float4  cur   = reinterpret_cast<device const float4*>(k_raw)[(ulong(b) * p.n_kv + hkv) * 32ul + lane];

        float4 v4;
        if (p.is_local != 0u) {
            const float inv_s = 0.4472135955f;
            v4.x = (fma(st0.x, wt0.x, fma(st0.y, wt0.y, fma(st0.z, wt0.z, cur.x * wt0.w))) + cur.x) * inv_s;
            v4.y = (fma(st1.x, wt1.x, fma(st1.y, wt1.y, fma(st1.z, wt1.z, cur.y * wt1.w))) + cur.y) * inv_s;
            v4.z = (fma(st2.x, wt2.x, fma(st2.y, wt2.y, fma(st2.z, wt2.z, cur.z * wt2.w))) + cur.z) * inv_s;
            v4.w = (fma(st3.x, wt3.x, fma(st3.y, wt3.y, fma(st3.z, wt3.z, cur.w * wt3.w))) + cur.w) * inv_s;
        } else {
            const float4 dyn  = k_dyn[ulong(b) * p.n_kv + hkv];
            const float4 t0   = wt0 + dyn, t1 = wt1 + dyn, t2 = wt2 + dyn, t3 = wt3 + dyn;
            const float  inv3 = 1.0f / 3.0f;
            v4.x = (fma(cur.x, t0.x, fma(st0.z, t0.y, fma(st0.y, t0.z, st0.x * t0.w))) + cur.x) * inv3;
            v4.y = (fma(cur.y, t1.x, fma(st1.z, t1.y, fma(st1.y, t1.z, st1.x * t1.w))) + cur.y) * inv3;
            v4.z = (fma(cur.z, t2.x, fma(st2.z, t2.y, fma(st2.y, t2.z, st2.x * t2.w))) + cur.z) * inv3;
            v4.w = (fma(cur.w, t3.x, fma(st3.z, t3.y, fma(st3.y, t3.z, st3.x * t3.w))) + cur.w) * inv3;
        }
        st_p[c0 + 0u] = float4(st0.y, st0.z, cur.x, 0.0f);
        st_p[c0 + 1u] = float4(st1.y, st1.z, cur.y, 0.0f);
        st_p[c0 + 2u] = float4(st2.y, st2.z, cur.z, 0.0f);
        st_p[c0 + 3u] = float4(st3.y, st3.z, cur.w, 0.0f);

        const float  tot   = simd_sum(dot(v4, v4));
        const float  k_fac = min(1.0f, p.qk_clip / (sqrt(tot) + p.eps));
        const uint   slot  = (p.is_local != 0u) ? (pos & (p.kv_stride - 1u)) : pos;
        const ulong  hs    = ulong(hkv) * p.kv_stride + slot;
        device uchar* k_lyr = ((p.is_local != 0u) ? sm.swa_k : sm.glob_k) + p.kv_lyr_off;
        if (p.kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0 && p.is_local == 0u) {
            sik_store_kv_q8_head_sg32(v4 * k_fac,
                                      reinterpret_cast<device char4*>(k_lyr),
                                      reinterpret_cast<device half*>(k_lyr + p.kv_q8_data_b),
                                      hs, lane);
        } else {
            reinterpret_cast<device half4*>(k_lyr)[hs * 32ul + lane] = sik_clamp_f16x4(v4 * k_fac);
        }
    } else if (idx < b3) {
        const uint    hkv   = idx - b2;
        const uint    c0    = hkv * 128u + lane * 4u;
        device float4* st_p = sm.sconv + ulong(p.sconv_k_off_f4) + ulong(p.v_state_off_f4);
        const float4  wt0   = v_sconv_w[c0 + 0u], wt1 = v_sconv_w[c0 + 1u];
        const float4  wt2   = v_sconv_w[c0 + 2u], wt3 = v_sconv_w[c0 + 3u];
        const float4  st0   = st_p[c0 + 0u],      st1 = st_p[c0 + 1u];
        const float4  st2   = st_p[c0 + 2u],      st3 = st_p[c0 + 3u];
        const float4  cur   = reinterpret_cast<device const float4*>(v_raw)[(ulong(b) * p.n_kv + hkv) * 32ul + lane];

        float4 v4;
        if (p.is_local != 0u) {
            const float inv_s = 0.4472135955f;
            v4.x = (fma(st0.x, wt0.x, fma(st0.y, wt0.y, fma(st0.z, wt0.z, cur.x * wt0.w))) + cur.x) * inv_s;
            v4.y = (fma(st1.x, wt1.x, fma(st1.y, wt1.y, fma(st1.z, wt1.z, cur.y * wt1.w))) + cur.y) * inv_s;
            v4.z = (fma(st2.x, wt2.x, fma(st2.y, wt2.y, fma(st2.z, wt2.z, cur.z * wt2.w))) + cur.z) * inv_s;
            v4.w = (fma(st3.x, wt3.x, fma(st3.y, wt3.y, fma(st3.z, wt3.z, cur.w * wt3.w))) + cur.w) * inv_s;
        } else {
            const float4 dyn  = v_dyn[ulong(b) * p.n_kv + hkv];
            const float4 t0   = wt0 + dyn, t1 = wt1 + dyn, t2 = wt2 + dyn, t3 = wt3 + dyn;
            const float  inv3 = 1.0f / 3.0f;
            v4.x = (fma(cur.x, t0.x, fma(st0.z, t0.y, fma(st0.y, t0.z, st0.x * t0.w))) + cur.x) * inv3;
            v4.y = (fma(cur.y, t1.x, fma(st1.z, t1.y, fma(st1.y, t1.z, st1.x * t1.w))) + cur.y) * inv3;
            v4.z = (fma(cur.z, t2.x, fma(st2.z, t2.y, fma(st2.y, t2.z, st2.x * t2.w))) + cur.z) * inv3;
            v4.w = (fma(cur.w, t3.x, fma(st3.z, t3.y, fma(st3.y, t3.z, st3.x * t3.w))) + cur.w) * inv3;
        }
        st_p[c0 + 0u] = float4(st0.y, st0.z, cur.x, 0.0f);
        st_p[c0 + 1u] = float4(st1.y, st1.z, cur.y, 0.0f);
        st_p[c0 + 2u] = float4(st2.y, st2.z, cur.z, 0.0f);
        st_p[c0 + 3u] = float4(st3.y, st3.z, cur.w, 0.0f);

        const float  tot   = simd_sum(dot(v4, v4));
        const float  v_fac = rsqrt(tot / float(p.head_dim) + p.eps);
        const uint   slot  = (p.is_local != 0u) ? (pos & (p.kv_stride - 1u)) : pos;
        const ulong  hs    = ulong(hkv) * p.kv_stride + slot;
        device uchar* v_lyr = ((p.is_local != 0u) ? sm.swa_v : sm.glob_v) + p.kv_lyr_off;
        if (p.kv_dtype == STRATA_INKLING2_KV_DTYPE_Q8_0 && p.is_local == 0u) {
            sik_store_kv_q8_head_sg32(v4 * v_fac,
                                      reinterpret_cast<device char4*>(v_lyr),
                                      reinterpret_cast<device half*>(v_lyr + p.kv_q8_data_b),
                                      hs, lane);
        } else {
            reinterpret_cast<device half4*>(v_lyr)[hs * 32ul + lane] = sik_clamp_f16x4(v4 * v_fac);
        }
    } else if (idx < b3 + p.n_q) {
        const uint  hq  = idx - b3;
        const float tau = (p.is_local == 0u && p.log_n_floor > 0.0f)
            ? (1.0f + p.log_alpha * log(max(1.0f, float(pos + 1u) / p.log_n_floor))) : 1.0f;
        const ulong off = (ulong(b) * p.n_q + hq) * p.d_rel + lane;
        const float val = (lane < p.d_rel) ? (r_raw[off] * r_scale[ulong(hq) * p.d_rel + lane]) : 0.0f;
        const float tot = simd_sum(val * val);
        const float fac = min(1.0f, p.rel_clip / (sqrt(tot) + p.eps)) * tau;
        if (lane < p.d_rel) r_out[off] = val * fac;
    }
}

// Per-Stream Independent Single-Pass GQA-3 Attention (`qp < 256`, F16 KV, 100.0000% bit-exact w/ M=1!)
// Grid: `(n_kv, B, 1)`, `256` threads/TG (`8` SGs)
kernel void sik_attn_decode_single_gqa3_batched(
    device const float4*         q_in     [[buffer(0)]],
    device const float*          r_in     [[buffer(1)]],
    device const float*          rel_proj [[buffer(2)]],
    device const float*          sink_in  [[buffer(3)]],
    device const float4*         gate_in  [[buffer(4)]],
    device half4*                out_f16  [[buffer(5)]],
    device float*                wo_xsum  [[buffer(6)]],
    constant SikBatchSlotMeta*   slots    [[buffer(7)]],
    constant SikBatchAttnParams& p        [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint hkv = tgid.x;
    const uint b   = tgid.y;
    if (hkv >= p.n_kv || b >= p.B) return;

    const SikBatchSlotMeta sm = slots[b];
    const uint qp = sm.seq_pos;
    if (qp >= 256u) return;

    const uint ws = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;

    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        const ulong qrow = ulong(b) * p.n_q + (hkv * 3u + h);
        qv[h] = q_in[qrow * 32ul + lane] * p.attn_scale;
        rl[h] = (lane < p.d_rel) ? r_in[qrow * p.d_rel + lane] : 0.0f;
    }

    device const half4* k_lyr = reinterpret_cast<device const half4*>(((p.is_local != 0u) ? sm.swa_k : sm.glob_k) + p.kv_lyr_off);
    device const half4* v_lyr = reinterpret_cast<device const half4*>(((p.is_local != 0u) ? sm.swa_v : sm.glob_v) + p.kv_lyr_off);
    device const half4* kh    = k_lyr + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* vh    = v_lyr + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    const uint mask = p.kv_stride - 1u;

    for (uint k = ws + sgid; k <= qp; k += 8u) {
        const uint   slot    = (p.is_local != 0u) ? (k & mask) : k;
        const ulong  off     = ulong(slot) * 32ul + lane;
        const float4 kv      = float4(kh[off]);
        const float4 vv      = float4(vh[off]);
        const uint   dist    = qp - k;
        const bool   has_rel = (dist < p.rel_extent);
        const float  relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    threadgroup float  tg_m[3][8], tg_l[3][8];
    threadgroup float4 tg_o[3][8][32];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        if (lane == 0u) { tg_m[h][sgid] = mv[h]; tg_l[h][sgid] = lv[h]; }
        tg_o[h][sgid][lane] = ov[h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid < 3u) {
        const uint  h      = sgid;
        const uint  hq     = hkv * 3u + h;
        const float sink_v = sink_in[ulong(b) * p.n_q + hq];
        float m_max = sink_v;
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) m_max = max(m_max, tg_m[h][s]);
        float  l_tot = exp(sink_v - m_max);
        float4 o_tot = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint s = 0u; s < 8u; ++s) {
            if (tg_m[h][s] > -INFINITY) {
                const float w = exp(tg_m[h][s] - m_max);
                l_tot = fma(tg_l[h][s], w, l_tot);
                o_tot = fma(tg_o[h][s][lane], float4(w), o_tot);
            }
        }
        const float  inv_l   = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
        const ulong  out_off = (ulong(b) * p.n_q + hq) * 32ul + lane;
        const float4 gv      = gate_in[out_off];
        const float4 sig_g   = float4(1.0f) / (float4(1.0f) + exp(-gv));
        const half4  oh      = sik_clamp_f16x4((o_tot * inv_l) * sig_g);
        out_f16[out_off]     = oh;
        if (p.B <= 64u) {
            const float bs = sik_reduce16((float(oh.x) + float(oh.y)) + (float(oh.z) + float(oh.w)));
            if ((lane & 15u) == 0u) {
                wo_xsum[ulong(b) * (p.n_q * 2u) + hq * 2u + (lane >> 4u)] = bs;
            }
        }
    }
}

// Per-Stream Independent Register-Direct GQA-3 Split-KV Attention (`qp >= 256`, F16 KV, 100.0000% bit-exact w/ M=1!)
// Grid: `(max_chunks, n_kv, B)`, `128` threads/TG (`4` SGs)
kernel void sik_attn_decode_split_gqa3_batched(
    device const float4*         q_in     [[buffer(0)]],
    device const float*          r_in     [[buffer(1)]],
    device const float*          rel_proj [[buffer(2)]],
    device float*                part_m   [[buffer(3)]],
    device float*                part_l   [[buffer(4)]],
    device float4*               part_o   [[buffer(5)]],
    constant SikBatchSlotMeta*   slots    [[buffer(6)]],
    constant SikBatchAttnParams& p        [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint s_slot = tgid.x;
    const uint hkv    = tgid.y;
    const uint b      = tgid.z;
    if (hkv >= p.n_kv || b >= p.B) return;

    const SikBatchSlotMeta sm = slots[b];
    const uint qp = sm.seq_pos;
    if (qp < 256u) return;

    const uint B_c     = sik_attn_chunk_size(qp, p.is_local);
    const uint ws      = (p.is_local != 0u && qp + 1u > p.swa_win) ? (qp + 1u - p.swa_win) : 0u;
    const uint c_first = ws / B_c;
    const uint c_last  = qp / B_c;
    const uint c       = c_first + s_slot;
    if (c > c_last) return;

    const uint kv_beg = c * B_c;
    const uint kv_end = min(kv_beg + B_c, qp + 1u);
    const uint kv_lo  = max(kv_beg, ws);

    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        const ulong qrow = ulong(b) * p.n_q + (hkv * 3u + h);
        qv[h] = q_in[qrow * 32ul + lane] * p.attn_scale;
        rl[h] = (lane < p.d_rel) ? r_in[qrow * p.d_rel + lane] : 0.0f;
    }

    device const half4* k_lyr = reinterpret_cast<device const half4*>(((p.is_local != 0u) ? sm.swa_k : sm.glob_k) + p.kv_lyr_off);
    device const half4* v_lyr = reinterpret_cast<device const half4*>(((p.is_local != 0u) ? sm.swa_v : sm.glob_v) + p.kv_lyr_off);
    device const half4* kh    = k_lyr + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half4* vh    = v_lyr + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    const uint mask = p.kv_stride - 1u;

    uint k = kv_beg + sgid;
    if (kv_beg >= kv_lo && c != c_last) {
        for (; k + 4u < kv_end; k += 8u) {
            const uint   s0   = (p.is_local != 0u) ? (k & mask) : k;
            const uint   s1   = (p.is_local != 0u) ? ((k + 4u) & mask) : (k + 4u);
            const ulong  o0   = ulong(s0) * 32ul + lane;
            const ulong  o1   = ulong(s1) * 32ul + lane;
            const float4 kv0  = float4(kh[o0]), vv0 = float4(vh[o0]);
            const float4 kv1  = float4(kh[o1]), vv1 = float4(vh[o1]);
            const uint   d0   = qp - k, d1 = d0 - 4u;
            const bool   hr0  = (d0 < p.rel_extent), hr1 = (d1 < p.rel_extent);
            const float  rw0  = (hr0 && lane < p.d_rel) ? rel_proj[ulong(d0) * p.d_rel + lane] : 0.0f;
            const float  rw1  = (hr1 && lane < p.d_rel) ? rel_proj[ulong(d1) * p.d_rel + lane] : 0.0f;
            #pragma clang loop unroll(full)
            for (uint h = 0u; h < 3u; ++h) {
                float p0 = dot(qv[h], kv0);
                if (hr0 && lane < p.d_rel) p0 = fma(rl[h], rw0, p0);
                const float sc0 = simd_sum(p0);
                const float mn0 = max(mv[h], sc0);
                const float al0 = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn0);
                const float w0  = exp(sc0 - mn0);
                mv[h] = mn0; lv[h] = fma(lv[h], al0, w0); ov[h] = fma(ov[h], al0, w0 * vv0);

                float p1 = dot(qv[h], kv1);
                if (hr1 && lane < p.d_rel) p1 = fma(rl[h], rw1, p1);
                const float sc1 = simd_sum(p1);
                const float mn1 = max(mv[h], sc1);
                const float al1 = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn1);
                const float w1  = exp(sc1 - mn1);
                mv[h] = mn1; lv[h] = fma(lv[h], al1, w1); ov[h] = fma(ov[h], al1, w1 * vv1);
            }
        }
    }
    for (; k < kv_end; k += 4u) {
        if (k < kv_lo) continue;
        const uint   slot    = (p.is_local != 0u) ? (k & mask) : k;
        const ulong  off     = ulong(slot) * 32ul + lane;
        const float4 kv      = float4(kh[off]);
        const float4 vv      = float4(vh[off]);
        const uint   dist    = qp - k;
        const bool   has_rel = (dist < p.rel_extent);
        const float  relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    threadgroup float  tg_m[4], tg_l[4];
    threadgroup float4 tg_o[4][32];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        if (h > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) { tg_m[sgid] = mv[h]; tg_l[sgid] = lv[h]; }
        tg_o[sgid][lane] = ov[h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float mm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
            float  ll = 0.0f;
            float4 oo(0.0f);
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 4u; ++s) {
                if (tg_m[s] > -INFINITY) {
                    const float w = exp(tg_m[s] - mm);
                    ll = fma(tg_l[s], w, ll);
                    oo = fma(tg_o[s][lane], float4(w), oo);
                }
            }
            const uint  hq  = hkv * 3u + h;
            const ulong idx = (ulong(b) * p.n_q + hq) * p.max_splits + s_slot;
            if (lane == 0u) { part_m[idx] = mm; part_l[idx] = ll; }
            part_o[idx * 32ul + lane] = oo;
        }
    }
}

// Per-Stream Independent Register-Direct Q8_0 Global KV Split-KV Attention (`100.0000%` bit-exact w/ M=1!)
// Grid: `(max_chunks, n_kv, B)`, `128` threads/TG (`4` SGs)
kernel void sik_attn_decode_q8_reg_batched(
    device const float4*         q_in     [[buffer(0)]],
    device const float*          r_in     [[buffer(1)]],
    device const float*          rel_proj [[buffer(2)]],
    device float*                part_m   [[buffer(3)]],
    device float*                part_l   [[buffer(4)]],
    device float4*               part_o   [[buffer(5)]],
    constant SikBatchSlotMeta*   slots    [[buffer(6)]],
    constant SikBatchAttnParams& p        [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint c   = tgid.x;
    const uint hkv = tgid.y;
    const uint b   = tgid.z;
    if (hkv >= p.n_kv || b >= p.B) return;

    const SikBatchSlotMeta sm = slots[b];
    const uint qp  = sm.seq_pos;
    const uint B_c = sik_attn_chunk_size(qp, 0u);
    if (c > (qp / B_c)) return;

    const uint kv_beg = c * B_c;
    const uint kv_end = min(kv_beg + B_c, qp + 1u);

    float4 qv[3];
    float  rl[3];
    float  mv[3] = {-INFINITY, -INFINITY, -INFINITY};
    float  lv[3] = {0.0f, 0.0f, 0.0f};
    float4 ov[3] = {float4(0.0f), float4(0.0f), float4(0.0f)};
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        const ulong qrow = ulong(b) * p.n_q + (hkv * 3u + h);
        qv[h] = q_in[qrow * 32ul + lane] * p.attn_scale;
        rl[h] = (lane < p.d_rel) ? r_in[qrow * p.d_rel + lane] : 0.0f;
    }

    device const uchar* k_lyr  = sm.glob_k + p.kv_lyr_off;
    device const uchar* v_lyr  = sm.glob_v + p.kv_lyr_off;
    device const char4* k_head = reinterpret_cast<device const char4*>(k_lyr) + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const char4* v_head = reinterpret_cast<device const char4*>(v_lyr) + ulong(hkv) * ulong(p.kv_stride) * 32ul;
    device const half*  k_s    = reinterpret_cast<device const half*>(k_lyr + p.kv_q8_data_b) + ulong(hkv) * ulong(p.kv_stride);
    device const half*  v_s    = reinterpret_cast<device const half*>(v_lyr + p.kv_q8_data_b) + ulong(hkv) * ulong(p.kv_stride);

    for (uint k = kv_beg + sgid; k < kv_end; k += 4u) {
        const ulong  off     = ulong(k) * 32ul + lane;
        const float4 kv      = float4(half4(k_head[off]) * k_s[k]);
        const float4 vv      = float4(half4(v_head[off]) * v_s[k]);
        const uint   dist    = qp - k;
        const bool   has_rel = (dist < p.rel_extent);
        const float  relw    = (has_rel && lane < p.d_rel) ? rel_proj[ulong(dist) * p.d_rel + lane] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0u; h < 3u; ++h) {
            float part = dot(qv[h], kv);
            if (has_rel && lane < p.d_rel) part = fma(rl[h], relw, part);
            const float s  = simd_sum(part);
            const float mn = max(mv[h], s);
            const float al = (mv[h] == -INFINITY) ? 0.0f : exp(mv[h] - mn);
            const float w  = exp(s - mn);
            mv[h] = mn;
            lv[h] = fma(lv[h], al, w);
            ov[h] = fma(ov[h], al, w * vv);
        }
    }

    threadgroup float  tg_m[4], tg_l[4];
    threadgroup float4 tg_o[4][32];
    #pragma clang loop unroll(full)
    for (uint h = 0u; h < 3u; ++h) {
        if (h > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) { tg_m[sgid] = mv[h]; tg_l[sgid] = lv[h]; }
        tg_o[sgid][lane] = ov[h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            const float mm = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
            float  ll = 0.0f;
            float4 oo(0.0f);
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 4u; ++s) {
                if (tg_m[s] > -INFINITY) {
                    const float w = exp(tg_m[s] - mm);
                    ll = fma(tg_l[s], w, ll);
                    oo = fma(tg_o[s][lane], float4(w), oo);
                }
            }
            const uint  hq  = hkv * 3u + h;
            const ulong idx = (ulong(b) * p.n_q + hq) * p.max_splits + c;
            if (lane == 0u) { part_m[idx] = mm; part_l[idx] = ll; }
            part_o[idx * 32ul + lane] = oo;
        }
    }
}

// Per-Stream Independent Split-KV Reduce + Virtual Sink + Sigmoid Gate + `wo_xsum` (`B <= 8`)
// Grid: `(n_q, B, 1)`, `32` threads/TG (`1` SG, ZERO TG mem!)
kernel void sik_attn_decode_reduce_batched(
    device const float*          part_m  [[buffer(0)]],
    device const float*          part_l  [[buffer(1)]],
    device const float4*         part_o  [[buffer(2)]],
    device const float*          sink_in [[buffer(3)]],
    device const float4*         gate_in [[buffer(4)]],
    device half4*                out_f16 [[buffer(5)]],
    device float*                wo_xsum [[buffer(6)]],
    constant SikBatchSlotMeta*   slots   [[buffer(7)]],
    constant SikBatchAttnParams& p       [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint hq = tgid.x;
    const uint b  = tgid.y;
    if (hq >= p.n_q || b >= p.B) return;

    const uint q_pos = slots[b].seq_pos;
    if (p.skip_under_256 != 0u && q_pos < 256u) return;

    const uint  B_c       = sik_attn_chunk_size(q_pos, p.is_local);
    const uint  win_start = (p.is_local != 0u && q_pos + 1u > p.swa_win) ? (q_pos + 1u - p.swa_win) : 0u;
    const uint  my_splits = (q_pos / B_c) - (win_start / B_c) + 1u;
    const ulong head_base = (ulong(b) * p.n_q + hq) * p.max_splits;

    const float sink_v = sink_in[ulong(b) * p.n_q + hq];
    const float m0     = (lane < my_splits) ? part_m[head_base + lane] : -INFINITY;
    const float m1     = (lane + 32u < my_splits) ? part_m[head_base + lane + 32u] : -INFINITY;
    const float m_max  = max(sink_v, simd_max(max(m0, m1)));

    const float w0     = (lane < my_splits && m0 > -INFINITY) ? exp(m0 - m_max) : 0.0f;
    const float w1     = (lane + 32u < my_splits && m1 > -INFINITY) ? exp(m1 - m_max) : 0.0f;
    const float l0     = (w0 > 0.0f) ? (part_l[head_base + lane] * w0) : 0.0f;
    const float l1     = (w1 > 0.0f) ? (part_l[head_base + lane + 32u] * w1) : 0.0f;
    const float l_tot  = exp(sink_v - m_max) + simd_sum(l0 + l1);
    const float inv_l  = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;

    float4 o_tot(0.0f);
    for (uint s = 0u; s < my_splits; ++s) {
        const float w = (s < 32u) ? simd_broadcast(w0, ushort(s)) : simd_broadcast(w1, ushort(s - 32u));
        if (w > 0.0f) {
            o_tot = fma(part_o[(head_base + s) * 32ul + lane], float4(w), o_tot);
        }
    }

    const ulong  out_off = (ulong(b) * p.n_q + hq) * 32ul + lane;
    const float4 gv      = gate_in[out_off];
    const float4 sig_g   = float4(1.0f) / (float4(1.0f) + exp(-gv));
    const half4  h       = sik_clamp_f16x4((o_tot * inv_l) * sig_g);
    out_f16[out_off]     = h;
    if (p.B <= 64u) {
        const float bs = sik_reduce16((float(h.x) + float(h.y)) + (float(h.z) + float(h.w)));
        if ((lane & 15u) == 0u) {
            wo_xsum[ulong(b) * (p.n_q * 2u) + hq * 2u + (lane >> 4u)] = bs;
        }
    }
}

// Per-Stream Independent Fused Q4_A `Wo + attn_sconv + Residual/Block` (`B = 1..8`, works w/ `--no-refusal` alt_wo!)
template <uint M_FIX>
inline void sik_q4a_wo_sconv_batched_fixed(
    device const uint2*          w_u2,
    device const half*           w_s,
    device const half*           w_b,
    device const half*           x_in,
    device const float*          x_sum,
    device const float4*         sconv_w,
    device float*                out_or_cur,
    device float*                block_dst,
    constant SikBatchSlotMeta*   slots,
    constant SikBatchSconvParams& p,
    threadgroup float*           tg_row,
    uint r0,
    uint lane
) {
    const uint r1      = min(r0 + 1u, p.channels - 1u);
    const uint k16     = p.cols >> 4u;
    const uint nblk    = p.cols >> 6u;
    const bool do_bias = ((lane & 3u) == 0u);
    device const uint2* rw0 = w_u2 + ulong(r0) * k16;
    device const uint2* rw1 = w_u2 + ulong(r1) * k16;
    device const half*  rs0 = w_s  + ulong(r0) * nblk;
    device const half*  rs1 = w_s  + ulong(r1) * nblk;
    device const half*  rb0 = w_b  + ulong(r0) * nblk;
    device const half*  rb1 = w_b  + ulong(r1) * nblk;

    float2 acc[M_FIX];
    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) acc[m] = float2(0.0f);

    for (uint i = lane; i < k16; i += 32u) {
        const uint2  q0  = rw0[i], q1 = rw1[i];
        const uint   grp = i >> 2u;
        const float2 sc  = float2(float(rs0[grp]), float(rs1[grp]));
        half4 a0, a1, a2, a3, b0, b1, b2, b3;
        sik_unpack_u4_u32_h4(q0.x, a0, a1); sik_unpack_u4_u32_h4(q0.y, a2, a3);
        sik_unpack_u4_u32_h4(q1.x, b0, b1); sik_unpack_u4_u32_h4(q1.y, b2, b3);
        #pragma clang loop unroll(full)
        for (uint m = 0u; m < M_FIX; ++m) {
            device const half4* xm = reinterpret_cast<device const half4*>(x_in + ulong(m) * p.cols + i * 16u);
            const half4 x0 = xm[0], x1 = xm[1], x2 = xm[2], x3 = xm[3];
            const half4 dh0 = fma(a0, x0, a1 * x1) + fma(a2, x2, a3 * x3);
            const half4 dh1 = fma(b0, x0, b1 * x1) + fma(b2, x2, b3 * x3);
            acc[m].x = fma(sc.x, (float(dh0.x) + float(dh0.y)) + (float(dh0.z) + float(dh0.w)), acc[m].x);
            acc[m].y = fma(sc.y, (float(dh1.x) + float(dh1.y)) + (float(dh1.z) + float(dh1.w)), acc[m].y);
        }
        if (do_bias) {
            const float2 bi = float2(float(rb0[grp]), float(rb1[grp]));
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                acc[m] = fma(bi, float2(x_sum[ulong(m) * nblk + grp]), acc[m]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint m = 0u; m < M_FIX; ++m) {
        const float vx = simd_sum(acc[m].x);
        const float vy = simd_sum(acc[m].y);
        if (lane < 2u) tg_row[lane * 8u + m] = (lane == 0u) ? vx : vy;
    }

    if (lane < 2u) {
        const uint c = r0 + lane;
        if (c < p.channels) {
            threadgroup const float* tv = tg_row + lane * 8u;
            const uint   C     = p.channels;
            const float4 w     = sconv_w[c];
            const float  inv_s = 0.4472135955f;
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < M_FIX; ++m) {
                device float4* st_p = slots[m].sconv + ulong(p.sconv_off_f4);
                const float4   st   = st_p[c];
                const float    cur  = tv[m];
                const float    sum  = fma(st.x, w.x, fma(st.y, w.y, fma(st.z, w.z, cur * w.w)));
                const float    y    = (sum + cur) * inv_s;
                st_p[c]             = float4(st.y, st.z, cur, 0.0f);
                const ulong    off  = ulong(m) * C + c;
                if (p.res_mode == 2u) {
                    block_dst[off]  = out_or_cur[off];
                    out_or_cur[off] = y;
                } else if (p.res_mode == 1u) {
                    out_or_cur[off] = out_or_cur[off] + y;
                } else {
                    out_or_cur[off] = y;
                }
            }
        }
    }
}

kernel void sik_q4a_gemv_wo_sconv_batched(
    device const uint2*           w_u2       [[buffer(0)]],
    device const half*            w_s        [[buffer(1)]],
    device const half*            w_b        [[buffer(2)]],
    device const half*            x_in       [[buffer(3)]],
    device const float*           x_sum      [[buffer(4)]],
    device const float4*          sconv_w    [[buffer(5)]],
    device float*                 out_or_cur [[buffer(6)]],
    device float*                 block_dst  [[buffer(7)]],
    constant SikBatchSlotMeta*    slots      [[buffer(8)]],
    constant SikBatchSconvParams& p          [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_val[4][16];
    const uint r0 = (tgid * sgs + sgid) * 2u;
    if (r0 >= p.channels) return;
    #define SIK_WOSCB_CASE(MF) { sik_q4a_wo_sconv_batched_fixed<MF>(w_u2, w_s, w_b, x_in, x_sum, sconv_w, out_or_cur, block_dst, slots, p, tg_val[sgid], r0, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_WOSCB_CASE(1u) case 2u: SIK_WOSCB_CASE(2u) case 3u: SIK_WOSCB_CASE(3u) case 4u: SIK_WOSCB_CASE(4u)
        case 5u: SIK_WOSCB_CASE(5u) case 6u: SIK_WOSCB_CASE(6u) case 7u: SIK_WOSCB_CASE(7u) case 8u: SIK_WOSCB_CASE(8u)
        default: break;
    }
    #undef SIK_WOSCB_CASE
}

// Per-Stream Independent Unfused Static SConv (`B = 1..64`, used for `B >= 9` NAX `wo` & FP16/Q8_0 `wo`!)
// Grid: `((channels + 255) / 256, B, 1)`, `256` threads/TG
kernel void sik_sconv_static_batched(
    device const float*           x_in       [[buffer(0)]],
    device const float4*          weight     [[buffer(1)]],
    device float*                 out_or_cur [[buffer(2)]],
    device float*                 block_dst  [[buffer(3)]],
    constant SikBatchSlotMeta*    slots      [[buffer(4)]],
    constant SikBatchSconvParams& p          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint c = tgid.x * 256u + tid;
    const uint b = tgid.y;
    if (c >= p.channels || b >= p.B) return;

    const ulong    off   = ulong(b) * p.channels + c;
    const float    cur   = x_in[off];
    const float4   w     = weight[c];
    device float4* st_p  = slots[b].sconv + ulong(p.sconv_off_f4);
    const float4   st    = st_p[c];
    const float    sum   = fma(st.x, w.x, fma(st.y, w.y, fma(st.z, w.z, cur * w.w)));
    const float    y     = (sum + cur) * 0.4472135955f;
    st_p[c]              = float4(st.y, st.z, cur, 0.0f);

    if (p.res_mode == 2u) {
        block_dst[off]  = out_or_cur[off];
        out_or_cur[off] = y;
    } else if (p.res_mode == 1u) {
        out_or_cur[off] = out_or_cur[off] + y;
    } else {
        out_or_cur[off] = y;
    }
}

// Per-Stream Independent Fused `MoE Top-K Combine + mlp_sconv + Residual Add` (`B = 1..64`)
// Grid: `((D + 63) / 64, B, 1)`, `64` threads/TG (`32` B TG mem, 0 spills, 100% parallel across `(c, b)`!)
kernel void sik_moe_combine_sconv_batched(
    device const float*           down_part [[buffer(0)]],
    device const float*           topk_w    [[buffer(1)]],
    device const float4*          weight    [[buffer(2)]],
    device float*                 cur_io    [[buffer(3)]],
    constant SikBatchSlotMeta*    slots     [[buffer(4)]],
    constant SikBatchSconvParams& p         [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint c = tgid.x * 64u + tid;
    const uint b = tgid.y;
    if (b >= p.B) return;

    const uint K = min(p.top_k, 8u);
    threadgroup float tg_tw[8];
    if (tid < K) tg_tw[tid] = topk_w[ulong(b) * p.top_k + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (c >= p.channels) return;
    const uint  C         = p.channels;
    const ulong base_pair = ulong(b) * p.top_k;
    float cur = 0.0f;
    if (K == 8u) {
        cur = fma(tg_tw[0], down_part[(base_pair + 0u) * C + c], cur);
        cur = fma(tg_tw[1], down_part[(base_pair + 1u) * C + c], cur);
        cur = fma(tg_tw[2], down_part[(base_pair + 2u) * C + c], cur);
        cur = fma(tg_tw[3], down_part[(base_pair + 3u) * C + c], cur);
        cur = fma(tg_tw[4], down_part[(base_pair + 4u) * C + c], cur);
        cur = fma(tg_tw[5], down_part[(base_pair + 5u) * C + c], cur);
        cur = fma(tg_tw[6], down_part[(base_pair + 6u) * C + c], cur);
        cur = fma(tg_tw[7], down_part[(base_pair + 7u) * C + c], cur);
    } else {
        for (uint k = 0u; k < p.top_k; ++k) {
            cur = fma(topk_w[base_pair + k], down_part[(base_pair + k) * C + c], cur);
        }
    }

    const float4   w    = weight[c];
    device float4* st_p = slots[b].sconv + ulong(p.sconv_off_f4);
    const float4   st   = st_p[c];
    const float    sum  = fma(st.x, w.x, fma(st.y, w.y, fma(st.z, w.z, cur * w.w)));
    const float    y    = (sum + cur) * 0.4472135955f;
    st_p[c]             = float4(st.y, st.z, cur, 0.0f);
    const ulong    off  = ulong(b) * C + c;
    cur_io[off]         = cur_io[off] + y;
}

// Per-Stream Independent Sparse Sliding-Window Logit Penalties (ZERO cross-stream contamination!)
// Grid: `((max_pen_len + 255) / 256, B_chunk, 1)`, `256` threads/TG
kernel void sik_apply_logit_penalties_batched(
    device float*                 logits    [[buffer(0)]], // [B_chunk, vocab_size]
    device const uint*            pen_toks  [[buffer(1)]],
    device const ushort*          pen_cnts  [[buffer(2)]],
    constant SikBatchPenItem*     pen_items [[buffer(3)]],
    constant SikBatchPenParams&   p         [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint m   = tgid.y;
    const uint gid = tgid.x * 256u + tid;
    if (m >= p.B_chunk) return;

    const SikBatchPenItem bi = pen_items[p.item_offset + m];
    if (gid >= bi.pen_len) return;

    const uint t = pen_toks[bi.pen_off + gid];
    const uint c = uint(pen_cnts[bi.pen_off + gid]);
    if (t < p.vocab_size && c > 0u) {
        const ulong idx = ulong(m) * p.vocab_size + t;
        logits[idx] = sik_apply_pen_val(logits[idx], c, bi.pres_pen, bi.freq_pen, bi.rep_pen);
    }
}

// Heterogeneous Per-Stream Sampler Stage 1 (supports mixed greedy + distinct top-K/top-P/temp/seed in ONE batch!)
// Grid: `(64, B_chunk, 1)`, `256` threads/TG
kernel void sik_sample_batched_stage1(
    device const float*            logits       [[buffer(0)]], // [B_chunk, vocab_size]
    device float*                  stage_v      [[buffer(1)]], // [B_chunk, 4096]
    device uint*                   stage_i      [[buffer(2)]], // [B_chunk, 4096]
    constant SikBatchSampleItem*   sample_items [[buffer(3)]],
    constant SikBatchSampleParams& p            [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint blk = tgid.x;
    const uint m   = tgid.y;
    if (blk >= 64u || m >= p.B_chunk) return;

    const SikBatchSampleItem si = sample_items[p.item_offset + m];
    const uint chunk = max(1u, (p.vocab_size + 63u) >> 6u);
    const uint beg   = blk * chunk;
    const uint end   = min(p.vocab_size, beg + chunk);
    device const float* row = logits + ulong(m) * p.vocab_size;
    device float*       sv  = stage_v + ulong(m) * 4096ul;
    device uint*        si_o = stage_i + ulong(m) * 4096ul;

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];

    if (si.greedy != 0u) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;
        for (uint i = beg + tid; i < end; i += 256u) {
            const float v = row[i];
            if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) { sv[blk] = v; si_o[blk] = idx; }
        }
        return;
    }

    const uint  K        = min(p.vocab_size, (si.top_k > 0u && si.top_k <= 64u) ? si.top_k : 20u);
    const ulong base_out = ulong(blk) * K;

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
            const float v = rv[j]; const uint idx = ri[j];
            if (v > best_v || (v == best_v && idx < best_i)) { best_v = v; best_i = idx; best_j = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_i[0]          = idx;
                sv[base_out + k] = v;
                si_o[base_out + k] = idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint win_i = tg_i[0];
        if (best_j < 16u && ri[best_j] == win_i) {
            rv[best_j] = -INFINITY;
            ri[best_j] = 0xFFFFFFFFu;
        }
        if (k + 1u < K) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Heterogeneous Per-Stream Sampler Stage 2 (`salt = 0u` matches solo `M=1` stochastic RNG bit-for-bit!)
// Grid: `(B_chunk, 1, 1)`, `256` threads/TG
kernel void sik_sample_batched_stage2(
    device const float*            stage_v      [[buffer(0)]], // [B_chunk, 4096]
    device const uint*             stage_i      [[buffer(1)]], // [B_chunk, 4096]
    device uint*                   out_tok      [[buffer(2)]], // [B_total]
    constant SikBatchSampleItem*   sample_items [[buffer(3)]],
    constant SikBatchSampleParams& p            [[buffer(4)]],
    uint m    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (m >= p.B_chunk) return;

    const SikBatchSampleItem si = sample_items[p.item_offset + m];
    device const float* sv      = stage_v + ulong(m) * 4096ul;
    device const uint*  si_in   = stage_i + ulong(m) * 4096ul;

    threadgroup float tg_v[8];
    threadgroup uint  tg_i[8];

    if (si.greedy != 0u) {
        const float my_v = (tid < 64u) ? sv[tid] : -INFINITY;
        const uint  my_i = (tid < 64u) ? si_in[tid] : 0xFFFFFFFFu;
        float best_v = my_v;
        uint  best_i = my_i;
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                out_tok[p.item_offset + m] = (idx < p.vocab_size) ? idx : 0u;
            }
        }
        return;
    }

    const uint K   = min(p.vocab_size, (si.top_k > 0u && si.top_k <= 64u) ? si.top_k : 20u);
    const uint tot = 64u * K;

    float rv[16];
    uint  ri[16];
    uint  cnt = 0u;
    for (uint i = tid; i < tot && cnt < 16u; i += 256u) {
        rv[cnt] = sv[i];
        ri[cnt] = si_in[i];
        ++cnt;
    }

    threadgroup float top_vals[64];
    threadgroup uint  top_ids[64];
    threadgroup float tg_probs[64];

    for (uint k = 0u; k < K; ++k) {
        float best_v = -INFINITY;
        uint  best_i = 0xFFFFFFFFu;
        uint  best_j = 16u;
        for (uint j = 0u; j < cnt; ++j) {
            const float v = rv[j]; const uint idx = ri[j];
            if (v > best_v || (v == best_v && idx < best_i)) { best_v = v; best_i = idx; best_j = j; }
        }
        #pragma clang loop unroll(full)
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(best_v, off);
            const uint  oi = simd_shuffle_xor(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { tg_v[sgid] = best_v; tg_i[sgid] = best_i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u) {
            float v   = (lane < 8u) ? tg_v[lane] : -INFINITY;
            uint  idx = (lane < 8u) ? tg_i[lane] : 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (ushort off = 4u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(v, off);
                const uint  oi = simd_shuffle_xor(idx, off);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0u) {
                tg_i[0]     = idx;
                top_vals[k] = v;
                top_ids[k]  = (idx < p.vocab_size) ? idx : 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint win_i = tg_i[0];
        if (best_j < 16u && ri[best_j] == win_i) {
            rv[best_j] = -INFINITY;
            ri[best_j] = 0xFFFFFFFFu;
        }
        if (k + 1u < K) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float max_v = top_vals[0];
        float sum_p = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            const float w = exp((top_vals[k] - max_v) * si.inv_temp);
            tg_probs[k] = w;
            sum_p      += w;
        }
        const float inv_sum = (sum_p > 0.0f) ? (1.0f / sum_p) : 1.0f;
        for (uint k = 0u; k < K; ++k) tg_probs[k] *= inv_sum;

        uint  keep_k    = K;
        float kept_mass = 1.0f;
        if (si.top_p > 0.0f && si.top_p < 1.0f) {
            float cum = 0.0f;
            for (uint k = 0u; k < K; ++k) {
                cum += tg_probs[k];
                if (cum >= si.top_p) {
                    keep_k    = k + 1u;
                    kept_mass = cum;
                    break;
                }
            }
        }
        const float inv_kept = (kept_mass > 0.0f) ? (1.0f / kept_mass) : 1.0f;
        const float u        = sik_rand_u01(si.rng_lo, si.rng_hi, 0u);
        float       cdf      = 0.0f;
        uint        chosen_k = 0u;
        for (uint k = 0u; k < keep_k; ++k) {
            cdf += tg_probs[k] * inv_kept;
            if (u <= cdf || k + 1u == keep_k) {
                chosen_k = k;
                break;
            }
        }
        out_tok[p.item_offset + m] = top_ids[chosen_k];
    }
}

// ============================================================================
// 12b. 2D SLC-Shared Exact-M8 Batched Kernels (`B = 8..64`, `tgid.x = m_chunk`
//      varies fastest so all 8-stream chunks hit L1/SLC on the SAME weight rows
//      simultaneously -> 1x DRAM weight read AND 0.0000 ULP bit-exact w/ M=1!)
// ============================================================================

// 1. 2D SLC-Shared Fused Attention Q/Gate/R/Sink/K/V/Dyn Projections (`M_FIX = 8`)
// Grid: `(B / 8, (tot_rows + 7) / 8, 1)`, `128` threads/TG
kernel void sik_gemv_attn_qgrs_mixed_2d(
    device const uchar*              uma    [[buffer(0)]],
    device const half*               x_q    [[buffer(1)]],
    device const half*               x_k    [[buffer(2)]],
    device const half*               x_v    [[buffer(3)]],
    device float*                    q_out  [[buffer(4)]],
    device float*                    g_out  [[buffer(5)]],
    device float*                    r_out  [[buffer(6)]],
    device float*                    s_out  [[buffer(7)]],
    device float*                    k_out  [[buffer(8)]],
    device float*                    v_out  [[buffer(9)]],
    device float*                    ks_out [[buffer(10)]],
    device float*                    vs_out [[buffer(11)]],
    constant SikAttnQgrsMixedParams& p      [[buffer(12)]],
    device const float*              xs_q   [[buffer(13)]],
    device const float*              xs_k   [[buffer(14)]],
    device const float*              xs_v   [[buffer(15)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m_base = tgid.x * 8u;
    const uint vrow   = (tgid.y * 4u + sgid) * 2u + (lane >> 4u);
    const uint nblk   = p.cols >> 6u;
    sik_attn_qgrs_mixed_impl<8u>(
        uma,
        x_q + ulong(m_base) * p.cols, x_k + ulong(m_base) * p.cols, x_v + ulong(m_base) * p.cols,
        xs_q + ulong(m_base) * nblk,  xs_k + ulong(m_base) * nblk,  xs_v + ulong(m_base) * nblk,
        q_out  + ulong(m_base) * p.rows[0], g_out  + ulong(m_base) * p.rows[1],
        r_out  + ulong(m_base) * p.rows[2], s_out  + ulong(m_base) * p.rows[3],
        k_out  + ulong(m_base) * p.rows[4], v_out  + ulong(m_base) * p.rows[5],
        ks_out + ulong(m_base) * p.rows[6], vs_out + ulong(m_base) * p.rows[7],
        p, vrow, lane & 15u
    );
}

// 2. 2D SLC-Shared Fused Q4_A `Wo + Per-Stream attn_sconv + Residual/Block` (`M_FIX = 8`)
// Grid: `(B / 8, (channels + 7) / 8, 1)`, `128` threads/TG
kernel void sik_q4a_gemv_wo_sconv_batched_2d(
    device const uint2*           w_u2       [[buffer(0)]],
    device const half*            w_s        [[buffer(1)]],
    device const half*            w_b        [[buffer(2)]],
    device const half*            x_in       [[buffer(3)]],
    device const float*           x_sum      [[buffer(4)]],
    device const float4*          sconv_w    [[buffer(5)]],
    device float*                 out_or_cur [[buffer(6)]],
    device float*                 block_dst  [[buffer(7)]],
    constant SikBatchSlotMeta*    slots      [[buffer(8)]],
    constant SikBatchSconvParams& p          [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgs  [[simdgroups_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_val[4][16];
    const uint m_base = tgid.x * 8u;
    const uint r0     = (tgid.y * sgs + sgid) * 2u;
    if (r0 >= p.channels) return;
    const uint nblk = p.cols >> 6u;
    sik_q4a_wo_sconv_batched_fixed<8u>(
        w_u2, w_s, w_b,
        x_in + ulong(m_base) * p.cols,
        x_sum + ulong(m_base) * nblk,
        sconv_w,
        out_or_cur + ulong(m_base) * p.channels,
        block_dst + ulong(m_base) * p.channels,
        slots + m_base,
        p, tg_val[sgid], r0, lane
    );
}

// 3. 2D SLC-Shared Q8_0 Unembed / Dense GEMV (`M_FIX = 8`, 1x DRAM read across all chunks!)
// Grid: `(B_c / 8, (rows + 7) / 8, 1)`, `128` threads/TG
kernel void sik_q8_gemv_batch_2d(
    device const char*           w_q   [[buffer(0)]],
    device const half*           w_s   [[buffer(1)]],
    device const half*           x_in  [[buffer(2)]],
    device half*                 y_f16 [[buffer(3)]],
    device float*                y_f32 [[buffer(4)]],
    constant SikGemvBatchParams& p     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgs  [[simdgroups_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m_base = tgid.x * 8u;
    const uint row    = (tgid.y * sgs + sgid) * 2u + (lane >> 4u);
    if (row >= p.rows) return;
    sik_q8_gemv_l16_fixed<8u>(
        w_q, w_s,
        x_in + ulong(m_base) * p.cols,
        y_f16 + ulong(m_base) * p.rows,
        y_f32 + ulong(m_base) * p.rows,
        row, p.rows, p.cols, p.out_mode, p.softcap, lane & 15u
    );
}

// 4. 2D SLC-Shared Pure-FP32 MoE Router (`M_FIX = 8`)
// Grid: `(B / 8, num_experts, 1)`, `128` threads/TG
kernel void sik_moe_router_f32_2d(
    device const float4*         gate_w [[buffer(0)]],
    device const float4*         x_f32  [[buffer(1)]],
    device float*                logits [[buffer(2)]],
    constant SikMoeRouterParams& p      [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m_base = tgid.x * 8u;
    const uint e      = tgid.y;
    if (e >= p.num_experts) return;
    threadgroup float tg_red[8][4];
    sik_moe_router_f32_fixed<8u>(
        gate_w,
        x_f32 + ulong(m_base) * (p.D >> 2u),
        logits + ulong(m_base) * p.num_experts,
        p, tg_red, e, tid, sgid, lane
    );
}

// 5. 2D SLC-Shared Batch-12 `down_w_right @ rms(msg)` (`SIK_CONST_M = 1..8`, 0-ULP bit-exact w/ solo M=1!)
// Grid: `(n_chunks, gate_rank, num_layers * 4)`, `32` threads/TG
kernel void sik_loop_ctrl_down_right_batch12_q8_2d(
    device const uchar*             uma        [[buffer(0)]],
    device const half*              msg_norm12 [[buffer(1)]], // [12, 4, M_valid, D]
    device float*                   h_msg12    [[buffer(2)]], // [12, 4, M_valid, R]
    constant SikLoopBatch12Offsets& p          [[buffer(3)]],
    constant uint&                  m_off      [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m_base = m_off + tgid.x * 8u;
    const uint row    = tgid.y;
    const uint slot   = tgid.z;
    const uint li     = slot >> 2u;
    const uint j      = slot & 3u;
    if (row >= p.gate_rank || li >= p.num_layers) return;
    device const char* w_q   = reinterpret_cast<device const char*>(uma + p.w_off[li][j]);
    device const half* w_s   = reinterpret_cast<device const half*>(uma + p.s_off[li][j]);
    device const half* x_in  = msg_norm12 + (ulong(slot) * p.M_valid + m_base) * p.D;
    device float*      y_f32 = h_msg12    + (ulong(slot) * p.M_valid + m_base) * p.gate_rank;
    #define SIK_DR12B_CASE(MF) { sik_q8_half_row_tg32_impl<MF>(w_q, w_s, x_in, nullptr, y_f32, row, p.gate_rank, p.D, p.D, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_DR12B_CASE(1u) case 2u: SIK_DR12B_CASE(2u) case 3u: SIK_DR12B_CASE(3u) case 4u: SIK_DR12B_CASE(4u)
        case 5u: SIK_DR12B_CASE(5u) case 6u: SIK_DR12B_CASE(6u) case 7u: SIK_DR12B_CASE(7u) case 8u: SIK_DR12B_CASE(8u)
        default: break;
    }
    #undef SIK_DR12B_CASE
}

// 6. 2D SLC-Shared 3-Way Half-Width `down_w_left @ rms(cur) + h_msg` (`SIK_CONST_M = 1..8`, 0-ULP bit-exact w/ solo M=1!)
// Grid: `(n_chunks, gate_rank, 3)`, `64` threads/TG
kernel void sik_loop_ctrl_down3_left_q8_2d(
    device const uchar*           uma      [[buffer(0)]],
    device const half*            cat_left [[buffer(1)]], // [3, M_valid, D]
    device const float*           h_msg    [[buffer(2)]], // [3, M_valid, R]
    device float*                 h_out    [[buffer(3)]], // [3, M_valid, R]
    constant SikLoopCtrl3Offsets& p        [[buffer(4)]],
    constant uint&                m_off    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m_base = m_off + tgid.x * 8u;
    const uint row    = tgid.y;
    const uint j      = tgid.z;
    if (row >= p.gate_rank || j >= p.num_ctrls) return;
    threadgroup float tg_red[8][4];
    device const char*  w_q   = reinterpret_cast<device const char*>(uma + p.w_off[j]);
    device const half*  w_s   = reinterpret_cast<device const half*>(uma + p.s_off[j]);
    device const half*  x_in  = cat_left + (ulong(j) * p.M_valid + m_base) * p.D;
    device const float* hm    = h_msg    + (ulong(j) * p.M_valid + m_base) * p.gate_rank;
    device float*       y_f32 = h_out    + (ulong(j) * p.M_valid + m_base) * p.gate_rank;
    #define SIK_DL3B_CASE(MF) { sik_q8_left_add_impl<MF, 2u>(w_q, w_s, x_in, hm, y_f32, tg_red, row, p.gate_rank, p.D, tid, sgid, lane); break; }
    switch (SIK_CONST_M) {
        case 1u: SIK_DL3B_CASE(1u) case 2u: SIK_DL3B_CASE(2u) case 3u: SIK_DL3B_CASE(3u) case 4u: SIK_DL3B_CASE(4u)
        case 5u: SIK_DL3B_CASE(5u) case 6u: SIK_DL3B_CASE(6u) case 7u: SIK_DL3B_CASE(7u) case 8u: SIK_DL3B_CASE(8u)
        default: break;
    }
    #undef SIK_DL3B_CASE
}

// 7. 2D SLC-Shared 1-Way Half-Width `down_w_left @ rms(cur) + h_msg` (`M_FIX = 8`, 0-ULP bit-exact w/ solo M=1!)
// Grid: `(B / 8, gate_rank, 1)`, `128` threads/TG
kernel void sik_loop_ctrl_down1_left_q8_2d(
    device const uchar*           uma      [[buffer(0)]],
    device const half*            cat_left [[buffer(1)]], // [M_valid, D]
    device const float*           h_msg    [[buffer(2)]], // [M_valid, R]
    device float*                 h_out    [[buffer(3)]], // [M_valid, R]
    constant SikLoopCtrl3Offsets& p        [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint m_base = tgid.x * 8u;
    const uint row    = tgid.y;
    if (row >= p.gate_rank) return;
    threadgroup float tg_red[8][4];
    device const char* w_q = reinterpret_cast<device const char*>(uma + p.w_off[0]);
    device const half* w_s = reinterpret_cast<device const half*>(uma + p.s_off[0]);
    sik_q8_left_add_impl<8u, 4u>(
        w_q, w_s,
        cat_left + ulong(m_base) * p.D,
        h_msg    + ulong(m_base) * p.gate_rank,
        h_out    + ulong(m_base) * p.gate_rank,
        tg_red, row, p.gate_rank, p.D, tid, sgid, lane
    );
}

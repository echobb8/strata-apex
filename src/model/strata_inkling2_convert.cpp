/*
 * STRATA-M5 v2.0 — Native Apple M5 Max Converter, Zero-Copy UMA Bundle Loader/Inspector &
 * Exact o200k_base ByteLevel BPE Tokenizer for `thinkingmachines/Inkling-2-Mini`
 *
 * File: src/model/strata_inkling2_convert.cpp
 *
 * Implements the frozen C ABI in `include/strata_inkling2.h`:
 *   1. Multi-threaded ARM NEON HF-to-STRATA converter (`strata_inkling2_convert_hf`):
 *        - Converts 24-layer `inkling2_loop_text` (48 logical visits: 40 SWA-1024 + 8 Global,
 *          looped core L6..17 x 3 passes, 128E top-8 MoE) + optional 3-layer DSpark Block-8
 *          `rnn_hidden` MTP (`mtp.safetensors`) into a single 16 KiB page-aligned zero-copy
 *          `inkling2_uma.strata` (`~18.2 GiB`, 128 KiB header + 256 B tensor alignment).
 *        - Native mixed precision preserved losslessly:
 *            * Layers 0..19 MoE Experts  : Native NVFP4 (de-interleaved `w13` into `[gate; up]`,
 *                                          `F8_E4M3` block-16 scales, `F32` `amax / denom`)
 *            * Layers 20..23 MoE Experts : Native MXFP8 (de-interleaved `w13` into `[gate; up]`,
 *                                          `F8_E4M3` weights, `UE8M0` block-32 scales)
 *            * Dense Projections / Embed / Trimmed Unembed (`200058`) / MTP : IEEE FP16
 *            * Norms, L2 Clips, Transposed Rel-Bias `[1024, 16]`, SConv (static oldest-first,
 *              global dynamic transposed lag-first without flip), AttnRes (`attn_queries.{i-1}`
 *              shift), Router, MTP `fuse_alpha` (row-wise softmax) : Lossless FP32
 *        - Skips all `model.visual.*` vision tensors; validates 0 NaN/Inf across all weights/scales.
 *   2. Zero-copy 16 KiB-aligned `mmap` bundle loader, freer & diagnostic inspector.
 *   3. Exact `o200k_base` ByteLevel BPE tokenizer (`tokenizer.json`, `ignore_merges = true`,
 *      60 atomic `added_tokens` `199998..200057`, 424,413 merges, UTF-8 streaming decoder).
 */

#include "../../include/strata_inkling2.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

static_assert(sizeof(StrataInkling2TensorSlice) == 32,
              "StrataInkling2TensorSlice ABI size must be 32 bytes");
static_assert(sizeof(StrataInkling2ExpertMatrix) == 72,
              "StrataInkling2ExpertMatrix ABI size must be 72 bytes");
static_assert(sizeof(StrataInkling2Header) <= STRATA_INKLING2_HEADER_PADDED_BYTES,
              "StrataInkling2Header must fit within 128 KiB padded header");

namespace {

/* ============================================================================
 * 1. Low-Level IEEE-754 BF16/FP16/FP32 NEON Kernels, FP8 Validation & I/O
 * ============================================================================ */

static inline uint64_t align_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1ULL) & ~(align - 1ULL);
}

static inline float bf16_to_f32_scalar(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

static inline bool bf16_is_finite(uint16_t bf) {
    return (bf & 0x7F80u) != 0x7F80u;
}

static inline bool fp16_is_finite(uint16_t h) {
    return (h & 0x7C00u) != 0x7C00u;
}

#if !defined(__aarch64__)
static inline uint16_t f32_to_fp16_scalar(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t round_bit = 1u << (shift - 1u);
        uint32_t m = (mant + round_bit) >> shift;
        return static_cast<uint16_t>(sign | m);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7BFFu);
    }
    uint32_t m = (mant + 0x0FFFu + ((mant >> 13) & 1u)) >> 13;
    if (m == 0x400u) {
        m = 0;
        ++exp;
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (m & 0x3FFu));
}

static inline float fp16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            float v = std::ldexp(static_cast<float>(mant), -24);
            return (h & 0x8000u) ? -v : v;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}
#endif

static inline uint16_t f32_to_fp16_fast(float f) {
    if (f > 65504.0f) f = 65504.0f;
    else if (f < -65504.0f) f = -65504.0f;
#if defined(__aarch64__)
    float16x4_t h = vcvt_f16_f32(vdupq_n_f32(f));
    return vget_lane_u16(vreinterpret_u16_f16(h), 0);
#else
    return f32_to_fp16_scalar(f);
#endif
}

static inline float fp16_to_f32_fast(uint16_t h) {
#if defined(__aarch64__)
    float16x4_t v = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(v), 0);
#else
    return fp16_to_f32_scalar(h);
#endif
}

static bool convert_bf16_to_fp16_span_checked(const uint16_t* src_bf16, uint16_t* dst_fp16, size_t n) {
    size_t i = 0;
    bool all_finite = true;
#if defined(__aarch64__)
    const uint16x8_t  v_exp_mask = vdupq_n_u16(0x7F80u);
    const float32x4_t v_max      = vdupq_n_f32(65504.0f);
    const float32x4_t v_min      = vdupq_n_f32(-65504.0f);
    uint16x8_t bad_acc = vdupq_n_u16(0);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bf = vld1q_u16(src_bf16 + i);
        bad_acc = vorrq_u16(bad_acc, vceqq_u16(vandq_u16(bf, v_exp_mask), v_exp_mask));
        float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
        float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
        lo = vminq_f32(v_max, vmaxq_f32(v_min, lo));
        hi = vminq_f32(v_max, vmaxq_f32(v_min, hi));
        float16x8_t h8 = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_u16(dst_fp16 + i, vreinterpretq_u16_f16(h8));
    }
    if (vmaxvq_u16(bad_acc) != 0u) all_finite = false;
#endif
    for (; i < n; ++i) {
        if (!bf16_is_finite(src_bf16[i])) all_finite = false;
        dst_fp16[i] = f32_to_fp16_fast(bf16_to_f32_scalar(src_bf16[i]));
    }
    return all_finite;
}

static bool convert_f32_to_fp16_span_checked(const float* src_f32, uint16_t* dst_fp16, size_t n) {
    size_t i = 0;
    bool all_finite = true;
#if defined(__aarch64__)
    const uint32x4_t  v_exp_mask = vdupq_n_u32(0x7F800000u);
    const float32x4_t v_max      = vdupq_n_f32(65504.0f);
    const float32x4_t v_min      = vdupq_n_f32(-65504.0f);
    uint32x4_t bad_acc = vdupq_n_u32(0);
    for (; i + 8 <= n; i += 8) {
        float32x4_t f0 = vld1q_f32(src_f32 + i);
        float32x4_t f1 = vld1q_f32(src_f32 + i + 4);
        bad_acc = vorrq_u32(bad_acc, vceqq_u32(vandq_u32(vreinterpretq_u32_f32(f0), v_exp_mask), v_exp_mask));
        bad_acc = vorrq_u32(bad_acc, vceqq_u32(vandq_u32(vreinterpretq_u32_f32(f1), v_exp_mask), v_exp_mask));
        float32x4_t lo = vminq_f32(v_max, vmaxq_f32(v_min, f0));
        float32x4_t hi = vminq_f32(v_max, vmaxq_f32(v_min, f1));
        float16x8_t h8 = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_u16(dst_fp16 + i, vreinterpretq_u16_f16(h8));
    }
    if (vmaxvq_u32(bad_acc) != 0u) all_finite = false;
#endif
    for (; i < n; ++i) {
        if (!std::isfinite(src_f32[i])) all_finite = false;
        dst_fp16[i] = f32_to_fp16_fast(src_f32[i]);
    }
    return all_finite;
}

static bool convert_bf16_to_f32_span_checked(const uint16_t* src_bf16, float* dst_f32, size_t n) {
    size_t i = 0;
    bool all_finite = true;
#if defined(__aarch64__)
    const uint16x8_t v_exp_mask = vdupq_n_u16(0x7F80u);
    uint16x8_t bad_acc = vdupq_n_u16(0);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bf = vld1q_u16(src_bf16 + i);
        bad_acc = vorrq_u16(bad_acc, vceqq_u16(vandq_u16(bf, v_exp_mask), v_exp_mask));
        float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
        float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
        vst1q_f32(dst_f32 + i,     lo);
        vst1q_f32(dst_f32 + i + 4, hi);
    }
    if (vmaxvq_u16(bad_acc) != 0u) all_finite = false;
#endif
    for (; i < n; ++i) {
        if (!bf16_is_finite(src_bf16[i])) all_finite = false;
        dst_f32[i] = bf16_to_f32_scalar(src_bf16[i]);
    }
    return all_finite;
}

static bool convert_fp16_to_f32_span_checked(const uint16_t* src_fp16, float* dst_f32, size_t n) {
    bool all_finite = true;
    for (size_t i = 0; i < n; ++i) {
        if (!fp16_is_finite(src_fp16[i])) all_finite = false;
        dst_f32[i] = fp16_to_f32_fast(src_fp16[i]);
    }
    return all_finite;
}

static bool check_f16_span_finite(const uint16_t* p, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    const uint16x8_t v_exp_mask = vdupq_n_u16(0x7C00u);
    uint16x8_t bad_acc = vdupq_n_u16(0);
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(p + i);
        bad_acc = vorrq_u16(bad_acc, vceqq_u16(vandq_u16(h, v_exp_mask), v_exp_mask));
    }
    if (vmaxvq_u16(bad_acc) != 0u) return false;
#endif
    for (; i < n; ++i) {
        if (!fp16_is_finite(p[i])) return false;
    }
    return true;
}

static bool check_f32_span_finite(const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

static bool check_e4m3fn_span_no_nan(const uint8_t* p, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    const uint8x16_t v_mask = vdupq_n_u8(0x7Fu);
    uint8x16_t bad = vdupq_n_u8(0);
    for (; i + 16 <= n; i += 16) {
        uint8x16_t v = vld1q_u8(p + i);
        bad = vorrq_u8(bad, vceqq_u8(vandq_u8(v, v_mask), v_mask));
    }
    if (vmaxvq_u8(bad) != 0u) return false;
#endif
    for (; i < n; ++i) {
        if ((p[i] & 0x7Fu) == 0x7Fu) return false;
    }
    return true;
}

static bool check_ue8m0_span_no_nan(const uint8_t* p, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    const uint8x16_t v_nan = vdupq_n_u8(0xFFu);
    uint8x16_t bad = vdupq_n_u8(0);
    for (; i + 16 <= n; i += 16) {
        uint8x16_t v = vld1q_u8(p + i);
        bad = vorrq_u8(bad, vceqq_u8(v, v_nan));
    }
    if (vmaxvq_u8(bad) != 0u) return false;
#endif
    for (; i < n; ++i) {
        if (p[i] == 0xFFu) return false;
    }
    return true;
}

/* ============================================================================
 * 1b. Quantization Kernels: Q8_0 (gs=64), Q4_A (gs=64) & MXFP8 -> NVFP4 Repack
 * ============================================================================ */

struct Fp8LutTables {
    float e4m3[256];
    float ue8m0[256];

    Fp8LutTables() {
        for (uint32_t c = 0; c < 256u; ++c) {
            const float sign = (c & 0x80u) ? -1.0f : 1.0f;
            const uint32_t exp  = (c >> 3) & 0xFu;
            const uint32_t mant = c & 0x7u;
            if (exp == 0u) {
                e4m3[c] = sign * static_cast<float>(mant) * 0.001953125f; /* mant * 2^-9 */
            } else if (exp == 15u && mant == 7u) {
                e4m3[c] = std::numeric_limits<float>::quiet_NaN();
            } else {
                const float v = (1.0f + static_cast<float>(mant) * 0.125f) *
                                std::ldexp(1.0f, static_cast<int>(exp) - 7);
                e4m3[c] = sign * v;
            }
        }
        for (uint32_t s = 0; s < 256u; ++s) {
            if (s == 0xFFu) {
                ue8m0[s] = std::numeric_limits<float>::quiet_NaN();
            } else if (s == 0u) {
                uint32_t bits = 0x00400000u; /* 2^-127 */
                std::memcpy(&ue8m0[s], &bits, sizeof(float));
            } else {
                uint32_t bits = s << 23; /* 2^(s - 127) */
                std::memcpy(&ue8m0[s], &bits, sizeof(float));
            }
        }
    }
};

static const Fp8LutTables& get_fp8_luts() {
    static const Fp8LutTables kLuts;
    return kLuts;
}

static inline uint8_t f32_to_e4m3fn_pos(float x) {
    if (!(x > 0.0f) || !std::isfinite(x)) return 0u;
    if (x >= 448.0f) return 0x7Eu;
    if (x < 0.015625f) {
        uint32_t m = static_cast<uint32_t>(std::nearbyintf(x * 512.0f));
        if (m > 8u) m = 8u;
        return static_cast<uint8_t>(m);
    }
    uint32_t bits = 0u;
    std::memcpy(&bits, &x, sizeof(bits));
    int32_t  e4     = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 7;
    uint32_t mant23 = bits & 0x7FFFFFu;
    uint32_t top3   = mant23 >> 20;
    uint32_t rem    = mant23 & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (top3 & 1u) != 0u)) {
        ++top3;
        if (top3 == 8u) {
            top3 = 0u;
            ++e4;
        }
    }
    if (e4 <= 0) return 0u;
    uint32_t code = (static_cast<uint32_t>(e4) << 3) | top3;
    if (code > 0x7Eu) code = 0x7Eu;
    return static_cast<uint8_t>(code);
}

static inline void quantize_group64_f32_to_q8(
    const float* vals,
    int8_t*      dst_i8,
    uint16_t*    dst_scale
) {
    float max_abs = 0.0f;
#if defined(__aarch64__)
    float32x4_t vmax4 = vdupq_n_f32(0.0f);
    for (int k = 0; k < 16; ++k) {
        vmax4 = vmaxq_f32(vmax4, vabsq_f32(vld1q_f32(vals + k * 4)));
    }
    max_abs = vmaxvq_f32(vmax4);
#else
    for (int i = 0; i < 64; ++i) {
        float ax = std::fabs(vals[i]);
        if (ax > max_abs) max_abs = ax;
    }
#endif
    const uint16_t scale_h = f32_to_fp16_fast(max_abs * (1.0f / 127.0f));
    *dst_scale = scale_h;
    const float scale_eff = fp16_to_f32_fast(scale_h);
    if (!(scale_eff > 0.0f)) {
        std::memset(dst_i8, 0, 64);
        return;
    }
    const float inv_s = 1.0f / scale_eff;
#if defined(__aarch64__)
    const float32x4_t v_invs = vdupq_n_f32(inv_s);
    const float32x4_t v_lo   = vdupq_n_f32(-127.0f);
    const float32x4_t v_hi   = vdupq_n_f32(127.0f);
    for (int k = 0; k < 16; ++k) {
        float32x4_t qf = vrndnq_f32(vmulq_f32(vld1q_f32(vals + k * 4), v_invs));
        qf = vminq_f32(v_hi, vmaxq_f32(v_lo, qf));
        int32x4_t qi = vcvtq_s32_f32(qf);
        dst_i8[k * 4 + 0] = static_cast<int8_t>(vgetq_lane_s32(qi, 0));
        dst_i8[k * 4 + 1] = static_cast<int8_t>(vgetq_lane_s32(qi, 1));
        dst_i8[k * 4 + 2] = static_cast<int8_t>(vgetq_lane_s32(qi, 2));
        dst_i8[k * 4 + 3] = static_cast<int8_t>(vgetq_lane_s32(qi, 3));
    }
#else
    for (int i = 0; i < 64; ++i) {
        float qf = std::nearbyintf(vals[i] * inv_s);
        if (qf < -127.0f) qf = -127.0f;
        else if (qf > 127.0f) qf = 127.0f;
        dst_i8[i] = static_cast<int8_t>(qf);
    }
#endif
}

static inline void quantize_group64_fp16_to_q8(
    const uint16_t* src_fp16,
    int8_t*         dst_i8,
    uint16_t*       dst_scale
) {
    float vals[64];
#if defined(__aarch64__)
    for (int k = 0; k < 16; ++k) {
        float16x4_t h4 = vreinterpret_f16_u16(vld1_u16(src_fp16 + k * 4));
        vst1q_f32(vals + k * 4, vcvt_f32_f16(h4));
    }
#else
    for (int i = 0; i < 64; ++i) vals[i] = fp16_to_f32_fast(src_fp16[i]);
#endif
    quantize_group64_f32_to_q8(vals, dst_i8, dst_scale);
}

static inline void quantize_group64_f32_to_q4a(
    const float* vals,
    uint32_t*    dst_u32,
    uint16_t*    dst_scale,
    uint16_t*    dst_bias
) {
    float w_min = vals[0];
    float w_max = vals[0];
#if defined(__aarch64__)
    float32x4_t vmin4 = vld1q_f32(vals);
    float32x4_t vmax4 = vmin4;
    for (int k = 1; k < 16; ++k) {
        float32x4_t f4 = vld1q_f32(vals + k * 4);
        vmin4 = vminq_f32(vmin4, f4);
        vmax4 = vmaxq_f32(vmax4, f4);
    }
    w_min = vminvq_f32(vmin4);
    w_max = vmaxvq_f32(vmax4);
#else
    for (int i = 1; i < 64; ++i) {
        if (vals[i] < w_min) w_min = vals[i];
        if (vals[i] > w_max) w_max = vals[i];
    }
#endif
    const uint16_t bias_h  = f32_to_fp16_fast(w_min);
    const uint16_t scale_h = f32_to_fp16_fast((w_max - w_min) * (1.0f / 15.0f));
    *dst_scale = scale_h;
    *dst_bias  = bias_h;
    const float scale_eff = fp16_to_f32_fast(scale_h);
    const float bias_eff  = fp16_to_f32_fast(bias_h);
    if (!(scale_eff > 0.0f)) {
        std::memset(dst_u32, 0, 8u * sizeof(uint32_t));
        return;
    }
    const float inv_s = 1.0f / scale_eff;
#if defined(__aarch64__)
    const float32x4_t v_bias = vdupq_n_f32(bias_eff);
    const float32x4_t v_invs = vdupq_n_f32(inv_s);
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    const float32x4_t v_15   = vdupq_n_f32(15.0f);
    for (int u = 0; u < 8; ++u) {
        float32x4_t f0 = vrndnq_f32(vmulq_f32(vsubq_f32(vld1q_f32(vals + u * 8 + 0), v_bias), v_invs));
        float32x4_t f1 = vrndnq_f32(vmulq_f32(vsubq_f32(vld1q_f32(vals + u * 8 + 4), v_bias), v_invs));
        f0 = vminq_f32(v_15, vmaxq_f32(v_zero, f0));
        f1 = vminq_f32(v_15, vmaxq_f32(v_zero, f1));
        uint32x4_t q0 = vcvtq_u32_f32(f0);
        uint32x4_t q1 = vcvtq_u32_f32(f1);
        dst_u32[u] =
            (vgetq_lane_u32(q0, 0) << 0u)  | (vgetq_lane_u32(q0, 1) << 4u)  |
            (vgetq_lane_u32(q0, 2) << 8u)  | (vgetq_lane_u32(q0, 3) << 12u) |
            (vgetq_lane_u32(q1, 0) << 16u) | (vgetq_lane_u32(q1, 1) << 20u) |
            (vgetq_lane_u32(q1, 2) << 24u) | (vgetq_lane_u32(q1, 3) << 28u);
    }
#else
    for (int u = 0; u < 8; ++u) {
        uint32_t packed = 0u;
        for (int j = 0; j < 8; ++j) {
            float qf = std::nearbyintf((vals[u * 8 + j] - bias_eff) * inv_s);
            if (qf < 0.0f) qf = 0.0f;
            else if (qf > 15.0f) qf = 15.0f;
            packed |= (static_cast<uint32_t>(qf) & 0xFu) << (4u * j);
        }
        dst_u32[u] = packed;
    }
#endif
}

static inline void quantize_group64_fp16_to_q4a(
    const uint16_t* src_fp16,
    uint32_t*       dst_u32,
    uint16_t*       dst_scale,
    uint16_t*       dst_bias
) {
    float vals[64];
#if defined(__aarch64__)
    for (int k = 0; k < 16; ++k) {
        float16x4_t h4 = vreinterpret_f16_u16(vld1_u16(src_fp16 + k * 4));
        vst1q_f32(vals + k * 4, vcvt_f32_f16(h4));
    }
#else
    for (int i = 0; i < 64; ++i) vals[i] = fp16_to_f32_fast(src_fp16[i]);
#endif
    quantize_group64_f32_to_q4a(vals, dst_u32, dst_scale, dst_bias);
}

#if !defined(__aarch64__)
static inline uint8_t f32_to_e2m1_nibble_scalar(float v) {
    const uint8_t sign = (v < 0.0f) ? 8u : 0u;
    const float a = std::fabs(v);
    uint8_t mag = 0u;
    if (a > 0.25f) {
        if (a < 0.75f)       mag = 1u;
        else if (a <= 1.25f) mag = 2u;
        else if (a < 1.75f)  mag = 3u;
        else if (a <= 2.50f) mag = 4u;
        else if (a < 3.50f)  mag = 5u;
        else if (a <= 5.00f) mag = 6u;
        else                 mag = 7u;
    }
    return static_cast<uint8_t>(sign | mag);
}
#endif

static bool repack_mxfp8_expert_to_nvfp4(
    const uint8_t* src_w,
    const uint8_t* src_s,
    uint32_t       rows,
    uint32_t       cols,
    uint8_t*       dst_w,
    uint8_t*       dst_s,
    float*         dst_outer_scale
) {
    if ((cols % 32u) != 0u) return false;
    const Fp8LutTables& lut = get_fp8_luts();
    const uint64_t blk32_total = static_cast<uint64_t>(rows) * (cols / 32u);

    /* Pass 1: Find per-expert amax & verify 0 NaNs (F8_E4M3FN magnitude is monotonic in c & 0x7F!) */
    float expert_amax = 0.0f;
#if defined(__aarch64__)
    const uint8x16_t v_mag_mask = vdupq_n_u8(0x7Fu);
#endif
    for (uint64_t b32 = 0; b32 < blk32_total; ++b32) {
        const uint8_t ues = src_s[b32];
        if (ues == 0xFFu) return false;
        const uint8_t* wp = src_w + b32 * 32ULL;
        uint8_t max_mag = 0u;
#if defined(__aarch64__)
        uint8x16_t m0 = vandq_u8(vld1q_u8(wp),      v_mag_mask);
        uint8x16_t m1 = vandq_u8(vld1q_u8(wp + 16), v_mag_mask);
        max_mag = vmaxvq_u8(vmaxq_u8(m0, m1));
#else
        for (int j = 0; j < 32; ++j) {
            uint8_t m = wp[j] & 0x7Fu;
            if (m > max_mag) max_mag = m;
        }
#endif
        if (max_mag == 0x7Fu) return false;
        const float bmax = lut.e4m3[max_mag] * lut.ue8m0[ues];
        if (bmax > expert_amax) expert_amax = bmax;
    }

    if (!(expert_amax > 0.0f) || !std::isfinite(expert_amax)) {
        if (!std::isfinite(expert_amax)) return false;
        *dst_outer_scale = 0.0f;
        std::memset(dst_w, 0, static_cast<size_t>(rows) * (cols / 2u));
        std::memset(dst_s, 0, static_cast<size_t>(rows) * (cols / 16u));
        return true;
    }

    const float outer_scale = expert_amax / (6.0f * 448.0f);
    if (!std::isfinite(outer_scale) || !(outer_scale > 0.0f)) return false;
    *dst_outer_scale = outer_scale;
    const float inv_6_outer = 1.0f / (6.0f * outer_scale);

#if defined(__aarch64__)
    const float32x4_t v_t0 = vdupq_n_f32(0.25f);
    const float32x4_t v_t1 = vdupq_n_f32(0.75f);
    const float32x4_t v_t2 = vdupq_n_f32(1.25f);
    const float32x4_t v_t3 = vdupq_n_f32(1.75f);
    const float32x4_t v_t4 = vdupq_n_f32(2.50f);
    const float32x4_t v_t5 = vdupq_n_f32(3.50f);
    const float32x4_t v_t6 = vdupq_n_f32(5.00f);
#endif

    /* Pass 2: Quantize each 16-element half-block to F8_E4M3FN scale + packed E2M1 nibbles */
    for (uint64_t b32 = 0; b32 < blk32_total; ++b32) {
        const float ue_scale = lut.ue8m0[src_s[b32]];
        for (uint64_t h = 0; h < 2ULL; ++h) {
            const uint64_t b16 = b32 * 2ULL + h;
            const uint8_t* wp16 = src_w + b16 * 16ULL;
            uint8_t*       dw8  = dst_w + b16 * 8ULL;

            uint8_t mag16 = 0u;
#if defined(__aarch64__)
            mag16 = vmaxvq_u8(vandq_u8(vld1q_u8(wp16), v_mag_mask));
#else
            for (int j = 0; j < 16; ++j) {
                uint8_t m = wp16[j] & 0x7Fu;
                if (m > mag16) mag16 = m;
            }
#endif
            const float blk_amax = lut.e4m3[mag16] * ue_scale;
            const uint8_t sc_byte = f32_to_e4m3fn_pos(blk_amax * inv_6_outer);
            dst_s[b16] = sc_byte;

            if (sc_byte == 0u) {
                std::memset(dw8, 0, 8);
                continue;
            }
            const float mult = ue_scale / (lut.e4m3[sc_byte] * outer_scale);
#if defined(__aarch64__)
            const float32x4_t v_mult = vdupq_n_f32(mult);
            for (int q = 0; q < 4; ++q) {
                const float raw4[4] = {
                    lut.e4m3[wp16[q * 4 + 0]],
                    lut.e4m3[wp16[q * 4 + 1]],
                    lut.e4m3[wp16[q * 4 + 2]],
                    lut.e4m3[wp16[q * 4 + 3]]
                };
                const float32x4_t v    = vmulq_f32(vld1q_f32(raw4), v_mult);
                const uint32x4_t  sign = vshlq_n_u32(vshrq_n_u32(vreinterpretq_u32_f32(v), 31), 3); /* 0 or 8 */
                const float32x4_t a    = vabsq_f32(v);
                uint32x4_t mag = vdupq_n_u32(0u);
                mag = vsubq_u32(mag, vcgtq_f32(a, v_t0));
                mag = vsubq_u32(mag, vcgeq_f32(a, v_t1));
                mag = vsubq_u32(mag, vcgtq_f32(a, v_t2));
                mag = vsubq_u32(mag, vcgeq_f32(a, v_t3));
                mag = vsubq_u32(mag, vcgtq_f32(a, v_t4));
                mag = vsubq_u32(mag, vcgeq_f32(a, v_t5));
                mag = vsubq_u32(mag, vcgtq_f32(a, v_t6));
                const uint32x4_t nib = vorrq_u32(sign, mag);
                const uint32_t n0 = vgetq_lane_u32(nib, 0);
                const uint32_t n1 = vgetq_lane_u32(nib, 1);
                const uint32_t n2 = vgetq_lane_u32(nib, 2);
                const uint32_t n3 = vgetq_lane_u32(nib, 3);
                dw8[q * 2 + 0] = static_cast<uint8_t>((n0 & 0xFu) | ((n1 & 0xFu) << 4));
                dw8[q * 2 + 1] = static_cast<uint8_t>((n2 & 0xFu) | ((n3 & 0xFu) << 4));
            }
#else
            for (int k = 0; k < 8; ++k) {
                const uint8_t n0 = f32_to_e2m1_nibble_scalar(lut.e4m3[wp16[k * 2 + 0]] * mult);
                const uint8_t n1 = f32_to_e2m1_nibble_scalar(lut.e4m3[wp16[k * 2 + 1]] * mult);
                dw8[k] = static_cast<uint8_t>((n0 & 0xFu) | ((n1 & 0xFu) << 4));
            }
#endif
        }
    }
    return true;
}

static double current_peak_rss_gib() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#if defined(__APPLE__)
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
#else
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
#endif
}

static bool pread_exact(int fd, void* buf, size_t count, uint64_t offset) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t r = pread(fd, p + done, count - done, static_cast<off_t>(offset + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;
        done += static_cast<size_t>(r);
    }
    return true;
}

static bool pwrite_exact(int fd, const void* buf, size_t count, uint64_t offset) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t w = pwrite(fd, p + done, count - done, static_cast<off_t>(offset + done));
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        done += static_cast<size_t>(w);
    }
    return true;
}

static bool pwrite_zeros(int fd, uint64_t offset, uint64_t count) {
    if (count == 0) return true;
    uint8_t zbuf[65536];
    std::memset(zbuf, 0, sizeof(zbuf));
    uint64_t rem = count;
    uint64_t cur = offset;
    while (rem > 0) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(sizeof(zbuf), rem));
        if (!pwrite_exact(fd, zbuf, chunk, cur)) return false;
        cur += chunk;
        rem -= chunk;
    }
    return true;
}

static bool file_exists_nonempty(const std::string& path, uint64_t* size_out = nullptr) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) {
        return false;
    }
    if (size_out) *size_out = static_cast<uint64_t>(st.st_size);
    return true;
}

static bool mkdirs(const std::string& dir) {
    if (dir.empty()) return false;
    std::string cur;
    if (dir[0] == '/') cur = "/";
    size_t i = (dir[0] == '/') ? 1 : 0;
    while (i <= dir.size()) {
        size_t slash = dir.find('/', i);
        std::string part = (slash == std::string::npos) ? dir.substr(i) : dir.substr(i, slash - i);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') cur.push_back('/');
            cur += part;
            struct stat st;
            if (stat(cur.c_str(), &st) != 0) {
                if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
            } else if (!S_ISDIR(st.st_mode)) {
                return false;
            }
        }
        if (slash == std::string::npos) break;
        i = slash + 1;
    }
    return true;
}

static bool read_whole_file(const std::string& path, std::string& out) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        return false;
    }
    out.resize(static_cast<size_t>(st.st_size));
    bool ok = out.empty() ? true : pread_exact(fd, out.data(), out.size(), 0);
    close(fd);
    return ok;
}

static bool copy_file_if_exists(const std::string& src, const std::string& dst, bool required) {
    if (src == dst) return file_exists_nonempty(src) || !required;
    struct stat st_src, st_dst;
    if (stat(src.c_str(), &st_src) == 0 && stat(dst.c_str(), &st_dst) == 0 &&
        st_src.st_dev == st_dst.st_dev && st_src.st_ino == st_dst.st_ino) {
        return true;
    }
    int in_fd = open(src.c_str(), O_RDONLY);
    if (in_fd < 0) return !required;
    if (fstat(in_fd, &st_src) != 0) {
        close(in_fd);
        return false;
    }
    std::string tmp_dst = dst + ".part";
    int out_fd = open(tmp_dst.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (out_fd < 0) {
        close(in_fd);
        return false;
    }
    std::vector<uint8_t> buf(1024 * 1024);
    uint64_t off = 0;
    uint64_t total = static_cast<uint64_t>(st_src.st_size);
    bool ok = true;
    while (off < total) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(buf.size(), total - off));
        if (!pread_exact(in_fd, buf.data(), chunk, off) ||
            !pwrite_exact(out_fd, buf.data(), chunk, off)) {
            ok = false;
            break;
        }
        off += chunk;
    }
    close(out_fd);
    close(in_fd);
    if (!ok || rename(tmp_dst.c_str(), dst.c_str()) != 0) {
        unlink(tmp_dst.c_str());
        return false;
    }
    return true;
}

static void try_preallocate_apfs(int fd, uint64_t length_bytes) {
#if defined(__APPLE__) && defined(F_PREALLOCATE)
    if (fd < 0 || length_bytes == 0) return;
    fstore_t fst;
    std::memset(&fst, 0, sizeof(fst));
    fst.fst_flags   = F_ALLOCATECONTIG | F_ALLOCATEALL;
    fst.fst_posmode = F_PEOFPOSMODE;
    fst.fst_offset  = 0;
    fst.fst_length  = static_cast<off_t>(length_bytes);
    if (fcntl(fd, F_PREALLOCATE, &fst) != 0) {
        fst.fst_flags = F_ALLOCATEALL;
        (void)fcntl(fd, F_PREALLOCATE, &fst);
    }
#else
    (void)fd;
    (void)length_bytes;
#endif
}

/* ============================================================================
 * 2. UTF-8 Helpers & Lightweight RFC-8259 JSON Scanner + Small DOM
 * ============================================================================ */

static inline void append_utf8_codepoint(std::string& out, uint32_t cp) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

static inline uint32_t decode_utf8_one(const uint8_t* s, size_t len, size_t* adv) {
    if (len == 0) {
        *adv = 0;
        return 0;
    }
    uint8_t b0 = s[0];
    if (b0 < 0x80u) {
        *adv = 1;
        return b0;
    }
    if ((b0 & 0xE0u) == 0xC0u && len >= 2 && (s[1] & 0xC0u) == 0x80u) {
        uint32_t cp = ((static_cast<uint32_t>(b0 & 0x1Fu)) << 6) |
                      static_cast<uint32_t>(s[1] & 0x3Fu);
        if (cp >= 0x80u) {
            *adv = 2;
            return cp;
        }
    } else if ((b0 & 0xF0u) == 0xE0u && len >= 3 &&
               (s[1] & 0xC0u) == 0x80u && (s[2] & 0xC0u) == 0x80u) {
        uint32_t cp = ((static_cast<uint32_t>(b0 & 0x0Fu)) << 12) |
                      ((static_cast<uint32_t>(s[1] & 0x3Fu)) << 6) |
                      static_cast<uint32_t>(s[2] & 0x3Fu);
        if (cp >= 0x800u) {
            *adv = 3;
            return cp;
        }
    } else if ((b0 & 0xF8u) == 0xF0u && len >= 4 &&
               (s[1] & 0xC0u) == 0x80u && (s[2] & 0xC0u) == 0x80u && (s[3] & 0xC0u) == 0x80u) {
        uint32_t cp = ((static_cast<uint32_t>(b0 & 0x07u)) << 18) |
                      ((static_cast<uint32_t>(s[1] & 0x3Fu)) << 12) |
                      ((static_cast<uint32_t>(s[2] & 0x3Fu)) << 6) |
                      static_cast<uint32_t>(s[3] & 0x3Fu);
        if (cp >= 0x10000u && cp <= 0x10FFFFu) {
            *adv = 4;
            return cp;
        }
    }
    *adv = 1;
    return 0xFFFDu;
}

struct JsonCursor {
    const char* s;
    size_t      n;
    size_t      p;

    JsonCursor(const char* str, size_t len) : s(str), n(len), p(0) {}

    inline void skip_ws() {
        while (p < n) {
            unsigned char c = static_cast<unsigned char>(s[p]);
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p;
            else break;
        }
    }

    inline bool consume(char expected) {
        skip_ws();
        if (p < n && s[p] == expected) {
            ++p;
            return true;
        }
        return false;
    }

    inline char peek() {
        skip_ws();
        return (p < n) ? s[p] : '\0';
    }

    static inline int hex_val(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    }

    bool parse_hex4(uint32_t& out) {
        if (p + 4 > n) return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            int h = hex_val(s[p + i]);
            if (h < 0) return false;
            v = (v << 4) | static_cast<uint32_t>(h);
        }
        p += 4;
        out = v;
        return true;
    }

    bool parse_string(std::string& out) {
        skip_ws();
        if (p >= n || s[p] != '"') return false;
        ++p;
        out.clear();
        while (p < n) {
            const char* start = s + p;
            size_t run = 0;
            while (p + run < n) {
                char c = s[p + run];
                if (c == '"' || c == '\\') break;
                ++run;
            }
            if (run > 0) {
                out.append(start, run);
                p += run;
            }
            if (p >= n) return false;
            char c = s[p++];
            if (c == '"') return true;
            if (p >= n) return false;
            char esc = s[p++];
            switch (esc) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t u1 = 0;
                    if (!parse_hex4(u1)) return false;
                    if (u1 >= 0xD800u && u1 <= 0xDBFFu) {
                        if (p + 2 <= n && s[p] == '\\' && s[p + 1] == 'u') {
                            p += 2;
                            uint32_t u2 = 0;
                            if (!parse_hex4(u2)) return false;
                            if (u2 >= 0xDC00u && u2 <= 0xDFFFu) {
                                uint32_t cp = 0x10000u + (((u1 - 0xD800u) << 10) | (u2 - 0xDC00u));
                                append_utf8_codepoint(out, cp);
                                break;
                            }
                        }
                    }
                    append_utf8_codepoint(out, u1);
                    break;
                }
                default:
                    return false;
            }
        }
        return false;
    }

    bool parse_number(double& out_d, int64_t& out_i) {
        skip_ws();
        if (p >= n) return false;
        const char* start = s + p;
        char* end_d = nullptr;
        out_d = std::strtod(start, &end_d);
        if (end_d == start) return false;
        char* end_i = nullptr;
        out_i = std::strtoll(start, &end_i, 10);
        if (end_i != end_d) out_i = static_cast<int64_t>(out_d);
        p = static_cast<size_t>(end_d - s);
        return true;
    }

    bool parse_uint64(uint64_t& out_u) {
        skip_ws();
        if (p >= n) return false;
        const char* start = s + p;
        char* end_u = nullptr;
        unsigned long long v = std::strtoull(start, &end_u, 10);
        if (end_u == start) return false;
        out_u = static_cast<uint64_t>(v);
        p = static_cast<size_t>(end_u - s);
        return true;
    }

    bool parse_bool(bool& out_b) {
        skip_ws();
        if (p + 4 <= n && std::memcmp(s + p, "true", 4) == 0) {
            p += 4;
            out_b = true;
            return true;
        }
        if (p + 5 <= n && std::memcmp(s + p, "false", 5) == 0) {
            p += 5;
            out_b = false;
            return true;
        }
        return false;
    }

    bool skip_value(int depth = 0) {
        if (depth > 128) return false;
        skip_ws();
        if (p >= n) return false;
        char c = s[p];
        if (c == '"') {
            std::string tmp;
            return parse_string(tmp);
        }
        if (c == '{') {
            ++p;
            if (consume('}')) return true;
            std::string key;
            while (true) {
                if (!parse_string(key)) return false;
                if (!consume(':')) return false;
                if (!skip_value(depth + 1)) return false;
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == '[') {
            ++p;
            if (consume(']')) return true;
            while (true) {
                if (!skip_value(depth + 1)) return false;
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == 't' || c == 'f') {
            bool b = false;
            return parse_bool(b);
        }
        if (c == 'n') {
            if (p + 4 <= n && std::memcmp(s + p, "null", 4) == 0) {
                p += 4;
                return true;
            }
            return false;
        }
        double d = 0.0;
        int64_t iv = 0;
        return parse_number(d, iv);
    }
};

struct SmallJson {
    enum Type { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } type = J_NULL;
    bool b_val = false;
    double d_val = 0.0;
    int64_t i_val = 0;
    std::string s_val;
    std::vector<SmallJson> arr;
    std::map<std::string, SmallJson> obj;

    const SmallJson* get(const std::string& k) const {
        if (type != J_OBJ) return nullptr;
        auto it = obj.find(k);
        return (it == obj.end()) ? nullptr : &it->second;
    }
    int64_t get_i64(const std::string& k, int64_t def) const {
        const SmallJson* v = get(k);
        return (v && v->type == J_NUM) ? v->i_val : def;
    }
    double get_f64(const std::string& k, double def) const {
        const SmallJson* v = get(k);
        return (v && v->type == J_NUM) ? v->d_val : def;
    }
    bool get_bool(const std::string& k, bool def) const {
        const SmallJson* v = get(k);
        return (v && v->type == J_BOOL) ? v->b_val : def;
    }
};

static bool parse_small_json_rec(JsonCursor& cur, SmallJson& out, int depth = 0) {
    if (depth > 128) return false;
    char c = cur.peek();
    if (c == '\0') return false;
    if (c == '"') {
        out.type = SmallJson::J_STR;
        return cur.parse_string(out.s_val);
    }
    if (c == '{') {
        cur.consume('{');
        out.type = SmallJson::J_OBJ;
        if (cur.consume('}')) return true;
        while (true) {
            std::string k;
            if (!cur.parse_string(k)) return false;
            if (!cur.consume(':')) return false;
            SmallJson child;
            if (!parse_small_json_rec(cur, child, depth + 1)) return false;
            out.obj.emplace(std::move(k), std::move(child));
            if (cur.consume('}')) return true;
            if (!cur.consume(',')) return false;
        }
    }
    if (c == '[') {
        cur.consume('[');
        out.type = SmallJson::J_ARR;
        if (cur.consume(']')) return true;
        while (true) {
            SmallJson elem;
            if (!parse_small_json_rec(cur, elem, depth + 1)) return false;
            out.arr.push_back(std::move(elem));
            if (cur.consume(']')) return true;
            if (!cur.consume(',')) return false;
        }
    }
    if (c == 't' || c == 'f') {
        out.type = SmallJson::J_BOOL;
        return cur.parse_bool(out.b_val);
    }
    if (c == 'n') {
        if (cur.p + 4 <= cur.n && std::memcmp(cur.s + cur.p, "null", 4) == 0) {
            cur.p += 4;
            out.type = SmallJson::J_NULL;
            return true;
        }
        return false;
    }
    out.type = SmallJson::J_NUM;
    return cur.parse_number(out.d_val, out.i_val);
}

static bool parse_small_json(const std::string& text, SmallJson& out) {
    JsonCursor cur(text.data(), text.size());
    return parse_small_json_rec(cur, out);
}

/* ============================================================================
 * 3. Safetensors Header Parser
 * ============================================================================ */

enum SafetensorDtype {
    ST_UNKNOWN = 0,
    ST_BF16    = 1,
    ST_F16     = 2,
    ST_F32     = 3,
    ST_U8      = 4,
    ST_F8_E4M3 = 5,
    ST_U32     = 6,
    ST_I64     = 7
};

static uint64_t st_dtype_bpe(SafetensorDtype dt) {
    switch (dt) {
        case ST_U8:
        case ST_F8_E4M3: return 1ULL;
        case ST_BF16:
        case ST_F16:     return 2ULL;
        case ST_F32:
        case ST_U32:     return 4ULL;
        case ST_I64:     return 8ULL;
        default:         return 0ULL;
    }
}

static SafetensorDtype parse_st_dtype(const std::string& s) {
    if (s == "BF16")    return ST_BF16;
    if (s == "F16")     return ST_F16;
    if (s == "F32")     return ST_F32;
    if (s == "U8")      return ST_U8;
    if (s == "F8_E4M3") return ST_F8_E4M3;
    if (s == "U32")     return ST_U32;
    if (s == "I64")     return ST_I64;
    return ST_UNKNOWN;
}

struct SafetensorEntry {
    std::string           raw_name;
    std::string           canon_name;
    int                   shard_fd   = -1;
    SafetensorDtype       dtype      = ST_UNKNOWN;
    std::vector<uint64_t> shape;
    uint64_t              abs_offset = 0;
    uint64_t              byte_len   = 0;
    bool                  used       = false;
};

static bool is_vision_tensor_name(const std::string& key) {
    return key.rfind("model.visual.", 0) == 0 ||
           key.rfind("visual.", 0) == 0 ||
           key.rfind("vision_tower.", 0) == 0 ||
           key.rfind("model.vision_tower.", 0) == 0;
}

static std::string canonicalize_backbone_name(const std::string& key) {
    std::string out = key;
    if (out.rfind("model.llm.", 0) == 0) {
        out = std::string("model.") + out.substr(std::strlen("model.llm."));
    } else if (out.rfind("model.language_model.", 0) == 0) {
        out = std::string("model.") + out.substr(std::strlen("model.language_model."));
    } else if (out.rfind("language_model.model.", 0) == 0) {
        out = std::string("model.") + out.substr(std::strlen("language_model.model."));
    }
    const std::string alias = ".mlp.gate.e_score_correction_bias";
    if (out.size() >= alias.size() &&
        out.compare(out.size() - alias.size(), alias.size(), alias) == 0) {
        out.replace(out.size() - alias.size(), alias.size(), ".mlp.gate.bias");
    }
    return out;
}

static std::string canonicalize_mtp_name(const std::string& key) {
    if (key.rfind("mtp.", 0) == 0) {
        return key.substr(4);
    }
    return key;
}

static bool parse_safetensors_file(
    int fd,
    uint64_t file_bytes,
    bool is_mtp,
    std::unordered_map<std::string, SafetensorEntry>& out_map,
    uint32_t* skipped_vision_out
) {
    uint8_t len_buf[8];
    if (!pread_exact(fd, len_buf, 8, 0)) return false;
    uint64_t hdr_len = 0;
    for (int i = 0; i < 8; ++i) {
        hdr_len |= static_cast<uint64_t>(len_buf[i]) << (8 * i);
    }
    if (hdr_len == 0 || hdr_len > 128ULL * 1024ULL * 1024ULL || 8ULL + hdr_len > file_bytes) {
        return false;
    }
    std::string hdr_json(static_cast<size_t>(hdr_len), '\0');
    if (!pread_exact(fd, hdr_json.data(), static_cast<size_t>(hdr_len), 8)) return false;

    const uint64_t data_base = 8ULL + hdr_len;
    JsonCursor cur(hdr_json.data(), hdr_json.size());
    if (!cur.consume('{')) return false;
    if (cur.consume('}')) return true;

    std::string key, prop, dt_str;
    while (true) {
        if (!cur.parse_string(key)) return false;
        if (!cur.consume(':')) return false;
        if (key == "__metadata__") {
            if (!cur.skip_value()) return false;
        } else {
            if (!cur.consume('{')) return false;
            SafetensorEntry ent;
            ent.raw_name = key;
            ent.shard_fd = fd;
            uint64_t off0 = 0, off1 = 0;
            bool has_offsets = false;
            if (!cur.consume('}')) {
                while (true) {
                    if (!cur.parse_string(prop)) return false;
                    if (!cur.consume(':')) return false;
                    if (prop == "dtype") {
                        if (!cur.parse_string(dt_str)) return false;
                        ent.dtype = parse_st_dtype(dt_str);
                    } else if (prop == "shape") {
                        if (!cur.consume('[')) return false;
                        if (!cur.consume(']')) {
                            while (true) {
                                uint64_t dim = 0;
                                if (!cur.parse_uint64(dim)) return false;
                                ent.shape.push_back(dim);
                                if (cur.consume(']')) break;
                                if (!cur.consume(',')) return false;
                            }
                        }
                    } else if (prop == "data_offsets") {
                        if (!cur.consume('[')) return false;
                        if (!cur.parse_uint64(off0)) return false;
                        if (!cur.consume(',')) return false;
                        if (!cur.parse_uint64(off1)) return false;
                        if (!cur.consume(']')) return false;
                        has_offsets = true;
                    } else {
                        if (!cur.skip_value()) return false;
                    }
                    if (cur.consume('}')) break;
                    if (!cur.consume(',')) return false;
                }
            }
            if (!has_offsets || off1 < off0 || data_base + off1 > file_bytes) {
                return false;
            }
            ent.abs_offset = data_base + off0;
            ent.byte_len   = off1 - off0;

            if (!is_mtp && is_vision_tensor_name(key)) {
                if (skipped_vision_out) ++(*skipped_vision_out);
            } else {
                const uint64_t bpe = st_dtype_bpe(ent.dtype);
                uint64_t elems = 1ULL;
                for (uint64_t d : ent.shape) elems *= d;
                if (bpe == 0ULL || elems * bpe != ent.byte_len) {
                    std::fprintf(stderr, "[inkling2-conv] ERROR: invalid dtype/byte_len on '%s'\n",
                                 key.c_str());
                    return false;
                }
                ent.canon_name = is_mtp ? canonicalize_mtp_name(key) : canonicalize_backbone_name(key);
                out_map.emplace(ent.canon_name, std::move(ent));
            }
        }
        if (cur.consume('}')) break;
        if (!cur.consume(',')) return false;
    }
    return true;
}

/* ============================================================================
 * 4. Config Parsing & Deterministic 16 KiB-Aligned UMA Layout Planner
 * ============================================================================ */

static void init_default_config(StrataInkling2Config& cfg, StrataInkling2MtpDesc& mtp) {
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.num_phys_layers         = 24u;
    cfg.num_logical_layers      = 48u;
    cfg.hidden_size             = 2304u;
    cfg.vocab_size_padded       = 201024u;
    cfg.vocab_size_unpadded     = 200058u;
    cfg.num_attention_heads     = 24u;
    cfg.num_key_value_heads     = 8u;
    cfg.head_dim                = 128u;
    cfg.d_rel                   = 16u;
    cfg.rel_extent              = 1024u;
    cfg.sliding_window_size     = 1024u;
    cfg.sconv_kernel_size       = 4u;
    cfg.attn_res_block_size     = 6u;
    cfg.num_experts             = 128u;
    cfg.num_experts_per_tok     = 8u;
    cfg.intermediate_size       = 1152u;
    cfg.loop_core_start         = 6u;
    cfg.loop_core_end           = 18u;
    cfg.loop_num_passes         = 3u;
    cfg.loop_gate_rank          = 64u;
    cfg.log_scaling_n_floor     = 8192u;
    cfg.max_position_embeddings = 1048576u;
    cfg.eos_token_id            = STRATA_INKLING2_TOKEN_EOS;
    cfg.pad_token_id            = STRATA_INKLING2_TOKEN_ENDOFTEXT;
    cfg.rms_norm_eps            = 1e-6f;
    cfg.qk_l2_clip_threshold    = 22.627417f;
    cfg.rel_l2_clip_threshold   = 8.0f;
    cfg.log_scaling_alpha       = 0.1f;
    cfg.mlp_gate_softcap        = 4.0f;
    cfg.mlp_up_softcap          = 25.0f;
    cfg.route_scale             = 8.0f;
    cfg.final_logit_softcapping = 30.0f;

    std::memset(&mtp, 0, sizeof(mtp));
    mtp.enabled             = 0u;
    mtp.num_layers          = 3u;
    mtp.intermediate_size   = 9216u;
    mtp.block_size          = 8u;
    mtp.markov_rank         = 256u;
    mtp.sliding_window      = 1024u;
    mtp.rope_theta          = 1000000.0f;
    mtp.rms_norm_eps        = 1e-6f;
    mtp.target_layer_ids[0] = 5u;
    mtp.target_layer_ids[1] = 23u;
    mtp.target_layer_ids[2] = 41u;
    mtp.target_layer_ids[3] = 48u;
}

static bool parse_inkling2_configs(
    const std::string& hf_dir,
    StrataInkling2Config& cfg,
    StrataInkling2MtpDesc& mtp,
    uint32_t phys_is_local[STRATA_INKLING2_MAX_PHYS_LAYERS],
    uint32_t mtp_is_local[STRATA_INKLING2_MAX_MTP_LAYERS],
    bool* use_gate_bias_out
) {
    init_default_config(cfg, mtp);
    std::memset(phys_is_local, 0, sizeof(uint32_t) * STRATA_INKLING2_MAX_PHYS_LAYERS);
    std::memset(mtp_is_local,  0, sizeof(uint32_t) * STRATA_INKLING2_MAX_MTP_LAYERS);

    std::string raw;
    if (!read_whole_file(hf_dir + "/config.json", raw)) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: cannot read '%s/config.json'\n", hf_dir.c_str());
        return false;
    }
    SmallJson root;
    if (!parse_small_json(raw, root) || root.type != SmallJson::J_OBJ) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: invalid JSON in '%s/config.json'\n", hf_dir.c_str());
        return false;
    }

    const SmallJson* tc = root.get("text_config");
    if (!tc || tc->type != SmallJson::J_OBJ) tc = &root;

    cfg.num_phys_layers         = static_cast<uint32_t>(tc->get_i64("num_hidden_layers", cfg.num_phys_layers));
    cfg.hidden_size             = static_cast<uint32_t>(tc->get_i64("hidden_size", cfg.hidden_size));
    cfg.vocab_size_padded       = static_cast<uint32_t>(tc->get_i64("vocab_size", cfg.vocab_size_padded));
    cfg.vocab_size_unpadded     = static_cast<uint32_t>(tc->get_i64("unpadded_vocab_size", cfg.vocab_size_unpadded));
    cfg.num_attention_heads     = static_cast<uint32_t>(tc->get_i64("num_attention_heads", cfg.num_attention_heads));
    cfg.num_key_value_heads     = static_cast<uint32_t>(tc->get_i64("num_key_value_heads", cfg.num_key_value_heads));
    cfg.head_dim                = static_cast<uint32_t>(tc->get_i64("head_dim", cfg.head_dim));
    cfg.d_rel                   = static_cast<uint32_t>(tc->get_i64("d_rel", cfg.d_rel));
    cfg.rel_extent              = static_cast<uint32_t>(tc->get_i64("rel_extent", cfg.rel_extent));
    cfg.sliding_window_size     = static_cast<uint32_t>(tc->get_i64("sliding_window_size", cfg.sliding_window_size));
    cfg.sconv_kernel_size       = static_cast<uint32_t>(tc->get_i64("sconv_kernel_size",
                                                                    tc->get_i64("conv_kernel_size", cfg.sconv_kernel_size)));
    cfg.attn_res_block_size     = static_cast<uint32_t>(tc->get_i64("n_layers_per_attn_res_block", cfg.attn_res_block_size));
    cfg.num_experts             = static_cast<uint32_t>(tc->get_i64("n_routed_experts", cfg.num_experts));
    cfg.num_experts_per_tok     = static_cast<uint32_t>(tc->get_i64("num_experts_per_tok", cfg.num_experts_per_tok));
    cfg.intermediate_size       = static_cast<uint32_t>(tc->get_i64("moe_intermediate_size",
                                                                    tc->get_i64("intermediate_size", cfg.intermediate_size)));
    cfg.log_scaling_n_floor     = static_cast<uint32_t>(tc->get_i64("log_scaling_n_floor", cfg.log_scaling_n_floor));
    cfg.max_position_embeddings = static_cast<uint32_t>(tc->get_i64("model_max_length",
                                                                    tc->get_i64("max_position_embeddings", cfg.max_position_embeddings)));
    cfg.eos_token_id            = static_cast<uint32_t>(root.get_i64("eos_token_id",
                                                                     tc->get_i64("eos_token_id", cfg.eos_token_id)));
    cfg.pad_token_id            = static_cast<uint32_t>(root.get_i64("pad_token_id",
                                                                     tc->get_i64("pad_token_id", cfg.pad_token_id)));

    cfg.rms_norm_eps            = static_cast<float>(tc->get_f64("rms_norm_eps", cfg.rms_norm_eps));
    cfg.qk_l2_clip_threshold    = static_cast<float>(tc->get_f64("qk_l2_clip_threshold", cfg.qk_l2_clip_threshold));
    cfg.rel_l2_clip_threshold   = static_cast<float>(tc->get_f64("rel_l2_clip_threshold", cfg.rel_l2_clip_threshold));
    cfg.log_scaling_alpha       = static_cast<float>(tc->get_f64("log_scaling_alpha", cfg.log_scaling_alpha));
    cfg.mlp_gate_softcap        = static_cast<float>(tc->get_f64("mlp_gate_softcap", cfg.mlp_gate_softcap));
    cfg.mlp_up_softcap          = static_cast<float>(tc->get_f64("mlp_up_softcap", cfg.mlp_up_softcap));
    cfg.route_scale             = static_cast<float>(tc->get_f64("route_scale", cfg.route_scale));
    cfg.final_logit_softcapping = static_cast<float>(tc->get_f64("final_logit_softcapping", cfg.final_logit_softcapping));

    if (use_gate_bias_out) {
        *use_gate_bias_out = tc->get_bool("use_gate_bias", true);
    }

    if (cfg.num_phys_layers == 0 || cfg.num_phys_layers > STRATA_INKLING2_MAX_PHYS_LAYERS) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: unsupported num_hidden_layers=%u\n", cfg.num_phys_layers);
        return false;
    }

    /* Loop config */
    if (const SmallJson* lc = tc->get("loop_config")) {
        if (lc->type == SmallJson::J_OBJ) {
            cfg.loop_core_start = static_cast<uint32_t>(lc->get_i64("core_start", cfg.loop_core_start));
            cfg.loop_core_end   = static_cast<uint32_t>(lc->get_i64("core_end", cfg.loop_core_end));
            cfg.loop_num_passes = static_cast<uint32_t>(lc->get_i64("num_passes", cfg.loop_num_passes));
            cfg.loop_gate_rank  = static_cast<uint32_t>(lc->get_i64("gate_rank", cfg.loop_gate_rank));
        } else {
            cfg.loop_core_start = 0u;
            cfg.loop_core_end   = 0u;
            cfg.loop_num_passes = 1u;
            cfg.loop_gate_rank  = 0u;
        }
    } else {
        cfg.loop_core_start = 0u;
        cfg.loop_core_end   = 0u;
        cfg.loop_num_passes = 1u;
        cfg.loop_gate_rank  = 0u;
    }

    /* Local vs Global physical layers */
    if (const SmallJson* ltypes = tc->get("layer_types"); ltypes && ltypes->type == SmallJson::J_ARR &&
        ltypes->arr.size() == cfg.num_phys_layers) {
        for (uint32_t i = 0; i < cfg.num_phys_layers; ++i) {
            phys_is_local[i] = (ltypes->arr[i].type == SmallJson::J_STR &&
                                ltypes->arr[i].s_val == "hybrid_sliding") ? 1u : 0u;
        }
    } else if (const SmallJson* lids = tc->get("local_layer_ids"); lids && lids->type == SmallJson::J_ARR) {
        for (const auto& elem : lids->arr) {
            if (elem.type == SmallJson::J_NUM && elem.i_val >= 0 &&
                static_cast<uint32_t>(elem.i_val) < cfg.num_phys_layers) {
                phys_is_local[static_cast<uint32_t>(elem.i_val)] = 1u;
            }
        }
    } else {
        for (uint32_t i = 0; i < cfg.num_phys_layers; ++i) {
            phys_is_local[i] = ((i % 6u) == 5u) ? 0u : 1u;
        }
    }

    /* Build 48-visit logical layer schedule */
    uint32_t il = 0u;
    auto emit_visit = [&](uint32_t phys, uint32_t pass_idx) -> bool {
        if (il >= STRATA_INKLING2_MAX_LOGICAL_LAYERS) return false;
        cfg.logical_to_phys[il]  = phys;
        cfg.logical_pass_idx[il] = pass_idx;
        cfg.logical_is_local[il] = phys_is_local[phys];
        ++il;
        return true;
    };

    if (cfg.loop_num_passes > 1u) {
        if (cfg.loop_core_start >= cfg.loop_core_end || cfg.loop_core_end > cfg.num_phys_layers) {
            std::fprintf(stderr, "[inkling2-conv] ERROR: invalid loop_config [%u..%u) passes=%u\n",
                         cfg.loop_core_start, cfg.loop_core_end, cfg.loop_num_passes);
            return false;
        }
        for (uint32_t p = 0; p < cfg.loop_core_start; ++p) {
            if (!emit_visit(p, 0u)) return false;
        }
        for (uint32_t pass = 0; pass < cfg.loop_num_passes; ++pass) {
            for (uint32_t p = cfg.loop_core_start; p < cfg.loop_core_end; ++p) {
                if (!emit_visit(p, pass)) return false;
            }
        }
        for (uint32_t p = cfg.loop_core_end; p < cfg.num_phys_layers; ++p) {
            if (!emit_visit(p, 0u)) return false;
        }
    } else {
        for (uint32_t p = 0; p < cfg.num_phys_layers; ++p) {
            if (!emit_visit(p, 0u)) return false;
        }
    }
    cfg.num_logical_layers = il;

    uint32_t local_slot  = 0u;
    uint32_t global_slot = 0u;
    for (uint32_t v = 0; v < cfg.num_logical_layers; ++v) {
        if (cfg.logical_is_local[v]) {
            cfg.logical_kv_slot[v] = local_slot++;
        } else {
            cfg.logical_kv_slot[v] = global_slot++;
        }
    }
    cfg.num_local_visits  = local_slot;
    cfg.num_global_visits = global_slot;

    /* Optional generation_config.json overrides for eos/pad */
    std::string gen_raw;
    if (read_whole_file(hf_dir + "/generation_config.json", gen_raw)) {
        SmallJson gen_root;
        if (parse_small_json(gen_raw, gen_root) && gen_root.type == SmallJson::J_OBJ) {
            cfg.eos_token_id = static_cast<uint32_t>(gen_root.get_i64("eos_token_id", cfg.eos_token_id));
            cfg.pad_token_id = static_cast<uint32_t>(gen_root.get_i64("pad_token_id", cfg.pad_token_id));
        }
    }

    /* MTP config */
    if (const SmallJson* mc = root.get("mtp_config"); mc && mc->type == SmallJson::J_OBJ) {
        mtp.num_layers        = static_cast<uint32_t>(mc->get_i64("num_hidden_layers", mtp.num_layers));
        mtp.intermediate_size = static_cast<uint32_t>(mc->get_i64("intermediate_size", mtp.intermediate_size));
        mtp.block_size        = static_cast<uint32_t>(mc->get_i64("block_size", mtp.block_size));
        mtp.markov_rank       = static_cast<uint32_t>(mc->get_i64("markov_rank", mtp.markov_rank));
        mtp.sliding_window    = static_cast<uint32_t>(mc->get_i64("sliding_window_size", mtp.sliding_window));
        mtp.rope_theta        = static_cast<float>(mc->get_f64("rope_theta", mtp.rope_theta));
        mtp.rms_norm_eps      = static_cast<float>(mc->get_f64("rms_norm_eps", mtp.rms_norm_eps));
        if (const SmallJson* tids = mc->get("target_layer_ids"); tids && tids->type == SmallJson::J_ARR) {
            for (size_t k = 0; k < tids->arr.size() && k < STRATA_INKLING2_MAX_MTP_TAPS; ++k) {
                if (tids->arr[k].type == SmallJson::J_NUM) {
                    mtp.target_layer_ids[k] = static_cast<uint32_t>(tids->arr[k].i_val);
                }
            }
        }
        if (const SmallJson* mlids = mc->get("local_layer_ids"); mlids && mlids->type == SmallJson::J_ARR) {
            for (const auto& elem : mlids->arr) {
                if (elem.type == SmallJson::J_NUM && elem.i_val >= 0 &&
                    static_cast<uint32_t>(elem.i_val) < STRATA_INKLING2_MAX_MTP_LAYERS) {
                    mtp_is_local[static_cast<uint32_t>(elem.i_val)] = 1u;
                }
            }
        } else {
            mtp_is_local[0] = 1u;
            mtp_is_local[1] = 0u;
            mtp_is_local[2] = 1u;
        }
    } else {
        mtp_is_local[0] = 1u;
        mtp_is_local[1] = 0u;
        mtp_is_local[2] = 1u;
    }

    if (mtp.num_layers > STRATA_INKLING2_MAX_MTP_LAYERS) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: unsupported mtp.num_layers=%u\n", mtp.num_layers);
        return false;
    }

    return true;
}

static StrataInkling2TensorSlice alloc_slice(uint64_t& cur, uint32_t dtype, uint32_t shape0, uint32_t shape1) {
    StrataInkling2TensorSlice s{};
    cur = align_up_u64(cur, STRATA_INKLING2_TENSOR_ALIGN);
    const uint64_t elems = static_cast<uint64_t>(shape0) * static_cast<uint64_t>(shape1);
    uint64_t bytes = 0ULL;
    if (dtype == STRATA_INKLING2_DTYPE_Q8_0) {
        uint64_t wb = 0ULL, sb = 0ULL;
        strata_inkling2_q8_layout(shape0, shape1, &wb, &sb);
        bytes = wb + sb;
    } else if (dtype == STRATA_INKLING2_DTYPE_Q4_A) {
        uint64_t wb = 0ULL, sb = 0ULL, bb = 0ULL;
        strata_inkling2_q4a_layout(shape0, shape1, &wb, &sb, &bb);
        bytes = wb + sb + bb;
    } else {
        const uint64_t bpe = (dtype == STRATA_INKLING2_DTYPE_F32) ? 4ULL : 2ULL;
        bytes = elems * bpe;
    }
    s.offset     = cur;
    s.bytes      = bytes;
    s.elem_count = static_cast<uint32_t>(elems);
    s.dtype      = dtype;
    s.shape0     = shape0;
    s.shape1     = shape1;
    cur += s.bytes;
    return s;
}

static StrataInkling2ExpertMatrix alloc_expert_matrix(
    uint64_t& cur,
    uint32_t num_experts,
    uint32_t rows_per_expert,
    uint32_t cols,
    uint32_t dtype
) {
    StrataInkling2ExpertMatrix em{};
    em.num_experts     = num_experts;
    em.rows_per_expert = rows_per_expert;
    em.cols            = cols;
    em.dtype           = dtype;

    if (dtype == STRATA_INKLING2_DTYPE_NVFP4) {
        em.block_size = 16u;
        cur = align_up_u64(cur, STRATA_INKLING2_TENSOR_ALIGN);
        em.weight_off   = cur;
        em.weight_bytes = static_cast<uint64_t>(num_experts) * rows_per_expert * (cols / 2u);
        cur += em.weight_bytes;

        cur = align_up_u64(cur, STRATA_INKLING2_TENSOR_ALIGN);
        em.scales_off   = cur;
        em.scales_bytes = static_cast<uint64_t>(num_experts) * rows_per_expert * (cols / 16u);
        cur += em.scales_bytes;

        cur = align_up_u64(cur, STRATA_INKLING2_TENSOR_ALIGN);
        em.expert_scale_off   = cur;
        em.expert_scale_bytes = static_cast<uint64_t>(num_experts) * sizeof(float);
        cur += em.expert_scale_bytes;
    } else {
        em.block_size = 32u;
        cur = align_up_u64(cur, STRATA_INKLING2_TENSOR_ALIGN);
        em.weight_off   = cur;
        em.weight_bytes = static_cast<uint64_t>(num_experts) * rows_per_expert * cols;
        cur += em.weight_bytes;

        cur = align_up_u64(cur, STRATA_INKLING2_TENSOR_ALIGN);
        em.scales_off   = cur;
        em.scales_bytes = static_cast<uint64_t>(num_experts) * rows_per_expert * (cols / 32u);
        cur += em.scales_bytes;

        em.expert_scale_off   = 0ULL;
        em.expert_scale_bytes = 0ULL;
    }
    return em;
}

static uint64_t embed_row_active_bytes(const StrataInkling2TensorSlice& emb) {
    const uint64_t cols = emb.shape1;
    if (emb.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
        return (cols / 2ULL) + (cols / STRATA_INKLING2_QUANT_GROUP_SIZE) * 4ULL;
    }
    if (emb.dtype == STRATA_INKLING2_DTYPE_Q8_0) {
        return cols + (cols / STRATA_INKLING2_QUANT_GROUP_SIZE) * 2ULL;
    }
    if (emb.dtype == STRATA_INKLING2_DTYPE_F32) {
        return cols * 4ULL;
    }
    return cols * 2ULL;
}

static void compute_active_bytes(StrataInkling2Header& hdr) {
    const StrataInkling2Config& c = hdr.cfg;
    const uint32_t H = c.hidden_size;
    const uint64_t emb_row = embed_row_active_bytes(hdr.embed_tokens);

    /* Exact M=1 active physical DRAM bytes per token across all logical visits + LM head */
    uint64_t active = emb_row + hdr.embed_norm.bytes +
                      hdr.final_norm.bytes + hdr.output_res_query.bytes + hdr.unembed.bytes;

    for (uint32_t v = 0; v < c.num_logical_layers; ++v) {
        const uint32_t p    = c.logical_to_phys[v];
        const uint32_t pass = c.logical_pass_idx[v];
        const StrataInkling2LayerDesc& ld = hdr.layers[p];

        uint64_t layer_dense =
            ld.attn_res_query.bytes + ld.mlp_res_query.bytes +
            ld.attn_norm.bytes + ld.q_input_norm.bytes + ld.k_input_norm.bytes + ld.v_input_norm.bytes +
            ld.mlp_norm.bytes +
            ld.wq.bytes + ld.wk.bytes + ld.wv.bytes + ld.wo.bytes +
            ld.gate_attn.bytes + ld.wr.bytes + ld.attn_sink_proj.bytes +
            ld.q_l2_clip_scale.bytes + ld.rel_l2_clip_scale.bytes + ld.rel_logits_proj.bytes +
            ld.k_sconv_w.bytes + ld.v_sconv_w.bytes +
            ld.k_sconv_proj.bytes + ld.v_sconv_proj.bytes +
            ld.attn_sconv_w.bytes + ld.mlp_sconv_w.bytes +
            ld.moe_gate_weight.bytes + ld.moe_gate_bias.bytes;

        if (pass > 0u && ld.in_loop_core) {
            for (uint32_t m = 0; m < 4u; ++m) {
                layer_dense += ld.controllers[m].down_weight.bytes +
                               ld.controllers[m].gain_weight.bytes +
                               ld.controllers[m].forget_weight.bytes;
            }
            layer_dense += 4ULL * static_cast<uint64_t>(H) * sizeof(float);
        }

        const uint64_t all_exp_bytes =
            ld.experts_w13.weight_bytes + ld.experts_w13.scales_bytes + ld.experts_w13.expert_scale_bytes +
            ld.experts_w2.weight_bytes  + ld.experts_w2.scales_bytes  + ld.experts_w2.expert_scale_bytes;
        const uint64_t topk_exp_bytes = (c.num_experts > 0u)
                                            ? (all_exp_bytes * c.num_experts_per_tok) / c.num_experts
                                            : 0ULL;

        active += layer_dense + topk_exp_bytes;
    }

    uint64_t mtp_active = 0ULL;
    if (hdr.mtp.enabled) {
        const StrataInkling2MtpDesc& mtp = hdr.mtp;
        const uint32_t MR = mtp.markov_rank;
        const uint64_t draft_steps = static_cast<uint64_t>(mtp.block_size > 1u ? mtp.block_size - 1u : 1u);

        mtp_active += mtp.fuse_alpha.bytes + mtp.mask_embed.bytes + mtp.final_norm.bytes +
                      mtp.markov_dflare_norm.bytes + emb_row + hdr.embed_norm.bytes;

        if (mtp.frspec.enabled) {
            /* 32K FR-Spec folded head: unembed_q4a read once; fold_weight_q8 & rnn_weight SLC-resident */
            mtp_active += mtp.frspec.token_ids.bytes +
                          mtp.frspec.unembed_q4a.bytes +
                          mtp.frspec.fold_weight_q8.bytes +
                          mtp.frspec.fold_bias_f32.bytes +
                          mtp.markov_rnn_weight.bytes +
                          draft_steps * (static_cast<uint64_t>(MR) * 2ULL + mtp.markov_rnn_bias.bytes);
        } else {
            mtp_active += draft_steps *
                              (static_cast<uint64_t>(MR) * 2ULL + mtp.markov_rnn_weight.bytes +
                               mtp.markov_rnn_bias.bytes + mtp.markov_hidden_out.bytes +
                               mtp.markov_hidden_bias.bytes) +
                          hdr.unembed.bytes;
        }

        for (uint32_t l = 0; l < mtp.num_layers; ++l) {
            const StrataInkling2MtpLayerDesc& ml = mtp.layers[l];
            mtp_active += ml.dflare_norm.bytes + ml.input_layernorm.bytes + ml.post_attn_norm.bytes +
                          ml.q_norm.bytes + ml.k_norm.bytes + ml.k_norm_ctx.bytes +
                          ml.q_proj.bytes + ml.k_proj.bytes + ml.v_proj.bytes +
                          ml.k_proj_ctx.bytes + ml.v_proj_ctx.bytes + ml.o_proj.bytes +
                          ml.gate_proj.bytes + ml.up_proj.bytes + ml.down_proj.bytes;
        }
    }

    hdr.active_bytes_per_tok     = active;
    hdr.mtp_active_bytes_per_blk = mtp_active;
}

static void plan_uma_layout(
    StrataInkling2Header& hdr,
    const uint32_t phys_is_local[STRATA_INKLING2_MAX_PHYS_LAYERS],
    const uint32_t layer_expert_dtype[STRATA_INKLING2_MAX_PHYS_LAYERS],
    bool has_gate_bias,
    bool has_mtp,
    const uint32_t mtp_is_local[STRATA_INKLING2_MAX_MTP_LAYERS]
) {
    const StrataInkling2Config& c = hdr.cfg;
    const uint32_t H    = c.hidden_size;
    const uint32_t NH   = c.num_attention_heads;
    const uint32_t NKV  = c.num_key_value_heads;
    const uint32_t HD   = c.head_dim;
    const uint32_t DR   = c.d_rel;
    const uint32_t KS   = c.sconv_kernel_size;
    const uint32_t E    = c.num_experts;
    const uint32_t IFF  = c.intermediate_size;
    const uint32_t QDIM = NH * HD;
    const uint32_t KVDIM = NKV * HD;

    hdr.magic               = STRATA_INKLING2_MAGIC_U64;
    hdr.version             = STRATA_INKLING2_FORMAT_VERSION;
    hdr.header_padded_bytes = STRATA_INKLING2_HEADER_PADDED_BYTES;

    uint64_t cur = STRATA_INKLING2_HEADER_PADDED_BYTES;

    /* Global backbone tensors */
    hdr.embed_tokens     = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, c.vocab_size_padded, H);
    hdr.embed_norm       = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
    hdr.final_norm       = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
    hdr.output_res_query = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
    hdr.unembed          = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, c.vocab_size_unpadded, H);

    if (c.loop_num_passes > 1u) {
        const uint32_t n_cp_rows = 2u * (c.loop_core_end - c.loop_core_start);
        hdr.cross_pass_queries    = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, n_cp_rows, H);
        hdr.cross_pass_kv_queries = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, n_cp_rows, H);
    }

    /* 24 Physical decoder layers (each 16 KiB page-aligned) */
    for (uint32_t i = 0; i < c.num_phys_layers; ++i) {
        cur = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        StrataInkling2LayerDesc& ld = hdr.layers[i];
        std::memset(&ld, 0, sizeof(ld));

        const bool is_local = (phys_is_local[i] != 0u);
        const bool in_core  = (c.loop_num_passes > 1u && i >= c.loop_core_start && i < c.loop_core_end);
        const uint32_t extent = is_local ? c.sliding_window_size : c.rel_extent;

        ld.phys_layer_idx = i;
        ld.is_local_swa   = is_local ? 1u : 0u;
        ld.in_loop_core   = in_core ? 1u : 0u;
        ld.expert_dtype   = layer_expert_dtype[i];
        ld.layer_page_off = cur;

        if (i > 0u) {
            ld.attn_res_query = is_local
                                    ? alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u)
                                    : alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, 3u, H);
        }
        ld.mlp_res_query = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);

        if (is_local) {
            ld.attn_norm = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
        } else {
            ld.q_input_norm = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
            ld.k_input_norm = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
            ld.v_input_norm = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
        }
        ld.mlp_norm = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);

        ld.wq             = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, QDIM, H);
        ld.wk             = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
        ld.wv             = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
        ld.wo             = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, H, QDIM);
        ld.gate_attn      = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, QDIM, H);
        ld.wr             = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, NH * DR, H);
        ld.attn_sink_proj = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, NH, H);

        ld.q_l2_clip_scale   = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, NH, HD);
        ld.rel_l2_clip_scale = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, NH, DR);
        ld.rel_logits_proj   = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, extent, DR);

        ld.k_sconv_w = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, KVDIM, KS);
        ld.v_sconv_w = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, KVDIM, KS);
        if (!is_local) {
            ld.k_sconv_proj = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, NKV * KS, H);
            ld.v_sconv_proj = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, NKV * KS, H);
        }
        ld.attn_sconv_w = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, KS);
        ld.mlp_sconv_w  = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, KS);

        ld.moe_gate_weight = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, E, H);
        if (has_gate_bias) {
            ld.moe_gate_bias = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, E, 1u);
        }
        ld.experts_w13 = alloc_expert_matrix(cur, E, 2u * IFF, H, ld.expert_dtype);
        ld.experts_w2  = alloc_expert_matrix(cur, E, H, IFF, ld.expert_dtype);

        if (in_core) {
            for (uint32_t m = 0; m < 4u; ++m) {
                ld.controllers[m].down_weight   = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, c.loop_gate_rank, 2u * H);
                ld.controllers[m].gain_weight   = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, H, c.loop_gate_rank);
                ld.controllers[m].forget_weight = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, H, c.loop_gate_rank);
            }
        }

        cur = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        ld.layer_page_bytes = cur - ld.layer_page_off;
    }

    /* Optional 3-Layer DSpark Block-8 MTP section (16 KiB page-aligned) */
    hdr.mtp.enabled        = has_mtp ? 1u : 0u;
    hdr.mtp.mtp_page_off   = 0ULL;
    hdr.mtp.mtp_page_bytes = 0ULL;
    std::memset(&hdr.mtp.frspec, 0, sizeof(hdr.mtp.frspec));

    if (has_mtp) {
        cur = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        StrataInkling2MtpDesc& mtp = hdr.mtp;
        mtp.mtp_page_off = cur;

        const uint32_t MR = mtp.markov_rank;
        const uint32_t MI = mtp.intermediate_size;

        mtp.fuse_alpha         = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, mtp.num_layers + 1u, STRATA_INKLING2_MAX_MTP_TAPS);
        mtp.mask_embed         = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, H, 1u);
        mtp.final_norm         = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
        mtp.markov_dflare_norm = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
        mtp.markov_w1          = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, c.vocab_size_padded, MR);
        mtp.markov_rnn_weight  = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, 3u * MR, 2u * MR + 2u * H);
        mtp.markov_rnn_bias    = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, 3u * MR, 1u);
        mtp.markov_hidden_out  = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, H, MR);
        mtp.markov_hidden_bias = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);

        for (uint32_t l = 0; l < mtp.num_layers; ++l) {
            StrataInkling2MtpLayerDesc& ml = mtp.layers[l];
            std::memset(&ml, 0, sizeof(ml));
            ml.layer_idx       = l;
            ml.is_local_swa    = mtp_is_local[l] ? 1u : 0u;
            ml.dflare_norm     = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
            ml.input_layernorm = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
            ml.post_attn_norm  = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, H, 1u);
            ml.q_norm          = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, HD, 1u);
            ml.k_norm          = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, HD, 1u);
            ml.k_norm_ctx      = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32, HD, 1u);
            ml.q_proj          = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, QDIM, H);
            ml.k_proj          = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
            ml.v_proj          = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
            ml.k_proj_ctx      = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
            ml.v_proj_ctx      = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, KVDIM, H);
            ml.o_proj          = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, H, QDIM);
            ml.gate_proj       = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, MI, H);
            ml.up_proj         = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, MI, H);
            ml.down_proj       = alloc_slice(cur, STRATA_INKLING2_DTYPE_F16, H, MI);
        }

        cur = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        mtp.mtp_page_bytes = cur - mtp.mtp_page_off;
    }

    hdr.total_file_bytes = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
    compute_active_bytes(hdr);
}

/* ============================================================================
 * 5. Parallel Conversion Task Builder & Multi-Threaded Streaming Executor
 * ============================================================================ */

enum ConvTaskKind {
    TASK_SLICE_F32 = 0,
    TASK_SLICE_F16,
    TASK_TRANSPOSE_2D_F32,
    TASK_FUSE_SOFTMAX_F32,
    TASK_EXPERT_DEINTERLEAVE,
    TASK_EXPERT_COPY,
    TASK_EXPERT_AMAX_SCALE
};

enum ByteCheckMode {
    CHK_NONE   = 0,
    CHK_E4M3FN = 1,
    CHK_UE8M0  = 2
};

struct ConvertTask {
    ConvTaskKind    kind        = TASK_SLICE_F32;
    ByteCheckMode   byte_check  = CHK_NONE;
    int             src_fd      = -1;
    int             src_fd2     = -1; /* outer_scale_denominator for TASK_EXPERT_AMAX_SCALE */
    SafetensorDtype src_dtype   = ST_UNKNOWN;
    uint64_t        src_off     = 0;
    uint64_t        src_off2    = 0;
    uint64_t        dst_off     = 0;
    uint64_t        elem_count  = 0; /* or total bytes for TASK_EXPERT_COPY */
    uint32_t        dim0        = 0; /* src rows (transpose/softmax) or num_experts_in_chunk */
    uint32_t        dim1        = 0; /* src cols (transpose/softmax) or half_rows (deinterleave) */
    uint32_t        row_bytes   = 0; /* bytes per row (deinterleave) */
    std::string     debug_name;
};

static bool check_shape_eq(const std::vector<uint64_t>& actual, std::initializer_list<uint64_t> expected) {
    if (actual.size() != expected.size()) return false;
    size_t i = 0;
    for (uint64_t e : expected) {
        if (actual[i++] != e) return false;
    }
    return true;
}

static bool enqueue_slice_tasks(
    std::unordered_map<std::string, SafetensorEntry>& src_map,
    const std::string& name,
    const StrataInkling2TensorSlice& dst,
    std::initializer_list<uint64_t> expected_shape,
    bool allow_trim_rows,
    std::vector<ConvertTask>& tasks
) {
    auto it = src_map.find(name);
    if (it == src_map.end()) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: missing tensor '%s'\n", name.c_str());
        return false;
    }
    SafetensorEntry& ent = it->second;
    if (!check_shape_eq(ent.shape, expected_shape)) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: shape mismatch on '%s'\n", name.c_str());
        return false;
    }
    if (ent.dtype != ST_BF16 && ent.dtype != ST_F16 && ent.dtype != ST_F32) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: unsupported dtype on dense tensor '%s'\n", name.c_str());
        return false;
    }
    uint64_t src_elems = 1ULL;
    for (uint64_t d : ent.shape) src_elems *= d;
    if (!allow_trim_rows && src_elems != dst.elem_count) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: element count mismatch on '%s' (%llu vs %u)\n",
                     name.c_str(), static_cast<unsigned long long>(src_elems), dst.elem_count);
        return false;
    }
    if (allow_trim_rows && dst.elem_count > src_elems) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: trimmed elem_count exceeds source on '%s'\n", name.c_str());
        return false;
    }

    ent.used = true;
    const uint64_t src_bpe = st_dtype_bpe(ent.dtype);
    const uint64_t dst_bpe = (dst.dtype == STRATA_INKLING2_DTYPE_F32) ? 4ULL : 2ULL;
    const ConvTaskKind kind = (dst.dtype == STRATA_INKLING2_DTYPE_F32) ? TASK_SLICE_F32 : TASK_SLICE_F16;

    /* Chunk large dense matrices (e.g. embed/unembed/markov_w1) into 4M-element slices for parallel I/O + NEON */
    constexpr uint64_t kMaxChunkElems = 4ULL * 1024ULL * 1024ULL;
    uint64_t done = 0ULL;
    while (done < dst.elem_count) {
        const uint64_t n = std::min<uint64_t>(kMaxChunkElems, dst.elem_count - done);
        ConvertTask t;
        t.kind       = kind;
        t.src_fd     = ent.shard_fd;
        t.src_dtype  = ent.dtype;
        t.src_off    = ent.abs_offset + done * src_bpe;
        t.dst_off    = dst.offset + done * dst_bpe;
        t.elem_count = n;
        t.debug_name = name;
        tasks.push_back(std::move(t));
        done += n;
    }
    return true;
}

static bool enqueue_transpose_2d_f32(
    std::unordered_map<std::string, SafetensorEntry>& src_map,
    const std::string& name,
    const StrataInkling2TensorSlice& dst,
    uint32_t src_rows,
    uint32_t src_cols,
    std::vector<ConvertTask>& tasks
) {
    auto it = src_map.find(name);
    if (it == src_map.end()) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: missing tensor '%s'\n", name.c_str());
        return false;
    }
    SafetensorEntry& ent = it->second;
    if (!check_shape_eq(ent.shape, {src_rows, src_cols}) ||
        (ent.dtype != ST_BF16 && ent.dtype != ST_F16 && ent.dtype != ST_F32) ||
        dst.shape0 != src_cols || dst.shape1 != src_rows) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: transpose shape/dtype mismatch on '%s'\n", name.c_str());
        return false;
    }
    ent.used = true;
    ConvertTask t;
    t.kind       = TASK_TRANSPOSE_2D_F32;
    t.src_fd     = ent.shard_fd;
    t.src_dtype  = ent.dtype;
    t.src_off    = ent.abs_offset;
    t.dst_off    = dst.offset;
    t.elem_count = static_cast<uint64_t>(src_rows) * src_cols;
    t.dim0       = src_rows;
    t.dim1       = src_cols;
    t.debug_name = name;
    tasks.push_back(std::move(t));
    return true;
}

static bool enqueue_expert_matrix_tasks(
    std::unordered_map<std::string, SafetensorEntry>& src_map,
    const std::string& prefix, /* e.g. "model.layers.0.mlp.experts.w13" */
    const StrataInkling2ExpertMatrix& em,
    bool deinterleave_w13,
    std::vector<ConvertTask>& tasks
) {
    const std::string w_name = prefix + "_weight";
    const std::string s_name = prefix + "_weight_scale";
    auto it_w = src_map.find(w_name);
    auto it_s = src_map.find(s_name);
    if (it_w == src_map.end() || it_s == src_map.end()) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: missing expert tensors for '%s'\n", prefix.c_str());
        return false;
    }
    SafetensorEntry& ew = it_w->second;
    SafetensorEntry& es = it_s->second;

    const bool is_nvfp4 = (em.dtype == STRATA_INKLING2_DTYPE_NVFP4);
    const uint32_t w_row_b = is_nvfp4 ? (em.cols / 2u)  : em.cols;
    const uint32_t s_row_b = is_nvfp4 ? (em.cols / 16u) : (em.cols / 32u);
    const SafetensorDtype exp_w_dt = is_nvfp4 ? ST_U8      : ST_F8_E4M3;
    const SafetensorDtype exp_s_dt = is_nvfp4 ? ST_F8_E4M3 : ST_U8;

    if (ew.dtype != exp_w_dt || es.dtype != exp_s_dt ||
        !check_shape_eq(ew.shape, {em.num_experts, em.rows_per_expert, w_row_b}) ||
        !check_shape_eq(es.shape, {em.num_experts, em.rows_per_expert, s_row_b})) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: expert weight/scale layout mismatch on '%s'\n", prefix.c_str());
        return false;
    }
    ew.used = true;
    es.used = true;

    const uint64_t w_exp_bytes = static_cast<uint64_t>(em.rows_per_expert) * w_row_b;
    const uint64_t s_exp_bytes = static_cast<uint64_t>(em.rows_per_expert) * s_row_b;

    /* Group experts into ~4-8 MiB tasks for high parallel I/O + bounded thread scratch */
    const uint32_t w_grp = std::max<uint32_t>(1u, static_cast<uint32_t>((8ULL * 1024ULL * 1024ULL) / std::max<uint64_t>(1ULL, w_exp_bytes)));
    const uint32_t s_grp = std::max<uint32_t>(1u, static_cast<uint32_t>((4ULL * 1024ULL * 1024ULL) / std::max<uint64_t>(1ULL, s_exp_bytes)));

    if (deinterleave_w13) {
        if ((em.rows_per_expert & 1u) != 0u) return false;
        const uint32_t half_rows = em.rows_per_expert / 2u;
        for (uint32_t e0 = 0; e0 < em.num_experts; e0 += w_grp) {
            const uint32_t cnt = std::min<uint32_t>(w_grp, em.num_experts - e0);
            ConvertTask tw;
            tw.kind       = TASK_EXPERT_DEINTERLEAVE;
            tw.byte_check = is_nvfp4 ? CHK_NONE : CHK_E4M3FN;
            tw.src_fd     = ew.shard_fd;
            tw.src_off    = ew.abs_offset + static_cast<uint64_t>(e0) * w_exp_bytes;
            tw.dst_off    = em.weight_off + static_cast<uint64_t>(e0) * w_exp_bytes;
            tw.dim0       = cnt;
            tw.dim1       = half_rows;
            tw.row_bytes  = w_row_b;
            tw.debug_name = w_name;
            tasks.push_back(std::move(tw));
        }
        for (uint32_t e0 = 0; e0 < em.num_experts; e0 += s_grp) {
            const uint32_t cnt = std::min<uint32_t>(s_grp, em.num_experts - e0);
            ConvertTask ts;
            ts.kind       = TASK_EXPERT_DEINTERLEAVE;
            ts.byte_check = is_nvfp4 ? CHK_E4M3FN : CHK_UE8M0;
            ts.src_fd     = es.shard_fd;
            ts.src_off    = es.abs_offset + static_cast<uint64_t>(e0) * s_exp_bytes;
            ts.dst_off    = em.scales_off + static_cast<uint64_t>(e0) * s_exp_bytes;
            ts.dim0       = cnt;
            ts.dim1       = half_rows;
            ts.row_bytes  = s_row_b;
            ts.debug_name = s_name;
            tasks.push_back(std::move(ts));
        }
    } else {
        for (uint32_t e0 = 0; e0 < em.num_experts; e0 += w_grp) {
            const uint32_t cnt = std::min<uint32_t>(w_grp, em.num_experts - e0);
            ConvertTask tw;
            tw.kind       = TASK_EXPERT_COPY;
            tw.byte_check = is_nvfp4 ? CHK_NONE : CHK_E4M3FN;
            tw.src_fd     = ew.shard_fd;
            tw.src_off    = ew.abs_offset + static_cast<uint64_t>(e0) * w_exp_bytes;
            tw.dst_off    = em.weight_off + static_cast<uint64_t>(e0) * w_exp_bytes;
            tw.elem_count = static_cast<uint64_t>(cnt) * w_exp_bytes;
            tw.debug_name = w_name;
            tasks.push_back(std::move(tw));
        }
        for (uint32_t e0 = 0; e0 < em.num_experts; e0 += s_grp) {
            const uint32_t cnt = std::min<uint32_t>(s_grp, em.num_experts - e0);
            ConvertTask ts;
            ts.kind       = TASK_EXPERT_COPY;
            ts.byte_check = is_nvfp4 ? CHK_E4M3FN : CHK_UE8M0;
            ts.src_fd     = es.shard_fd;
            ts.src_off    = es.abs_offset + static_cast<uint64_t>(e0) * s_exp_bytes;
            ts.dst_off    = em.scales_off + static_cast<uint64_t>(e0) * s_exp_bytes;
            ts.elem_count = static_cast<uint64_t>(cnt) * s_exp_bytes;
            ts.debug_name = s_name;
            tasks.push_back(std::move(ts));
        }
    }

    if (is_nvfp4) {
        const std::string a_name = prefix + "_weight_amax";
        const std::string d_name = prefix + "_weight_outer_scale_denominator";
        auto it_a = src_map.find(a_name);
        auto it_d = src_map.find(d_name);
        if (it_a == src_map.end() || it_d == src_map.end()) {
            std::fprintf(stderr, "[inkling2-conv] ERROR: missing NVFP4 amax/denominator for '%s'\n", prefix.c_str());
            return false;
        }
        SafetensorEntry& ea = it_a->second;
        SafetensorEntry& ed = it_d->second;
        if (ea.dtype != ST_F32 || ed.dtype != ST_F32 ||
            !check_shape_eq(ea.shape, {em.num_experts}) || !check_shape_eq(ed.shape, {1ULL})) {
            std::fprintf(stderr, "[inkling2-conv] ERROR: invalid NVFP4 amax/denominator shape on '%s'\n", prefix.c_str());
            return false;
        }
        ea.used = true;
        ed.used = true;

        ConvertTask ta;
        ta.kind       = TASK_EXPERT_AMAX_SCALE;
        ta.src_fd     = ea.shard_fd;
        ta.src_fd2    = ed.shard_fd;
        ta.src_off    = ea.abs_offset;
        ta.src_off2   = ed.abs_offset;
        ta.dst_off    = em.expert_scale_off;
        ta.elem_count = em.num_experts;
        ta.debug_name = a_name;
        tasks.push_back(std::move(ta));
    }

    return true;
}

static bool execute_convert_task(
    int out_fd,
    const ConvertTask& t,
    std::vector<uint8_t>& src_buf,
    std::vector<uint8_t>& dst_buf
) {
    switch (t.kind) {
        case TASK_SLICE_F32: {
            const uint64_t sbpe = st_dtype_bpe(t.src_dtype);
            const size_t in_bytes  = static_cast<size_t>(t.elem_count * sbpe);
            const size_t out_bytes = static_cast<size_t>(t.elem_count * sizeof(float));
            if (src_buf.size() < in_bytes)  src_buf.resize(in_bytes);
            if (dst_buf.size() < out_bytes) dst_buf.resize(out_bytes);
            if (!pread_exact(t.src_fd, src_buf.data(), in_bytes, t.src_off)) return false;

            float* dst = reinterpret_cast<float*>(dst_buf.data());
            bool finite = false;
            if (t.src_dtype == ST_BF16) {
                finite = convert_bf16_to_f32_span_checked(
                    reinterpret_cast<const uint16_t*>(src_buf.data()), dst, static_cast<size_t>(t.elem_count));
            } else if (t.src_dtype == ST_F32) {
                std::memcpy(dst, src_buf.data(), out_bytes);
                finite = check_f32_span_finite(dst, static_cast<size_t>(t.elem_count));
            } else if (t.src_dtype == ST_F16) {
                finite = convert_fp16_to_f32_span_checked(
                    reinterpret_cast<const uint16_t*>(src_buf.data()), dst, static_cast<size_t>(t.elem_count));
            }
            if (!finite) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: NaN/Inf detected in '%s'\n", t.debug_name.c_str());
                return false;
            }
            return pwrite_exact(out_fd, dst_buf.data(), out_bytes, t.dst_off);
        }

        case TASK_SLICE_F16: {
            const uint64_t sbpe = st_dtype_bpe(t.src_dtype);
            const size_t in_bytes  = static_cast<size_t>(t.elem_count * sbpe);
            const size_t out_bytes = static_cast<size_t>(t.elem_count * sizeof(uint16_t));
            if (src_buf.size() < in_bytes)  src_buf.resize(in_bytes);
            if (dst_buf.size() < out_bytes) dst_buf.resize(out_bytes);
            if (!pread_exact(t.src_fd, src_buf.data(), in_bytes, t.src_off)) return false;

            uint16_t* dst = reinterpret_cast<uint16_t*>(dst_buf.data());
            bool finite = false;
            if (t.src_dtype == ST_BF16) {
                finite = convert_bf16_to_fp16_span_checked(
                    reinterpret_cast<const uint16_t*>(src_buf.data()), dst, static_cast<size_t>(t.elem_count));
            } else if (t.src_dtype == ST_F16) {
                std::memcpy(dst, src_buf.data(), out_bytes);
                finite = check_f16_span_finite(dst, static_cast<size_t>(t.elem_count));
            } else if (t.src_dtype == ST_F32) {
                finite = convert_f32_to_fp16_span_checked(
                    reinterpret_cast<const float*>(src_buf.data()), dst, static_cast<size_t>(t.elem_count));
            }
            if (!finite) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: NaN/Inf detected in '%s'\n", t.debug_name.c_str());
                return false;
            }
            return pwrite_exact(out_fd, dst_buf.data(), out_bytes, t.dst_off);
        }

        case TASK_TRANSPOSE_2D_F32: {
            const uint32_t R = t.dim0;
            const uint32_t C = t.dim1;
            const uint64_t sbpe = st_dtype_bpe(t.src_dtype);
            const size_t in_bytes  = static_cast<size_t>(t.elem_count * sbpe);
            const size_t out_bytes = static_cast<size_t>(t.elem_count * sizeof(float));
            if (src_buf.size() < in_bytes)  src_buf.resize(in_bytes);
            if (dst_buf.size() < out_bytes) dst_buf.resize(out_bytes);
            if (!pread_exact(t.src_fd, src_buf.data(), in_bytes, t.src_off)) return false;

            std::vector<float> tmp(static_cast<size_t>(t.elem_count));
            bool finite = false;
            if (t.src_dtype == ST_BF16) {
                finite = convert_bf16_to_f32_span_checked(
                    reinterpret_cast<const uint16_t*>(src_buf.data()), tmp.data(), tmp.size());
            } else if (t.src_dtype == ST_F32) {
                std::memcpy(tmp.data(), src_buf.data(), out_bytes);
                finite = check_f32_span_finite(tmp.data(), tmp.size());
            } else if (t.src_dtype == ST_F16) {
                finite = convert_fp16_to_f32_span_checked(
                    reinterpret_cast<const uint16_t*>(src_buf.data()), tmp.data(), tmp.size());
            }
            if (!finite) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: NaN/Inf detected in '%s'\n", t.debug_name.c_str());
                return false;
            }
            float* dst = reinterpret_cast<float*>(dst_buf.data());
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t r = 0; r < R; ++r) {
                    dst[static_cast<size_t>(c) * R + r] = tmp[static_cast<size_t>(r) * C + c];
                }
            }
            return pwrite_exact(out_fd, dst_buf.data(), out_bytes, t.dst_off);
        }

        case TASK_FUSE_SOFTMAX_F32: {
            const uint32_t R = t.dim0;
            const uint32_t C = t.dim1;
            const uint64_t sbpe = st_dtype_bpe(t.src_dtype);
            const size_t in_bytes  = static_cast<size_t>(t.elem_count * sbpe);
            const size_t out_bytes = static_cast<size_t>(t.elem_count * sizeof(float));
            if (src_buf.size() < in_bytes)  src_buf.resize(in_bytes);
            if (dst_buf.size() < out_bytes) dst_buf.resize(out_bytes);
            if (!pread_exact(t.src_fd, src_buf.data(), in_bytes, t.src_off)) return false;

            std::vector<float> tmp(static_cast<size_t>(t.elem_count));
            bool finite = false;
            if (t.src_dtype == ST_BF16) {
                finite = convert_bf16_to_f32_span_checked(
                    reinterpret_cast<const uint16_t*>(src_buf.data()), tmp.data(), tmp.size());
            } else if (t.src_dtype == ST_F32) {
                std::memcpy(tmp.data(), src_buf.data(), out_bytes);
                finite = check_f32_span_finite(tmp.data(), tmp.size());
            } else if (t.src_dtype == ST_F16) {
                finite = convert_fp16_to_f32_span_checked(
                    reinterpret_cast<const uint16_t*>(src_buf.data()), tmp.data(), tmp.size());
            }
            if (!finite) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: NaN/Inf detected in '%s'\n", t.debug_name.c_str());
                return false;
            }
            std::vector<double> exps(C);
            float* dst = reinterpret_cast<float*>(dst_buf.data());
            for (uint32_t r = 0; r < R; ++r) {
                const float* row_in = tmp.data() + static_cast<size_t>(r) * C;
                float* row_out      = dst + static_cast<size_t>(r) * C;
                float mx = row_in[0];
                for (uint32_t c = 1; c < C; ++c) {
                    if (row_in[c] > mx) mx = row_in[c];
                }
                double sum = 0.0;
                for (uint32_t c = 0; c < C; ++c) {
                    exps[c] = std::exp(static_cast<double>(row_in[c] - mx));
                    sum += exps[c];
                }
                if (!(sum > 0.0) || !std::isfinite(sum)) return false;
                for (uint32_t c = 0; c < C; ++c) {
                    row_out[c] = static_cast<float>(exps[c] / sum);
                }
            }
            return pwrite_exact(out_fd, dst_buf.data(), out_bytes, t.dst_off);
        }

        case TASK_EXPERT_DEINTERLEAVE: {
            const uint32_t cnt       = t.dim0;
            const uint32_t half_rows = t.dim1;
            const uint32_t rb        = t.row_bytes;
            const uint64_t exp_bytes = static_cast<uint64_t>(2u * half_rows) * rb;
            const size_t total_bytes = static_cast<size_t>(cnt * exp_bytes);
            if (src_buf.size() < total_bytes) src_buf.resize(total_bytes);
            if (dst_buf.size() < total_bytes) dst_buf.resize(total_bytes);
            if (!pread_exact(t.src_fd, src_buf.data(), total_bytes, t.src_off)) return false;

            if (t.byte_check == CHK_E4M3FN && !check_e4m3fn_span_no_nan(src_buf.data(), total_bytes)) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: E4M3FN NaN in '%s'\n", t.debug_name.c_str());
                return false;
            }
            if (t.byte_check == CHK_UE8M0 && !check_ue8m0_span_no_nan(src_buf.data(), total_bytes)) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: UE8M0 NaN in '%s'\n", t.debug_name.c_str());
                return false;
            }

            for (uint32_t e = 0; e < cnt; ++e) {
                const uint8_t* se = src_buf.data() + static_cast<size_t>(e) * exp_bytes;
                uint8_t*       de = dst_buf.data() + static_cast<size_t>(e) * exp_bytes;
                uint8_t* de_gate  = de;
                uint8_t* de_up    = de + static_cast<size_t>(half_rows) * rb;
                for (uint32_t r = 0; r < half_rows; ++r) {
                    std::memcpy(de_gate + static_cast<size_t>(r) * rb,
                                se + static_cast<size_t>(2u * r) * rb, rb);
                    std::memcpy(de_up   + static_cast<size_t>(r) * rb,
                                se + static_cast<size_t>(2u * r + 1u) * rb, rb);
                }
            }
            return pwrite_exact(out_fd, dst_buf.data(), total_bytes, t.dst_off);
        }

        case TASK_EXPERT_COPY: {
            const size_t total_bytes = static_cast<size_t>(t.elem_count);
            if (src_buf.size() < total_bytes) src_buf.resize(total_bytes);
            if (!pread_exact(t.src_fd, src_buf.data(), total_bytes, t.src_off)) return false;

            if (t.byte_check == CHK_E4M3FN && !check_e4m3fn_span_no_nan(src_buf.data(), total_bytes)) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: E4M3FN NaN in '%s'\n", t.debug_name.c_str());
                return false;
            }
            if (t.byte_check == CHK_UE8M0 && !check_ue8m0_span_no_nan(src_buf.data(), total_bytes)) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: UE8M0 NaN in '%s'\n", t.debug_name.c_str());
                return false;
            }
            return pwrite_exact(out_fd, src_buf.data(), total_bytes, t.dst_off);
        }

        case TASK_EXPERT_AMAX_SCALE: {
            const size_t E = static_cast<size_t>(t.elem_count);
            std::vector<float> amax(E);
            float denom = 0.0f;
            if (!pread_exact(t.src_fd, amax.data(), E * sizeof(float), t.src_off)) return false;
            if (!pread_exact(t.src_fd2, &denom, sizeof(float), t.src_off2)) return false;
            if (!std::isfinite(denom) || !(denom > 0.0f)) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: invalid NVFP4 outer_scale_denominator=%f on '%s'\n",
                             denom, t.debug_name.c_str());
                return false;
            }
            for (size_t e = 0; e < E; ++e) {
                if (!std::isfinite(amax[e]) || amax[e] < 0.0f) {
                    std::fprintf(stderr, "[inkling2-conv] ERROR: invalid NVFP4 amax[%zu]=%f on '%s'\n",
                                 e, amax[e], t.debug_name.c_str());
                    return false;
                }
                amax[e] = amax[e] / denom;
                if (!std::isfinite(amax[e])) return false;
            }
            return pwrite_exact(out_fd, amax.data(), E * sizeof(float), t.dst_off);
        }
    }
    return false;
}

/* ============================================================================
 * 6. Exact `o200k_base` ByteLevel BPE Tokenizer (`tokenizer.json`)
 *
 * Pre-Tokenizer Split Regex (from `tokenizer.json`):
 *   `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`
 *  |`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`
 *  |`\p{N}{1,3}`
 *  |` ?[^\s\p{L}\p{N}]+[\r\n/]*`
 *  |`\s*[\r\n]+`
 *  |`\s+(?!\S)`
 *  |`\s+`
 * + GPT-2 ByteLevel mapping + `ignore_merges = true` + 60 atomic `added_tokens`.
 * ============================================================================ */

struct UcdCatRange {
    uint32_t lo;
    uint32_t hi_cat; /* (hi << 3) | cat: 1=Lu|Lt, 2=Ll, 3=Lm|Lo, 4=M, 5=N */
};

static const UcdCatRange kUcdCatRanges[] = {
    {0x00AAu, (0x00AAu << 3) | 3u}, {0x00B2u, (0x00B3u << 3) | 5u}, {0x00B5u, (0x00B5u << 3) | 2u}, {0x00B9u, (0x00B9u << 3) | 5u},
    {0x00BAu, (0x00BAu << 3) | 3u}, {0x00BCu, (0x00BEu << 3) | 5u}, {0x00C0u, (0x00D6u << 3) | 1u}, {0x00D8u, (0x00DEu << 3) | 1u},
    {0x00DFu, (0x00F6u << 3) | 2u}, {0x00F8u, (0x00FFu << 3) | 2u}, {0x0100u, (0x0100u << 3) | 1u}, {0x0101u, (0x0101u << 3) | 2u},
    {0x0102u, (0x0102u << 3) | 1u}, {0x0103u, (0x0103u << 3) | 2u}, {0x0104u, (0x0104u << 3) | 1u}, {0x0105u, (0x0105u << 3) | 2u},
    {0x0106u, (0x0106u << 3) | 1u}, {0x0107u, (0x0107u << 3) | 2u}, {0x0108u, (0x0108u << 3) | 1u}, {0x0109u, (0x0109u << 3) | 2u},
    {0x010Au, (0x010Au << 3) | 1u}, {0x010Bu, (0x010Bu << 3) | 2u}, {0x010Cu, (0x010Cu << 3) | 1u}, {0x010Du, (0x010Du << 3) | 2u},
    {0x010Eu, (0x010Eu << 3) | 1u}, {0x010Fu, (0x010Fu << 3) | 2u}, {0x0110u, (0x0110u << 3) | 1u}, {0x0111u, (0x0111u << 3) | 2u},
    {0x0112u, (0x0112u << 3) | 1u}, {0x0113u, (0x0113u << 3) | 2u}, {0x0114u, (0x0114u << 3) | 1u}, {0x0115u, (0x0115u << 3) | 2u},
    {0x0116u, (0x0116u << 3) | 1u}, {0x0117u, (0x0117u << 3) | 2u}, {0x0118u, (0x0118u << 3) | 1u}, {0x0119u, (0x0119u << 3) | 2u},
    {0x011Au, (0x011Au << 3) | 1u}, {0x011Bu, (0x011Bu << 3) | 2u}, {0x011Cu, (0x011Cu << 3) | 1u}, {0x011Du, (0x011Du << 3) | 2u},
    {0x011Eu, (0x011Eu << 3) | 1u}, {0x011Fu, (0x011Fu << 3) | 2u}, {0x0120u, (0x0120u << 3) | 1u}, {0x0121u, (0x0121u << 3) | 2u},
    {0x0122u, (0x0122u << 3) | 1u}, {0x0123u, (0x0123u << 3) | 2u}, {0x0124u, (0x0124u << 3) | 1u}, {0x0125u, (0x0125u << 3) | 2u},
    {0x0126u, (0x0126u << 3) | 1u}, {0x0127u, (0x0127u << 3) | 2u}, {0x0128u, (0x0128u << 3) | 1u}, {0x0129u, (0x0129u << 3) | 2u},
    {0x012Au, (0x012Au << 3) | 1u}, {0x012Bu, (0x012Bu << 3) | 2u}, {0x012Cu, (0x012Cu << 3) | 1u}, {0x012Du, (0x012Du << 3) | 2u},
    {0x012Eu, (0x012Eu << 3) | 1u}, {0x012Fu, (0x012Fu << 3) | 2u}, {0x0130u, (0x0130u << 3) | 1u}, {0x0131u, (0x0131u << 3) | 2u},
    {0x0132u, (0x0132u << 3) | 1u}, {0x0133u, (0x0133u << 3) | 2u}, {0x0134u, (0x0134u << 3) | 1u}, {0x0135u, (0x0135u << 3) | 2u},
    {0x0136u, (0x0136u << 3) | 1u}, {0x0137u, (0x0138u << 3) | 2u}, {0x0139u, (0x0139u << 3) | 1u}, {0x013Au, (0x013Au << 3) | 2u},
    {0x013Bu, (0x013Bu << 3) | 1u}, {0x013Cu, (0x013Cu << 3) | 2u}, {0x013Du, (0x013Du << 3) | 1u}, {0x013Eu, (0x013Eu << 3) | 2u},
    {0x013Fu, (0x013Fu << 3) | 1u}, {0x0140u, (0x0140u << 3) | 2u}, {0x0141u, (0x0141u << 3) | 1u}, {0x0142u, (0x0142u << 3) | 2u},
    {0x0143u, (0x0143u << 3) | 1u}, {0x0144u, (0x0144u << 3) | 2u}, {0x0145u, (0x0145u << 3) | 1u}, {0x0146u, (0x0146u << 3) | 2u},
    {0x0147u, (0x0147u << 3) | 1u}, {0x0148u, (0x0149u << 3) | 2u}, {0x014Au, (0x014Au << 3) | 1u}, {0x014Bu, (0x014Bu << 3) | 2u},
    {0x014Cu, (0x014Cu << 3) | 1u}, {0x014Du, (0x014Du << 3) | 2u}, {0x014Eu, (0x014Eu << 3) | 1u}, {0x014Fu, (0x014Fu << 3) | 2u},
    {0x0150u, (0x0150u << 3) | 1u}, {0x0151u, (0x0151u << 3) | 2u}, {0x0152u, (0x0152u << 3) | 1u}, {0x0153u, (0x0153u << 3) | 2u},
    {0x0154u, (0x0154u << 3) | 1u}, {0x0155u, (0x0155u << 3) | 2u}, {0x0156u, (0x0156u << 3) | 1u}, {0x0157u, (0x0157u << 3) | 2u},
    {0x0158u, (0x0158u << 3) | 1u}, {0x0159u, (0x0159u << 3) | 2u}, {0x015Au, (0x015Au << 3) | 1u}, {0x015Bu, (0x015Bu << 3) | 2u},
    {0x015Cu, (0x015Cu << 3) | 1u}, {0x015Du, (0x015Du << 3) | 2u}, {0x015Eu, (0x015Eu << 3) | 1u}, {0x015Fu, (0x015Fu << 3) | 2u},
    {0x0160u, (0x0160u << 3) | 1u}, {0x0161u, (0x0161u << 3) | 2u}, {0x0162u, (0x0162u << 3) | 1u}, {0x0163u, (0x0163u << 3) | 2u},
    {0x0164u, (0x0164u << 3) | 1u}, {0x0165u, (0x0165u << 3) | 2u}, {0x0166u, (0x0166u << 3) | 1u}, {0x0167u, (0x0167u << 3) | 2u},
    {0x0168u, (0x0168u << 3) | 1u}, {0x0169u, (0x0169u << 3) | 2u}, {0x016Au, (0x016Au << 3) | 1u}, {0x016Bu, (0x016Bu << 3) | 2u},
    {0x016Cu, (0x016Cu << 3) | 1u}, {0x016Du, (0x016Du << 3) | 2u}, {0x016Eu, (0x016Eu << 3) | 1u}, {0x016Fu, (0x016Fu << 3) | 2u},
    {0x0170u, (0x0170u << 3) | 1u}, {0x0171u, (0x0171u << 3) | 2u}, {0x0172u, (0x0172u << 3) | 1u}, {0x0173u, (0x0173u << 3) | 2u},
    {0x0174u, (0x0174u << 3) | 1u}, {0x0175u, (0x0175u << 3) | 2u}, {0x0176u, (0x0176u << 3) | 1u}, {0x0177u, (0x0177u << 3) | 2u},
    {0x0178u, (0x0179u << 3) | 1u}, {0x017Au, (0x017Au << 3) | 2u}, {0x017Bu, (0x017Bu << 3) | 1u}, {0x017Cu, (0x017Cu << 3) | 2u},
    {0x017Du, (0x017Du << 3) | 1u}, {0x017Eu, (0x0180u << 3) | 2u}, {0x0181u, (0x0182u << 3) | 1u}, {0x0183u, (0x0183u << 3) | 2u},
    {0x0184u, (0x0184u << 3) | 1u}, {0x0185u, (0x0185u << 3) | 2u}, {0x0186u, (0x0187u << 3) | 1u}, {0x0188u, (0x0188u << 3) | 2u},
    {0x0189u, (0x018Bu << 3) | 1u}, {0x018Cu, (0x018Du << 3) | 2u}, {0x018Eu, (0x0191u << 3) | 1u}, {0x0192u, (0x0192u << 3) | 2u},
    {0x0193u, (0x0194u << 3) | 1u}, {0x0195u, (0x0195u << 3) | 2u}, {0x0196u, (0x0198u << 3) | 1u}, {0x0199u, (0x019Bu << 3) | 2u},
    {0x019Cu, (0x019Du << 3) | 1u}, {0x019Eu, (0x019Eu << 3) | 2u}, {0x019Fu, (0x01A0u << 3) | 1u}, {0x01A1u, (0x01A1u << 3) | 2u},
    {0x01A2u, (0x01A2u << 3) | 1u}, {0x01A3u, (0x01A3u << 3) | 2u}, {0x01A4u, (0x01A4u << 3) | 1u}, {0x01A5u, (0x01A5u << 3) | 2u},
    {0x01A6u, (0x01A7u << 3) | 1u}, {0x01A8u, (0x01A8u << 3) | 2u}, {0x01A9u, (0x01A9u << 3) | 1u}, {0x01AAu, (0x01ABu << 3) | 2u},
    {0x01ACu, (0x01ACu << 3) | 1u}, {0x01ADu, (0x01ADu << 3) | 2u}, {0x01AEu, (0x01AFu << 3) | 1u}, {0x01B0u, (0x01B0u << 3) | 2u},
    {0x01B1u, (0x01B3u << 3) | 1u}, {0x01B4u, (0x01B4u << 3) | 2u}, {0x01B5u, (0x01B5u << 3) | 1u}, {0x01B6u, (0x01B6u << 3) | 2u},
    {0x01B7u, (0x01B8u << 3) | 1u}, {0x01B9u, (0x01BAu << 3) | 2u}, {0x01BBu, (0x01BBu << 3) | 3u}, {0x01BCu, (0x01BCu << 3) | 1u},
    {0x01BDu, (0x01BFu << 3) | 2u}, {0x01C0u, (0x01C3u << 3) | 3u}, {0x01C4u, (0x01C5u << 3) | 1u}, {0x01C6u, (0x01C6u << 3) | 2u},
    {0x01C7u, (0x01C8u << 3) | 1u}, {0x01C9u, (0x01C9u << 3) | 2u}, {0x01CAu, (0x01CBu << 3) | 1u}, {0x01CCu, (0x01CCu << 3) | 2u},
    {0x01CDu, (0x01CDu << 3) | 1u}, {0x01CEu, (0x01CEu << 3) | 2u}, {0x01CFu, (0x01CFu << 3) | 1u}, {0x01D0u, (0x01D0u << 3) | 2u},
    {0x01D1u, (0x01D1u << 3) | 1u}, {0x01D2u, (0x01D2u << 3) | 2u}, {0x01D3u, (0x01D3u << 3) | 1u}, {0x01D4u, (0x01D4u << 3) | 2u},
    {0x01D5u, (0x01D5u << 3) | 1u}, {0x01D6u, (0x01D6u << 3) | 2u}, {0x01D7u, (0x01D7u << 3) | 1u}, {0x01D8u, (0x01D8u << 3) | 2u},
    {0x01D9u, (0x01D9u << 3) | 1u}, {0x01DAu, (0x01DAu << 3) | 2u}, {0x01DBu, (0x01DBu << 3) | 1u}, {0x01DCu, (0x01DDu << 3) | 2u},
    {0x01DEu, (0x01DEu << 3) | 1u}, {0x01DFu, (0x01DFu << 3) | 2u}, {0x01E0u, (0x01E0u << 3) | 1u}, {0x01E1u, (0x01E1u << 3) | 2u},
    {0x01E2u, (0x01E2u << 3) | 1u}, {0x01E3u, (0x01E3u << 3) | 2u}, {0x01E4u, (0x01E4u << 3) | 1u}, {0x01E5u, (0x01E5u << 3) | 2u},
    {0x01E6u, (0x01E6u << 3) | 1u}, {0x01E7u, (0x01E7u << 3) | 2u}, {0x01E8u, (0x01E8u << 3) | 1u}, {0x01E9u, (0x01E9u << 3) | 2u},
    {0x01EAu, (0x01EAu << 3) | 1u}, {0x01EBu, (0x01EBu << 3) | 2u}, {0x01ECu, (0x01ECu << 3) | 1u}, {0x01EDu, (0x01EDu << 3) | 2u},
    {0x01EEu, (0x01EEu << 3) | 1u}, {0x01EFu, (0x01F0u << 3) | 2u}, {0x01F1u, (0x01F2u << 3) | 1u}, {0x01F3u, (0x01F3u << 3) | 2u},
    {0x01F4u, (0x01F4u << 3) | 1u}, {0x01F5u, (0x01F5u << 3) | 2u}, {0x01F6u, (0x01F8u << 3) | 1u}, {0x01F9u, (0x01F9u << 3) | 2u},
    {0x01FAu, (0x01FAu << 3) | 1u}, {0x01FBu, (0x01FBu << 3) | 2u}, {0x01FCu, (0x01FCu << 3) | 1u}, {0x01FDu, (0x01FDu << 3) | 2u},
    {0x01FEu, (0x01FEu << 3) | 1u}, {0x01FFu, (0x01FFu << 3) | 2u}, {0x0200u, (0x0200u << 3) | 1u}, {0x0201u, (0x0201u << 3) | 2u},
    {0x0202u, (0x0202u << 3) | 1u}, {0x0203u, (0x0203u << 3) | 2u}, {0x0204u, (0x0204u << 3) | 1u}, {0x0205u, (0x0205u << 3) | 2u},
    {0x0206u, (0x0206u << 3) | 1u}, {0x0207u, (0x0207u << 3) | 2u}, {0x0208u, (0x0208u << 3) | 1u}, {0x0209u, (0x0209u << 3) | 2u},
    {0x020Au, (0x020Au << 3) | 1u}, {0x020Bu, (0x020Bu << 3) | 2u}, {0x020Cu, (0x020Cu << 3) | 1u}, {0x020Du, (0x020Du << 3) | 2u},
    {0x020Eu, (0x020Eu << 3) | 1u}, {0x020Fu, (0x020Fu << 3) | 2u}, {0x0210u, (0x0210u << 3) | 1u}, {0x0211u, (0x0211u << 3) | 2u},
    {0x0212u, (0x0212u << 3) | 1u}, {0x0213u, (0x0213u << 3) | 2u}, {0x0214u, (0x0214u << 3) | 1u}, {0x0215u, (0x0215u << 3) | 2u},
    {0x0216u, (0x0216u << 3) | 1u}, {0x0217u, (0x0217u << 3) | 2u}, {0x0218u, (0x0218u << 3) | 1u}, {0x0219u, (0x0219u << 3) | 2u},
    {0x021Au, (0x021Au << 3) | 1u}, {0x021Bu, (0x021Bu << 3) | 2u}, {0x021Cu, (0x021Cu << 3) | 1u}, {0x021Du, (0x021Du << 3) | 2u},
    {0x021Eu, (0x021Eu << 3) | 1u}, {0x021Fu, (0x021Fu << 3) | 2u}, {0x0220u, (0x0220u << 3) | 1u}, {0x0221u, (0x0221u << 3) | 2u},
    {0x0222u, (0x0222u << 3) | 1u}, {0x0223u, (0x0223u << 3) | 2u}, {0x0224u, (0x0224u << 3) | 1u}, {0x0225u, (0x0225u << 3) | 2u},
    {0x0226u, (0x0226u << 3) | 1u}, {0x0227u, (0x0227u << 3) | 2u}, {0x0228u, (0x0228u << 3) | 1u}, {0x0229u, (0x0229u << 3) | 2u},
    {0x022Au, (0x022Au << 3) | 1u}, {0x022Bu, (0x022Bu << 3) | 2u}, {0x022Cu, (0x022Cu << 3) | 1u}, {0x022Du, (0x022Du << 3) | 2u},
    {0x022Eu, (0x022Eu << 3) | 1u}, {0x022Fu, (0x022Fu << 3) | 2u}, {0x0230u, (0x0230u << 3) | 1u}, {0x0231u, (0x0231u << 3) | 2u},
    {0x0232u, (0x0232u << 3) | 1u}, {0x0233u, (0x0239u << 3) | 2u}, {0x023Au, (0x023Bu << 3) | 1u}, {0x023Cu, (0x023Cu << 3) | 2u},
    {0x023Du, (0x023Eu << 3) | 1u}, {0x023Fu, (0x0240u << 3) | 2u}, {0x0241u, (0x0241u << 3) | 1u}, {0x0242u, (0x0242u << 3) | 2u},
    {0x0243u, (0x0246u << 3) | 1u}, {0x0247u, (0x0247u << 3) | 2u}, {0x0248u, (0x0248u << 3) | 1u}, {0x0249u, (0x0249u << 3) | 2u},
    {0x024Au, (0x024Au << 3) | 1u}, {0x024Bu, (0x024Bu << 3) | 2u}, {0x024Cu, (0x024Cu << 3) | 1u}, {0x024Du, (0x024Du << 3) | 2u},
    {0x024Eu, (0x024Eu << 3) | 1u}, {0x024Fu, (0x0293u << 3) | 2u}, {0x0294u, (0x0294u << 3) | 3u}, {0x0295u, (0x02AFu << 3) | 2u},
    {0x02B0u, (0x02C1u << 3) | 3u}, {0x02C6u, (0x02D1u << 3) | 3u}, {0x02E0u, (0x02E4u << 3) | 3u}, {0x02ECu, (0x02ECu << 3) | 3u},
    {0x02EEu, (0x02EEu << 3) | 3u}, {0x0300u, (0x036Fu << 3) | 4u}, {0x0370u, (0x0370u << 3) | 1u}, {0x0371u, (0x0371u << 3) | 2u},
    {0x0372u, (0x0372u << 3) | 1u}, {0x0373u, (0x0373u << 3) | 2u}, {0x0374u, (0x0374u << 3) | 3u}, {0x0376u, (0x0376u << 3) | 1u},
    {0x0377u, (0x0377u << 3) | 2u}, {0x037Au, (0x037Au << 3) | 3u}, {0x037Bu, (0x037Du << 3) | 2u}, {0x037Fu, (0x037Fu << 3) | 1u},
    {0x0386u, (0x0386u << 3) | 1u}, {0x0388u, (0x038Au << 3) | 1u}, {0x038Cu, (0x038Cu << 3) | 1u}, {0x038Eu, (0x038Fu << 3) | 1u},
    {0x0390u, (0x0390u << 3) | 2u}, {0x0391u, (0x03A1u << 3) | 1u}, {0x03A3u, (0x03ABu << 3) | 1u}, {0x03ACu, (0x03CEu << 3) | 2u},
    {0x03CFu, (0x03CFu << 3) | 1u}, {0x03D0u, (0x03D1u << 3) | 2u}, {0x03D2u, (0x03D4u << 3) | 1u}, {0x03D5u, (0x03D7u << 3) | 2u},
    {0x03D8u, (0x03D8u << 3) | 1u}, {0x03D9u, (0x03D9u << 3) | 2u}, {0x03DAu, (0x03DAu << 3) | 1u}, {0x03DBu, (0x03DBu << 3) | 2u},
    {0x03DCu, (0x03DCu << 3) | 1u}, {0x03DDu, (0x03DDu << 3) | 2u}, {0x03DEu, (0x03DEu << 3) | 1u}, {0x03DFu, (0x03DFu << 3) | 2u},
    {0x03E0u, (0x03E0u << 3) | 1u}, {0x03E1u, (0x03E1u << 3) | 2u}, {0x03E2u, (0x03E2u << 3) | 1u}, {0x03E3u, (0x03E3u << 3) | 2u},
    {0x03E4u, (0x03E4u << 3) | 1u}, {0x03E5u, (0x03E5u << 3) | 2u}, {0x03E6u, (0x03E6u << 3) | 1u}, {0x03E7u, (0x03E7u << 3) | 2u},
    {0x03E8u, (0x03E8u << 3) | 1u}, {0x03E9u, (0x03E9u << 3) | 2u}, {0x03EAu, (0x03EAu << 3) | 1u}, {0x03EBu, (0x03EBu << 3) | 2u},
    {0x03ECu, (0x03ECu << 3) | 1u}, {0x03EDu, (0x03EDu << 3) | 2u}, {0x03EEu, (0x03EEu << 3) | 1u}, {0x03EFu, (0x03F3u << 3) | 2u},
    {0x03F4u, (0x03F4u << 3) | 1u}, {0x03F5u, (0x03F5u << 3) | 2u}, {0x03F7u, (0x03F7u << 3) | 1u}, {0x03F8u, (0x03F8u << 3) | 2u},
    {0x03F9u, (0x03FAu << 3) | 1u}, {0x03FBu, (0x03FCu << 3) | 2u}, {0x03FDu, (0x042Fu << 3) | 1u}, {0x0430u, (0x045Fu << 3) | 2u},
    {0x0460u, (0x0460u << 3) | 1u}, {0x0461u, (0x0461u << 3) | 2u}, {0x0462u, (0x0462u << 3) | 1u}, {0x0463u, (0x0463u << 3) | 2u},
    {0x0464u, (0x0464u << 3) | 1u}, {0x0465u, (0x0465u << 3) | 2u}, {0x0466u, (0x0466u << 3) | 1u}, {0x0467u, (0x0467u << 3) | 2u},
    {0x0468u, (0x0468u << 3) | 1u}, {0x0469u, (0x0469u << 3) | 2u}, {0x046Au, (0x046Au << 3) | 1u}, {0x046Bu, (0x046Bu << 3) | 2u},
    {0x046Cu, (0x046Cu << 3) | 1u}, {0x046Du, (0x046Du << 3) | 2u}, {0x046Eu, (0x046Eu << 3) | 1u}, {0x046Fu, (0x046Fu << 3) | 2u},
    {0x0470u, (0x0470u << 3) | 1u}, {0x0471u, (0x0471u << 3) | 2u}, {0x0472u, (0x0472u << 3) | 1u}, {0x0473u, (0x0473u << 3) | 2u},
    {0x0474u, (0x0474u << 3) | 1u}, {0x0475u, (0x0475u << 3) | 2u}, {0x0476u, (0x0476u << 3) | 1u}, {0x0477u, (0x0477u << 3) | 2u},
    {0x0478u, (0x0478u << 3) | 1u}, {0x0479u, (0x0479u << 3) | 2u}, {0x047Au, (0x047Au << 3) | 1u}, {0x047Bu, (0x047Bu << 3) | 2u},
    {0x047Cu, (0x047Cu << 3) | 1u}, {0x047Du, (0x047Du << 3) | 2u}, {0x047Eu, (0x047Eu << 3) | 1u}, {0x047Fu, (0x047Fu << 3) | 2u},
    {0x0480u, (0x0480u << 3) | 1u}, {0x0481u, (0x0481u << 3) | 2u}, {0x0483u, (0x0489u << 3) | 4u}, {0x048Au, (0x048Au << 3) | 1u},
    {0x048Bu, (0x048Bu << 3) | 2u}, {0x048Cu, (0x048Cu << 3) | 1u}, {0x048Du, (0x048Du << 3) | 2u}, {0x048Eu, (0x048Eu << 3) | 1u},
    {0x048Fu, (0x048Fu << 3) | 2u}, {0x0490u, (0x0490u << 3) | 1u}, {0x0491u, (0x0491u << 3) | 2u}, {0x0492u, (0x0492u << 3) | 1u},
    {0x0493u, (0x0493u << 3) | 2u}, {0x0494u, (0x0494u << 3) | 1u}, {0x0495u, (0x0495u << 3) | 2u}, {0x0496u, (0x0496u << 3) | 1u},
    {0x0497u, (0x0497u << 3) | 2u}, {0x0498u, (0x0498u << 3) | 1u}, {0x0499u, (0x0499u << 3) | 2u}, {0x049Au, (0x049Au << 3) | 1u},
    {0x049Bu, (0x049Bu << 3) | 2u}, {0x049Cu, (0x049Cu << 3) | 1u}, {0x049Du, (0x049Du << 3) | 2u}, {0x049Eu, (0x049Eu << 3) | 1u},
    {0x049Fu, (0x049Fu << 3) | 2u}, {0x04A0u, (0x04A0u << 3) | 1u}, {0x04A1u, (0x04A1u << 3) | 2u}, {0x04A2u, (0x04A2u << 3) | 1u},
    {0x04A3u, (0x04A3u << 3) | 2u}, {0x04A4u, (0x04A4u << 3) | 1u}, {0x04A5u, (0x04A5u << 3) | 2u}, {0x04A6u, (0x04A6u << 3) | 1u},
    {0x04A7u, (0x04A7u << 3) | 2u}, {0x04A8u, (0x04A8u << 3) | 1u}, {0x04A9u, (0x04A9u << 3) | 2u}, {0x04AAu, (0x04AAu << 3) | 1u},
    {0x04ABu, (0x04ABu << 3) | 2u}, {0x04ACu, (0x04ACu << 3) | 1u}, {0x04ADu, (0x04ADu << 3) | 2u}, {0x04AEu, (0x04AEu << 3) | 1u},
    {0x04AFu, (0x04AFu << 3) | 2u}, {0x04B0u, (0x04B0u << 3) | 1u}, {0x04B1u, (0x04B1u << 3) | 2u}, {0x04B2u, (0x04B2u << 3) | 1u},
    {0x04B3u, (0x04B3u << 3) | 2u}, {0x04B4u, (0x04B4u << 3) | 1u}, {0x04B5u, (0x04B5u << 3) | 2u}, {0x04B6u, (0x04B6u << 3) | 1u},
    {0x04B7u, (0x04B7u << 3) | 2u}, {0x04B8u, (0x04B8u << 3) | 1u}, {0x04B9u, (0x04B9u << 3) | 2u}, {0x04BAu, (0x04BAu << 3) | 1u},
    {0x04BBu, (0x04BBu << 3) | 2u}, {0x04BCu, (0x04BCu << 3) | 1u}, {0x04BDu, (0x04BDu << 3) | 2u}, {0x04BEu, (0x04BEu << 3) | 1u},
    {0x04BFu, (0x04BFu << 3) | 2u}, {0x04C0u, (0x04C1u << 3) | 1u}, {0x04C2u, (0x04C2u << 3) | 2u}, {0x04C3u, (0x04C3u << 3) | 1u},
    {0x04C4u, (0x04C4u << 3) | 2u}, {0x04C5u, (0x04C5u << 3) | 1u}, {0x04C6u, (0x04C6u << 3) | 2u}, {0x04C7u, (0x04C7u << 3) | 1u},
    {0x04C8u, (0x04C8u << 3) | 2u}, {0x04C9u, (0x04C9u << 3) | 1u}, {0x04CAu, (0x04CAu << 3) | 2u}, {0x04CBu, (0x04CBu << 3) | 1u},
    {0x04CCu, (0x04CCu << 3) | 2u}, {0x04CDu, (0x04CDu << 3) | 1u}, {0x04CEu, (0x04CFu << 3) | 2u}, {0x04D0u, (0x04D0u << 3) | 1u},
    {0x04D1u, (0x04D1u << 3) | 2u}, {0x04D2u, (0x04D2u << 3) | 1u}, {0x04D3u, (0x04D3u << 3) | 2u}, {0x04D4u, (0x04D4u << 3) | 1u},
    {0x04D5u, (0x04D5u << 3) | 2u}, {0x04D6u, (0x04D6u << 3) | 1u}, {0x04D7u, (0x04D7u << 3) | 2u}, {0x04D8u, (0x04D8u << 3) | 1u},
    {0x04D9u, (0x04D9u << 3) | 2u}, {0x04DAu, (0x04DAu << 3) | 1u}, {0x04DBu, (0x04DBu << 3) | 2u}, {0x04DCu, (0x04DCu << 3) | 1u},
    {0x04DDu, (0x04DDu << 3) | 2u}, {0x04DEu, (0x04DEu << 3) | 1u}, {0x04DFu, (0x04DFu << 3) | 2u}, {0x04E0u, (0x04E0u << 3) | 1u},
    {0x04E1u, (0x04E1u << 3) | 2u}, {0x04E2u, (0x04E2u << 3) | 1u}, {0x04E3u, (0x04E3u << 3) | 2u}, {0x04E4u, (0x04E4u << 3) | 1u},
    {0x04E5u, (0x04E5u << 3) | 2u}, {0x04E6u, (0x04E6u << 3) | 1u}, {0x04E7u, (0x04E7u << 3) | 2u}, {0x04E8u, (0x04E8u << 3) | 1u},
    {0x04E9u, (0x04E9u << 3) | 2u}, {0x04EAu, (0x04EAu << 3) | 1u}, {0x04EBu, (0x04EBu << 3) | 2u}, {0x04ECu, (0x04ECu << 3) | 1u},
    {0x04EDu, (0x04EDu << 3) | 2u}, {0x04EEu, (0x04EEu << 3) | 1u}, {0x04EFu, (0x04EFu << 3) | 2u}, {0x04F0u, (0x04F0u << 3) | 1u},
    {0x04F1u, (0x04F1u << 3) | 2u}, {0x04F2u, (0x04F2u << 3) | 1u}, {0x04F3u, (0x04F3u << 3) | 2u}, {0x04F4u, (0x04F4u << 3) | 1u},
    {0x04F5u, (0x04F5u << 3) | 2u}, {0x04F6u, (0x04F6u << 3) | 1u}, {0x04F7u, (0x04F7u << 3) | 2u}, {0x04F8u, (0x04F8u << 3) | 1u},
    {0x04F9u, (0x04F9u << 3) | 2u}, {0x04FAu, (0x04FAu << 3) | 1u}, {0x04FBu, (0x04FBu << 3) | 2u}, {0x04FCu, (0x04FCu << 3) | 1u},
    {0x04FDu, (0x04FDu << 3) | 2u}, {0x04FEu, (0x04FEu << 3) | 1u}, {0x04FFu, (0x04FFu << 3) | 2u}, {0x0500u, (0x0500u << 3) | 1u},
    {0x0501u, (0x0501u << 3) | 2u}, {0x0502u, (0x0502u << 3) | 1u}, {0x0503u, (0x0503u << 3) | 2u}, {0x0504u, (0x0504u << 3) | 1u},
    {0x0505u, (0x0505u << 3) | 2u}, {0x0506u, (0x0506u << 3) | 1u}, {0x0507u, (0x0507u << 3) | 2u}, {0x0508u, (0x0508u << 3) | 1u},
    {0x0509u, (0x0509u << 3) | 2u}, {0x050Au, (0x050Au << 3) | 1u}, {0x050Bu, (0x050Bu << 3) | 2u}, {0x050Cu, (0x050Cu << 3) | 1u},
    {0x050Du, (0x050Du << 3) | 2u}, {0x050Eu, (0x050Eu << 3) | 1u}, {0x050Fu, (0x050Fu << 3) | 2u}, {0x0510u, (0x0510u << 3) | 1u},
    {0x0511u, (0x0511u << 3) | 2u}, {0x0512u, (0x0512u << 3) | 1u}, {0x0513u, (0x0513u << 3) | 2u}, {0x0514u, (0x0514u << 3) | 1u},
    {0x0515u, (0x0515u << 3) | 2u}, {0x0516u, (0x0516u << 3) | 1u}, {0x0517u, (0x0517u << 3) | 2u}, {0x0518u, (0x0518u << 3) | 1u},
    {0x0519u, (0x0519u << 3) | 2u}, {0x051Au, (0x051Au << 3) | 1u}, {0x051Bu, (0x051Bu << 3) | 2u}, {0x051Cu, (0x051Cu << 3) | 1u},
    {0x051Du, (0x051Du << 3) | 2u}, {0x051Eu, (0x051Eu << 3) | 1u}, {0x051Fu, (0x051Fu << 3) | 2u}, {0x0520u, (0x0520u << 3) | 1u},
    {0x0521u, (0x0521u << 3) | 2u}, {0x0522u, (0x0522u << 3) | 1u}, {0x0523u, (0x0523u << 3) | 2u}, {0x0524u, (0x0524u << 3) | 1u},
    {0x0525u, (0x0525u << 3) | 2u}, {0x0526u, (0x0526u << 3) | 1u}, {0x0527u, (0x0527u << 3) | 2u}, {0x0528u, (0x0528u << 3) | 1u},
    {0x0529u, (0x0529u << 3) | 2u}, {0x052Au, (0x052Au << 3) | 1u}, {0x052Bu, (0x052Bu << 3) | 2u}, {0x052Cu, (0x052Cu << 3) | 1u},
    {0x052Du, (0x052Du << 3) | 2u}, {0x052Eu, (0x052Eu << 3) | 1u}, {0x052Fu, (0x052Fu << 3) | 2u}, {0x0531u, (0x0556u << 3) | 1u},
    {0x0559u, (0x0559u << 3) | 3u}, {0x0560u, (0x0588u << 3) | 2u}, {0x0591u, (0x05BDu << 3) | 4u}, {0x05BFu, (0x05BFu << 3) | 4u},
    {0x05C1u, (0x05C2u << 3) | 4u}, {0x05C4u, (0x05C5u << 3) | 4u}, {0x05C7u, (0x05C7u << 3) | 4u}, {0x05D0u, (0x05EAu << 3) | 3u},
    {0x05EFu, (0x05F2u << 3) | 3u}, {0x0610u, (0x061Au << 3) | 4u}, {0x0620u, (0x064Au << 3) | 3u}, {0x064Bu, (0x065Fu << 3) | 4u},
    {0x0660u, (0x0669u << 3) | 5u}, {0x066Eu, (0x066Fu << 3) | 3u}, {0x0670u, (0x0670u << 3) | 4u}, {0x0671u, (0x06D3u << 3) | 3u},
    {0x06D5u, (0x06D5u << 3) | 3u}, {0x06D6u, (0x06DCu << 3) | 4u}, {0x06DFu, (0x06E4u << 3) | 4u}, {0x06E5u, (0x06E6u << 3) | 3u},
    {0x06E7u, (0x06E8u << 3) | 4u}, {0x06EAu, (0x06EDu << 3) | 4u}, {0x06EEu, (0x06EFu << 3) | 3u}, {0x06F0u, (0x06F9u << 3) | 5u},
    {0x06FAu, (0x06FCu << 3) | 3u}, {0x06FFu, (0x06FFu << 3) | 3u}, {0x0710u, (0x0710u << 3) | 3u}, {0x0711u, (0x0711u << 3) | 4u},
    {0x0712u, (0x072Fu << 3) | 3u}, {0x0730u, (0x074Au << 3) | 4u}, {0x074Du, (0x07A5u << 3) | 3u}, {0x07A6u, (0x07B0u << 3) | 4u},
    {0x07B1u, (0x07B1u << 3) | 3u}, {0x07C0u, (0x07C9u << 3) | 5u}, {0x07CAu, (0x07EAu << 3) | 3u}, {0x07EBu, (0x07F3u << 3) | 4u},
    {0x07F4u, (0x07F5u << 3) | 3u}, {0x07FAu, (0x07FAu << 3) | 3u}, {0x07FDu, (0x07FDu << 3) | 4u}, {0x0800u, (0x0815u << 3) | 3u},
    {0x0816u, (0x0819u << 3) | 4u}, {0x081Au, (0x081Au << 3) | 3u}, {0x081Bu, (0x0823u << 3) | 4u}, {0x0824u, (0x0824u << 3) | 3u},
    {0x0825u, (0x0827u << 3) | 4u}, {0x0828u, (0x0828u << 3) | 3u}, {0x0829u, (0x082Du << 3) | 4u}, {0x0840u, (0x0858u << 3) | 3u},
    {0x0859u, (0x085Bu << 3) | 4u}, {0x0860u, (0x086Au << 3) | 3u}, {0x08A0u, (0x08B4u << 3) | 3u}, {0x08B6u, (0x08C7u << 3) | 3u},
    {0x08D3u, (0x08E1u << 3) | 4u}, {0x08E3u, (0x0903u << 3) | 4u}, {0x0904u, (0x0939u << 3) | 3u}, {0x093Au, (0x093Cu << 3) | 4u},
    {0x093Du, (0x093Du << 3) | 3u}, {0x093Eu, (0x094Fu << 3) | 4u}, {0x0950u, (0x0950u << 3) | 3u}, {0x0951u, (0x0957u << 3) | 4u},
    {0x0958u, (0x0961u << 3) | 3u}, {0x0962u, (0x0963u << 3) | 4u}, {0x0966u, (0x096Fu << 3) | 5u}, {0x0971u, (0x0980u << 3) | 3u},
    {0x0981u, (0x0983u << 3) | 4u}, {0x0985u, (0x098Cu << 3) | 3u}, {0x098Fu, (0x0990u << 3) | 3u}, {0x0993u, (0x09A8u << 3) | 3u},
    {0x09AAu, (0x09B0u << 3) | 3u}, {0x09B2u, (0x09B2u << 3) | 3u}, {0x09B6u, (0x09B9u << 3) | 3u}, {0x09BCu, (0x09BCu << 3) | 4u},
    {0x09BDu, (0x09BDu << 3) | 3u}, {0x09BEu, (0x09C4u << 3) | 4u}, {0x09C7u, (0x09C8u << 3) | 4u}, {0x09CBu, (0x09CDu << 3) | 4u},
    {0x09CEu, (0x09CEu << 3) | 3u}, {0x09D7u, (0x09D7u << 3) | 4u}, {0x09DCu, (0x09DDu << 3) | 3u}, {0x09DFu, (0x09E1u << 3) | 3u},
    {0x09E2u, (0x09E3u << 3) | 4u}, {0x09E6u, (0x09EFu << 3) | 5u}, {0x09F0u, (0x09F1u << 3) | 3u}, {0x09F4u, (0x09F9u << 3) | 5u},
    {0x09FCu, (0x09FCu << 3) | 3u}, {0x09FEu, (0x09FEu << 3) | 4u}, {0x0A01u, (0x0A03u << 3) | 4u}, {0x0A05u, (0x0A0Au << 3) | 3u},
    {0x0A0Fu, (0x0A10u << 3) | 3u}, {0x0A13u, (0x0A28u << 3) | 3u}, {0x0A2Au, (0x0A30u << 3) | 3u}, {0x0A32u, (0x0A33u << 3) | 3u},
    {0x0A35u, (0x0A36u << 3) | 3u}, {0x0A38u, (0x0A39u << 3) | 3u}, {0x0A3Cu, (0x0A3Cu << 3) | 4u}, {0x0A3Eu, (0x0A42u << 3) | 4u},
    {0x0A47u, (0x0A48u << 3) | 4u}, {0x0A4Bu, (0x0A4Du << 3) | 4u}, {0x0A51u, (0x0A51u << 3) | 4u}, {0x0A59u, (0x0A5Cu << 3) | 3u},
    {0x0A5Eu, (0x0A5Eu << 3) | 3u}, {0x0A66u, (0x0A6Fu << 3) | 5u}, {0x0A70u, (0x0A71u << 3) | 4u}, {0x0A72u, (0x0A74u << 3) | 3u},
    {0x0A75u, (0x0A75u << 3) | 4u}, {0x0A81u, (0x0A83u << 3) | 4u}, {0x0A85u, (0x0A8Du << 3) | 3u}, {0x0A8Fu, (0x0A91u << 3) | 3u},
    {0x0A93u, (0x0AA8u << 3) | 3u}, {0x0AAAu, (0x0AB0u << 3) | 3u}, {0x0AB2u, (0x0AB3u << 3) | 3u}, {0x0AB5u, (0x0AB9u << 3) | 3u},
    {0x0ABCu, (0x0ABCu << 3) | 4u}, {0x0ABDu, (0x0ABDu << 3) | 3u}, {0x0ABEu, (0x0AC5u << 3) | 4u}, {0x0AC7u, (0x0AC9u << 3) | 4u},
    {0x0ACBu, (0x0ACDu << 3) | 4u}, {0x0AD0u, (0x0AD0u << 3) | 3u}, {0x0AE0u, (0x0AE1u << 3) | 3u}, {0x0AE2u, (0x0AE3u << 3) | 4u},
    {0x0AE6u, (0x0AEFu << 3) | 5u}, {0x0AF9u, (0x0AF9u << 3) | 3u}, {0x0AFAu, (0x0AFFu << 3) | 4u}, {0x0B01u, (0x0B03u << 3) | 4u},
    {0x0B05u, (0x0B0Cu << 3) | 3u}, {0x0B0Fu, (0x0B10u << 3) | 3u}, {0x0B13u, (0x0B28u << 3) | 3u}, {0x0B2Au, (0x0B30u << 3) | 3u},
    {0x0B32u, (0x0B33u << 3) | 3u}, {0x0B35u, (0x0B39u << 3) | 3u}, {0x0B3Cu, (0x0B3Cu << 3) | 4u}, {0x0B3Du, (0x0B3Du << 3) | 3u},
    {0x0B3Eu, (0x0B44u << 3) | 4u}, {0x0B47u, (0x0B48u << 3) | 4u}, {0x0B4Bu, (0x0B4Du << 3) | 4u}, {0x0B55u, (0x0B57u << 3) | 4u},
    {0x0B5Cu, (0x0B5Du << 3) | 3u}, {0x0B5Fu, (0x0B61u << 3) | 3u}, {0x0B62u, (0x0B63u << 3) | 4u}, {0x0B66u, (0x0B6Fu << 3) | 5u},
    {0x0B71u, (0x0B71u << 3) | 3u}, {0x0B72u, (0x0B77u << 3) | 5u}, {0x0B82u, (0x0B82u << 3) | 4u}, {0x0B83u, (0x0B83u << 3) | 3u},
    {0x0B85u, (0x0B8Au << 3) | 3u}, {0x0B8Eu, (0x0B90u << 3) | 3u}, {0x0B92u, (0x0B95u << 3) | 3u}, {0x0B99u, (0x0B9Au << 3) | 3u},
    {0x0B9Cu, (0x0B9Cu << 3) | 3u}, {0x0B9Eu, (0x0B9Fu << 3) | 3u}, {0x0BA3u, (0x0BA4u << 3) | 3u}, {0x0BA8u, (0x0BAAu << 3) | 3u},
    {0x0BAEu, (0x0BB9u << 3) | 3u}, {0x0BBEu, (0x0BC2u << 3) | 4u}, {0x0BC6u, (0x0BC8u << 3) | 4u}, {0x0BCAu, (0x0BCDu << 3) | 4u},
    {0x0BD0u, (0x0BD0u << 3) | 3u}, {0x0BD7u, (0x0BD7u << 3) | 4u}, {0x0BE6u, (0x0BF2u << 3) | 5u}, {0x0C00u, (0x0C04u << 3) | 4u},
    {0x0C05u, (0x0C0Cu << 3) | 3u}, {0x0C0Eu, (0x0C10u << 3) | 3u}, {0x0C12u, (0x0C28u << 3) | 3u}, {0x0C2Au, (0x0C39u << 3) | 3u},
    {0x0C3Du, (0x0C3Du << 3) | 3u}, {0x0C3Eu, (0x0C44u << 3) | 4u}, {0x0C46u, (0x0C48u << 3) | 4u}, {0x0C4Au, (0x0C4Du << 3) | 4u},
    {0x0C55u, (0x0C56u << 3) | 4u}, {0x0C58u, (0x0C5Au << 3) | 3u}, {0x0C60u, (0x0C61u << 3) | 3u}, {0x0C62u, (0x0C63u << 3) | 4u},
    {0x0C66u, (0x0C6Fu << 3) | 5u}, {0x0C78u, (0x0C7Eu << 3) | 5u}, {0x0C80u, (0x0C80u << 3) | 3u}, {0x0C81u, (0x0C83u << 3) | 4u},
    {0x0C85u, (0x0C8Cu << 3) | 3u}, {0x0C8Eu, (0x0C90u << 3) | 3u}, {0x0C92u, (0x0CA8u << 3) | 3u}, {0x0CAAu, (0x0CB3u << 3) | 3u},
    {0x0CB5u, (0x0CB9u << 3) | 3u}, {0x0CBCu, (0x0CBCu << 3) | 4u}, {0x0CBDu, (0x0CBDu << 3) | 3u}, {0x0CBEu, (0x0CC4u << 3) | 4u},
    {0x0CC6u, (0x0CC8u << 3) | 4u}, {0x0CCAu, (0x0CCDu << 3) | 4u}, {0x0CD5u, (0x0CD6u << 3) | 4u}, {0x0CDEu, (0x0CDEu << 3) | 3u},
    {0x0CE0u, (0x0CE1u << 3) | 3u}, {0x0CE2u, (0x0CE3u << 3) | 4u}, {0x0CE6u, (0x0CEFu << 3) | 5u}, {0x0CF1u, (0x0CF2u << 3) | 3u},
    {0x0D00u, (0x0D03u << 3) | 4u}, {0x0D04u, (0x0D0Cu << 3) | 3u}, {0x0D0Eu, (0x0D10u << 3) | 3u}, {0x0D12u, (0x0D3Au << 3) | 3u},
    {0x0D3Bu, (0x0D3Cu << 3) | 4u}, {0x0D3Du, (0x0D3Du << 3) | 3u}, {0x0D3Eu, (0x0D44u << 3) | 4u}, {0x0D46u, (0x0D48u << 3) | 4u},
    {0x0D4Au, (0x0D4Du << 3) | 4u}, {0x0D4Eu, (0x0D4Eu << 3) | 3u}, {0x0D54u, (0x0D56u << 3) | 3u}, {0x0D57u, (0x0D57u << 3) | 4u},
    {0x0D58u, (0x0D5Eu << 3) | 5u}, {0x0D5Fu, (0x0D61u << 3) | 3u}, {0x0D62u, (0x0D63u << 3) | 4u}, {0x0D66u, (0x0D78u << 3) | 5u},
    {0x0D7Au, (0x0D7Fu << 3) | 3u}, {0x0D81u, (0x0D83u << 3) | 4u}, {0x0D85u, (0x0D96u << 3) | 3u}, {0x0D9Au, (0x0DB1u << 3) | 3u},
    {0x0DB3u, (0x0DBBu << 3) | 3u}, {0x0DBDu, (0x0DBDu << 3) | 3u}, {0x0DC0u, (0x0DC6u << 3) | 3u}, {0x0DCAu, (0x0DCAu << 3) | 4u},
    {0x0DCFu, (0x0DD4u << 3) | 4u}, {0x0DD6u, (0x0DD6u << 3) | 4u}, {0x0DD8u, (0x0DDFu << 3) | 4u}, {0x0DE6u, (0x0DEFu << 3) | 5u},
    {0x0DF2u, (0x0DF3u << 3) | 4u}, {0x0E01u, (0x0E30u << 3) | 3u}, {0x0E31u, (0x0E31u << 3) | 4u}, {0x0E32u, (0x0E33u << 3) | 3u},
    {0x0E34u, (0x0E3Au << 3) | 4u}, {0x0E40u, (0x0E46u << 3) | 3u}, {0x0E47u, (0x0E4Eu << 3) | 4u}, {0x0E50u, (0x0E59u << 3) | 5u},
    {0x0E81u, (0x0E82u << 3) | 3u}, {0x0E84u, (0x0E84u << 3) | 3u}, {0x0E86u, (0x0E8Au << 3) | 3u}, {0x0E8Cu, (0x0EA3u << 3) | 3u},
    {0x0EA5u, (0x0EA5u << 3) | 3u}, {0x0EA7u, (0x0EB0u << 3) | 3u}, {0x0EB1u, (0x0EB1u << 3) | 4u}, {0x0EB2u, (0x0EB3u << 3) | 3u},
    {0x0EB4u, (0x0EBCu << 3) | 4u}, {0x0EBDu, (0x0EBDu << 3) | 3u}, {0x0EC0u, (0x0EC4u << 3) | 3u}, {0x0EC6u, (0x0EC6u << 3) | 3u},
    {0x0EC8u, (0x0ECDu << 3) | 4u}, {0x0ED0u, (0x0ED9u << 3) | 5u}, {0x0EDCu, (0x0EDFu << 3) | 3u}, {0x0F00u, (0x0F00u << 3) | 3u},
    {0x0F18u, (0x0F19u << 3) | 4u}, {0x0F20u, (0x0F33u << 3) | 5u}, {0x0F35u, (0x0F35u << 3) | 4u}, {0x0F37u, (0x0F37u << 3) | 4u},
    {0x0F39u, (0x0F39u << 3) | 4u}, {0x0F3Eu, (0x0F3Fu << 3) | 4u}, {0x0F40u, (0x0F47u << 3) | 3u}, {0x0F49u, (0x0F6Cu << 3) | 3u},
    {0x0F71u, (0x0F84u << 3) | 4u}, {0x0F86u, (0x0F87u << 3) | 4u}, {0x0F88u, (0x0F8Cu << 3) | 3u}, {0x0F8Du, (0x0F97u << 3) | 4u},
    {0x0F99u, (0x0FBCu << 3) | 4u}, {0x0FC6u, (0x0FC6u << 3) | 4u}, {0x1000u, (0x102Au << 3) | 3u}, {0x102Bu, (0x103Eu << 3) | 4u},
    {0x103Fu, (0x103Fu << 3) | 3u}, {0x1040u, (0x1049u << 3) | 5u}, {0x1050u, (0x1055u << 3) | 3u}, {0x1056u, (0x1059u << 3) | 4u},
    {0x105Au, (0x105Du << 3) | 3u}, {0x105Eu, (0x1060u << 3) | 4u}, {0x1061u, (0x1061u << 3) | 3u}, {0x1062u, (0x1064u << 3) | 4u},
    {0x1065u, (0x1066u << 3) | 3u}, {0x1067u, (0x106Du << 3) | 4u}, {0x106Eu, (0x1070u << 3) | 3u}, {0x1071u, (0x1074u << 3) | 4u},
    {0x1075u, (0x1081u << 3) | 3u}, {0x1082u, (0x108Du << 3) | 4u}, {0x108Eu, (0x108Eu << 3) | 3u}, {0x108Fu, (0x108Fu << 3) | 4u},
    {0x1090u, (0x1099u << 3) | 5u}, {0x109Au, (0x109Du << 3) | 4u}, {0x10A0u, (0x10C5u << 3) | 1u}, {0x10C7u, (0x10C7u << 3) | 1u},
    {0x10CDu, (0x10CDu << 3) | 1u}, {0x10D0u, (0x10FAu << 3) | 2u}, {0x10FCu, (0x10FCu << 3) | 3u}, {0x10FDu, (0x10FFu << 3) | 2u},
    {0x1100u, (0x1248u << 3) | 3u}, {0x124Au, (0x124Du << 3) | 3u}, {0x1250u, (0x1256u << 3) | 3u}, {0x1258u, (0x1258u << 3) | 3u},
    {0x125Au, (0x125Du << 3) | 3u}, {0x1260u, (0x1288u << 3) | 3u}, {0x128Au, (0x128Du << 3) | 3u}, {0x1290u, (0x12B0u << 3) | 3u},
    {0x12B2u, (0x12B5u << 3) | 3u}, {0x12B8u, (0x12BEu << 3) | 3u}, {0x12C0u, (0x12C0u << 3) | 3u}, {0x12C2u, (0x12C5u << 3) | 3u},
    {0x12C8u, (0x12D6u << 3) | 3u}, {0x12D8u, (0x1310u << 3) | 3u}, {0x1312u, (0x1315u << 3) | 3u}, {0x1318u, (0x135Au << 3) | 3u},
    {0x135Du, (0x135Fu << 3) | 4u}, {0x1369u, (0x137Cu << 3) | 5u}, {0x1380u, (0x138Fu << 3) | 3u}, {0x13A0u, (0x13F5u << 3) | 1u},
    {0x13F8u, (0x13FDu << 3) | 2u}, {0x1401u, (0x166Cu << 3) | 3u}, {0x166Fu, (0x167Fu << 3) | 3u}, {0x1681u, (0x169Au << 3) | 3u},
    {0x16A0u, (0x16EAu << 3) | 3u}, {0x16EEu, (0x16F0u << 3) | 5u}, {0x16F1u, (0x16F8u << 3) | 3u}, {0x1700u, (0x170Cu << 3) | 3u},
    {0x170Eu, (0x1711u << 3) | 3u}, {0x1712u, (0x1714u << 3) | 4u}, {0x1720u, (0x1731u << 3) | 3u}, {0x1732u, (0x1734u << 3) | 4u},
    {0x1740u, (0x1751u << 3) | 3u}, {0x1752u, (0x1753u << 3) | 4u}, {0x1760u, (0x176Cu << 3) | 3u}, {0x176Eu, (0x1770u << 3) | 3u},
    {0x1772u, (0x1773u << 3) | 4u}, {0x1780u, (0x17B3u << 3) | 3u}, {0x17B4u, (0x17D3u << 3) | 4u}, {0x17D7u, (0x17D7u << 3) | 3u},
    {0x17DCu, (0x17DCu << 3) | 3u}, {0x17DDu, (0x17DDu << 3) | 4u}, {0x17E0u, (0x17E9u << 3) | 5u}, {0x17F0u, (0x17F9u << 3) | 5u},
    {0x180Bu, (0x180Du << 3) | 4u}, {0x1810u, (0x1819u << 3) | 5u}, {0x1820u, (0x1878u << 3) | 3u}, {0x1880u, (0x1884u << 3) | 3u},
    {0x1885u, (0x1886u << 3) | 4u}, {0x1887u, (0x18A8u << 3) | 3u}, {0x18A9u, (0x18A9u << 3) | 4u}, {0x18AAu, (0x18AAu << 3) | 3u},
    {0x18B0u, (0x18F5u << 3) | 3u}, {0x1900u, (0x191Eu << 3) | 3u}, {0x1920u, (0x192Bu << 3) | 4u}, {0x1930u, (0x193Bu << 3) | 4u},
    {0x1946u, (0x194Fu << 3) | 5u}, {0x1950u, (0x196Du << 3) | 3u}, {0x1970u, (0x1974u << 3) | 3u}, {0x1980u, (0x19ABu << 3) | 3u},
    {0x19B0u, (0x19C9u << 3) | 3u}, {0x19D0u, (0x19DAu << 3) | 5u}, {0x1A00u, (0x1A16u << 3) | 3u}, {0x1A17u, (0x1A1Bu << 3) | 4u},
    {0x1A20u, (0x1A54u << 3) | 3u}, {0x1A55u, (0x1A5Eu << 3) | 4u}, {0x1A60u, (0x1A7Cu << 3) | 4u}, {0x1A7Fu, (0x1A7Fu << 3) | 4u},
    {0x1A80u, (0x1A89u << 3) | 5u}, {0x1A90u, (0x1A99u << 3) | 5u}, {0x1AA7u, (0x1AA7u << 3) | 3u}, {0x1AB0u, (0x1AC0u << 3) | 4u},
    {0x1B00u, (0x1B04u << 3) | 4u}, {0x1B05u, (0x1B33u << 3) | 3u}, {0x1B34u, (0x1B44u << 3) | 4u}, {0x1B45u, (0x1B4Bu << 3) | 3u},
    {0x1B50u, (0x1B59u << 3) | 5u}, {0x1B6Bu, (0x1B73u << 3) | 4u}, {0x1B80u, (0x1B82u << 3) | 4u}, {0x1B83u, (0x1BA0u << 3) | 3u},
    {0x1BA1u, (0x1BADu << 3) | 4u}, {0x1BAEu, (0x1BAFu << 3) | 3u}, {0x1BB0u, (0x1BB9u << 3) | 5u}, {0x1BBAu, (0x1BE5u << 3) | 3u},
    {0x1BE6u, (0x1BF3u << 3) | 4u}, {0x1C00u, (0x1C23u << 3) | 3u}, {0x1C24u, (0x1C37u << 3) | 4u}, {0x1C40u, (0x1C49u << 3) | 5u},
    {0x1C4Du, (0x1C4Fu << 3) | 3u}, {0x1C50u, (0x1C59u << 3) | 5u}, {0x1C5Au, (0x1C7Du << 3) | 3u}, {0x1C80u, (0x1C88u << 3) | 2u},
    {0x1C90u, (0x1CBAu << 3) | 1u}, {0x1CBDu, (0x1CBFu << 3) | 1u}, {0x1CD0u, (0x1CD2u << 3) | 4u}, {0x1CD4u, (0x1CE8u << 3) | 4u},
    {0x1CE9u, (0x1CECu << 3) | 3u}, {0x1CEDu, (0x1CEDu << 3) | 4u}, {0x1CEEu, (0x1CF3u << 3) | 3u}, {0x1CF4u, (0x1CF4u << 3) | 4u},
    {0x1CF5u, (0x1CF6u << 3) | 3u}, {0x1CF7u, (0x1CF9u << 3) | 4u}, {0x1CFAu, (0x1CFAu << 3) | 3u}, {0x1D00u, (0x1D2Bu << 3) | 2u},
    {0x1D2Cu, (0x1D6Au << 3) | 3u}, {0x1D6Bu, (0x1D77u << 3) | 2u}, {0x1D78u, (0x1D78u << 3) | 3u}, {0x1D79u, (0x1D9Au << 3) | 2u},
    {0x1D9Bu, (0x1DBFu << 3) | 3u}, {0x1DC0u, (0x1DF9u << 3) | 4u}, {0x1DFBu, (0x1DFFu << 3) | 4u}, {0x1E00u, (0x1E00u << 3) | 1u},
    {0x1E01u, (0x1E01u << 3) | 2u}, {0x1E02u, (0x1E02u << 3) | 1u}, {0x1E03u, (0x1E03u << 3) | 2u}, {0x1E04u, (0x1E04u << 3) | 1u},
    {0x1E05u, (0x1E05u << 3) | 2u}, {0x1E06u, (0x1E06u << 3) | 1u}, {0x1E07u, (0x1E07u << 3) | 2u}, {0x1E08u, (0x1E08u << 3) | 1u},
    {0x1E09u, (0x1E09u << 3) | 2u}, {0x1E0Au, (0x1E0Au << 3) | 1u}, {0x1E0Bu, (0x1E0Bu << 3) | 2u}, {0x1E0Cu, (0x1E0Cu << 3) | 1u},
    {0x1E0Du, (0x1E0Du << 3) | 2u}, {0x1E0Eu, (0x1E0Eu << 3) | 1u}, {0x1E0Fu, (0x1E0Fu << 3) | 2u}, {0x1E10u, (0x1E10u << 3) | 1u},
    {0x1E11u, (0x1E11u << 3) | 2u}, {0x1E12u, (0x1E12u << 3) | 1u}, {0x1E13u, (0x1E13u << 3) | 2u}, {0x1E14u, (0x1E14u << 3) | 1u},
    {0x1E15u, (0x1E15u << 3) | 2u}, {0x1E16u, (0x1E16u << 3) | 1u}, {0x1E17u, (0x1E17u << 3) | 2u}, {0x1E18u, (0x1E18u << 3) | 1u},
    {0x1E19u, (0x1E19u << 3) | 2u}, {0x1E1Au, (0x1E1Au << 3) | 1u}, {0x1E1Bu, (0x1E1Bu << 3) | 2u}, {0x1E1Cu, (0x1E1Cu << 3) | 1u},
    {0x1E1Du, (0x1E1Du << 3) | 2u}, {0x1E1Eu, (0x1E1Eu << 3) | 1u}, {0x1E1Fu, (0x1E1Fu << 3) | 2u}, {0x1E20u, (0x1E20u << 3) | 1u},
    {0x1E21u, (0x1E21u << 3) | 2u}, {0x1E22u, (0x1E22u << 3) | 1u}, {0x1E23u, (0x1E23u << 3) | 2u}, {0x1E24u, (0x1E24u << 3) | 1u},
    {0x1E25u, (0x1E25u << 3) | 2u}, {0x1E26u, (0x1E26u << 3) | 1u}, {0x1E27u, (0x1E27u << 3) | 2u}, {0x1E28u, (0x1E28u << 3) | 1u},
    {0x1E29u, (0x1E29u << 3) | 2u}, {0x1E2Au, (0x1E2Au << 3) | 1u}, {0x1E2Bu, (0x1E2Bu << 3) | 2u}, {0x1E2Cu, (0x1E2Cu << 3) | 1u},
    {0x1E2Du, (0x1E2Du << 3) | 2u}, {0x1E2Eu, (0x1E2Eu << 3) | 1u}, {0x1E2Fu, (0x1E2Fu << 3) | 2u}, {0x1E30u, (0x1E30u << 3) | 1u},
    {0x1E31u, (0x1E31u << 3) | 2u}, {0x1E32u, (0x1E32u << 3) | 1u}, {0x1E33u, (0x1E33u << 3) | 2u}, {0x1E34u, (0x1E34u << 3) | 1u},
    {0x1E35u, (0x1E35u << 3) | 2u}, {0x1E36u, (0x1E36u << 3) | 1u}, {0x1E37u, (0x1E37u << 3) | 2u}, {0x1E38u, (0x1E38u << 3) | 1u},
    {0x1E39u, (0x1E39u << 3) | 2u}, {0x1E3Au, (0x1E3Au << 3) | 1u}, {0x1E3Bu, (0x1E3Bu << 3) | 2u}, {0x1E3Cu, (0x1E3Cu << 3) | 1u},
    {0x1E3Du, (0x1E3Du << 3) | 2u}, {0x1E3Eu, (0x1E3Eu << 3) | 1u}, {0x1E3Fu, (0x1E3Fu << 3) | 2u}, {0x1E40u, (0x1E40u << 3) | 1u},
    {0x1E41u, (0x1E41u << 3) | 2u}, {0x1E42u, (0x1E42u << 3) | 1u}, {0x1E43u, (0x1E43u << 3) | 2u}, {0x1E44u, (0x1E44u << 3) | 1u},
    {0x1E45u, (0x1E45u << 3) | 2u}, {0x1E46u, (0x1E46u << 3) | 1u}, {0x1E47u, (0x1E47u << 3) | 2u}, {0x1E48u, (0x1E48u << 3) | 1u},
    {0x1E49u, (0x1E49u << 3) | 2u}, {0x1E4Au, (0x1E4Au << 3) | 1u}, {0x1E4Bu, (0x1E4Bu << 3) | 2u}, {0x1E4Cu, (0x1E4Cu << 3) | 1u},
    {0x1E4Du, (0x1E4Du << 3) | 2u}, {0x1E4Eu, (0x1E4Eu << 3) | 1u}, {0x1E4Fu, (0x1E4Fu << 3) | 2u}, {0x1E50u, (0x1E50u << 3) | 1u},
    {0x1E51u, (0x1E51u << 3) | 2u}, {0x1E52u, (0x1E52u << 3) | 1u}, {0x1E53u, (0x1E53u << 3) | 2u}, {0x1E54u, (0x1E54u << 3) | 1u},
    {0x1E55u, (0x1E55u << 3) | 2u}, {0x1E56u, (0x1E56u << 3) | 1u}, {0x1E57u, (0x1E57u << 3) | 2u}, {0x1E58u, (0x1E58u << 3) | 1u},
    {0x1E59u, (0x1E59u << 3) | 2u}, {0x1E5Au, (0x1E5Au << 3) | 1u}, {0x1E5Bu, (0x1E5Bu << 3) | 2u}, {0x1E5Cu, (0x1E5Cu << 3) | 1u},
    {0x1E5Du, (0x1E5Du << 3) | 2u}, {0x1E5Eu, (0x1E5Eu << 3) | 1u}, {0x1E5Fu, (0x1E5Fu << 3) | 2u}, {0x1E60u, (0x1E60u << 3) | 1u},
    {0x1E61u, (0x1E61u << 3) | 2u}, {0x1E62u, (0x1E62u << 3) | 1u}, {0x1E63u, (0x1E63u << 3) | 2u}, {0x1E64u, (0x1E64u << 3) | 1u},
    {0x1E65u, (0x1E65u << 3) | 2u}, {0x1E66u, (0x1E66u << 3) | 1u}, {0x1E67u, (0x1E67u << 3) | 2u}, {0x1E68u, (0x1E68u << 3) | 1u},
    {0x1E69u, (0x1E69u << 3) | 2u}, {0x1E6Au, (0x1E6Au << 3) | 1u}, {0x1E6Bu, (0x1E6Bu << 3) | 2u}, {0x1E6Cu, (0x1E6Cu << 3) | 1u},
    {0x1E6Du, (0x1E6Du << 3) | 2u}, {0x1E6Eu, (0x1E6Eu << 3) | 1u}, {0x1E6Fu, (0x1E6Fu << 3) | 2u}, {0x1E70u, (0x1E70u << 3) | 1u},
    {0x1E71u, (0x1E71u << 3) | 2u}, {0x1E72u, (0x1E72u << 3) | 1u}, {0x1E73u, (0x1E73u << 3) | 2u}, {0x1E74u, (0x1E74u << 3) | 1u},
    {0x1E75u, (0x1E75u << 3) | 2u}, {0x1E76u, (0x1E76u << 3) | 1u}, {0x1E77u, (0x1E77u << 3) | 2u}, {0x1E78u, (0x1E78u << 3) | 1u},
    {0x1E79u, (0x1E79u << 3) | 2u}, {0x1E7Au, (0x1E7Au << 3) | 1u}, {0x1E7Bu, (0x1E7Bu << 3) | 2u}, {0x1E7Cu, (0x1E7Cu << 3) | 1u},
    {0x1E7Du, (0x1E7Du << 3) | 2u}, {0x1E7Eu, (0x1E7Eu << 3) | 1u}, {0x1E7Fu, (0x1E7Fu << 3) | 2u}, {0x1E80u, (0x1E80u << 3) | 1u},
    {0x1E81u, (0x1E81u << 3) | 2u}, {0x1E82u, (0x1E82u << 3) | 1u}, {0x1E83u, (0x1E83u << 3) | 2u}, {0x1E84u, (0x1E84u << 3) | 1u},
    {0x1E85u, (0x1E85u << 3) | 2u}, {0x1E86u, (0x1E86u << 3) | 1u}, {0x1E87u, (0x1E87u << 3) | 2u}, {0x1E88u, (0x1E88u << 3) | 1u},
    {0x1E89u, (0x1E89u << 3) | 2u}, {0x1E8Au, (0x1E8Au << 3) | 1u}, {0x1E8Bu, (0x1E8Bu << 3) | 2u}, {0x1E8Cu, (0x1E8Cu << 3) | 1u},
    {0x1E8Du, (0x1E8Du << 3) | 2u}, {0x1E8Eu, (0x1E8Eu << 3) | 1u}, {0x1E8Fu, (0x1E8Fu << 3) | 2u}, {0x1E90u, (0x1E90u << 3) | 1u},
    {0x1E91u, (0x1E91u << 3) | 2u}, {0x1E92u, (0x1E92u << 3) | 1u}, {0x1E93u, (0x1E93u << 3) | 2u}, {0x1E94u, (0x1E94u << 3) | 1u},
    {0x1E95u, (0x1E9Du << 3) | 2u}, {0x1E9Eu, (0x1E9Eu << 3) | 1u}, {0x1E9Fu, (0x1E9Fu << 3) | 2u}, {0x1EA0u, (0x1EA0u << 3) | 1u},
    {0x1EA1u, (0x1EA1u << 3) | 2u}, {0x1EA2u, (0x1EA2u << 3) | 1u}, {0x1EA3u, (0x1EA3u << 3) | 2u}, {0x1EA4u, (0x1EA4u << 3) | 1u},
    {0x1EA5u, (0x1EA5u << 3) | 2u}, {0x1EA6u, (0x1EA6u << 3) | 1u}, {0x1EA7u, (0x1EA7u << 3) | 2u}, {0x1EA8u, (0x1EA8u << 3) | 1u},
    {0x1EA9u, (0x1EA9u << 3) | 2u}, {0x1EAAu, (0x1EAAu << 3) | 1u}, {0x1EABu, (0x1EABu << 3) | 2u}, {0x1EACu, (0x1EACu << 3) | 1u},
    {0x1EADu, (0x1EADu << 3) | 2u}, {0x1EAEu, (0x1EAEu << 3) | 1u}, {0x1EAFu, (0x1EAFu << 3) | 2u}, {0x1EB0u, (0x1EB0u << 3) | 1u},
    {0x1EB1u, (0x1EB1u << 3) | 2u}, {0x1EB2u, (0x1EB2u << 3) | 1u}, {0x1EB3u, (0x1EB3u << 3) | 2u}, {0x1EB4u, (0x1EB4u << 3) | 1u},
    {0x1EB5u, (0x1EB5u << 3) | 2u}, {0x1EB6u, (0x1EB6u << 3) | 1u}, {0x1EB7u, (0x1EB7u << 3) | 2u}, {0x1EB8u, (0x1EB8u << 3) | 1u},
    {0x1EB9u, (0x1EB9u << 3) | 2u}, {0x1EBAu, (0x1EBAu << 3) | 1u}, {0x1EBBu, (0x1EBBu << 3) | 2u}, {0x1EBCu, (0x1EBCu << 3) | 1u},
    {0x1EBDu, (0x1EBDu << 3) | 2u}, {0x1EBEu, (0x1EBEu << 3) | 1u}, {0x1EBFu, (0x1EBFu << 3) | 2u}, {0x1EC0u, (0x1EC0u << 3) | 1u},
    {0x1EC1u, (0x1EC1u << 3) | 2u}, {0x1EC2u, (0x1EC2u << 3) | 1u}, {0x1EC3u, (0x1EC3u << 3) | 2u}, {0x1EC4u, (0x1EC4u << 3) | 1u},
    {0x1EC5u, (0x1EC5u << 3) | 2u}, {0x1EC6u, (0x1EC6u << 3) | 1u}, {0x1EC7u, (0x1EC7u << 3) | 2u}, {0x1EC8u, (0x1EC8u << 3) | 1u},
    {0x1EC9u, (0x1EC9u << 3) | 2u}, {0x1ECAu, (0x1ECAu << 3) | 1u}, {0x1ECBu, (0x1ECBu << 3) | 2u}, {0x1ECCu, (0x1ECCu << 3) | 1u},
    {0x1ECDu, (0x1ECDu << 3) | 2u}, {0x1ECEu, (0x1ECEu << 3) | 1u}, {0x1ECFu, (0x1ECFu << 3) | 2u}, {0x1ED0u, (0x1ED0u << 3) | 1u},
    {0x1ED1u, (0x1ED1u << 3) | 2u}, {0x1ED2u, (0x1ED2u << 3) | 1u}, {0x1ED3u, (0x1ED3u << 3) | 2u}, {0x1ED4u, (0x1ED4u << 3) | 1u},
    {0x1ED5u, (0x1ED5u << 3) | 2u}, {0x1ED6u, (0x1ED6u << 3) | 1u}, {0x1ED7u, (0x1ED7u << 3) | 2u}, {0x1ED8u, (0x1ED8u << 3) | 1u},
    {0x1ED9u, (0x1ED9u << 3) | 2u}, {0x1EDAu, (0x1EDAu << 3) | 1u}, {0x1EDBu, (0x1EDBu << 3) | 2u}, {0x1EDCu, (0x1EDCu << 3) | 1u},
    {0x1EDDu, (0x1EDDu << 3) | 2u}, {0x1EDEu, (0x1EDEu << 3) | 1u}, {0x1EDFu, (0x1EDFu << 3) | 2u}, {0x1EE0u, (0x1EE0u << 3) | 1u},
    {0x1EE1u, (0x1EE1u << 3) | 2u}, {0x1EE2u, (0x1EE2u << 3) | 1u}, {0x1EE3u, (0x1EE3u << 3) | 2u}, {0x1EE4u, (0x1EE4u << 3) | 1u},
    {0x1EE5u, (0x1EE5u << 3) | 2u}, {0x1EE6u, (0x1EE6u << 3) | 1u}, {0x1EE7u, (0x1EE7u << 3) | 2u}, {0x1EE8u, (0x1EE8u << 3) | 1u},
    {0x1EE9u, (0x1EE9u << 3) | 2u}, {0x1EEAu, (0x1EEAu << 3) | 1u}, {0x1EEBu, (0x1EEBu << 3) | 2u}, {0x1EECu, (0x1EECu << 3) | 1u},
    {0x1EEDu, (0x1EEDu << 3) | 2u}, {0x1EEEu, (0x1EEEu << 3) | 1u}, {0x1EEFu, (0x1EEFu << 3) | 2u}, {0x1EF0u, (0x1EF0u << 3) | 1u},
    {0x1EF1u, (0x1EF1u << 3) | 2u}, {0x1EF2u, (0x1EF2u << 3) | 1u}, {0x1EF3u, (0x1EF3u << 3) | 2u}, {0x1EF4u, (0x1EF4u << 3) | 1u},
    {0x1EF5u, (0x1EF5u << 3) | 2u}, {0x1EF6u, (0x1EF6u << 3) | 1u}, {0x1EF7u, (0x1EF7u << 3) | 2u}, {0x1EF8u, (0x1EF8u << 3) | 1u},
    {0x1EF9u, (0x1EF9u << 3) | 2u}, {0x1EFAu, (0x1EFAu << 3) | 1u}, {0x1EFBu, (0x1EFBu << 3) | 2u}, {0x1EFCu, (0x1EFCu << 3) | 1u},
    {0x1EFDu, (0x1EFDu << 3) | 2u}, {0x1EFEu, (0x1EFEu << 3) | 1u}, {0x1EFFu, (0x1F07u << 3) | 2u}, {0x1F08u, (0x1F0Fu << 3) | 1u},
    {0x1F10u, (0x1F15u << 3) | 2u}, {0x1F18u, (0x1F1Du << 3) | 1u}, {0x1F20u, (0x1F27u << 3) | 2u}, {0x1F28u, (0x1F2Fu << 3) | 1u},
    {0x1F30u, (0x1F37u << 3) | 2u}, {0x1F38u, (0x1F3Fu << 3) | 1u}, {0x1F40u, (0x1F45u << 3) | 2u}, {0x1F48u, (0x1F4Du << 3) | 1u},
    {0x1F50u, (0x1F57u << 3) | 2u}, {0x1F59u, (0x1F59u << 3) | 1u}, {0x1F5Bu, (0x1F5Bu << 3) | 1u}, {0x1F5Du, (0x1F5Du << 3) | 1u},
    {0x1F5Fu, (0x1F5Fu << 3) | 1u}, {0x1F60u, (0x1F67u << 3) | 2u}, {0x1F68u, (0x1F6Fu << 3) | 1u}, {0x1F70u, (0x1F7Du << 3) | 2u},
    {0x1F80u, (0x1F87u << 3) | 2u}, {0x1F88u, (0x1F8Fu << 3) | 1u}, {0x1F90u, (0x1F97u << 3) | 2u}, {0x1F98u, (0x1F9Fu << 3) | 1u},
    {0x1FA0u, (0x1FA7u << 3) | 2u}, {0x1FA8u, (0x1FAFu << 3) | 1u}, {0x1FB0u, (0x1FB4u << 3) | 2u}, {0x1FB6u, (0x1FB7u << 3) | 2u},
    {0x1FB8u, (0x1FBCu << 3) | 1u}, {0x1FBEu, (0x1FBEu << 3) | 2u}, {0x1FC2u, (0x1FC4u << 3) | 2u}, {0x1FC6u, (0x1FC7u << 3) | 2u},
    {0x1FC8u, (0x1FCCu << 3) | 1u}, {0x1FD0u, (0x1FD3u << 3) | 2u}, {0x1FD6u, (0x1FD7u << 3) | 2u}, {0x1FD8u, (0x1FDBu << 3) | 1u},
    {0x1FE0u, (0x1FE7u << 3) | 2u}, {0x1FE8u, (0x1FECu << 3) | 1u}, {0x1FF2u, (0x1FF4u << 3) | 2u}, {0x1FF6u, (0x1FF7u << 3) | 2u},
    {0x1FF8u, (0x1FFCu << 3) | 1u}, {0x2070u, (0x2070u << 3) | 5u}, {0x2071u, (0x2071u << 3) | 3u}, {0x2074u, (0x2079u << 3) | 5u},
    {0x207Fu, (0x207Fu << 3) | 3u}, {0x2080u, (0x2089u << 3) | 5u}, {0x2090u, (0x209Cu << 3) | 3u}, {0x20D0u, (0x20F0u << 3) | 4u},
    {0x2102u, (0x2102u << 3) | 1u}, {0x2107u, (0x2107u << 3) | 1u}, {0x210Au, (0x210Au << 3) | 2u}, {0x210Bu, (0x210Du << 3) | 1u},
    {0x210Eu, (0x210Fu << 3) | 2u}, {0x2110u, (0x2112u << 3) | 1u}, {0x2113u, (0x2113u << 3) | 2u}, {0x2115u, (0x2115u << 3) | 1u},
    {0x2119u, (0x211Du << 3) | 1u}, {0x2124u, (0x2124u << 3) | 1u}, {0x2126u, (0x2126u << 3) | 1u}, {0x2128u, (0x2128u << 3) | 1u},
    {0x212Au, (0x212Du << 3) | 1u}, {0x212Fu, (0x212Fu << 3) | 2u}, {0x2130u, (0x2133u << 3) | 1u}, {0x2134u, (0x2134u << 3) | 2u},
    {0x2135u, (0x2138u << 3) | 3u}, {0x2139u, (0x2139u << 3) | 2u}, {0x213Cu, (0x213Du << 3) | 2u}, {0x213Eu, (0x213Fu << 3) | 1u},
    {0x2145u, (0x2145u << 3) | 1u}, {0x2146u, (0x2149u << 3) | 2u}, {0x214Eu, (0x214Eu << 3) | 2u}, {0x2150u, (0x2182u << 3) | 5u},
    {0x2183u, (0x2183u << 3) | 1u}, {0x2184u, (0x2184u << 3) | 2u}, {0x2185u, (0x2189u << 3) | 5u}, {0x2460u, (0x249Bu << 3) | 5u},
    {0x24EAu, (0x24FFu << 3) | 5u}, {0x2776u, (0x2793u << 3) | 5u}, {0x2C00u, (0x2C2Eu << 3) | 1u}, {0x2C30u, (0x2C5Eu << 3) | 2u},
    {0x2C60u, (0x2C60u << 3) | 1u}, {0x2C61u, (0x2C61u << 3) | 2u}, {0x2C62u, (0x2C64u << 3) | 1u}, {0x2C65u, (0x2C66u << 3) | 2u},
    {0x2C67u, (0x2C67u << 3) | 1u}, {0x2C68u, (0x2C68u << 3) | 2u}, {0x2C69u, (0x2C69u << 3) | 1u}, {0x2C6Au, (0x2C6Au << 3) | 2u},
    {0x2C6Bu, (0x2C6Bu << 3) | 1u}, {0x2C6Cu, (0x2C6Cu << 3) | 2u}, {0x2C6Du, (0x2C70u << 3) | 1u}, {0x2C71u, (0x2C71u << 3) | 2u},
    {0x2C72u, (0x2C72u << 3) | 1u}, {0x2C73u, (0x2C74u << 3) | 2u}, {0x2C75u, (0x2C75u << 3) | 1u}, {0x2C76u, (0x2C7Bu << 3) | 2u},
    {0x2C7Cu, (0x2C7Du << 3) | 3u}, {0x2C7Eu, (0x2C80u << 3) | 1u}, {0x2C81u, (0x2C81u << 3) | 2u}, {0x2C82u, (0x2C82u << 3) | 1u},
    {0x2C83u, (0x2C83u << 3) | 2u}, {0x2C84u, (0x2C84u << 3) | 1u}, {0x2C85u, (0x2C85u << 3) | 2u}, {0x2C86u, (0x2C86u << 3) | 1u},
    {0x2C87u, (0x2C87u << 3) | 2u}, {0x2C88u, (0x2C88u << 3) | 1u}, {0x2C89u, (0x2C89u << 3) | 2u}, {0x2C8Au, (0x2C8Au << 3) | 1u},
    {0x2C8Bu, (0x2C8Bu << 3) | 2u}, {0x2C8Cu, (0x2C8Cu << 3) | 1u}, {0x2C8Du, (0x2C8Du << 3) | 2u}, {0x2C8Eu, (0x2C8Eu << 3) | 1u},
    {0x2C8Fu, (0x2C8Fu << 3) | 2u}, {0x2C90u, (0x2C90u << 3) | 1u}, {0x2C91u, (0x2C91u << 3) | 2u}, {0x2C92u, (0x2C92u << 3) | 1u},
    {0x2C93u, (0x2C93u << 3) | 2u}, {0x2C94u, (0x2C94u << 3) | 1u}, {0x2C95u, (0x2C95u << 3) | 2u}, {0x2C96u, (0x2C96u << 3) | 1u},
    {0x2C97u, (0x2C97u << 3) | 2u}, {0x2C98u, (0x2C98u << 3) | 1u}, {0x2C99u, (0x2C99u << 3) | 2u}, {0x2C9Au, (0x2C9Au << 3) | 1u},
    {0x2C9Bu, (0x2C9Bu << 3) | 2u}, {0x2C9Cu, (0x2C9Cu << 3) | 1u}, {0x2C9Du, (0x2C9Du << 3) | 2u}, {0x2C9Eu, (0x2C9Eu << 3) | 1u},
    {0x2C9Fu, (0x2C9Fu << 3) | 2u}, {0x2CA0u, (0x2CA0u << 3) | 1u}, {0x2CA1u, (0x2CA1u << 3) | 2u}, {0x2CA2u, (0x2CA2u << 3) | 1u},
    {0x2CA3u, (0x2CA3u << 3) | 2u}, {0x2CA4u, (0x2CA4u << 3) | 1u}, {0x2CA5u, (0x2CA5u << 3) | 2u}, {0x2CA6u, (0x2CA6u << 3) | 1u},
    {0x2CA7u, (0x2CA7u << 3) | 2u}, {0x2CA8u, (0x2CA8u << 3) | 1u}, {0x2CA9u, (0x2CA9u << 3) | 2u}, {0x2CAAu, (0x2CAAu << 3) | 1u},
    {0x2CABu, (0x2CABu << 3) | 2u}, {0x2CACu, (0x2CACu << 3) | 1u}, {0x2CADu, (0x2CADu << 3) | 2u}, {0x2CAEu, (0x2CAEu << 3) | 1u},
    {0x2CAFu, (0x2CAFu << 3) | 2u}, {0x2CB0u, (0x2CB0u << 3) | 1u}, {0x2CB1u, (0x2CB1u << 3) | 2u}, {0x2CB2u, (0x2CB2u << 3) | 1u},
    {0x2CB3u, (0x2CB3u << 3) | 2u}, {0x2CB4u, (0x2CB4u << 3) | 1u}, {0x2CB5u, (0x2CB5u << 3) | 2u}, {0x2CB6u, (0x2CB6u << 3) | 1u},
    {0x2CB7u, (0x2CB7u << 3) | 2u}, {0x2CB8u, (0x2CB8u << 3) | 1u}, {0x2CB9u, (0x2CB9u << 3) | 2u}, {0x2CBAu, (0x2CBAu << 3) | 1u},
    {0x2CBBu, (0x2CBBu << 3) | 2u}, {0x2CBCu, (0x2CBCu << 3) | 1u}, {0x2CBDu, (0x2CBDu << 3) | 2u}, {0x2CBEu, (0x2CBEu << 3) | 1u},
    {0x2CBFu, (0x2CBFu << 3) | 2u}, {0x2CC0u, (0x2CC0u << 3) | 1u}, {0x2CC1u, (0x2CC1u << 3) | 2u}, {0x2CC2u, (0x2CC2u << 3) | 1u},
    {0x2CC3u, (0x2CC3u << 3) | 2u}, {0x2CC4u, (0x2CC4u << 3) | 1u}, {0x2CC5u, (0x2CC5u << 3) | 2u}, {0x2CC6u, (0x2CC6u << 3) | 1u},
    {0x2CC7u, (0x2CC7u << 3) | 2u}, {0x2CC8u, (0x2CC8u << 3) | 1u}, {0x2CC9u, (0x2CC9u << 3) | 2u}, {0x2CCAu, (0x2CCAu << 3) | 1u},
    {0x2CCBu, (0x2CCBu << 3) | 2u}, {0x2CCCu, (0x2CCCu << 3) | 1u}, {0x2CCDu, (0x2CCDu << 3) | 2u}, {0x2CCEu, (0x2CCEu << 3) | 1u},
    {0x2CCFu, (0x2CCFu << 3) | 2u}, {0x2CD0u, (0x2CD0u << 3) | 1u}, {0x2CD1u, (0x2CD1u << 3) | 2u}, {0x2CD2u, (0x2CD2u << 3) | 1u},
    {0x2CD3u, (0x2CD3u << 3) | 2u}, {0x2CD4u, (0x2CD4u << 3) | 1u}, {0x2CD5u, (0x2CD5u << 3) | 2u}, {0x2CD6u, (0x2CD6u << 3) | 1u},
    {0x2CD7u, (0x2CD7u << 3) | 2u}, {0x2CD8u, (0x2CD8u << 3) | 1u}, {0x2CD9u, (0x2CD9u << 3) | 2u}, {0x2CDAu, (0x2CDAu << 3) | 1u},
    {0x2CDBu, (0x2CDBu << 3) | 2u}, {0x2CDCu, (0x2CDCu << 3) | 1u}, {0x2CDDu, (0x2CDDu << 3) | 2u}, {0x2CDEu, (0x2CDEu << 3) | 1u},
    {0x2CDFu, (0x2CDFu << 3) | 2u}, {0x2CE0u, (0x2CE0u << 3) | 1u}, {0x2CE1u, (0x2CE1u << 3) | 2u}, {0x2CE2u, (0x2CE2u << 3) | 1u},
    {0x2CE3u, (0x2CE4u << 3) | 2u}, {0x2CEBu, (0x2CEBu << 3) | 1u}, {0x2CECu, (0x2CECu << 3) | 2u}, {0x2CEDu, (0x2CEDu << 3) | 1u},
    {0x2CEEu, (0x2CEEu << 3) | 2u}, {0x2CEFu, (0x2CF1u << 3) | 4u}, {0x2CF2u, (0x2CF2u << 3) | 1u}, {0x2CF3u, (0x2CF3u << 3) | 2u},
    {0x2CFDu, (0x2CFDu << 3) | 5u}, {0x2D00u, (0x2D25u << 3) | 2u}, {0x2D27u, (0x2D27u << 3) | 2u}, {0x2D2Du, (0x2D2Du << 3) | 2u},
    {0x2D30u, (0x2D67u << 3) | 3u}, {0x2D6Fu, (0x2D6Fu << 3) | 3u}, {0x2D7Fu, (0x2D7Fu << 3) | 4u}, {0x2D80u, (0x2D96u << 3) | 3u},
    {0x2DA0u, (0x2DA6u << 3) | 3u}, {0x2DA8u, (0x2DAEu << 3) | 3u}, {0x2DB0u, (0x2DB6u << 3) | 3u}, {0x2DB8u, (0x2DBEu << 3) | 3u},
    {0x2DC0u, (0x2DC6u << 3) | 3u}, {0x2DC8u, (0x2DCEu << 3) | 3u}, {0x2DD0u, (0x2DD6u << 3) | 3u}, {0x2DD8u, (0x2DDEu << 3) | 3u},
    {0x2DE0u, (0x2DFFu << 3) | 4u}, {0x2E2Fu, (0x2E2Fu << 3) | 3u}, {0x3005u, (0x3006u << 3) | 3u}, {0x3007u, (0x3007u << 3) | 5u},
    {0x3021u, (0x3029u << 3) | 5u}, {0x302Au, (0x302Fu << 3) | 4u}, {0x3031u, (0x3035u << 3) | 3u}, {0x3038u, (0x303Au << 3) | 5u},
    {0x303Bu, (0x303Cu << 3) | 3u}, {0x3041u, (0x3096u << 3) | 3u}, {0x3099u, (0x309Au << 3) | 4u}, {0x309Du, (0x309Fu << 3) | 3u},
    {0x30A1u, (0x30FAu << 3) | 3u}, {0x30FCu, (0x30FFu << 3) | 3u}, {0x3105u, (0x312Fu << 3) | 3u}, {0x3131u, (0x318Eu << 3) | 3u},
    {0x3192u, (0x3195u << 3) | 5u}, {0x31A0u, (0x31BFu << 3) | 3u}, {0x31F0u, (0x31FFu << 3) | 3u}, {0x3220u, (0x3229u << 3) | 5u},
    {0x3248u, (0x324Fu << 3) | 5u}, {0x3251u, (0x325Fu << 3) | 5u}, {0x3280u, (0x3289u << 3) | 5u}, {0x32B1u, (0x32BFu << 3) | 5u},
    {0x3400u, (0x4DBFu << 3) | 3u}, {0x4E00u, (0x9FFCu << 3) | 3u}, {0xA000u, (0xA48Cu << 3) | 3u}, {0xA4D0u, (0xA4FDu << 3) | 3u},
    {0xA500u, (0xA60Cu << 3) | 3u}, {0xA610u, (0xA61Fu << 3) | 3u}, {0xA620u, (0xA629u << 3) | 5u}, {0xA62Au, (0xA62Bu << 3) | 3u},
    {0xA640u, (0xA640u << 3) | 1u}, {0xA641u, (0xA641u << 3) | 2u}, {0xA642u, (0xA642u << 3) | 1u}, {0xA643u, (0xA643u << 3) | 2u},
    {0xA644u, (0xA644u << 3) | 1u}, {0xA645u, (0xA645u << 3) | 2u}, {0xA646u, (0xA646u << 3) | 1u}, {0xA647u, (0xA647u << 3) | 2u},
    {0xA648u, (0xA648u << 3) | 1u}, {0xA649u, (0xA649u << 3) | 2u}, {0xA64Au, (0xA64Au << 3) | 1u}, {0xA64Bu, (0xA64Bu << 3) | 2u},
    {0xA64Cu, (0xA64Cu << 3) | 1u}, {0xA64Du, (0xA64Du << 3) | 2u}, {0xA64Eu, (0xA64Eu << 3) | 1u}, {0xA64Fu, (0xA64Fu << 3) | 2u},
    {0xA650u, (0xA650u << 3) | 1u}, {0xA651u, (0xA651u << 3) | 2u}, {0xA652u, (0xA652u << 3) | 1u}, {0xA653u, (0xA653u << 3) | 2u},
    {0xA654u, (0xA654u << 3) | 1u}, {0xA655u, (0xA655u << 3) | 2u}, {0xA656u, (0xA656u << 3) | 1u}, {0xA657u, (0xA657u << 3) | 2u},
    {0xA658u, (0xA658u << 3) | 1u}, {0xA659u, (0xA659u << 3) | 2u}, {0xA65Au, (0xA65Au << 3) | 1u}, {0xA65Bu, (0xA65Bu << 3) | 2u},
    {0xA65Cu, (0xA65Cu << 3) | 1u}, {0xA65Du, (0xA65Du << 3) | 2u}, {0xA65Eu, (0xA65Eu << 3) | 1u}, {0xA65Fu, (0xA65Fu << 3) | 2u},
    {0xA660u, (0xA660u << 3) | 1u}, {0xA661u, (0xA661u << 3) | 2u}, {0xA662u, (0xA662u << 3) | 1u}, {0xA663u, (0xA663u << 3) | 2u},
    {0xA664u, (0xA664u << 3) | 1u}, {0xA665u, (0xA665u << 3) | 2u}, {0xA666u, (0xA666u << 3) | 1u}, {0xA667u, (0xA667u << 3) | 2u},
    {0xA668u, (0xA668u << 3) | 1u}, {0xA669u, (0xA669u << 3) | 2u}, {0xA66Au, (0xA66Au << 3) | 1u}, {0xA66Bu, (0xA66Bu << 3) | 2u},
    {0xA66Cu, (0xA66Cu << 3) | 1u}, {0xA66Du, (0xA66Du << 3) | 2u}, {0xA66Eu, (0xA66Eu << 3) | 3u}, {0xA66Fu, (0xA672u << 3) | 4u},
    {0xA674u, (0xA67Du << 3) | 4u}, {0xA67Fu, (0xA67Fu << 3) | 3u}, {0xA680u, (0xA680u << 3) | 1u}, {0xA681u, (0xA681u << 3) | 2u},
    {0xA682u, (0xA682u << 3) | 1u}, {0xA683u, (0xA683u << 3) | 2u}, {0xA684u, (0xA684u << 3) | 1u}, {0xA685u, (0xA685u << 3) | 2u},
    {0xA686u, (0xA686u << 3) | 1u}, {0xA687u, (0xA687u << 3) | 2u}, {0xA688u, (0xA688u << 3) | 1u}, {0xA689u, (0xA689u << 3) | 2u},
    {0xA68Au, (0xA68Au << 3) | 1u}, {0xA68Bu, (0xA68Bu << 3) | 2u}, {0xA68Cu, (0xA68Cu << 3) | 1u}, {0xA68Du, (0xA68Du << 3) | 2u},
    {0xA68Eu, (0xA68Eu << 3) | 1u}, {0xA68Fu, (0xA68Fu << 3) | 2u}, {0xA690u, (0xA690u << 3) | 1u}, {0xA691u, (0xA691u << 3) | 2u},
    {0xA692u, (0xA692u << 3) | 1u}, {0xA693u, (0xA693u << 3) | 2u}, {0xA694u, (0xA694u << 3) | 1u}, {0xA695u, (0xA695u << 3) | 2u},
    {0xA696u, (0xA696u << 3) | 1u}, {0xA697u, (0xA697u << 3) | 2u}, {0xA698u, (0xA698u << 3) | 1u}, {0xA699u, (0xA699u << 3) | 2u},
    {0xA69Au, (0xA69Au << 3) | 1u}, {0xA69Bu, (0xA69Bu << 3) | 2u}, {0xA69Cu, (0xA69Du << 3) | 3u}, {0xA69Eu, (0xA69Fu << 3) | 4u},
    {0xA6A0u, (0xA6E5u << 3) | 3u}, {0xA6E6u, (0xA6EFu << 3) | 5u}, {0xA6F0u, (0xA6F1u << 3) | 4u}, {0xA717u, (0xA71Fu << 3) | 3u},
    {0xA722u, (0xA722u << 3) | 1u}, {0xA723u, (0xA723u << 3) | 2u}, {0xA724u, (0xA724u << 3) | 1u}, {0xA725u, (0xA725u << 3) | 2u},
    {0xA726u, (0xA726u << 3) | 1u}, {0xA727u, (0xA727u << 3) | 2u}, {0xA728u, (0xA728u << 3) | 1u}, {0xA729u, (0xA729u << 3) | 2u},
    {0xA72Au, (0xA72Au << 3) | 1u}, {0xA72Bu, (0xA72Bu << 3) | 2u}, {0xA72Cu, (0xA72Cu << 3) | 1u}, {0xA72Du, (0xA72Du << 3) | 2u},
    {0xA72Eu, (0xA72Eu << 3) | 1u}, {0xA72Fu, (0xA731u << 3) | 2u}, {0xA732u, (0xA732u << 3) | 1u}, {0xA733u, (0xA733u << 3) | 2u},
    {0xA734u, (0xA734u << 3) | 1u}, {0xA735u, (0xA735u << 3) | 2u}, {0xA736u, (0xA736u << 3) | 1u}, {0xA737u, (0xA737u << 3) | 2u},
    {0xA738u, (0xA738u << 3) | 1u}, {0xA739u, (0xA739u << 3) | 2u}, {0xA73Au, (0xA73Au << 3) | 1u}, {0xA73Bu, (0xA73Bu << 3) | 2u},
    {0xA73Cu, (0xA73Cu << 3) | 1u}, {0xA73Du, (0xA73Du << 3) | 2u}, {0xA73Eu, (0xA73Eu << 3) | 1u}, {0xA73Fu, (0xA73Fu << 3) | 2u},
    {0xA740u, (0xA740u << 3) | 1u}, {0xA741u, (0xA741u << 3) | 2u}, {0xA742u, (0xA742u << 3) | 1u}, {0xA743u, (0xA743u << 3) | 2u},
    {0xA744u, (0xA744u << 3) | 1u}, {0xA745u, (0xA745u << 3) | 2u}, {0xA746u, (0xA746u << 3) | 1u}, {0xA747u, (0xA747u << 3) | 2u},
    {0xA748u, (0xA748u << 3) | 1u}, {0xA749u, (0xA749u << 3) | 2u}, {0xA74Au, (0xA74Au << 3) | 1u}, {0xA74Bu, (0xA74Bu << 3) | 2u},
    {0xA74Cu, (0xA74Cu << 3) | 1u}, {0xA74Du, (0xA74Du << 3) | 2u}, {0xA74Eu, (0xA74Eu << 3) | 1u}, {0xA74Fu, (0xA74Fu << 3) | 2u},
    {0xA750u, (0xA750u << 3) | 1u}, {0xA751u, (0xA751u << 3) | 2u}, {0xA752u, (0xA752u << 3) | 1u}, {0xA753u, (0xA753u << 3) | 2u},
    {0xA754u, (0xA754u << 3) | 1u}, {0xA755u, (0xA755u << 3) | 2u}, {0xA756u, (0xA756u << 3) | 1u}, {0xA757u, (0xA757u << 3) | 2u},
    {0xA758u, (0xA758u << 3) | 1u}, {0xA759u, (0xA759u << 3) | 2u}, {0xA75Au, (0xA75Au << 3) | 1u}, {0xA75Bu, (0xA75Bu << 3) | 2u},
    {0xA75Cu, (0xA75Cu << 3) | 1u}, {0xA75Du, (0xA75Du << 3) | 2u}, {0xA75Eu, (0xA75Eu << 3) | 1u}, {0xA75Fu, (0xA75Fu << 3) | 2u},
    {0xA760u, (0xA760u << 3) | 1u}, {0xA761u, (0xA761u << 3) | 2u}, {0xA762u, (0xA762u << 3) | 1u}, {0xA763u, (0xA763u << 3) | 2u},
    {0xA764u, (0xA764u << 3) | 1u}, {0xA765u, (0xA765u << 3) | 2u}, {0xA766u, (0xA766u << 3) | 1u}, {0xA767u, (0xA767u << 3) | 2u},
    {0xA768u, (0xA768u << 3) | 1u}, {0xA769u, (0xA769u << 3) | 2u}, {0xA76Au, (0xA76Au << 3) | 1u}, {0xA76Bu, (0xA76Bu << 3) | 2u},
    {0xA76Cu, (0xA76Cu << 3) | 1u}, {0xA76Du, (0xA76Du << 3) | 2u}, {0xA76Eu, (0xA76Eu << 3) | 1u}, {0xA76Fu, (0xA76Fu << 3) | 2u},
    {0xA770u, (0xA770u << 3) | 3u}, {0xA771u, (0xA778u << 3) | 2u}, {0xA779u, (0xA779u << 3) | 1u}, {0xA77Au, (0xA77Au << 3) | 2u},
    {0xA77Bu, (0xA77Bu << 3) | 1u}, {0xA77Cu, (0xA77Cu << 3) | 2u}, {0xA77Du, (0xA77Eu << 3) | 1u}, {0xA77Fu, (0xA77Fu << 3) | 2u},
    {0xA780u, (0xA780u << 3) | 1u}, {0xA781u, (0xA781u << 3) | 2u}, {0xA782u, (0xA782u << 3) | 1u}, {0xA783u, (0xA783u << 3) | 2u},
    {0xA784u, (0xA784u << 3) | 1u}, {0xA785u, (0xA785u << 3) | 2u}, {0xA786u, (0xA786u << 3) | 1u}, {0xA787u, (0xA787u << 3) | 2u},
    {0xA788u, (0xA788u << 3) | 3u}, {0xA78Bu, (0xA78Bu << 3) | 1u}, {0xA78Cu, (0xA78Cu << 3) | 2u}, {0xA78Du, (0xA78Du << 3) | 1u},
    {0xA78Eu, (0xA78Eu << 3) | 2u}, {0xA78Fu, (0xA78Fu << 3) | 3u}, {0xA790u, (0xA790u << 3) | 1u}, {0xA791u, (0xA791u << 3) | 2u},
    {0xA792u, (0xA792u << 3) | 1u}, {0xA793u, (0xA795u << 3) | 2u}, {0xA796u, (0xA796u << 3) | 1u}, {0xA797u, (0xA797u << 3) | 2u},
    {0xA798u, (0xA798u << 3) | 1u}, {0xA799u, (0xA799u << 3) | 2u}, {0xA79Au, (0xA79Au << 3) | 1u}, {0xA79Bu, (0xA79Bu << 3) | 2u},
    {0xA79Cu, (0xA79Cu << 3) | 1u}, {0xA79Du, (0xA79Du << 3) | 2u}, {0xA79Eu, (0xA79Eu << 3) | 1u}, {0xA79Fu, (0xA79Fu << 3) | 2u},
    {0xA7A0u, (0xA7A0u << 3) | 1u}, {0xA7A1u, (0xA7A1u << 3) | 2u}, {0xA7A2u, (0xA7A2u << 3) | 1u}, {0xA7A3u, (0xA7A3u << 3) | 2u},
    {0xA7A4u, (0xA7A4u << 3) | 1u}, {0xA7A5u, (0xA7A5u << 3) | 2u}, {0xA7A6u, (0xA7A6u << 3) | 1u}, {0xA7A7u, (0xA7A7u << 3) | 2u},
    {0xA7A8u, (0xA7A8u << 3) | 1u}, {0xA7A9u, (0xA7A9u << 3) | 2u}, {0xA7AAu, (0xA7AEu << 3) | 1u}, {0xA7AFu, (0xA7AFu << 3) | 2u},
    {0xA7B0u, (0xA7B4u << 3) | 1u}, {0xA7B5u, (0xA7B5u << 3) | 2u}, {0xA7B6u, (0xA7B6u << 3) | 1u}, {0xA7B7u, (0xA7B7u << 3) | 2u},
    {0xA7B8u, (0xA7B8u << 3) | 1u}, {0xA7B9u, (0xA7B9u << 3) | 2u}, {0xA7BAu, (0xA7BAu << 3) | 1u}, {0xA7BBu, (0xA7BBu << 3) | 2u},
    {0xA7BCu, (0xA7BCu << 3) | 1u}, {0xA7BDu, (0xA7BDu << 3) | 2u}, {0xA7BEu, (0xA7BEu << 3) | 1u}, {0xA7BFu, (0xA7BFu << 3) | 2u},
    {0xA7C2u, (0xA7C2u << 3) | 1u}, {0xA7C3u, (0xA7C3u << 3) | 2u}, {0xA7C4u, (0xA7C7u << 3) | 1u}, {0xA7C8u, (0xA7C8u << 3) | 2u},
    {0xA7C9u, (0xA7C9u << 3) | 1u}, {0xA7CAu, (0xA7CAu << 3) | 2u}, {0xA7F5u, (0xA7F5u << 3) | 1u}, {0xA7F6u, (0xA7F6u << 3) | 2u},
    {0xA7F7u, (0xA7F9u << 3) | 3u}, {0xA7FAu, (0xA7FAu << 3) | 2u}, {0xA7FBu, (0xA801u << 3) | 3u}, {0xA802u, (0xA802u << 3) | 4u},
    {0xA803u, (0xA805u << 3) | 3u}, {0xA806u, (0xA806u << 3) | 4u}, {0xA807u, (0xA80Au << 3) | 3u}, {0xA80Bu, (0xA80Bu << 3) | 4u},
    {0xA80Cu, (0xA822u << 3) | 3u}, {0xA823u, (0xA827u << 3) | 4u}, {0xA82Cu, (0xA82Cu << 3) | 4u}, {0xA830u, (0xA835u << 3) | 5u},
    {0xA840u, (0xA873u << 3) | 3u}, {0xA880u, (0xA881u << 3) | 4u}, {0xA882u, (0xA8B3u << 3) | 3u}, {0xA8B4u, (0xA8C5u << 3) | 4u},
    {0xA8D0u, (0xA8D9u << 3) | 5u}, {0xA8E0u, (0xA8F1u << 3) | 4u}, {0xA8F2u, (0xA8F7u << 3) | 3u}, {0xA8FBu, (0xA8FBu << 3) | 3u},
    {0xA8FDu, (0xA8FEu << 3) | 3u}, {0xA8FFu, (0xA8FFu << 3) | 4u}, {0xA900u, (0xA909u << 3) | 5u}, {0xA90Au, (0xA925u << 3) | 3u},
    {0xA926u, (0xA92Du << 3) | 4u}, {0xA930u, (0xA946u << 3) | 3u}, {0xA947u, (0xA953u << 3) | 4u}, {0xA960u, (0xA97Cu << 3) | 3u},
    {0xA980u, (0xA983u << 3) | 4u}, {0xA984u, (0xA9B2u << 3) | 3u}, {0xA9B3u, (0xA9C0u << 3) | 4u}, {0xA9CFu, (0xA9CFu << 3) | 3u},
    {0xA9D0u, (0xA9D9u << 3) | 5u}, {0xA9E0u, (0xA9E4u << 3) | 3u}, {0xA9E5u, (0xA9E5u << 3) | 4u}, {0xA9E6u, (0xA9EFu << 3) | 3u},
    {0xA9F0u, (0xA9F9u << 3) | 5u}, {0xA9FAu, (0xA9FEu << 3) | 3u}, {0xAA00u, (0xAA28u << 3) | 3u}, {0xAA29u, (0xAA36u << 3) | 4u},
    {0xAA40u, (0xAA42u << 3) | 3u}, {0xAA43u, (0xAA43u << 3) | 4u}, {0xAA44u, (0xAA4Bu << 3) | 3u}, {0xAA4Cu, (0xAA4Du << 3) | 4u},
    {0xAA50u, (0xAA59u << 3) | 5u}, {0xAA60u, (0xAA76u << 3) | 3u}, {0xAA7Au, (0xAA7Au << 3) | 3u}, {0xAA7Bu, (0xAA7Du << 3) | 4u},
    {0xAA7Eu, (0xAAAFu << 3) | 3u}, {0xAAB0u, (0xAAB0u << 3) | 4u}, {0xAAB1u, (0xAAB1u << 3) | 3u}, {0xAAB2u, (0xAAB4u << 3) | 4u},
    {0xAAB5u, (0xAAB6u << 3) | 3u}, {0xAAB7u, (0xAAB8u << 3) | 4u}, {0xAAB9u, (0xAABDu << 3) | 3u}, {0xAABEu, (0xAABFu << 3) | 4u},
    {0xAAC0u, (0xAAC0u << 3) | 3u}, {0xAAC1u, (0xAAC1u << 3) | 4u}, {0xAAC2u, (0xAAC2u << 3) | 3u}, {0xAADBu, (0xAADDu << 3) | 3u},
    {0xAAE0u, (0xAAEAu << 3) | 3u}, {0xAAEBu, (0xAAEFu << 3) | 4u}, {0xAAF2u, (0xAAF4u << 3) | 3u}, {0xAAF5u, (0xAAF6u << 3) | 4u},
    {0xAB01u, (0xAB06u << 3) | 3u}, {0xAB09u, (0xAB0Eu << 3) | 3u}, {0xAB11u, (0xAB16u << 3) | 3u}, {0xAB20u, (0xAB26u << 3) | 3u},
    {0xAB28u, (0xAB2Eu << 3) | 3u}, {0xAB30u, (0xAB5Au << 3) | 2u}, {0xAB5Cu, (0xAB5Fu << 3) | 3u}, {0xAB60u, (0xAB68u << 3) | 2u},
    {0xAB69u, (0xAB69u << 3) | 3u}, {0xAB70u, (0xABBFu << 3) | 2u}, {0xABC0u, (0xABE2u << 3) | 3u}, {0xABE3u, (0xABEAu << 3) | 4u},
    {0xABECu, (0xABEDu << 3) | 4u}, {0xABF0u, (0xABF9u << 3) | 5u}, {0xAC00u, (0xD7A3u << 3) | 3u}, {0xD7B0u, (0xD7C6u << 3) | 3u},
    {0xD7CBu, (0xD7FBu << 3) | 3u}, {0xF900u, (0xFA6Du << 3) | 3u}, {0xFA70u, (0xFAD9u << 3) | 3u}, {0xFB00u, (0xFB06u << 3) | 2u},
    {0xFB13u, (0xFB17u << 3) | 2u}, {0xFB1Du, (0xFB1Du << 3) | 3u}, {0xFB1Eu, (0xFB1Eu << 3) | 4u}, {0xFB1Fu, (0xFB28u << 3) | 3u},
    {0xFB2Au, (0xFB36u << 3) | 3u}, {0xFB38u, (0xFB3Cu << 3) | 3u}, {0xFB3Eu, (0xFB3Eu << 3) | 3u}, {0xFB40u, (0xFB41u << 3) | 3u},
    {0xFB43u, (0xFB44u << 3) | 3u}, {0xFB46u, (0xFBB1u << 3) | 3u}, {0xFBD3u, (0xFD3Du << 3) | 3u}, {0xFD50u, (0xFD8Fu << 3) | 3u},
    {0xFD92u, (0xFDC7u << 3) | 3u}, {0xFDF0u, (0xFDFBu << 3) | 3u}, {0xFE00u, (0xFE0Fu << 3) | 4u}, {0xFE20u, (0xFE2Fu << 3) | 4u},
    {0xFE70u, (0xFE74u << 3) | 3u}, {0xFE76u, (0xFEFCu << 3) | 3u}, {0xFF10u, (0xFF19u << 3) | 5u}, {0xFF21u, (0xFF3Au << 3) | 1u},
    {0xFF41u, (0xFF5Au << 3) | 2u}, {0xFF66u, (0xFFBEu << 3) | 3u}, {0xFFC2u, (0xFFC7u << 3) | 3u}, {0xFFCAu, (0xFFCFu << 3) | 3u},
    {0xFFD2u, (0xFFD7u << 3) | 3u}, {0xFFDAu, (0xFFDCu << 3) | 3u}, {0x10000u, (0x1000Bu << 3) | 3u}, {0x1000Du, (0x10026u << 3) | 3u},
    {0x10028u, (0x1003Au << 3) | 3u}, {0x1003Cu, (0x1003Du << 3) | 3u}, {0x1003Fu, (0x1004Du << 3) | 3u}, {0x10050u, (0x1005Du << 3) | 3u},
    {0x10080u, (0x100FAu << 3) | 3u}, {0x10107u, (0x10133u << 3) | 5u}, {0x10140u, (0x10178u << 3) | 5u}, {0x1018Au, (0x1018Bu << 3) | 5u},
    {0x101FDu, (0x101FDu << 3) | 4u}, {0x10280u, (0x1029Cu << 3) | 3u}, {0x102A0u, (0x102D0u << 3) | 3u}, {0x102E0u, (0x102E0u << 3) | 4u},
    {0x102E1u, (0x102FBu << 3) | 5u}, {0x10300u, (0x1031Fu << 3) | 3u}, {0x10320u, (0x10323u << 3) | 5u}, {0x1032Du, (0x10340u << 3) | 3u},
    {0x10341u, (0x10341u << 3) | 5u}, {0x10342u, (0x10349u << 3) | 3u}, {0x1034Au, (0x1034Au << 3) | 5u}, {0x10350u, (0x10375u << 3) | 3u},
    {0x10376u, (0x1037Au << 3) | 4u}, {0x10380u, (0x1039Du << 3) | 3u}, {0x103A0u, (0x103C3u << 3) | 3u}, {0x103C8u, (0x103CFu << 3) | 3u},
    {0x103D1u, (0x103D5u << 3) | 5u}, {0x10400u, (0x10427u << 3) | 1u}, {0x10428u, (0x1044Fu << 3) | 2u}, {0x10450u, (0x1049Du << 3) | 3u},
    {0x104A0u, (0x104A9u << 3) | 5u}, {0x104B0u, (0x104D3u << 3) | 1u}, {0x104D8u, (0x104FBu << 3) | 2u}, {0x10500u, (0x10527u << 3) | 3u},
    {0x10530u, (0x10563u << 3) | 3u}, {0x10600u, (0x10736u << 3) | 3u}, {0x10740u, (0x10755u << 3) | 3u}, {0x10760u, (0x10767u << 3) | 3u},
    {0x10800u, (0x10805u << 3) | 3u}, {0x10808u, (0x10808u << 3) | 3u}, {0x1080Au, (0x10835u << 3) | 3u}, {0x10837u, (0x10838u << 3) | 3u},
    {0x1083Cu, (0x1083Cu << 3) | 3u}, {0x1083Fu, (0x10855u << 3) | 3u}, {0x10858u, (0x1085Fu << 3) | 5u}, {0x10860u, (0x10876u << 3) | 3u},
    {0x10879u, (0x1087Fu << 3) | 5u}, {0x10880u, (0x1089Eu << 3) | 3u}, {0x108A7u, (0x108AFu << 3) | 5u}, {0x108E0u, (0x108F2u << 3) | 3u},
    {0x108F4u, (0x108F5u << 3) | 3u}, {0x108FBu, (0x108FFu << 3) | 5u}, {0x10900u, (0x10915u << 3) | 3u}, {0x10916u, (0x1091Bu << 3) | 5u},
    {0x10920u, (0x10939u << 3) | 3u}, {0x10980u, (0x109B7u << 3) | 3u}, {0x109BCu, (0x109BDu << 3) | 5u}, {0x109BEu, (0x109BFu << 3) | 3u},
    {0x109C0u, (0x109CFu << 3) | 5u}, {0x109D2u, (0x109FFu << 3) | 5u}, {0x10A00u, (0x10A00u << 3) | 3u}, {0x10A01u, (0x10A03u << 3) | 4u},
    {0x10A05u, (0x10A06u << 3) | 4u}, {0x10A0Cu, (0x10A0Fu << 3) | 4u}, {0x10A10u, (0x10A13u << 3) | 3u}, {0x10A15u, (0x10A17u << 3) | 3u},
    {0x10A19u, (0x10A35u << 3) | 3u}, {0x10A38u, (0x10A3Au << 3) | 4u}, {0x10A3Fu, (0x10A3Fu << 3) | 4u}, {0x10A40u, (0x10A48u << 3) | 5u},
    {0x10A60u, (0x10A7Cu << 3) | 3u}, {0x10A7Du, (0x10A7Eu << 3) | 5u}, {0x10A80u, (0x10A9Cu << 3) | 3u}, {0x10A9Du, (0x10A9Fu << 3) | 5u},
    {0x10AC0u, (0x10AC7u << 3) | 3u}, {0x10AC9u, (0x10AE4u << 3) | 3u}, {0x10AE5u, (0x10AE6u << 3) | 4u}, {0x10AEBu, (0x10AEFu << 3) | 5u},
    {0x10B00u, (0x10B35u << 3) | 3u}, {0x10B40u, (0x10B55u << 3) | 3u}, {0x10B58u, (0x10B5Fu << 3) | 5u}, {0x10B60u, (0x10B72u << 3) | 3u},
    {0x10B78u, (0x10B7Fu << 3) | 5u}, {0x10B80u, (0x10B91u << 3) | 3u}, {0x10BA9u, (0x10BAFu << 3) | 5u}, {0x10C00u, (0x10C48u << 3) | 3u},
    {0x10C80u, (0x10CB2u << 3) | 1u}, {0x10CC0u, (0x10CF2u << 3) | 2u}, {0x10CFAu, (0x10CFFu << 3) | 5u}, {0x10D00u, (0x10D23u << 3) | 3u},
    {0x10D24u, (0x10D27u << 3) | 4u}, {0x10D30u, (0x10D39u << 3) | 5u}, {0x10E60u, (0x10E7Eu << 3) | 5u}, {0x10E80u, (0x10EA9u << 3) | 3u},
    {0x10EABu, (0x10EACu << 3) | 4u}, {0x10EB0u, (0x10EB1u << 3) | 3u}, {0x10F00u, (0x10F1Cu << 3) | 3u}, {0x10F1Du, (0x10F26u << 3) | 5u},
    {0x10F27u, (0x10F27u << 3) | 3u}, {0x10F30u, (0x10F45u << 3) | 3u}, {0x10F46u, (0x10F50u << 3) | 4u}, {0x10F51u, (0x10F54u << 3) | 5u},
    {0x10FB0u, (0x10FC4u << 3) | 3u}, {0x10FC5u, (0x10FCBu << 3) | 5u}, {0x10FE0u, (0x10FF6u << 3) | 3u}, {0x11000u, (0x11002u << 3) | 4u},
    {0x11003u, (0x11037u << 3) | 3u}, {0x11038u, (0x11046u << 3) | 4u}, {0x11052u, (0x1106Fu << 3) | 5u}, {0x1107Fu, (0x11082u << 3) | 4u},
    {0x11083u, (0x110AFu << 3) | 3u}, {0x110B0u, (0x110BAu << 3) | 4u}, {0x110D0u, (0x110E8u << 3) | 3u}, {0x110F0u, (0x110F9u << 3) | 5u},
    {0x11100u, (0x11102u << 3) | 4u}, {0x11103u, (0x11126u << 3) | 3u}, {0x11127u, (0x11134u << 3) | 4u}, {0x11136u, (0x1113Fu << 3) | 5u},
    {0x11144u, (0x11144u << 3) | 3u}, {0x11145u, (0x11146u << 3) | 4u}, {0x11147u, (0x11147u << 3) | 3u}, {0x11150u, (0x11172u << 3) | 3u},
    {0x11173u, (0x11173u << 3) | 4u}, {0x11176u, (0x11176u << 3) | 3u}, {0x11180u, (0x11182u << 3) | 4u}, {0x11183u, (0x111B2u << 3) | 3u},
    {0x111B3u, (0x111C0u << 3) | 4u}, {0x111C1u, (0x111C4u << 3) | 3u}, {0x111C9u, (0x111CCu << 3) | 4u}, {0x111CEu, (0x111CFu << 3) | 4u},
    {0x111D0u, (0x111D9u << 3) | 5u}, {0x111DAu, (0x111DAu << 3) | 3u}, {0x111DCu, (0x111DCu << 3) | 3u}, {0x111E1u, (0x111F4u << 3) | 5u},
    {0x11200u, (0x11211u << 3) | 3u}, {0x11213u, (0x1122Bu << 3) | 3u}, {0x1122Cu, (0x11237u << 3) | 4u}, {0x1123Eu, (0x1123Eu << 3) | 4u},
    {0x11280u, (0x11286u << 3) | 3u}, {0x11288u, (0x11288u << 3) | 3u}, {0x1128Au, (0x1128Du << 3) | 3u}, {0x1128Fu, (0x1129Du << 3) | 3u},
    {0x1129Fu, (0x112A8u << 3) | 3u}, {0x112B0u, (0x112DEu << 3) | 3u}, {0x112DFu, (0x112EAu << 3) | 4u}, {0x112F0u, (0x112F9u << 3) | 5u},
    {0x11300u, (0x11303u << 3) | 4u}, {0x11305u, (0x1130Cu << 3) | 3u}, {0x1130Fu, (0x11310u << 3) | 3u}, {0x11313u, (0x11328u << 3) | 3u},
    {0x1132Au, (0x11330u << 3) | 3u}, {0x11332u, (0x11333u << 3) | 3u}, {0x11335u, (0x11339u << 3) | 3u}, {0x1133Bu, (0x1133Cu << 3) | 4u},
    {0x1133Du, (0x1133Du << 3) | 3u}, {0x1133Eu, (0x11344u << 3) | 4u}, {0x11347u, (0x11348u << 3) | 4u}, {0x1134Bu, (0x1134Du << 3) | 4u},
    {0x11350u, (0x11350u << 3) | 3u}, {0x11357u, (0x11357u << 3) | 4u}, {0x1135Du, (0x11361u << 3) | 3u}, {0x11362u, (0x11363u << 3) | 4u},
    {0x11366u, (0x1136Cu << 3) | 4u}, {0x11370u, (0x11374u << 3) | 4u}, {0x11400u, (0x11434u << 3) | 3u}, {0x11435u, (0x11446u << 3) | 4u},
    {0x11447u, (0x1144Au << 3) | 3u}, {0x11450u, (0x11459u << 3) | 5u}, {0x1145Eu, (0x1145Eu << 3) | 4u}, {0x1145Fu, (0x11461u << 3) | 3u},
    {0x11480u, (0x114AFu << 3) | 3u}, {0x114B0u, (0x114C3u << 3) | 4u}, {0x114C4u, (0x114C5u << 3) | 3u}, {0x114C7u, (0x114C7u << 3) | 3u},
    {0x114D0u, (0x114D9u << 3) | 5u}, {0x11580u, (0x115AEu << 3) | 3u}, {0x115AFu, (0x115B5u << 3) | 4u}, {0x115B8u, (0x115C0u << 3) | 4u},
    {0x115D8u, (0x115DBu << 3) | 3u}, {0x115DCu, (0x115DDu << 3) | 4u}, {0x11600u, (0x1162Fu << 3) | 3u}, {0x11630u, (0x11640u << 3) | 4u},
    {0x11644u, (0x11644u << 3) | 3u}, {0x11650u, (0x11659u << 3) | 5u}, {0x11680u, (0x116AAu << 3) | 3u}, {0x116ABu, (0x116B7u << 3) | 4u},
    {0x116B8u, (0x116B8u << 3) | 3u}, {0x116C0u, (0x116C9u << 3) | 5u}, {0x11700u, (0x1171Au << 3) | 3u}, {0x1171Du, (0x1172Bu << 3) | 4u},
    {0x11730u, (0x1173Bu << 3) | 5u}, {0x11800u, (0x1182Bu << 3) | 3u}, {0x1182Cu, (0x1183Au << 3) | 4u}, {0x118A0u, (0x118BFu << 3) | 1u},
    {0x118C0u, (0x118DFu << 3) | 2u}, {0x118E0u, (0x118F2u << 3) | 5u}, {0x118FFu, (0x11906u << 3) | 3u}, {0x11909u, (0x11909u << 3) | 3u},
    {0x1190Cu, (0x11913u << 3) | 3u}, {0x11915u, (0x11916u << 3) | 3u}, {0x11918u, (0x1192Fu << 3) | 3u}, {0x11930u, (0x11935u << 3) | 4u},
    {0x11937u, (0x11938u << 3) | 4u}, {0x1193Bu, (0x1193Eu << 3) | 4u}, {0x1193Fu, (0x1193Fu << 3) | 3u}, {0x11940u, (0x11940u << 3) | 4u},
    {0x11941u, (0x11941u << 3) | 3u}, {0x11942u, (0x11943u << 3) | 4u}, {0x11950u, (0x11959u << 3) | 5u}, {0x119A0u, (0x119A7u << 3) | 3u},
    {0x119AAu, (0x119D0u << 3) | 3u}, {0x119D1u, (0x119D7u << 3) | 4u}, {0x119DAu, (0x119E0u << 3) | 4u}, {0x119E1u, (0x119E1u << 3) | 3u},
    {0x119E3u, (0x119E3u << 3) | 3u}, {0x119E4u, (0x119E4u << 3) | 4u}, {0x11A00u, (0x11A00u << 3) | 3u}, {0x11A01u, (0x11A0Au << 3) | 4u},
    {0x11A0Bu, (0x11A32u << 3) | 3u}, {0x11A33u, (0x11A39u << 3) | 4u}, {0x11A3Au, (0x11A3Au << 3) | 3u}, {0x11A3Bu, (0x11A3Eu << 3) | 4u},
    {0x11A47u, (0x11A47u << 3) | 4u}, {0x11A50u, (0x11A50u << 3) | 3u}, {0x11A51u, (0x11A5Bu << 3) | 4u}, {0x11A5Cu, (0x11A89u << 3) | 3u},
    {0x11A8Au, (0x11A99u << 3) | 4u}, {0x11A9Du, (0x11A9Du << 3) | 3u}, {0x11AC0u, (0x11AF8u << 3) | 3u}, {0x11C00u, (0x11C08u << 3) | 3u},
    {0x11C0Au, (0x11C2Eu << 3) | 3u}, {0x11C2Fu, (0x11C36u << 3) | 4u}, {0x11C38u, (0x11C3Fu << 3) | 4u}, {0x11C40u, (0x11C40u << 3) | 3u},
    {0x11C50u, (0x11C6Cu << 3) | 5u}, {0x11C72u, (0x11C8Fu << 3) | 3u}, {0x11C92u, (0x11CA7u << 3) | 4u}, {0x11CA9u, (0x11CB6u << 3) | 4u},
    {0x11D00u, (0x11D06u << 3) | 3u}, {0x11D08u, (0x11D09u << 3) | 3u}, {0x11D0Bu, (0x11D30u << 3) | 3u}, {0x11D31u, (0x11D36u << 3) | 4u},
    {0x11D3Au, (0x11D3Au << 3) | 4u}, {0x11D3Cu, (0x11D3Du << 3) | 4u}, {0x11D3Fu, (0x11D45u << 3) | 4u}, {0x11D46u, (0x11D46u << 3) | 3u},
    {0x11D47u, (0x11D47u << 3) | 4u}, {0x11D50u, (0x11D59u << 3) | 5u}, {0x11D60u, (0x11D65u << 3) | 3u}, {0x11D67u, (0x11D68u << 3) | 3u},
    {0x11D6Au, (0x11D89u << 3) | 3u}, {0x11D8Au, (0x11D8Eu << 3) | 4u}, {0x11D90u, (0x11D91u << 3) | 4u}, {0x11D93u, (0x11D97u << 3) | 4u},
    {0x11D98u, (0x11D98u << 3) | 3u}, {0x11DA0u, (0x11DA9u << 3) | 5u}, {0x11EE0u, (0x11EF2u << 3) | 3u}, {0x11EF3u, (0x11EF6u << 3) | 4u},
    {0x11FB0u, (0x11FB0u << 3) | 3u}, {0x11FC0u, (0x11FD4u << 3) | 5u}, {0x12000u, (0x12399u << 3) | 3u}, {0x12400u, (0x1246Eu << 3) | 5u},
    {0x12480u, (0x12543u << 3) | 3u}, {0x13000u, (0x1342Eu << 3) | 3u}, {0x14400u, (0x14646u << 3) | 3u}, {0x16800u, (0x16A38u << 3) | 3u},
    {0x16A40u, (0x16A5Eu << 3) | 3u}, {0x16A60u, (0x16A69u << 3) | 5u}, {0x16AD0u, (0x16AEDu << 3) | 3u}, {0x16AF0u, (0x16AF4u << 3) | 4u},
    {0x16B00u, (0x16B2Fu << 3) | 3u}, {0x16B30u, (0x16B36u << 3) | 4u}, {0x16B40u, (0x16B43u << 3) | 3u}, {0x16B50u, (0x16B59u << 3) | 5u},
    {0x16B5Bu, (0x16B61u << 3) | 5u}, {0x16B63u, (0x16B77u << 3) | 3u}, {0x16B7Du, (0x16B8Fu << 3) | 3u}, {0x16E40u, (0x16E5Fu << 3) | 1u},
    {0x16E60u, (0x16E7Fu << 3) | 2u}, {0x16E80u, (0x16E96u << 3) | 5u}, {0x16F00u, (0x16F4Au << 3) | 3u}, {0x16F4Fu, (0x16F4Fu << 3) | 4u},
    {0x16F50u, (0x16F50u << 3) | 3u}, {0x16F51u, (0x16F87u << 3) | 4u}, {0x16F8Fu, (0x16F92u << 3) | 4u}, {0x16F93u, (0x16F9Fu << 3) | 3u},
    {0x16FE0u, (0x16FE1u << 3) | 3u}, {0x16FE3u, (0x16FE3u << 3) | 3u}, {0x16FE4u, (0x16FE4u << 3) | 4u}, {0x16FF0u, (0x16FF1u << 3) | 4u},
    {0x17000u, (0x187F7u << 3) | 3u}, {0x18800u, (0x18CD5u << 3) | 3u}, {0x18D00u, (0x18D08u << 3) | 3u}, {0x1B000u, (0x1B11Eu << 3) | 3u},
    {0x1B150u, (0x1B152u << 3) | 3u}, {0x1B164u, (0x1B167u << 3) | 3u}, {0x1B170u, (0x1B2FBu << 3) | 3u}, {0x1BC00u, (0x1BC6Au << 3) | 3u},
    {0x1BC70u, (0x1BC7Cu << 3) | 3u}, {0x1BC80u, (0x1BC88u << 3) | 3u}, {0x1BC90u, (0x1BC99u << 3) | 3u}, {0x1BC9Du, (0x1BC9Eu << 3) | 4u},
    {0x1D165u, (0x1D169u << 3) | 4u}, {0x1D16Du, (0x1D172u << 3) | 4u}, {0x1D17Bu, (0x1D182u << 3) | 4u}, {0x1D185u, (0x1D18Bu << 3) | 4u},
    {0x1D1AAu, (0x1D1ADu << 3) | 4u}, {0x1D242u, (0x1D244u << 3) | 4u}, {0x1D2E0u, (0x1D2F3u << 3) | 5u}, {0x1D360u, (0x1D378u << 3) | 5u},
    {0x1D400u, (0x1D419u << 3) | 1u}, {0x1D41Au, (0x1D433u << 3) | 2u}, {0x1D434u, (0x1D44Du << 3) | 1u}, {0x1D44Eu, (0x1D454u << 3) | 2u},
    {0x1D456u, (0x1D467u << 3) | 2u}, {0x1D468u, (0x1D481u << 3) | 1u}, {0x1D482u, (0x1D49Bu << 3) | 2u}, {0x1D49Cu, (0x1D49Cu << 3) | 1u},
    {0x1D49Eu, (0x1D49Fu << 3) | 1u}, {0x1D4A2u, (0x1D4A2u << 3) | 1u}, {0x1D4A5u, (0x1D4A6u << 3) | 1u}, {0x1D4A9u, (0x1D4ACu << 3) | 1u},
    {0x1D4AEu, (0x1D4B5u << 3) | 1u}, {0x1D4B6u, (0x1D4B9u << 3) | 2u}, {0x1D4BBu, (0x1D4BBu << 3) | 2u}, {0x1D4BDu, (0x1D4C3u << 3) | 2u},
    {0x1D4C5u, (0x1D4CFu << 3) | 2u}, {0x1D4D0u, (0x1D4E9u << 3) | 1u}, {0x1D4EAu, (0x1D503u << 3) | 2u}, {0x1D504u, (0x1D505u << 3) | 1u},
    {0x1D507u, (0x1D50Au << 3) | 1u}, {0x1D50Du, (0x1D514u << 3) | 1u}, {0x1D516u, (0x1D51Cu << 3) | 1u}, {0x1D51Eu, (0x1D537u << 3) | 2u},
    {0x1D538u, (0x1D539u << 3) | 1u}, {0x1D53Bu, (0x1D53Eu << 3) | 1u}, {0x1D540u, (0x1D544u << 3) | 1u}, {0x1D546u, (0x1D546u << 3) | 1u},
    {0x1D54Au, (0x1D550u << 3) | 1u}, {0x1D552u, (0x1D56Bu << 3) | 2u}, {0x1D56Cu, (0x1D585u << 3) | 1u}, {0x1D586u, (0x1D59Fu << 3) | 2u},
    {0x1D5A0u, (0x1D5B9u << 3) | 1u}, {0x1D5BAu, (0x1D5D3u << 3) | 2u}, {0x1D5D4u, (0x1D5EDu << 3) | 1u}, {0x1D5EEu, (0x1D607u << 3) | 2u},
    {0x1D608u, (0x1D621u << 3) | 1u}, {0x1D622u, (0x1D63Bu << 3) | 2u}, {0x1D63Cu, (0x1D655u << 3) | 1u}, {0x1D656u, (0x1D66Fu << 3) | 2u},
    {0x1D670u, (0x1D689u << 3) | 1u}, {0x1D68Au, (0x1D6A5u << 3) | 2u}, {0x1D6A8u, (0x1D6C0u << 3) | 1u}, {0x1D6C2u, (0x1D6DAu << 3) | 2u},
    {0x1D6DCu, (0x1D6E1u << 3) | 2u}, {0x1D6E2u, (0x1D6FAu << 3) | 1u}, {0x1D6FCu, (0x1D714u << 3) | 2u}, {0x1D716u, (0x1D71Bu << 3) | 2u},
    {0x1D71Cu, (0x1D734u << 3) | 1u}, {0x1D736u, (0x1D74Eu << 3) | 2u}, {0x1D750u, (0x1D755u << 3) | 2u}, {0x1D756u, (0x1D76Eu << 3) | 1u},
    {0x1D770u, (0x1D788u << 3) | 2u}, {0x1D78Au, (0x1D78Fu << 3) | 2u}, {0x1D790u, (0x1D7A8u << 3) | 1u}, {0x1D7AAu, (0x1D7C2u << 3) | 2u},
    {0x1D7C4u, (0x1D7C9u << 3) | 2u}, {0x1D7CAu, (0x1D7CAu << 3) | 1u}, {0x1D7CBu, (0x1D7CBu << 3) | 2u}, {0x1D7CEu, (0x1D7FFu << 3) | 5u},
    {0x1DA00u, (0x1DA36u << 3) | 4u}, {0x1DA3Bu, (0x1DA6Cu << 3) | 4u}, {0x1DA75u, (0x1DA75u << 3) | 4u}, {0x1DA84u, (0x1DA84u << 3) | 4u},
    {0x1DA9Bu, (0x1DA9Fu << 3) | 4u}, {0x1DAA1u, (0x1DAAFu << 3) | 4u}, {0x1E000u, (0x1E006u << 3) | 4u}, {0x1E008u, (0x1E018u << 3) | 4u},
    {0x1E01Bu, (0x1E021u << 3) | 4u}, {0x1E023u, (0x1E024u << 3) | 4u}, {0x1E026u, (0x1E02Au << 3) | 4u}, {0x1E100u, (0x1E12Cu << 3) | 3u},
    {0x1E130u, (0x1E136u << 3) | 4u}, {0x1E137u, (0x1E13Du << 3) | 3u}, {0x1E140u, (0x1E149u << 3) | 5u}, {0x1E14Eu, (0x1E14Eu << 3) | 3u},
    {0x1E2C0u, (0x1E2EBu << 3) | 3u}, {0x1E2ECu, (0x1E2EFu << 3) | 4u}, {0x1E2F0u, (0x1E2F9u << 3) | 5u}, {0x1E800u, (0x1E8C4u << 3) | 3u},
    {0x1E8C7u, (0x1E8CFu << 3) | 5u}, {0x1E8D0u, (0x1E8D6u << 3) | 4u}, {0x1E900u, (0x1E921u << 3) | 1u}, {0x1E922u, (0x1E943u << 3) | 2u},
    {0x1E944u, (0x1E94Au << 3) | 4u}, {0x1E94Bu, (0x1E94Bu << 3) | 3u}, {0x1E950u, (0x1E959u << 3) | 5u}, {0x1EC71u, (0x1ECABu << 3) | 5u},
    {0x1ECADu, (0x1ECAFu << 3) | 5u}, {0x1ECB1u, (0x1ECB4u << 3) | 5u}, {0x1ED01u, (0x1ED2Du << 3) | 5u}, {0x1ED2Fu, (0x1ED3Du << 3) | 5u},
    {0x1EE00u, (0x1EE03u << 3) | 3u}, {0x1EE05u, (0x1EE1Fu << 3) | 3u}, {0x1EE21u, (0x1EE22u << 3) | 3u}, {0x1EE24u, (0x1EE24u << 3) | 3u},
    {0x1EE27u, (0x1EE27u << 3) | 3u}, {0x1EE29u, (0x1EE32u << 3) | 3u}, {0x1EE34u, (0x1EE37u << 3) | 3u}, {0x1EE39u, (0x1EE39u << 3) | 3u},
    {0x1EE3Bu, (0x1EE3Bu << 3) | 3u}, {0x1EE42u, (0x1EE42u << 3) | 3u}, {0x1EE47u, (0x1EE47u << 3) | 3u}, {0x1EE49u, (0x1EE49u << 3) | 3u},
    {0x1EE4Bu, (0x1EE4Bu << 3) | 3u}, {0x1EE4Du, (0x1EE4Fu << 3) | 3u}, {0x1EE51u, (0x1EE52u << 3) | 3u}, {0x1EE54u, (0x1EE54u << 3) | 3u},
    {0x1EE57u, (0x1EE57u << 3) | 3u}, {0x1EE59u, (0x1EE59u << 3) | 3u}, {0x1EE5Bu, (0x1EE5Bu << 3) | 3u}, {0x1EE5Du, (0x1EE5Du << 3) | 3u},
    {0x1EE5Fu, (0x1EE5Fu << 3) | 3u}, {0x1EE61u, (0x1EE62u << 3) | 3u}, {0x1EE64u, (0x1EE64u << 3) | 3u}, {0x1EE67u, (0x1EE6Au << 3) | 3u},
    {0x1EE6Cu, (0x1EE72u << 3) | 3u}, {0x1EE74u, (0x1EE77u << 3) | 3u}, {0x1EE79u, (0x1EE7Cu << 3) | 3u}, {0x1EE7Eu, (0x1EE7Eu << 3) | 3u},
    {0x1EE80u, (0x1EE89u << 3) | 3u}, {0x1EE8Bu, (0x1EE9Bu << 3) | 3u}, {0x1EEA1u, (0x1EEA3u << 3) | 3u}, {0x1EEA5u, (0x1EEA9u << 3) | 3u},
    {0x1EEABu, (0x1EEBBu << 3) | 3u}, {0x1F100u, (0x1F10Cu << 3) | 5u}, {0x1FBF0u, (0x1FBF9u << 3) | 5u}, {0x20000u, (0x2A6DDu << 3) | 3u},
    {0x2A700u, (0x2B734u << 3) | 3u}, {0x2B740u, (0x2B81Du << 3) | 3u}, {0x2B820u, (0x2CEA1u << 3) | 3u}, {0x2CEB0u, (0x2EBE0u << 3) | 3u},
    {0x2F800u, (0x2FA1Du << 3) | 3u}, {0x30000u, (0x3134Au << 3) | 3u}, {0xE0100u, (0xE01EFu << 3) | 4u},
};

static inline uint8_t unicode_category_class(uint32_t cp) {
    if (cp < 0x80u) {
        if (cp >= 'A' && cp <= 'Z') return 1u;
        if (cp >= 'a' && cp <= 'z') return 2u;
        if (cp >= '0' && cp <= '9') return 5u;
        return 0u;
    }
    size_t lo = 0;
    size_t hi = sizeof(kUcdCatRanges) / sizeof(kUcdCatRanges[0]);
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        const uint32_t r_lo = kUcdCatRanges[mid].lo;
        const uint32_t r_hi = kUcdCatRanges[mid].hi_cat >> 3;
        if (cp < r_lo) {
            hi = mid;
        } else if (cp > r_hi) {
            lo = mid + 1;
        } else {
            return static_cast<uint8_t>(kUcdCatRanges[mid].hi_cat & 7u);
        }
    }
    return 0u;
}

static inline bool is_unicode_whitespace(uint32_t cp) {
    if (cp <= 0x20u) {
        return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f';
    }
    return cp == 0x0085u || cp == 0x00A0u || cp == 0x1680u ||
           (cp >= 0x2000u && cp <= 0x200Au) ||
           cp == 0x2028u || cp == 0x2029u || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u;
}

static inline char ascii_lower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return static_cast<char>(cp + ('a' - 'A'));
    return (cp < 0x80u) ? static_cast<char>(cp) : '\0';
}

struct CodepointSpan {
    uint32_t cp;
    uint32_t byte_off;
    uint32_t byte_len;
    uint8_t  cat;
    bool     is_ws;
};

static inline bool cp_is_L(const CodepointSpan& c)   { return c.cat >= 1u && c.cat <= 3u; }
static inline bool cp_is_N(const CodepointSpan& c)   { return c.cat == 5u; }
static inline bool cp_is_U(const CodepointSpan& c)   { return c.cat == 1u || c.cat == 3u || c.cat == 4u; }
static inline bool cp_is_LoL(const CodepointSpan& c) { return c.cat == 2u || c.cat == 3u || c.cat == 4u; }
static inline bool cp_is_opt_prefix(const CodepointSpan& c) {
    return c.cp != '\r' && c.cp != '\n' && !cp_is_L(c) && !cp_is_N(c);
}
static inline bool cp_is_punct(const CodepointSpan& c) {
    return !c.is_ws && !cp_is_L(c) && !cp_is_N(c);
}

static inline size_t match_contraction_at(const std::vector<CodepointSpan>& cps, size_t p) {
    const size_t N = cps.size();
    if (p + 1 >= N || cps[p].cp != '\'') return 0;
    const char c1 = ascii_lower(cps[p + 1].cp);
    if (p + 2 < N) {
        const char c2 = ascii_lower(cps[p + 2].cp);
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
            return 3;
        }
    }
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
        return 2;
    }
    return 0;
}

static void o200k_pretokenize_span(
    const char* text,
    size_t len,
    std::vector<std::pair<uint32_t, uint32_t>>& pieces_out
) {
    pieces_out.clear();
    if (len == 0) return;

    std::vector<CodepointSpan> cps;
    cps.reserve(len);
    size_t off = 0;
    const uint8_t* u = reinterpret_cast<const uint8_t*>(text);
    while (off < len) {
        size_t adv = 0;
        uint32_t cp = decode_utf8_one(u + off, len - off, &adv);
        cps.push_back({
            cp,
            static_cast<uint32_t>(off),
            static_cast<uint32_t>(adv),
            unicode_category_class(cp),
            is_unicode_whitespace(cp)
        });
        off += adv;
    }

    const size_t N = cps.size();

    auto match_b1 = [&](size_t start) -> size_t {
        size_t u_end = start;
        while (u_end < N && cp_is_U(cps[u_end])) ++u_end;
        size_t k = u_end;
        while (true) {
            if (k < N && cp_is_LoL(cps[k])) {
                size_t end = k + 1;
                while (end < N && cp_is_LoL(cps[end])) ++end;
                end += match_contraction_at(cps, end);
                return end;
            }
            if (k == start) return 0;
            --k;
        }
    };

    auto match_b2 = [&](size_t start) -> size_t {
        if (start >= N || !cp_is_U(cps[start])) return 0;
        size_t end = start + 1;
        while (end < N && cp_is_U(cps[end])) ++end;
        while (end < N && cp_is_LoL(cps[end])) ++end;
        end += match_contraction_at(cps, end);
        return end;
    };

    size_t i = 0;
    while (i < N) {
        const CodepointSpan& c0 = cps[i];

        /* Branch 1: [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)? */
        size_t end = 0;
        if (cp_is_opt_prefix(c0)) end = match_b1(i + 1);
        if (end == 0)             end = match_b1(i);
        if (end > 0) {
            const uint32_t b0 = cps[i].byte_off;
            const uint32_t b1 = cps[end - 1].byte_off + cps[end - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = end;
            continue;
        }

        /* Branch 2: [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)? */
        if (cp_is_opt_prefix(c0)) end = match_b2(i + 1);
        if (end == 0)             end = match_b2(i);
        if (end > 0) {
            const uint32_t b0 = cps[i].byte_off;
            const uint32_t b1 = cps[end - 1].byte_off + cps[end - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = end;
            continue;
        }

        /* Branch 3: \p{N}{1,3} */
        if (cp_is_N(c0)) {
            size_t j = i + 1;
            while (j < N && (j - i) < 3 && cp_is_N(cps[j])) ++j;
            const uint32_t b0 = cps[i].byte_off;
            const uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }

        /* Branch 4:  ?[^\s\p{L}\p{N}]+[\r\n/]* */
        if (cp_is_punct(c0) || (c0.cp == ' ' && i + 1 < N && cp_is_punct(cps[i + 1]))) {
            size_t j = (c0.cp == ' ') ? (i + 2) : (i + 1);
            while (j < N && cp_is_punct(cps[j])) ++j;
            while (j < N && (cps[j].cp == '\r' || cps[j].cp == '\n' || cps[j].cp == '/')) ++j;
            const uint32_t b0 = cps[i].byte_off;
            const uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }

        /* Branches 5/6/7: \s*[\r\n]+ | \s+(?!\S) | \s+ */
        if (c0.is_ws) {
            size_t ws_end = i + 1;
            ssize_t last_crlf = (c0.cp == '\r' || c0.cp == '\n') ? static_cast<ssize_t>(i) : -1;
            while (ws_end < N && cps[ws_end].is_ws) {
                if (cps[ws_end].cp == '\r' || cps[ws_end].cp == '\n') {
                    last_crlf = static_cast<ssize_t>(ws_end);
                }
                ++ws_end;
            }
            size_t match_end = ws_end;
            if (last_crlf >= 0) {
                match_end = static_cast<size_t>(last_crlf) + 1;
            } else if (ws_end < N && (ws_end - i) >= 2) {
                match_end = ws_end - 1;
            }
            const uint32_t b0 = cps[i].byte_off;
            const uint32_t b1 = cps[match_end - 1].byte_off + cps[match_end - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = match_end;
            continue;
        }

        pieces_out.emplace_back(c0.byte_off, c0.byte_len);
        ++i;
    }
}

struct MergeSlot {
    uint64_t pair_key;
    uint32_t rank;
    uint32_t merged_id;
};

struct AddedTokenEntry {
    std::string content;
    uint32_t    id;
    bool        special;
};

struct TransparentSvHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }
    size_t operator()(const std::string& s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
};

static void build_gpt2_byte_unicode_tables(std::string b2u_utf8[256], int16_t u2b_cp[512]) {
    for (int i = 0; i < 512; ++i) u2b_cp[i] = -1;
    bool direct[256] = {};
    for (int b = 33;  b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;

    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++);
        b2u_utf8[b].clear();
        append_utf8_codepoint(b2u_utf8[b], cp);
        u2b_cp[cp] = static_cast<int16_t>(b);
    }
}

static bool decode_bpe_vocab_key_to_bytes(
    const std::string& vocab_key,
    const int16_t u2b_cp[512],
    std::string& raw_out
) {
    raw_out.clear();
    raw_out.reserve(vocab_key.size());
    const uint8_t* u = reinterpret_cast<const uint8_t*>(vocab_key.data());
    size_t len = vocab_key.size();
    size_t off = 0;
    while (off < len) {
        size_t adv = 0;
        uint32_t cp = decode_utf8_one(u + off, len - off, &adv);
        if (cp >= 512u || u2b_cp[cp] < 0) return false;
        raw_out.push_back(static_cast<char>(static_cast<uint8_t>(u2b_cp[cp])));
        off += adv;
    }
    return true;
}

} /* anonymous namespace */

struct StrataInkling2Tokenizer {
    uint32_t                     vocab_size          = 0;
    bool                         ignore_merges       = true;
    uint32_t                     byte_token_ids[256] = {};
    std::vector<std::string>     id_to_raw_bytes;
    std::vector<uint8_t>         id_is_special;
    std::unordered_map<std::string, uint32_t, TransparentSvHash, std::equal_to<>> raw_piece_to_id;
    std::vector<MergeSlot>       merge_ht;
    uint64_t                     merge_mask  = 0;
    uint32_t                     merge_shift = 32;
    std::vector<AddedTokenEntry> added_sorted;
    std::vector<uint32_t>        added_by_b0[256];

    inline uint64_t hash_pair(uint64_t key) const {
        return (key * 11400714819323198485ULL) >> merge_shift;
    }

    inline bool lookup_merge(uint32_t left_id, uint32_t right_id, uint32_t* rank_out, uint32_t* merged_out) const {
        const uint64_t key = (static_cast<uint64_t>(left_id) << 32) | static_cast<uint64_t>(right_id);
        uint64_t idx = hash_pair(key) & merge_mask;
        while (true) {
            const MergeSlot& s = merge_ht[idx];
            if (s.pair_key == key) {
                *rank_out   = s.rank;
                *merged_out = s.merged_id;
                return true;
            }
            if (s.pair_key == std::numeric_limits<uint64_t>::max()) {
                return false;
            }
            idx = (idx + 1ULL) & merge_mask;
        }
    }

    inline void insert_merge(uint32_t left_id, uint32_t right_id, uint32_t rank, uint32_t merged_id) {
        const uint64_t key = (static_cast<uint64_t>(left_id) << 32) | static_cast<uint64_t>(right_id);
        uint64_t idx = hash_pair(key) & merge_mask;
        while (true) {
            MergeSlot& s = merge_ht[idx];
            if (s.pair_key == std::numeric_limits<uint64_t>::max()) {
                s.pair_key  = key;
                s.rank      = rank;
                s.merged_id = merged_id;
                return;
            }
            if (s.pair_key == key) {
                if (rank < s.rank) {
                    s.rank      = rank;
                    s.merged_id = merged_id;
                }
                return;
            }
            idx = (idx + 1ULL) & merge_mask;
        }
    }
};

namespace {

static void bpe_encode_piece(
    const StrataInkling2Tokenizer* tok,
    const uint8_t* bytes,
    uint32_t len,
    std::vector<uint32_t>& out_ids
) {
    if (len == 0) return;
    if (len == 1) {
        out_ids.push_back(tok->byte_token_ids[bytes[0]]);
        return;
    }

    if (tok->ignore_merges) {
        std::string_view sv(reinterpret_cast<const char*>(bytes), len);
        auto it = tok->raw_piece_to_id.find(sv);
        if (it != tok->raw_piece_to_id.end()) {
            out_ids.push_back(it->second);
            return;
        }
    }

    if (len <= 64) {
        uint32_t ids[64];
        uint32_t n = len;
        for (uint32_t i = 0; i < n; ++i) ids[i] = tok->byte_token_ids[bytes[i]];

        while (n >= 2) {
            uint32_t best_rank   = std::numeric_limits<uint32_t>::max();
            uint32_t best_merged = 0;
            uint32_t best_pos    = 0;
            for (uint32_t i = 0; i + 1 < n; ++i) {
                uint32_t r = 0, m = 0;
                if (tok->lookup_merge(ids[i], ids[i + 1], &r, &m) && r < best_rank) {
                    best_rank   = r;
                    best_merged = m;
                    best_pos    = i;
                }
            }
            if (best_rank == std::numeric_limits<uint32_t>::max()) break;
            ids[best_pos] = best_merged;
            for (uint32_t j = best_pos + 1; j + 1 < n; ++j) {
                ids[j] = ids[j + 1];
            }
            --n;
        }
        for (uint32_t i = 0; i < n; ++i) out_ids.push_back(ids[i]);
        return;
    }

    struct Node {
        uint32_t id;
        int32_t  prev;
        int32_t  next;
        uint32_t ver;
        bool     alive;
    };
    struct HeapItem {
        uint32_t rank;
        int32_t  left;
        uint32_t left_ver;
        uint32_t right_ver;
        uint32_t merged_id;
        bool operator>(const HeapItem& o) const {
            return (rank != o.rank) ? (rank > o.rank) : (left > o.left);
        }
    };

    std::vector<Node> nodes(len);
    for (uint32_t i = 0; i < len; ++i) {
        nodes[i] = {
            tok->byte_token_ids[bytes[i]],
            static_cast<int32_t>(i) - 1,
            (i + 1 < len) ? static_cast<int32_t>(i + 1) : -1,
            0u,
            true
        };
    }

    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> pq;
    auto try_push = [&](int32_t l) {
        if (l < 0) return;
        int32_t r = nodes[l].next;
        if (r < 0) return;
        uint32_t rank = 0, merged = 0;
        if (tok->lookup_merge(nodes[l].id, nodes[r].id, &rank, &merged)) {
            pq.push({rank, l, nodes[l].ver, nodes[r].ver, merged});
        }
    };
    for (uint32_t i = 0; i + 1 < len; ++i) {
        try_push(static_cast<int32_t>(i));
    }

    while (!pq.empty()) {
        HeapItem top = pq.top();
        pq.pop();
        int32_t l = top.left;
        if (!nodes[l].alive || nodes[l].ver != top.left_ver) continue;
        int32_t r = nodes[l].next;
        if (r < 0 || !nodes[r].alive || nodes[r].ver != top.right_ver) continue;

        nodes[l].id = top.merged_id;
        ++nodes[l].ver;
        nodes[r].alive = false;
        int32_t rn = nodes[r].next;
        nodes[l].next = rn;
        if (rn >= 0) nodes[rn].prev = l;

        try_push(nodes[l].prev);
        try_push(l);
    }

    for (int32_t cur = 0; cur >= 0; cur = nodes[cur].next) {
        if (nodes[cur].alive) out_ids.push_back(nodes[cur].id);
    }
}

static const char* dtype_name(uint32_t dt) {
    switch (dt) {
        case STRATA_INKLING2_DTYPE_NONE:  return "NONE";
        case STRATA_INKLING2_DTYPE_F32:   return "F32";
        case STRATA_INKLING2_DTYPE_F16:   return "F16";
        case STRATA_INKLING2_DTYPE_NVFP4: return "NVFP4";
        case STRATA_INKLING2_DTYPE_MXFP8: return "MXFP8";
        case STRATA_INKLING2_DTYPE_Q8_0:  return "Q8_0";
        case STRATA_INKLING2_DTYPE_Q4_A:  return "Q4_A";
        default:                          return "UNKNOWN";
    }
}

static bool validate_slice_bounds(
    const StrataInkling2TensorSlice& sl,
    uint32_t allowed_dtype_mask,
    uint64_t exp_shape0,
    uint64_t exp_shape1,
    uint64_t file_size
) {
    if (sl.dtype == STRATA_INKLING2_DTYPE_NONE) {
        if ((allowed_dtype_mask & (1U << STRATA_INKLING2_DTYPE_NONE)) == 0U) return false;
        return sl.offset == 0ULL && sl.bytes == 0ULL && sl.elem_count == 0u;
    }
    if (sl.dtype >= 16U || ((allowed_dtype_mask & (1U << sl.dtype)) == 0U)) return false;
    if (exp_shape0 == 0ULL || exp_shape1 == 0ULL ||
        exp_shape0 > 0xFFFFFFFFULL || exp_shape1 > 0xFFFFFFFFULL) return false;
    if (static_cast<uint64_t>(sl.shape0) != exp_shape0 ||
        static_cast<uint64_t>(sl.shape1) != exp_shape1) return false;
    const uint64_t exp_elems = exp_shape0 * exp_shape1;
    if (exp_elems == 0ULL || exp_elems > 0xFFFFFFFFULL ||
        static_cast<uint64_t>(sl.elem_count) != exp_elems) return false;
    if ((sl.offset % STRATA_INKLING2_TENSOR_ALIGN) != 0ULL) return false;
    if (sl.offset < STRATA_INKLING2_HEADER_PADDED_BYTES) return false;
    if (sl.bytes > file_size || sl.offset > file_size - sl.bytes) return false;

    if (sl.dtype == STRATA_INKLING2_DTYPE_F32 || sl.dtype == STRATA_INKLING2_DTYPE_F16) {
        const uint64_t bpe = (sl.dtype == STRATA_INKLING2_DTYPE_F32) ? 4ULL : 2ULL;
        return exp_elems * bpe == sl.bytes;
    }
    if (sl.dtype == STRATA_INKLING2_DTYPE_Q8_0) {
        if ((sl.shape1 % STRATA_INKLING2_QUANT_GROUP_SIZE) != 0u) return false;
        uint64_t wb = 0ULL, sb = 0ULL;
        strata_inkling2_q8_layout(sl.shape0, sl.shape1, &wb, &sb);
        return sl.bytes == (wb + sb);
    }
    if (sl.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
        if ((sl.shape1 % STRATA_INKLING2_QUANT_GROUP_SIZE) != 0u) return false;
        uint64_t wb = 0ULL, sb = 0ULL, bb = 0ULL;
        strata_inkling2_q4a_layout(sl.shape0, sl.shape1, &wb, &sb, &bb);
        return sl.bytes == (wb + sb + bb);
    }
    return false;
}

static bool validate_expert_bounds(
    const StrataInkling2ExpertMatrix& em,
    uint32_t exp_num_experts,
    uint32_t exp_rows_per_expert,
    uint32_t exp_cols,
    uint32_t exp_dtype,
    uint64_t file_size
) {
    if (exp_dtype != STRATA_INKLING2_DTYPE_NVFP4 && exp_dtype != STRATA_INKLING2_DTYPE_MXFP8) return false;
    if (em.dtype != exp_dtype ||
        em.num_experts != exp_num_experts ||
        em.rows_per_expert != exp_rows_per_expert ||
        em.cols != exp_cols ||
        em.num_experts == 0u || em.rows_per_expert == 0u || em.cols == 0u) return false;
    if ((em.weight_off % STRATA_INKLING2_TENSOR_ALIGN) != 0ULL ||
        (em.scales_off % STRATA_INKLING2_TENSOR_ALIGN) != 0ULL) return false;
    if (em.weight_off < STRATA_INKLING2_HEADER_PADDED_BYTES ||
        em.scales_off < STRATA_INKLING2_HEADER_PADDED_BYTES) return false;
    if (em.weight_bytes > file_size || em.weight_off > file_size - em.weight_bytes) return false;
    if (em.scales_bytes > file_size || em.scales_off > file_size - em.scales_bytes) return false;
    const uint64_t E = em.num_experts;
    const uint64_t R = em.rows_per_expert;
    const uint64_t C = em.cols;
    if (em.dtype == STRATA_INKLING2_DTYPE_NVFP4) {
        if (em.block_size != 16u || (C % 16ULL) != 0ULL) return false;
        const uint64_t exp_wb = E * R * (C / 2ULL);
        const uint64_t exp_sb = E * R * (C / 16ULL);
        const uint64_t exp_es = E * sizeof(float);
        if (em.weight_bytes != exp_wb || em.scales_bytes != exp_sb || em.expert_scale_bytes != exp_es ||
            (em.expert_scale_off % STRATA_INKLING2_TENSOR_ALIGN) != 0ULL ||
            em.expert_scale_off < STRATA_INKLING2_HEADER_PADDED_BYTES ||
            em.expert_scale_bytes > file_size ||
            em.expert_scale_off > file_size - em.expert_scale_bytes) return false;
    } else {
        if (em.block_size != 32u || (C % 32ULL) != 0ULL) return false;
        const uint64_t exp_wb = E * R * C;
        const uint64_t exp_sb = E * R * (C / 32ULL);
        if (em.weight_bytes != exp_wb || em.scales_bytes != exp_sb ||
            em.expert_scale_off != 0ULL || em.expert_scale_bytes != 0ULL) return false;
    }
    return true;
}

static bool validate_header(const StrataInkling2Header& h, uint64_t file_size) {
    if (h.magic != STRATA_INKLING2_MAGIC_U64 ||
        (h.version != STRATA_INKLING2_FORMAT_VERSION_V1 &&
         h.version != STRATA_INKLING2_FORMAT_VERSION) ||
        h.header_padded_bytes != STRATA_INKLING2_HEADER_PADDED_BYTES ||
        h.total_file_bytes != file_size ||
        (file_size % STRATA_INKLING2_PAGE_BYTES) != 0ULL) {
        return false;
    }
    const StrataInkling2Config& c = h.cfg;
    if (c.num_phys_layers == 0u || c.num_phys_layers > STRATA_INKLING2_MAX_PHYS_LAYERS ||
        c.num_logical_layers == 0u || c.num_logical_layers > STRATA_INKLING2_MAX_LOGICAL_LAYERS ||
        c.hidden_size == 0u || c.hidden_size > 16384u || (c.hidden_size % STRATA_INKLING2_QUANT_GROUP_SIZE) != 0u ||
        c.vocab_size_unpadded == 0u || c.vocab_size_unpadded > (1u << 22) ||
        c.vocab_size_padded < c.vocab_size_unpadded || c.vocab_size_padded > (1u << 22) ||
        c.num_attention_heads == 0u || c.num_attention_heads > 128u ||
        c.num_key_value_heads == 0u || c.num_key_value_heads > c.num_attention_heads ||
        (c.num_attention_heads % c.num_key_value_heads) != 0u ||
        c.head_dim != 128u ||
        c.d_rel != 16u ||
        c.rel_extent == 0u || c.rel_extent > 8192u ||
        c.sliding_window_size == 0u || c.sliding_window_size > 8192u ||
        c.sconv_kernel_size != 4u ||
        c.attn_res_block_size == 0u || c.attn_res_block_size > c.num_logical_layers ||
        (c.num_logical_layers / c.attn_res_block_size) + 2u > 18u ||
        c.num_experts == 0u || c.num_experts > 256u ||
        c.num_experts_per_tok == 0u || c.num_experts_per_tok > c.num_experts || c.num_experts_per_tok > 16u ||
        c.intermediate_size == 0u || c.intermediate_size > 65536u || (c.intermediate_size % 32u) != 0u ||
        c.loop_num_passes == 0u || c.loop_num_passes > 8u ||
        c.loop_core_start > c.loop_core_end || c.loop_core_end > c.num_phys_layers) {
        return false;
    }
    if (c.loop_num_passes > 1u) {
        if (c.loop_core_end <= c.loop_core_start ||
            c.loop_gate_rank == 0u || c.loop_gate_rank > 1024u) {
            return false;
        }
    }

    constexpr uint32_t kNone = 1U << STRATA_INKLING2_DTYPE_NONE;
    constexpr uint32_t kF32  = 1U << STRATA_INKLING2_DTYPE_F32;
    constexpr uint32_t kF16  = 1U << STRATA_INKLING2_DTYPE_F16;
    constexpr uint32_t kQ8   = 1U << STRATA_INKLING2_DTYPE_Q8_0;
    constexpr uint32_t kQ4   = 1U << STRATA_INKLING2_DTYPE_Q4_A;
    constexpr uint32_t kLin  = kF16 | kQ8 | kQ4;

    const uint64_t H     = c.hidden_size;
    const uint64_t NH    = c.num_attention_heads;
    const uint64_t NKV   = c.num_key_value_heads;
    const uint64_t HD    = c.head_dim;
    const uint64_t DR    = c.d_rel;
    const uint64_t KS    = c.sconv_kernel_size;
    const uint64_t E     = c.num_experts;
    const uint64_t QDIM  = NH * HD;
    const uint64_t KVDIM = NKV * HD;

    if (!validate_slice_bounds(h.embed_tokens, kF16 | kQ4, c.vocab_size_padded, H, file_size) ||
        !validate_slice_bounds(h.embed_norm, kF32, H, 1ULL, file_size) ||
        !validate_slice_bounds(h.final_norm, kF32, H, 1ULL, file_size) ||
        !validate_slice_bounds(h.output_res_query, kF32, H, 1ULL, file_size) ||
        !validate_slice_bounds(h.unembed, kLin, c.vocab_size_unpadded, H, file_size)) {
        return false;
    }
    if (c.loop_num_passes > 1u) {
        const uint64_t n_cp = 2ULL * static_cast<uint64_t>(c.loop_core_end - c.loop_core_start);
        if (!validate_slice_bounds(h.cross_pass_queries, kF32, n_cp, H, file_size) ||
            !validate_slice_bounds(h.cross_pass_kv_queries, kF32, n_cp, H, file_size)) {
            return false;
        }
    } else {
        if (!validate_slice_bounds(h.cross_pass_queries, kNone, 0ULL, 0ULL, file_size) ||
            !validate_slice_bounds(h.cross_pass_kv_queries, kNone, 0ULL, 0ULL, file_size)) {
            return false;
        }
    }

    uint32_t shared_ctrl_dtype = STRATA_INKLING2_DTYPE_NONE;
    for (uint32_t i = 0; i < c.num_phys_layers; ++i) {
        const StrataInkling2LayerDesc& ld = h.layers[i];
        const bool is_local = (ld.is_local_swa != 0u);
        const bool exp_core = (c.loop_num_passes > 1u && i >= c.loop_core_start && i < c.loop_core_end);
        if (ld.phys_layer_idx != i ||
            ld.is_local_swa > 1u ||
            (ld.in_loop_core != 0u) != exp_core ||
            (ld.layer_page_off % STRATA_INKLING2_PAGE_BYTES) != 0ULL ||
            (ld.layer_page_bytes % STRATA_INKLING2_PAGE_BYTES) != 0ULL ||
            ld.layer_page_off < STRATA_INKLING2_HEADER_PADDED_BYTES ||
            ld.layer_page_off > file_size ||
            ld.layer_page_bytes > file_size - ld.layer_page_off) {
            return false;
        }
        if (i == 0u) {
            if (!validate_slice_bounds(ld.attn_res_query, kNone, 0ULL, 0ULL, file_size)) return false;
        } else if (is_local) {
            if (!validate_slice_bounds(ld.attn_res_query, kF32, H, 1ULL, file_size)) return false;
        } else {
            if (!validate_slice_bounds(ld.attn_res_query, kF32, 3ULL, H, file_size)) return false;
        }
        if (!validate_slice_bounds(ld.mlp_res_query, kF32, H, 1ULL, file_size) ||
            !validate_slice_bounds(ld.mlp_norm, kF32, H, 1ULL, file_size)) {
            return false;
        }
        if (is_local) {
            if (!validate_slice_bounds(ld.attn_norm, kF32, H, 1ULL, file_size) ||
                !validate_slice_bounds(ld.q_input_norm, kNone, 0ULL, 0ULL, file_size) ||
                !validate_slice_bounds(ld.k_input_norm, kNone, 0ULL, 0ULL, file_size) ||
                !validate_slice_bounds(ld.v_input_norm, kNone, 0ULL, 0ULL, file_size)) {
                return false;
            }
        } else {
            if (!validate_slice_bounds(ld.attn_norm, kNone, 0ULL, 0ULL, file_size) ||
                !validate_slice_bounds(ld.q_input_norm, kF32, H, 1ULL, file_size) ||
                !validate_slice_bounds(ld.k_input_norm, kF32, H, 1ULL, file_size) ||
                !validate_slice_bounds(ld.v_input_norm, kF32, H, 1ULL, file_size)) {
                return false;
            }
        }
        const uint64_t extent = is_local ? c.sliding_window_size : c.rel_extent;
        if (!validate_slice_bounds(ld.wq, kLin, QDIM, H, file_size) ||
            !validate_slice_bounds(ld.wk, kLin, KVDIM, H, file_size) ||
            !validate_slice_bounds(ld.wv, kLin, KVDIM, H, file_size) ||
            !validate_slice_bounds(ld.wo, kLin, H, QDIM, file_size) ||
            !validate_slice_bounds(ld.gate_attn, kLin, QDIM, H, file_size) ||
            !validate_slice_bounds(ld.wr, kLin, NH * DR, H, file_size) ||
            !validate_slice_bounds(ld.attn_sink_proj, kLin, NH, H, file_size) ||
            !validate_slice_bounds(ld.q_l2_clip_scale, kF32, NH, HD, file_size) ||
            !validate_slice_bounds(ld.rel_l2_clip_scale, kF32, NH, DR, file_size) ||
            !validate_slice_bounds(ld.rel_logits_proj, kF32, extent, DR, file_size) ||
            !validate_slice_bounds(ld.k_sconv_w, kF32, KVDIM, KS, file_size) ||
            !validate_slice_bounds(ld.v_sconv_w, kF32, KVDIM, KS, file_size) ||
            !validate_slice_bounds(ld.attn_sconv_w, kF32, H, KS, file_size) ||
            !validate_slice_bounds(ld.mlp_sconv_w, kF32, H, KS, file_size) ||
            !validate_slice_bounds(ld.moe_gate_weight, kF32, E, H, file_size) ||
            !validate_slice_bounds(ld.moe_gate_bias, kNone | kF32, E, 1ULL, file_size) ||
            !validate_expert_bounds(ld.experts_w13, c.num_experts, 2u * c.intermediate_size, c.hidden_size,
                                    ld.expert_dtype, file_size) ||
            !validate_expert_bounds(ld.experts_w2, c.num_experts, c.hidden_size, c.intermediate_size,
                                    ld.expert_dtype, file_size)) {
            return false;
        }
        if (is_local) {
            if (!validate_slice_bounds(ld.k_sconv_proj, kNone, 0ULL, 0ULL, file_size) ||
                !validate_slice_bounds(ld.v_sconv_proj, kNone, 0ULL, 0ULL, file_size)) {
                return false;
            }
        } else {
            if (!validate_slice_bounds(ld.k_sconv_proj, kLin, NKV * KS, H, file_size) ||
                !validate_slice_bounds(ld.v_sconv_proj, kLin, NKV * KS, H, file_size)) {
                return false;
            }
        }
        for (uint32_t m = 0; m < 4u; ++m) {
            if (exp_core) {
                if (!validate_slice_bounds(ld.controllers[m].down_weight, kF16 | kQ8, c.loop_gate_rank, 2ULL * H, file_size) ||
                    !validate_slice_bounds(ld.controllers[m].gain_weight, kF16 | kQ8, H, c.loop_gate_rank, file_size) ||
                    !validate_slice_bounds(ld.controllers[m].forget_weight, kF16 | kQ8, H, c.loop_gate_rank, file_size)) {
                    return false;
                }
                const uint32_t dt = ld.controllers[m].down_weight.dtype;
                if (shared_ctrl_dtype == STRATA_INKLING2_DTYPE_NONE) {
                    shared_ctrl_dtype = dt;
                }
                if (dt != shared_ctrl_dtype ||
                    ld.controllers[m].gain_weight.dtype != shared_ctrl_dtype ||
                    ld.controllers[m].forget_weight.dtype != shared_ctrl_dtype) {
                    return false;
                }
            } else {
                if (!validate_slice_bounds(ld.controllers[m].down_weight, kNone, 0ULL, 0ULL, file_size) ||
                    !validate_slice_bounds(ld.controllers[m].gain_weight, kNone, 0ULL, 0ULL, file_size) ||
                    !validate_slice_bounds(ld.controllers[m].forget_weight, kNone, 0ULL, 0ULL, file_size)) {
                    return false;
                }
            }
        }
    }

    /* Validate logical-to-physical visit schedule, per-tier KV slot indices, and exact AttnRes block bound (I1) */
    uint32_t loc_slots = 0u, glob_slots = 0u;
    uint32_t num_blocks = 0u, prelude_blocks = 0u;
    for (uint32_t il = 0u; il < c.num_logical_layers; ++il) {
        const uint32_t ip   = c.logical_to_phys[il];
        const uint32_t pass = c.logical_pass_idx[il];
        if (ip >= c.num_phys_layers) return false;
        if (pass >= c.loop_num_passes) return false;
        const bool is_loc = (h.layers[ip].is_local_swa != 0u);
        if ((c.logical_is_local[il] != 0u) != is_loc) return false;
        if (is_loc) {
            if (c.logical_kv_slot[il] != loc_slots++) return false;
        } else {
            if (c.logical_kv_slot[il] != glob_slots++) return false;
        }
        if (c.loop_num_passes > 1u && ip == c.loop_core_start) {
            if (pass == 0u) prelude_blocks = num_blocks;
            else            num_blocks     = prelude_blocks;
        }
        if ((ip % c.attn_res_block_size) == 0u) {
            if (num_blocks >= 8u) return false;
            ++num_blocks;
        }
    }
    if (c.num_local_visits != loc_slots || c.num_global_visits != glob_slots) {
        return false;
    }

    if (!h.mtp.enabled) {
        if (h.mtp.frspec.enabled != 0u) return false;
    } else {
        const StrataInkling2MtpDesc& mtp = h.mtp;
        if (mtp.num_layers == 0u || mtp.num_layers > STRATA_INKLING2_MAX_MTP_LAYERS ||
            mtp.intermediate_size == 0u || mtp.intermediate_size > 65536u ||
            mtp.block_size < 2u || mtp.block_size > 16u ||
            mtp.markov_rank == 0u || mtp.markov_rank > 4096u ||
            mtp.sliding_window == 0u || mtp.sliding_window > 16384u) {
            return false;
        }
        for (uint32_t t = 0u; t < STRATA_INKLING2_MAX_MTP_TAPS; ++t) {
            if (mtp.target_layer_ids[t] > c.num_logical_layers) return false;
        }
        if ((mtp.mtp_page_off % STRATA_INKLING2_PAGE_BYTES) != 0ULL ||
            (mtp.mtp_page_bytes % STRATA_INKLING2_PAGE_BYTES) != 0ULL ||
            mtp.mtp_page_off < STRATA_INKLING2_HEADER_PADDED_BYTES ||
            mtp.mtp_page_off > file_size ||
            mtp.mtp_page_bytes > file_size - mtp.mtp_page_off) {
            return false;
        }
        const uint64_t MR = mtp.markov_rank;
        const uint64_t MI = mtp.intermediate_size;
        if (!validate_slice_bounds(mtp.fuse_alpha, kF32, static_cast<uint64_t>(mtp.num_layers) + 1ULL,
                                   STRATA_INKLING2_MAX_MTP_TAPS, file_size) ||
            !validate_slice_bounds(mtp.mask_embed, kF16, H, 1ULL, file_size) ||
            !validate_slice_bounds(mtp.final_norm, kF32, H, 1ULL, file_size) ||
            !validate_slice_bounds(mtp.markov_dflare_norm, kF32, H, 1ULL, file_size) ||
            !validate_slice_bounds(mtp.markov_w1, kF16, c.vocab_size_padded, MR, file_size) ||
            !validate_slice_bounds(mtp.markov_rnn_weight, kLin, 3ULL * MR, 2ULL * MR + 2ULL * H, file_size) ||
            !validate_slice_bounds(mtp.markov_rnn_bias, kF32, 3ULL * MR, 1ULL, file_size) ||
            !validate_slice_bounds(mtp.markov_hidden_out, kF16, H, MR, file_size) ||
            !validate_slice_bounds(mtp.markov_hidden_bias, kF32, H, 1ULL, file_size)) {
            return false;
        }
        for (uint32_t l = 0; l < mtp.num_layers; ++l) {
            const StrataInkling2MtpLayerDesc& ml = mtp.layers[l];
            if (ml.layer_idx != l || ml.is_local_swa > 1u) return false;
            if (!validate_slice_bounds(ml.dflare_norm, kF32, H, 1ULL, file_size) ||
                !validate_slice_bounds(ml.input_layernorm, kF32, H, 1ULL, file_size) ||
                !validate_slice_bounds(ml.post_attn_norm, kF32, H, 1ULL, file_size) ||
                !validate_slice_bounds(ml.q_norm, kF32, HD, 1ULL, file_size) ||
                !validate_slice_bounds(ml.k_norm, kF32, HD, 1ULL, file_size) ||
                !validate_slice_bounds(ml.k_norm_ctx, kF32, HD, 1ULL, file_size) ||
                !validate_slice_bounds(ml.q_proj, kLin, QDIM, H, file_size) ||
                !validate_slice_bounds(ml.k_proj, kLin, KVDIM, H, file_size) ||
                !validate_slice_bounds(ml.v_proj, kLin, KVDIM, H, file_size) ||
                !validate_slice_bounds(ml.k_proj_ctx, kLin, KVDIM, H, file_size) ||
                !validate_slice_bounds(ml.v_proj_ctx, kLin, KVDIM, H, file_size) ||
                !validate_slice_bounds(ml.o_proj, kLin, H, QDIM, file_size) ||
                !validate_slice_bounds(ml.gate_proj, kLin, MI, H, file_size) ||
                !validate_slice_bounds(ml.up_proj, kLin, MI, H, file_size) ||
                ml.gate_proj.dtype != ml.up_proj.dtype ||
                !validate_slice_bounds(ml.down_proj, kLin, H, MI, file_size)) {
                return false;
            }
        }
        if (mtp.frspec.enabled) {
            const StrataInkling2FrSpecDesc& fr = mtp.frspec;
            if (h.version < STRATA_INKLING2_FORMAT_VERSION) return false;
            if (fr.shortlist_size == 0u || (fr.shortlist_size % 256u) != 0u ||
                fr.shortlist_size > c.vocab_size_unpadded) {
                return false;
            }
            if (!validate_slice_bounds(fr.token_ids, kF32, fr.shortlist_size, 1ULL, file_size) ||
                !validate_slice_bounds(fr.unembed_q4a, kQ4, fr.shortlist_size, H, file_size) ||
                !validate_slice_bounds(fr.fold_weight_q8, kQ8, fr.shortlist_size, MR, file_size) ||
                !validate_slice_bounds(fr.fold_bias_f32, kF32, fr.shortlist_size, 1ULL, file_size)) {
                return false;
            }
        }
    }
    return true;
}

/* ============================================================================
 * 6b. Multi-Threaded Fast Repacker (`Cfg B++` / `Cfg C` + 32K FR-Spec Fold)
 * ============================================================================ */

static bool paths_refer_to_same_entry(const std::string& a, const std::string& b) {
    if (a == b) return true;
    struct stat sa, sb;
    if (stat(a.c_str(), &sa) == 0 && stat(b.c_str(), &sb) == 0 &&
        sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino) {
        return true;
    }
    char ra[PATH_MAX], rb[PATH_MAX];
    if (realpath(a.c_str(), ra) && realpath(b.c_str(), rb) && std::strcmp(ra, rb) == 0) {
        return true;
    }
    return false;
}

static StrataInkling2TensorSlice alloc_like_or_none(
    uint64_t& cur,
    const StrataInkling2TensorSlice& src,
    uint32_t dst_dtype
) {
    if (src.dtype == STRATA_INKLING2_DTYPE_NONE) {
        StrataInkling2TensorSlice empty{};
        return empty;
    }
    return alloc_slice(cur, dst_dtype, src.shape0, src.shape1);
}

static inline uint32_t pick_quant_if_aligned(
    const StrataInkling2TensorSlice& src,
    uint32_t target_dt
) {
    if (src.dtype == STRATA_INKLING2_DTYPE_NONE) return STRATA_INKLING2_DTYPE_NONE;
    if (src.dtype == STRATA_INKLING2_DTYPE_F16 &&
        src.shape1 > 0u &&
        (src.shape1 % STRATA_INKLING2_QUANT_GROUP_SIZE) == 0u) {
        return target_dt;
    }
    return src.dtype;
}

static void plan_repack_layout(
    const StrataInkling2Header& src,
    bool q4_unembed,
    uint32_t mtp_lin_dt,
    StrataInkling2Header& dst
) {
    std::memset(&dst, 0, sizeof(dst));
    dst.magic               = STRATA_INKLING2_MAGIC_U64;
    dst.version             = STRATA_INKLING2_FORMAT_VERSION;
    dst.header_padded_bytes = STRATA_INKLING2_HEADER_PADDED_BYTES;
    dst.cfg                 = src.cfg;

    const StrataInkling2Config& c = dst.cfg;
    const uint32_t H = c.hidden_size;
    const uint32_t unembed_dt = q4_unembed ? STRATA_INKLING2_DTYPE_Q4_A : STRATA_INKLING2_DTYPE_Q8_0;
    const uint32_t mtp_rnn_dt = (mtp_lin_dt == STRATA_INKLING2_DTYPE_F16)
                                    ? STRATA_INKLING2_DTYPE_F16
                                    : STRATA_INKLING2_DTYPE_Q8_0;

    uint64_t cur = STRATA_INKLING2_HEADER_PADDED_BYTES;

    /* Global backbone tensors */
    dst.embed_tokens          = alloc_like_or_none(cur, src.embed_tokens,          STRATA_INKLING2_DTYPE_Q4_A);
    dst.embed_norm            = alloc_like_or_none(cur, src.embed_norm,            STRATA_INKLING2_DTYPE_F32);
    dst.final_norm            = alloc_like_or_none(cur, src.final_norm,            STRATA_INKLING2_DTYPE_F32);
    dst.output_res_query      = alloc_like_or_none(cur, src.output_res_query,      STRATA_INKLING2_DTYPE_F32);
    dst.unembed               = alloc_like_or_none(cur, src.unembed,               unembed_dt);
    dst.cross_pass_queries    = alloc_like_or_none(cur, src.cross_pass_queries,    STRATA_INKLING2_DTYPE_F32);
    dst.cross_pass_kv_queries = alloc_like_or_none(cur, src.cross_pass_kv_queries, STRATA_INKLING2_DTYPE_F32);

    /* 24 Physical layers (each 16 KiB page-aligned; ALL L0..23 experts NVFP4) */
    for (uint32_t i = 0; i < c.num_phys_layers; ++i) {
        cur = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        const StrataInkling2LayerDesc& sl = src.layers[i];
        StrataInkling2LayerDesc&       dl = dst.layers[i];

        dl.phys_layer_idx = sl.phys_layer_idx;
        dl.is_local_swa   = sl.is_local_swa;
        dl.in_loop_core   = sl.in_loop_core;
        dl.expert_dtype   = STRATA_INKLING2_DTYPE_NVFP4;
        dl.layer_page_off = cur;

        dl.attn_res_query    = alloc_like_or_none(cur, sl.attn_res_query,    STRATA_INKLING2_DTYPE_F32);
        dl.mlp_res_query     = alloc_like_or_none(cur, sl.mlp_res_query,     STRATA_INKLING2_DTYPE_F32);
        dl.attn_norm         = alloc_like_or_none(cur, sl.attn_norm,         STRATA_INKLING2_DTYPE_F32);
        dl.q_input_norm      = alloc_like_or_none(cur, sl.q_input_norm,      STRATA_INKLING2_DTYPE_F32);
        dl.k_input_norm      = alloc_like_or_none(cur, sl.k_input_norm,      STRATA_INKLING2_DTYPE_F32);
        dl.v_input_norm      = alloc_like_or_none(cur, sl.v_input_norm,      STRATA_INKLING2_DTYPE_F32);
        dl.mlp_norm          = alloc_like_or_none(cur, sl.mlp_norm,          STRATA_INKLING2_DTYPE_F32);

        dl.wq                = alloc_like_or_none(cur, sl.wq,                STRATA_INKLING2_DTYPE_Q4_A);
        dl.wk                = alloc_like_or_none(cur, sl.wk,                STRATA_INKLING2_DTYPE_Q4_A);
        dl.wv                = alloc_like_or_none(cur, sl.wv,                STRATA_INKLING2_DTYPE_Q4_A);
        dl.wo                = alloc_like_or_none(cur, sl.wo,                STRATA_INKLING2_DTYPE_Q4_A);
        dl.gate_attn         = alloc_like_or_none(cur, sl.gate_attn,         STRATA_INKLING2_DTYPE_Q4_A);
        dl.wr                = alloc_like_or_none(cur, sl.wr,                STRATA_INKLING2_DTYPE_Q8_0);
        dl.attn_sink_proj    = alloc_like_or_none(cur, sl.attn_sink_proj,    STRATA_INKLING2_DTYPE_F16);

        dl.q_l2_clip_scale   = alloc_like_or_none(cur, sl.q_l2_clip_scale,   STRATA_INKLING2_DTYPE_F32);
        dl.rel_l2_clip_scale = alloc_like_or_none(cur, sl.rel_l2_clip_scale, STRATA_INKLING2_DTYPE_F32);
        dl.rel_logits_proj   = alloc_like_or_none(cur, sl.rel_logits_proj,   STRATA_INKLING2_DTYPE_F32);

        dl.k_sconv_w         = alloc_like_or_none(cur, sl.k_sconv_w,         STRATA_INKLING2_DTYPE_F32);
        dl.v_sconv_w         = alloc_like_or_none(cur, sl.v_sconv_w,         STRATA_INKLING2_DTYPE_F32);
        dl.k_sconv_proj      = alloc_like_or_none(cur, sl.k_sconv_proj,      STRATA_INKLING2_DTYPE_F16);
        dl.v_sconv_proj      = alloc_like_or_none(cur, sl.v_sconv_proj,      STRATA_INKLING2_DTYPE_F16);
        dl.attn_sconv_w      = alloc_like_or_none(cur, sl.attn_sconv_w,      STRATA_INKLING2_DTYPE_F32);
        dl.mlp_sconv_w       = alloc_like_or_none(cur, sl.mlp_sconv_w,       STRATA_INKLING2_DTYPE_F32);

        dl.moe_gate_weight   = alloc_like_or_none(cur, sl.moe_gate_weight,   STRATA_INKLING2_DTYPE_F32);
        dl.moe_gate_bias     = alloc_like_or_none(cur, sl.moe_gate_bias,     STRATA_INKLING2_DTYPE_F32);

        dl.experts_w13 = alloc_expert_matrix(cur, sl.experts_w13.num_experts,
                                             sl.experts_w13.rows_per_expert, sl.experts_w13.cols,
                                             STRATA_INKLING2_DTYPE_NVFP4);
        dl.experts_w2  = alloc_expert_matrix(cur, sl.experts_w2.num_experts,
                                             sl.experts_w2.rows_per_expert, sl.experts_w2.cols,
                                             STRATA_INKLING2_DTYPE_NVFP4);

        if (dl.in_loop_core) {
            for (uint32_t m = 0; m < 4u; ++m) {
                dl.controllers[m].down_weight   = alloc_like_or_none(cur, sl.controllers[m].down_weight,   STRATA_INKLING2_DTYPE_Q8_0);
                dl.controllers[m].gain_weight   = alloc_like_or_none(cur, sl.controllers[m].gain_weight,   STRATA_INKLING2_DTYPE_Q8_0);
                dl.controllers[m].forget_weight = alloc_like_or_none(cur, sl.controllers[m].forget_weight, STRATA_INKLING2_DTYPE_Q8_0);
            }
        }

        cur = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        dl.layer_page_bytes = cur - dl.layer_page_off;
    }

    /* 3-Layer DSpark MTP + 32K Weight-Intrinsic FR-Spec Folded Head */
    if (src.mtp.enabled) {
        cur = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        const StrataInkling2MtpDesc& sm = src.mtp;
        StrataInkling2MtpDesc&       dm = dst.mtp;

        dm.enabled           = 1u;
        dm.num_layers        = sm.num_layers;
        dm.intermediate_size = sm.intermediate_size;
        dm.block_size        = sm.block_size;
        dm.markov_rank       = sm.markov_rank;
        dm.sliding_window    = sm.sliding_window;
        dm.rope_theta        = sm.rope_theta;
        dm.rms_norm_eps      = sm.rms_norm_eps;
        for (uint32_t t = 0; t < STRATA_INKLING2_MAX_MTP_TAPS; ++t) {
            dm.target_layer_ids[t] = sm.target_layer_ids[t];
        }
        dm.mtp_page_off = cur;

        dm.fuse_alpha         = alloc_like_or_none(cur, sm.fuse_alpha,         STRATA_INKLING2_DTYPE_F32);
        dm.mask_embed         = alloc_like_or_none(cur, sm.mask_embed,         STRATA_INKLING2_DTYPE_F16);
        dm.final_norm         = alloc_like_or_none(cur, sm.final_norm,         STRATA_INKLING2_DTYPE_F32);
        dm.markov_dflare_norm = alloc_like_or_none(cur, sm.markov_dflare_norm, STRATA_INKLING2_DTYPE_F32);
        dm.markov_w1          = alloc_like_or_none(cur, sm.markov_w1,          STRATA_INKLING2_DTYPE_F16);
        dm.markov_rnn_weight  = alloc_like_or_none(cur, sm.markov_rnn_weight,  pick_quant_if_aligned(sm.markov_rnn_weight, mtp_rnn_dt));
        dm.markov_rnn_bias    = alloc_like_or_none(cur, sm.markov_rnn_bias,    STRATA_INKLING2_DTYPE_F32);
        dm.markov_hidden_out  = alloc_like_or_none(cur, sm.markov_hidden_out,  STRATA_INKLING2_DTYPE_F16);
        dm.markov_hidden_bias = alloc_like_or_none(cur, sm.markov_hidden_bias, STRATA_INKLING2_DTYPE_F32);

        for (uint32_t l = 0; l < sm.num_layers; ++l) {
            const StrataInkling2MtpLayerDesc& sml = sm.layers[l];
            StrataInkling2MtpLayerDesc&       dml = dm.layers[l];
            dml.layer_idx       = sml.layer_idx;
            dml.is_local_swa    = sml.is_local_swa;
            dml.dflare_norm     = alloc_like_or_none(cur, sml.dflare_norm,     STRATA_INKLING2_DTYPE_F32);
            dml.input_layernorm = alloc_like_or_none(cur, sml.input_layernorm, STRATA_INKLING2_DTYPE_F32);
            dml.post_attn_norm  = alloc_like_or_none(cur, sml.post_attn_norm,  STRATA_INKLING2_DTYPE_F32);
            dml.q_norm          = alloc_like_or_none(cur, sml.q_norm,          STRATA_INKLING2_DTYPE_F32);
            dml.k_norm          = alloc_like_or_none(cur, sml.k_norm,          STRATA_INKLING2_DTYPE_F32);
            dml.k_norm_ctx      = alloc_like_or_none(cur, sml.k_norm_ctx,      STRATA_INKLING2_DTYPE_F32);
            dml.q_proj          = alloc_like_or_none(cur, sml.q_proj,          pick_quant_if_aligned(sml.q_proj,     mtp_lin_dt));
            dml.k_proj          = alloc_like_or_none(cur, sml.k_proj,          pick_quant_if_aligned(sml.k_proj,     mtp_lin_dt));
            dml.v_proj          = alloc_like_or_none(cur, sml.v_proj,          pick_quant_if_aligned(sml.v_proj,     mtp_lin_dt));
            dml.k_proj_ctx      = alloc_like_or_none(cur, sml.k_proj_ctx,      pick_quant_if_aligned(sml.k_proj_ctx, mtp_lin_dt));
            dml.v_proj_ctx      = alloc_like_or_none(cur, sml.v_proj_ctx,      pick_quant_if_aligned(sml.v_proj_ctx, mtp_lin_dt));
            dml.o_proj          = alloc_like_or_none(cur, sml.o_proj,          pick_quant_if_aligned(sml.o_proj,     mtp_lin_dt));
            dml.gate_proj       = alloc_like_or_none(cur, sml.gate_proj,       pick_quant_if_aligned(sml.gate_proj,  mtp_lin_dt));
            dml.up_proj         = alloc_like_or_none(cur, sml.up_proj,         pick_quant_if_aligned(sml.up_proj,    mtp_lin_dt));
            dml.down_proj       = alloc_like_or_none(cur, sml.down_proj,       pick_quant_if_aligned(sml.down_proj,  mtp_lin_dt));
        }

        const uint32_t max_short = (c.vocab_size_unpadded / 256u) * 256u;
        const uint32_t shortlist = std::min<uint32_t>(STRATA_INKLING2_FRSPEC_SHORTLIST_SIZE, max_short);
        if (shortlist > 0u) {
            dm.frspec.enabled        = 1u;
            dm.frspec.shortlist_size = shortlist;
            dm.frspec.token_ids      = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32,  shortlist, 1u);
            dm.frspec.unembed_q4a    = alloc_slice(cur, STRATA_INKLING2_DTYPE_Q4_A, shortlist, H);
            dm.frspec.fold_weight_q8 = alloc_slice(cur, STRATA_INKLING2_DTYPE_Q8_0, shortlist, dm.markov_rank);
            dm.frspec.fold_bias_f32  = alloc_slice(cur, STRATA_INKLING2_DTYPE_F32,  shortlist, 1u);
        }

        cur = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
        dm.mtp_page_bytes = cur - dm.mtp_page_off;
    }

    dst.total_file_bytes = align_up_u64(cur, STRATA_INKLING2_PAGE_BYTES);
    compute_active_bytes(dst);
}

enum RepackTaskKind {
    RTASK_COPY_RAW = 0,
    RTASK_F16_TO_Q8,
    RTASK_F16_TO_Q4A,
    RTASK_MXFP8_TO_NVFP4
};

struct Inkling2WoLoraAdapter {
    bool               loaded          = false;
    uint32_t           rank            = 0;
    uint32_t           num_layers      = 0;
    float              effective_scale = 1.0f; /* (lora_alpha / r) * user_scale */
    bool               has_layer[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
    uint32_t           rows[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
    uint32_t           cols[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
    std::vector<float> a_mat[STRATA_INKLING2_MAX_PHYS_LAYERS]; /* [rank, cols] */
    std::vector<float> b_mat[STRATA_INKLING2_MAX_PHYS_LAYERS]; /* [rows, rank] */
};

static bool read_safetensor_as_f32(const SafetensorEntry& ent, std::vector<float>& out) {
    uint64_t elems = 1ULL;
    for (uint64_t d : ent.shape) elems *= d;
    out.resize(static_cast<size_t>(elems));
    if (ent.dtype == ST_F32) {
        if (!pread_exact(ent.shard_fd, out.data(), static_cast<size_t>(ent.byte_len), ent.abs_offset)) return false;
    } else if (ent.dtype == ST_F16) {
        std::vector<uint16_t> tmp(static_cast<size_t>(elems));
        if (!pread_exact(ent.shard_fd, tmp.data(), static_cast<size_t>(ent.byte_len), ent.abs_offset)) return false;
        for (size_t i = 0; i < tmp.size(); ++i) out[i] = fp16_to_f32_fast(tmp[i]);
    } else if (ent.dtype == ST_BF16) {
        std::vector<uint16_t> tmp(static_cast<size_t>(elems));
        if (!pread_exact(ent.shard_fd, tmp.data(), static_cast<size_t>(ent.byte_len), ent.abs_offset)) return false;
        for (size_t i = 0; i < tmp.size(); ++i) out[i] = bf16_to_f32_scalar(tmp[i]);
    } else {
        return false;
    }
    for (float v : out) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

static bool load_inkling2_wo_lora(
    const std::string& lora_dir,
    float user_scale,
    const StrataInkling2Header& src_hdr,
    Inkling2WoLoraAdapter& out
) {
    out = Inkling2WoLoraAdapter{};
    std::string cfg_raw;
    if (!read_whole_file(lora_dir + "/adapter_config.json", cfg_raw)) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: failed reading '%s/adapter_config.json'\n", lora_dir.c_str());
        return false;
    }
    SmallJson cfg_json;
    if (!parse_small_json(cfg_raw, cfg_json) || cfg_json.type != SmallJson::J_OBJ) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: invalid JSON in '%s/adapter_config.json'\n", lora_dir.c_str());
        return false;
    }
    double r_val = 0.0, alpha_val = 0.0;
    const SmallJson* rj = cfg_json.get("r");
    const SmallJson* aj = cfg_json.get("lora_alpha");
    if (!rj || rj->type != SmallJson::J_NUM || rj->d_val < 1.0) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: invalid LoRA rank `r` in adapter_config.json\n");
        return false;
    }
    r_val     = rj->d_val;
    alpha_val = (aj && aj->type == SmallJson::J_NUM) ? aj->d_val : r_val;
    const uint32_t rank = static_cast<uint32_t>(std::llround(r_val));
    if (rank == 0u || rank > 256u) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: unsupported LoRA rank %u\n", rank);
        return false;
    }
    const float mult = (user_scale > 0.0f) ? user_scale : 1.0f;
    out.rank            = rank;
    out.effective_scale = static_cast<float>(alpha_val / r_val) * mult;

    const std::string st_path = lora_dir + "/adapter_model.safetensors";
    int lfd = open(st_path.c_str(), O_RDONLY);
    if (lfd < 0) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: cannot open '%s' (%s)\n",
                     st_path.c_str(), std::strerror(errno));
        return false;
    }
    struct stat lst;
    if (fstat(lfd, &lst) != 0 || lst.st_size <= 8) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: invalid file size on '%s'\n", st_path.c_str());
        close(lfd);
        return false;
    }
    std::unordered_map<std::string, SafetensorEntry> lmap;
    if (!parse_safetensors_file(lfd, static_cast<uint64_t>(lst.st_size), false, lmap, nullptr)) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: failed parsing '%s'\n", st_path.c_str());
        close(lfd);
        return false;
    }

    bool has_a[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
    bool has_b[STRATA_INKLING2_MAX_PHYS_LAYERS]{};
    for (const auto& kv : lmap) {
        const SafetensorEntry& ent = kv.second;
        const std::string& raw = ent.raw_name;
        const size_t lp = raw.find("layers.");
        if (lp == std::string::npos) continue;
        const size_t num_start = lp + 7u;
        size_t num_end = num_start;
        while (num_end < raw.size() && raw[num_end] >= '0' && raw[num_end] <= '9') ++num_end;
        if (num_end == num_start || num_end >= raw.size() || raw[num_end] != '.') continue;
        const unsigned long lidx = std::strtoul(raw.substr(num_start, num_end - num_start).c_str(), nullptr, 10);
        if (lidx >= src_hdr.cfg.num_phys_layers) {
            std::fprintf(stderr, "[inkling2-repack] ERROR: LoRA layer index %lu out of range\n", lidx);
            close(lfd);
            return false;
        }
        const std::string suffix = raw.substr(num_end + 1u);
        const StrataInkling2TensorSlice& wo_sl = src_hdr.layers[lidx].wo;
        if (suffix == "attn.wo_ud.lora_A.weight" || suffix == "self_attn.o_proj.lora_A.weight") {
            if (ent.shape.size() != 2u || ent.shape[0] != rank || ent.shape[1] != wo_sl.shape1) {
                std::fprintf(stderr, "[inkling2-repack] ERROR: unexpected lora_A shape on '%s'\n", raw.c_str());
                close(lfd);
                return false;
            }
            if (!read_safetensor_as_f32(ent, out.a_mat[lidx])) {
                std::fprintf(stderr, "[inkling2-repack] ERROR: failed reading '%s'\n", raw.c_str());
                close(lfd);
                return false;
            }
            has_a[lidx] = true;
            out.cols[lidx] = wo_sl.shape1;
        } else if (suffix == "attn.wo_ud.lora_B.weight" || suffix == "self_attn.o_proj.lora_B.weight") {
            if (ent.shape.size() != 2u || ent.shape[0] != wo_sl.shape0 || ent.shape[1] != rank) {
                std::fprintf(stderr, "[inkling2-repack] ERROR: unexpected lora_B shape on '%s'\n", raw.c_str());
                close(lfd);
                return false;
            }
            if (!read_safetensor_as_f32(ent, out.b_mat[lidx])) {
                std::fprintf(stderr, "[inkling2-repack] ERROR: failed reading '%s'\n", raw.c_str());
                close(lfd);
                return false;
            }
            has_b[lidx] = true;
            out.rows[lidx] = wo_sl.shape0;
        } else {
            std::fprintf(stderr, "[inkling2-repack] ERROR: unsupported LoRA target tensor '%s' (only attn.wo_ud supported)\n",
                         raw.c_str());
            close(lfd);
            return false;
        }
    }
    close(lfd);

    for (uint32_t l = 0; l < src_hdr.cfg.num_phys_layers; ++l) {
        if (has_a[l] != has_b[l]) {
            std::fprintf(stderr, "[inkling2-repack] ERROR: mismatched lora_A/lora_B pair on layer %u\n", l);
            return false;
        }
        if (has_a[l]) {
            out.has_layer[l] = true;
            ++out.num_layers;
        }
    }
    if (out.num_layers == 0u) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: no attn.wo_ud LoRA tensors found in '%s'\n", st_path.c_str());
        return false;
    }
    out.loaded = true;
    return true;
}

struct RepackTask {
    RepackTaskKind kind       = RTASK_COPY_RAW;
    uint64_t       src_off    = 0;
    uint64_t       src_off2   = 0; /* MXFP8 scales_off */
    uint64_t       dst_off    = 0;
    uint64_t       dst_off2   = 0; /* scales_off */
    uint64_t       dst_off3   = 0; /* biases_off (Q4_A) or expert_scale_off (NVFP4) */
    uint64_t       bytes      = 0; /* raw copy byte count */
    uint32_t       dim0       = 0; /* rows in chunk or experts in chunk */
    uint32_t       dim1       = 0; /* cols or rows_per_expert */
    uint32_t       dim2       = 0; /* cols (for MXFP8->NVFP4) */
    uint32_t       row_start  = 0; /* starting row offset within parent 2D matrix */
    int32_t        lora_layer = -1;/* >= 0 iff `wo` LoRA delta applies to this task */
    uint64_t       pad_w      = 0; /* trailing zero-pad bytes after quants (last chunk) */
    uint64_t       pad_s      = 0; /* trailing zero-pad bytes after scales (last chunk) */
    std::string    name;
};

static void enqueue_raw_copy(
    uint64_t src_off,
    uint64_t dst_off,
    uint64_t total_bytes,
    const std::string& name,
    std::vector<RepackTask>& tasks
) {
    constexpr uint64_t kMaxCopyChunk = 8ULL * 1024ULL * 1024ULL;
    uint64_t done = 0ULL;
    while (done < total_bytes) {
        const uint64_t n = std::min<uint64_t>(kMaxCopyChunk, total_bytes - done);
        RepackTask t;
        t.kind    = RTASK_COPY_RAW;
        t.src_off = src_off + done;
        t.dst_off = dst_off + done;
        t.bytes   = n;
        t.name    = name;
        tasks.push_back(std::move(t));
        done += n;
    }
}

static bool enqueue_repack_slice(
    const StrataInkling2TensorSlice& ss,
    const StrataInkling2TensorSlice& ds,
    const std::string& name,
    std::vector<RepackTask>& tasks
) {
    if (ss.dtype == STRATA_INKLING2_DTYPE_NONE && ds.dtype == STRATA_INKLING2_DTYPE_NONE) return true;
    if (ss.shape0 != ds.shape0 || ss.shape1 != ds.shape1) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: shape mismatch on '%s'\n", name.c_str());
        return false;
    }
    if (ss.dtype == ds.dtype) {
        if (ss.bytes != ds.bytes) return false;
        enqueue_raw_copy(ss.offset, ds.offset, ss.bytes, name, tasks);
        return true;
    }
    if (ss.dtype != STRATA_INKLING2_DTYPE_F16) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: source '%s' must be FP16 (got %s)\n",
                     name.c_str(), dtype_name(ss.dtype));
        return false;
    }

    const uint32_t rows = ds.shape0;
    const uint32_t cols = ds.shape1;
    if ((cols % STRATA_INKLING2_QUANT_GROUP_SIZE) != 0u) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: cols=%u not divisible by 64 on '%s'\n",
                     cols, name.c_str());
        return false;
    }
    const uint32_t grps = cols / STRATA_INKLING2_QUANT_GROUP_SIZE;
    constexpr uint32_t kRowsPerTask = 4096u;

    if (ds.dtype == STRATA_INKLING2_DTYPE_Q8_0) {
        uint64_t wb = 0ULL, sb = 0ULL;
        strata_inkling2_q8_layout(rows, cols, &wb, &sb);
        const uint64_t raw_w = static_cast<uint64_t>(rows) * cols;
        const uint64_t raw_s = static_cast<uint64_t>(rows) * grps * 2ULL;
        for (uint32_t r0 = 0; r0 < rows; r0 += kRowsPerTask) {
            const uint32_t cnt = std::min<uint32_t>(kRowsPerTask, rows - r0);
            const bool is_last = (r0 + cnt == rows);
            RepackTask t;
            t.kind      = RTASK_F16_TO_Q8;
            t.src_off   = ss.offset + static_cast<uint64_t>(r0) * cols * 2ULL;
            t.dst_off   = ds.offset + static_cast<uint64_t>(r0) * cols;
            t.dst_off2  = ds.offset + wb + static_cast<uint64_t>(r0) * grps * 2ULL;
            t.dim0      = cnt;
            t.dim1      = cols;
            t.row_start = r0;
            t.pad_w     = is_last ? (wb - raw_w) : 0ULL;
            t.pad_s     = is_last ? (sb - raw_s) : 0ULL;
            t.name      = name;
            tasks.push_back(std::move(t));
        }
        return true;
    }

    if (ds.dtype == STRATA_INKLING2_DTYPE_Q4_A) {
        uint64_t wb = 0ULL, sb = 0ULL, bb = 0ULL;
        strata_inkling2_q4a_layout(rows, cols, &wb, &sb, &bb);
        const uint64_t raw_w = static_cast<uint64_t>(rows) * (cols / 2u);
        const uint64_t raw_s = static_cast<uint64_t>(rows) * grps * 2ULL;
        for (uint32_t r0 = 0; r0 < rows; r0 += kRowsPerTask) {
            const uint32_t cnt = std::min<uint32_t>(kRowsPerTask, rows - r0);
            const bool is_last = (r0 + cnt == rows);
            RepackTask t;
            t.kind      = RTASK_F16_TO_Q4A;
            t.src_off   = ss.offset + static_cast<uint64_t>(r0) * cols * 2ULL;
            t.dst_off   = ds.offset + static_cast<uint64_t>(r0) * (cols / 2u);
            t.dst_off2  = ds.offset + wb + static_cast<uint64_t>(r0) * grps * 2ULL;
            t.dst_off3  = ds.offset + wb + sb + static_cast<uint64_t>(r0) * grps * 2ULL;
            t.dim0      = cnt;
            t.dim1      = cols;
            t.row_start = r0;
            t.pad_w     = is_last ? (wb - raw_w) : 0ULL;
            t.pad_s     = is_last ? (sb - raw_s) : 0ULL;
            t.name      = name;
            tasks.push_back(std::move(t));
        }
        return true;
    }

    return false;
}

static bool enqueue_repack_experts(
    const StrataInkling2ExpertMatrix& sm,
    const StrataInkling2ExpertMatrix& dm,
    const std::string& name,
    std::vector<RepackTask>& tasks
) {
    if (sm.num_experts != dm.num_experts || sm.rows_per_expert != dm.rows_per_expert ||
        sm.cols != dm.cols || dm.dtype != STRATA_INKLING2_DTYPE_NVFP4) {
        return false;
    }
    if (sm.dtype == STRATA_INKLING2_DTYPE_NVFP4) {
        if (sm.weight_bytes != dm.weight_bytes || sm.scales_bytes != dm.scales_bytes ||
            sm.expert_scale_bytes != dm.expert_scale_bytes) {
            return false;
        }
        enqueue_raw_copy(sm.weight_off,       dm.weight_off,       sm.weight_bytes,       name + ".w", tasks);
        enqueue_raw_copy(sm.scales_off,       dm.scales_off,       sm.scales_bytes,       name + ".s", tasks);
        enqueue_raw_copy(sm.expert_scale_off, dm.expert_scale_off, sm.expert_scale_bytes, name + ".es", tasks);
        return true;
    }
    if (sm.dtype == STRATA_INKLING2_DTYPE_MXFP8) {
        const uint64_t sw_exp = static_cast<uint64_t>(sm.rows_per_expert) * sm.cols;
        const uint64_t ss_exp = static_cast<uint64_t>(sm.rows_per_expert) * (sm.cols / 32u);
        const uint64_t dw_exp = static_cast<uint64_t>(dm.rows_per_expert) * (dm.cols / 2u);
        const uint64_t ds_exp = static_cast<uint64_t>(dm.rows_per_expert) * (dm.cols / 16u);
        constexpr uint32_t kExpPerTask = 4u;
        for (uint32_t e0 = 0; e0 < sm.num_experts; e0 += kExpPerTask) {
            const uint32_t cnt = std::min<uint32_t>(kExpPerTask, sm.num_experts - e0);
            RepackTask t;
            t.kind     = RTASK_MXFP8_TO_NVFP4;
            t.src_off  = sm.weight_off + static_cast<uint64_t>(e0) * sw_exp;
            t.src_off2 = sm.scales_off + static_cast<uint64_t>(e0) * ss_exp;
            t.dst_off  = dm.weight_off + static_cast<uint64_t>(e0) * dw_exp;
            t.dst_off2 = dm.scales_off + static_cast<uint64_t>(e0) * ds_exp;
            t.dst_off3 = dm.expert_scale_off + static_cast<uint64_t>(e0) * sizeof(float);
            t.dim0     = cnt;
            t.dim1     = sm.rows_per_expert;
            t.dim2     = sm.cols;
            t.name     = name;
            tasks.push_back(std::move(t));
        }
        return true;
    }
    return false;
}

static bool apply_wo_lora_to_fp16_chunk(
    const Inkling2WoLoraAdapter* lora,
    int32_t layer_idx,
    uint32_t row_start,
    uint32_t rows,
    uint32_t cols,
    uint16_t* dst_h
) {
    if (!lora || !lora->loaded || layer_idx < 0) return true;
    const uint32_t l = static_cast<uint32_t>(layer_idx);
    if (l >= STRATA_INKLING2_MAX_PHYS_LAYERS || !lora->has_layer[l]) return true;
    if (cols != lora->cols[l] || row_start + rows > lora->rows[l]) return false;

    const uint32_t rank = lora->rank;
    const float    sc   = lora->effective_scale;
    const float*   A    = lora->a_mat[l].data(); /* [rank, cols] */
    const float*   B    = lora->b_mat[l].data(); /* [rows_full, rank] */

    for (uint32_t r = 0; r < rows; ++r) {
        const float* b_row = B + static_cast<size_t>(row_start + r) * rank;
        uint16_t*    w_row = dst_h + static_cast<size_t>(r) * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            float delta = 0.0f;
            for (uint32_t k = 0; k < rank; ++k) {
                delta += b_row[k] * A[static_cast<size_t>(k) * cols + c];
            }
            const float v = fp16_to_f32_fast(w_row[c]) + sc * delta;
            if (!std::isfinite(v)) return false;
            w_row[c] = f32_to_fp16_fast(v);
        }
    }
    return true;
}

static bool execute_repack_task(
    int src_fd,
    int out_fd,
    const RepackTask& t,
    const Inkling2WoLoraAdapter* lora,
    std::vector<uint8_t>& buf_in,
    std::vector<uint8_t>& buf_in2,
    std::vector<uint8_t>& buf_w,
    std::vector<uint8_t>& buf_s,
    std::vector<uint8_t>& buf_b
) {
    switch (t.kind) {
        case RTASK_COPY_RAW: {
            const size_t n = static_cast<size_t>(t.bytes);
            if (buf_in.size() < n) buf_in.resize(n);
            if (!pread_exact(src_fd, buf_in.data(), n, t.src_off)) return false;
            return pwrite_exact(out_fd, buf_in.data(), n, t.dst_off);
        }

        case RTASK_F16_TO_Q8: {
            const uint32_t rows = t.dim0;
            const uint32_t cols = t.dim1;
            const uint32_t grps = cols / STRATA_INKLING2_QUANT_GROUP_SIZE;
            const size_t in_bytes = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
            const size_t w_bytes  = static_cast<size_t>(rows) * cols + static_cast<size_t>(t.pad_w);
            const size_t s_bytes  = static_cast<size_t>(rows) * grps * sizeof(uint16_t) + static_cast<size_t>(t.pad_s);

            if (buf_in.size() < in_bytes) buf_in.resize(in_bytes);
            if (buf_w.size()  < w_bytes)  buf_w.resize(w_bytes);
            if (buf_s.size()  < s_bytes)  buf_s.resize(s_bytes);
            if (!pread_exact(src_fd, buf_in.data(), in_bytes, t.src_off)) return false;

            uint16_t* src_h = reinterpret_cast<uint16_t*>(buf_in.data());
            if (!check_f16_span_finite(src_h, static_cast<size_t>(rows) * cols)) {
                std::fprintf(stderr, "[inkling2-repack] ERROR: NaN/Inf in FP16 source '%s'\n", t.name.c_str());
                return false;
            }
            if (!apply_wo_lora_to_fp16_chunk(lora, t.lora_layer, t.row_start, rows, cols, src_h)) {
                std::fprintf(stderr, "[inkling2-repack] ERROR: LoRA merge failed on '%s'\n", t.name.c_str());
                return false;
            }
            int8_t*   w_out = reinterpret_cast<int8_t*>(buf_w.data());
            uint16_t* s_out = reinterpret_cast<uint16_t*>(buf_s.data());
            for (uint32_t r = 0; r < rows; ++r) {
                const uint16_t* rh = src_h + static_cast<size_t>(r) * cols;
                int8_t*         rw = w_out + static_cast<size_t>(r) * cols;
                uint16_t*       rs = s_out + static_cast<size_t>(r) * grps;
                for (uint32_t g = 0; g < grps; ++g) {
                    quantize_group64_fp16_to_q8(rh + g * 64u, rw + g * 64u, rs + g);
                }
            }
            if (t.pad_w > 0ULL) std::memset(buf_w.data() + static_cast<size_t>(rows) * cols, 0, static_cast<size_t>(t.pad_w));
            if (t.pad_s > 0ULL) std::memset(buf_s.data() + static_cast<size_t>(rows) * grps * 2u, 0, static_cast<size_t>(t.pad_s));
            return pwrite_exact(out_fd, buf_w.data(), w_bytes, t.dst_off) &&
                   pwrite_exact(out_fd, buf_s.data(), s_bytes, t.dst_off2);
        }

        case RTASK_F16_TO_Q4A: {
            const uint32_t rows = t.dim0;
            const uint32_t cols = t.dim1;
            const uint32_t grps = cols / STRATA_INKLING2_QUANT_GROUP_SIZE;
            const uint32_t u32_per_row = cols / 8u;
            const size_t in_bytes = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
            const size_t w_raw    = static_cast<size_t>(rows) * (cols / 2u);
            const size_t s_raw    = static_cast<size_t>(rows) * grps * sizeof(uint16_t);
            const size_t w_bytes  = w_raw + static_cast<size_t>(t.pad_w);
            const size_t s_bytes  = s_raw + static_cast<size_t>(t.pad_s);

            if (buf_in.size() < in_bytes) buf_in.resize(in_bytes);
            if (buf_w.size()  < w_bytes)  buf_w.resize(w_bytes);
            if (buf_s.size()  < s_bytes)  buf_s.resize(s_bytes);
            if (buf_b.size()  < s_bytes)  buf_b.resize(s_bytes);
            if (!pread_exact(src_fd, buf_in.data(), in_bytes, t.src_off)) return false;

            uint16_t* src_h = reinterpret_cast<uint16_t*>(buf_in.data());
            if (!check_f16_span_finite(src_h, static_cast<size_t>(rows) * cols)) {
                std::fprintf(stderr, "[inkling2-repack] ERROR: NaN/Inf in FP16 source '%s'\n", t.name.c_str());
                return false;
            }
            if (!apply_wo_lora_to_fp16_chunk(lora, t.lora_layer, t.row_start, rows, cols, src_h)) {
                std::fprintf(stderr, "[inkling2-repack] ERROR: LoRA merge failed on '%s'\n", t.name.c_str());
                return false;
            }
            uint32_t* w_out = reinterpret_cast<uint32_t*>(buf_w.data());
            uint16_t* s_out = reinterpret_cast<uint16_t*>(buf_s.data());
            uint16_t* b_out = reinterpret_cast<uint16_t*>(buf_b.data());
            for (uint32_t r = 0; r < rows; ++r) {
                const uint16_t* rh = src_h + static_cast<size_t>(r) * cols;
                uint32_t*       rw = w_out + static_cast<size_t>(r) * u32_per_row;
                uint16_t*       rs = s_out + static_cast<size_t>(r) * grps;
                uint16_t*       rb = b_out + static_cast<size_t>(r) * grps;
                for (uint32_t g = 0; g < grps; ++g) {
                    quantize_group64_fp16_to_q4a(rh + g * 64u, rw + g * 8u, rs + g, rb + g);
                }
            }
            if (t.pad_w > 0ULL) std::memset(buf_w.data() + w_raw, 0, static_cast<size_t>(t.pad_w));
            if (t.pad_s > 0ULL) {
                std::memset(buf_s.data() + s_raw, 0, static_cast<size_t>(t.pad_s));
                std::memset(buf_b.data() + s_raw, 0, static_cast<size_t>(t.pad_s));
            }
            return pwrite_exact(out_fd, buf_w.data(), w_bytes, t.dst_off) &&
                   pwrite_exact(out_fd, buf_s.data(), s_bytes, t.dst_off2) &&
                   pwrite_exact(out_fd, buf_b.data(), s_bytes, t.dst_off3);
        }

        case RTASK_MXFP8_TO_NVFP4: {
            const uint32_t cnt  = t.dim0;
            const uint32_t rows = t.dim1;
            const uint32_t cols = t.dim2;
            const size_t sw_exp = static_cast<size_t>(rows) * cols;
            const size_t ss_exp = static_cast<size_t>(rows) * (cols / 32u);
            const size_t dw_exp = static_cast<size_t>(rows) * (cols / 2u);
            const size_t ds_exp = static_cast<size_t>(rows) * (cols / 16u);

            if (buf_in.size()  < cnt * sw_exp) buf_in.resize(cnt * sw_exp);
            if (buf_in2.size() < cnt * ss_exp) buf_in2.resize(cnt * ss_exp);
            if (buf_w.size()   < cnt * dw_exp) buf_w.resize(cnt * dw_exp);
            if (buf_s.size()   < cnt * ds_exp) buf_s.resize(cnt * ds_exp);
            if (!pread_exact(src_fd, buf_in.data(),  cnt * sw_exp, t.src_off) ||
                !pread_exact(src_fd, buf_in2.data(), cnt * ss_exp, t.src_off2)) {
                return false;
            }

            std::vector<float> outer_scales(cnt, 0.0f);
            for (uint32_t e = 0; e < cnt; ++e) {
                if (!repack_mxfp8_expert_to_nvfp4(
                        buf_in.data()  + e * sw_exp,
                        buf_in2.data() + e * ss_exp,
                        rows, cols,
                        buf_w.data()   + e * dw_exp,
                        buf_s.data()   + e * ds_exp,
                        &outer_scales[e])) {
                    std::fprintf(stderr, "[inkling2-repack] ERROR: MXFP8->NVFP4 NaN/invalid on '%s'\n",
                                 t.name.c_str());
                    return false;
                }
            }
            return pwrite_exact(out_fd, buf_w.data(),        cnt * dw_exp,               t.dst_off) &&
                   pwrite_exact(out_fd, buf_s.data(),        cnt * ds_exp,               t.dst_off2) &&
                   pwrite_exact(out_fd, outer_scales.data(), cnt * sizeof(float),        t.dst_off3);
        }
    }
    return false;
}

static bool build_frspec_tables(
    int src_fd,
    int out_fd,
    const StrataInkling2Header& src_hdr,
    const StrataInkling2Header& dst_hdr,
    uint32_t num_threads
) {
    const StrataInkling2FrSpecDesc& fr = dst_hdr.mtp.frspec;
    if (!dst_hdr.mtp.enabled || !fr.enabled) return true;

    const uint32_t V_unpad = src_hdr.cfg.vocab_size_unpadded;
    const uint32_t H       = src_hdr.cfg.hidden_size;
    const uint32_t MR      = src_hdr.mtp.markov_rank;
    const uint32_t S       = fr.shortlist_size;

    if (S == 0u || S > V_unpad || (H % 64u) != 0u || (MR % 64u) != 0u) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: invalid FR-Spec dims S=%u V=%u H=%u MR=%u\n",
                     S, V_unpad, H, MR);
        return false;
    }

    /* 1. Compute weight-intrinsic `||markov_w1[v]||_2` shortlist:
     *    60 special tokens `[199998..200057]` FIRST + top `S - 60` base tokens `[0..199997]`. */
    const uint32_t n_special   = (V_unpad >= 60u && S >= 60u) ? 60u : 0u;
    const uint32_t k_base_end  = V_unpad - n_special; /* 199998 for V_unpad=200058 */
    const uint32_t n_base_pick = S - n_special;       /* 32708 for S=32768 */

    std::vector<uint32_t> token_ids(S, 0u);
    {
        const size_t w1_elems = static_cast<size_t>(k_base_end) * MR;
        std::vector<uint16_t> w1_f16(w1_elems);
        if (!pread_exact(src_fd, w1_f16.data(), w1_elems * sizeof(uint16_t), src_hdr.mtp.markov_w1.offset)) {
            std::fprintf(stderr, "[inkling2-repack] ERROR: failed reading markov_w1\n");
            return false;
        }
        if (!check_f16_span_finite(w1_f16.data(), w1_elems)) {
            std::fprintf(stderr, "[inkling2-repack] ERROR: NaN/Inf in markov_w1\n");
            return false;
        }

        std::vector<double> norm2(k_base_end, 0.0);
        std::atomic<uint32_t> next_v{0};
        auto norm_worker = [&]() {
            constexpr uint32_t kChunk = 4096u;
            while (true) {
                uint32_t v0 = next_v.fetch_add(kChunk, std::memory_order_relaxed);
                if (v0 >= k_base_end) break;
                uint32_t v1 = std::min<uint32_t>(k_base_end, v0 + kChunk);
                for (uint32_t v = v0; v < v1; ++v) {
                    const uint16_t* row = w1_f16.data() + static_cast<size_t>(v) * MR;
                    double acc = 0.0;
                    for (uint32_t k = 0; k < MR; ++k) {
                        const double x = static_cast<double>(fp16_to_f32_fast(row[k]));
                        acc += x * x;
                    }
                    norm2[v] = acc;
                }
            }
        };
        std::vector<std::thread> ths;
        for (uint32_t t = 1; t < num_threads; ++t) ths.emplace_back(norm_worker);
        norm_worker();
        for (auto& th : ths) if (th.joinable()) th.join();

        std::vector<uint32_t> base_ids(k_base_end);
        for (uint32_t v = 0; v < k_base_end; ++v) base_ids[v] = v;
        std::partial_sort(
            base_ids.begin(),
            base_ids.begin() + static_cast<std::ptrdiff_t>(n_base_pick),
            base_ids.end(),
            [&](uint32_t a, uint32_t b) {
                if (norm2[a] != norm2[b]) return norm2[a] > norm2[b];
                return a < b;
            }
        );

        for (uint32_t i = 0; i < n_special; ++i) token_ids[i] = k_base_end + i;
        for (uint32_t i = 0; i < n_base_pick; ++i) token_ids[n_special + i] = base_ids[i];
    }

    if (!pwrite_exact(out_fd, token_ids.data(), static_cast<size_t>(S) * sizeof(uint32_t), fr.token_ids.offset)) {
        return false;
    }

    /* 2. Sequentially stream FP16 `unembed` in 8K-row chunks and gather shortlisted `U_short_f16 [S, H]` */
    std::vector<int32_t> vocab_to_short(V_unpad, -1);
    for (uint32_t s = 0; s < S; ++s) {
        vocab_to_short[token_ids[s]] = static_cast<int32_t>(s);
    }

    std::vector<uint16_t> u_short_f16(static_cast<size_t>(S) * H);
    {
        constexpr uint32_t kStreamRows = 8192u;
        std::vector<uint16_t> chunk_buf(static_cast<size_t>(kStreamRows) * H);
        const size_t row_bytes = static_cast<size_t>(H) * sizeof(uint16_t);
        for (uint32_t v0 = 0; v0 < V_unpad; v0 += kStreamRows) {
            const uint32_t cnt = std::min<uint32_t>(kStreamRows, V_unpad - v0);
            if (!pread_exact(src_fd, chunk_buf.data(), static_cast<size_t>(cnt) * row_bytes,
                             src_hdr.unembed.offset + static_cast<uint64_t>(v0) * row_bytes)) {
                std::fprintf(stderr, "[inkling2-repack] ERROR: failed streaming unembed for FR-Spec\n");
                return false;
            }
            for (uint32_t i = 0; i < cnt; ++i) {
                const int32_t si = vocab_to_short[v0 + i];
                if (si >= 0) {
                    std::memcpy(u_short_f16.data() + static_cast<size_t>(si) * H,
                                chunk_buf.data() + static_cast<size_t>(i) * H,
                                row_bytes);
                }
            }
        }
    }
    if (!check_f16_span_finite(u_short_f16.data(), u_short_f16.size())) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: NaN/Inf in shortlisted unembed\n");
        return false;
    }

    /* 3. Load `markov_hidden_out [H, MR]` (FP16) & `markov_hidden_bias [H]` (F32) into F64 */
    const uint32_t mr_blks = MR / 8u;
    std::vector<double> h_packed(static_cast<size_t>(mr_blks) * H * 8u);
    std::vector<double> bias_f64(H);
    {
        std::vector<uint16_t> ho_f16(static_cast<size_t>(H) * MR);
        std::vector<float>    hb_f32(H);
        if (!pread_exact(src_fd, ho_f16.data(), ho_f16.size() * sizeof(uint16_t), src_hdr.mtp.markov_hidden_out.offset) ||
            !pread_exact(src_fd, hb_f32.data(), hb_f32.size() * sizeof(float),    src_hdr.mtp.markov_hidden_bias.offset)) {
            std::fprintf(stderr, "[inkling2-repack] ERROR: failed reading markov hidden_out/bias\n");
            return false;
        }
        if (!check_f16_span_finite(ho_f16.data(), ho_f16.size()) ||
            !check_f32_span_finite(hb_f32.data(), hb_f32.size())) {
            std::fprintf(stderr, "[inkling2-repack] ERROR: NaN/Inf in markov hidden_out/bias\n");
            return false;
        }
        for (uint32_t d = 0; d < H; ++d) {
            bias_f64[d] = static_cast<double>(hb_f32[d]);
        }
        for (uint32_t rb = 0; rb < mr_blks; ++rb) {
            for (uint32_t d = 0; d < H; ++d) {
                double* dst8 = h_packed.data() + (static_cast<size_t>(rb) * H + d) * 8u;
                const uint16_t* src8 = ho_f16.data() + static_cast<size_t>(d) * MR + rb * 8u;
                for (uint32_t j = 0; j < 8u; ++j) {
                    dst8[j] = static_cast<double>(fp16_to_f32_fast(src8[j]));
                }
            }
        }
    }

    /* 4. Allocate output arrays & run multi-threaded F64 fold + Q4_A / Q8_0 quantization */
    uint64_t u_wb = 0ULL, u_sb = 0ULL, u_bb = 0ULL;
    strata_inkling2_q4a_layout(S, H, &u_wb, &u_sb, &u_bb);
    uint64_t f_wb = 0ULL, f_sb = 0ULL;
    strata_inkling2_q8_layout(S, MR, &f_wb, &f_sb);

    std::vector<uint8_t> u_q4_w(static_cast<size_t>(u_wb), 0u);
    std::vector<uint8_t> u_q4_s(static_cast<size_t>(u_sb), 0u);
    std::vector<uint8_t> u_q4_b(static_cast<size_t>(u_bb), 0u);
    std::vector<uint8_t> f_q8_w(static_cast<size_t>(f_wb), 0u);
    std::vector<uint8_t> f_q8_s(static_cast<size_t>(f_sb), 0u);
    std::vector<float>   f_bias(S, 0.0f);

    const uint32_t u_u32_per_row = H / 8u;
    const uint32_t u_grp_per_row = H / STRATA_INKLING2_QUANT_GROUP_SIZE;
    const uint32_t f_grp_per_row = MR / STRATA_INKLING2_QUANT_GROUP_SIZE;

    std::atomic<uint32_t> next_row{0};
    std::atomic<bool>     any_bad{false};

    auto fold_worker = [&]() {
        std::vector<double> u4(static_cast<size_t>(4u) * H);
        std::vector<double> out4(static_cast<size_t>(4u) * MR);
        std::vector<float>  frow(MR);

        uint32_t* u_w_ptr = reinterpret_cast<uint32_t*>(u_q4_w.data());
        uint16_t* u_s_ptr = reinterpret_cast<uint16_t*>(u_q4_s.data());
        uint16_t* u_b_ptr = reinterpret_cast<uint16_t*>(u_q4_b.data());
        int8_t*   f_w_ptr = reinterpret_cast<int8_t*>(f_q8_w.data());
        uint16_t* f_s_ptr = reinterpret_cast<uint16_t*>(f_q8_s.data());

        constexpr uint32_t kRowChunk = 64u;
        while (!any_bad.load(std::memory_order_relaxed)) {
            uint32_t s0 = next_row.fetch_add(kRowChunk, std::memory_order_relaxed);
            if (s0 >= S) break;
            uint32_t s1 = std::min<uint32_t>(S, s0 + kRowChunk);

            for (uint32_t s = s0; s < s1; s += 4u) {
                const uint32_t br = std::min<uint32_t>(4u, s1 - s);

                /* Quantize shortlisted unembed rows -> Q4_A, unpack to F64, and compute F64 fold_bias */
                for (uint32_t bi = 0; bi < br; ++bi) {
                    const uint32_t row_idx = s + bi;
                    const uint16_t* rh = u_short_f16.data() + static_cast<size_t>(row_idx) * H;
                    uint32_t* rw = u_w_ptr + static_cast<size_t>(row_idx) * u_u32_per_row;
                    uint16_t* rs = u_s_ptr + static_cast<size_t>(row_idx) * u_grp_per_row;
                    uint16_t* rb = u_b_ptr + static_cast<size_t>(row_idx) * u_grp_per_row;
                    for (uint32_t g = 0; g < u_grp_per_row; ++g) {
                        quantize_group64_fp16_to_q4a(rh + g * 64u, rw + g * 8u, rs + g, rb + g);
                    }

                    double* ud = u4.data() + static_cast<size_t>(bi) * H;
                    double b_acc = 0.0;
                    for (uint32_t d = 0; d < H; ++d) {
                        const double v = static_cast<double>(fp16_to_f32_fast(rh[d]));
                        ud[d] = v;
                        b_acc += v * bias_f64[d];
                    }
                    const float bf = static_cast<float>(b_acc);
                    if (!std::isfinite(bf)) {
                        any_bad.store(true, std::memory_order_relaxed);
                        return;
                    }
                    f_bias[row_idx] = bf;
                }

                /* Exact F64 GEMM: `out4[0..br-1, 0..MR-1] = u4[0..br-1, 0..H-1] @ hidden_out[H, MR]` */
#if defined(__aarch64__)
                if (br == 4u) {
                    const double* u_r0 = u4.data() + static_cast<size_t>(0) * H;
                    const double* u_r1 = u4.data() + static_cast<size_t>(1) * H;
                    const double* u_r2 = u4.data() + static_cast<size_t>(2) * H;
                    const double* u_r3 = u4.data() + static_cast<size_t>(3) * H;
                    for (uint32_t rb = 0; rb < mr_blks; ++rb) {
                        float64x2_t c00 = vdupq_n_f64(0.0), c01 = vdupq_n_f64(0.0), c02 = vdupq_n_f64(0.0), c03 = vdupq_n_f64(0.0);
                        float64x2_t c10 = vdupq_n_f64(0.0), c11 = vdupq_n_f64(0.0), c12 = vdupq_n_f64(0.0), c13 = vdupq_n_f64(0.0);
                        float64x2_t c20 = vdupq_n_f64(0.0), c21 = vdupq_n_f64(0.0), c22 = vdupq_n_f64(0.0), c23 = vdupq_n_f64(0.0);
                        float64x2_t c30 = vdupq_n_f64(0.0), c31 = vdupq_n_f64(0.0), c32 = vdupq_n_f64(0.0), c33 = vdupq_n_f64(0.0);

                        const double* hp = h_packed.data() + static_cast<size_t>(rb) * H * 8u;
                        for (uint32_t d = 0; d < H; ++d) {
                            const float64x2_t b0 = vld1q_f64(hp + d * 8u + 0u);
                            const float64x2_t b1 = vld1q_f64(hp + d * 8u + 2u);
                            const float64x2_t b2 = vld1q_f64(hp + d * 8u + 4u);
                            const float64x2_t b3 = vld1q_f64(hp + d * 8u + 6u);

                            const double a0 = u_r0[d];
                            c00 = vfmaq_n_f64(c00, b0, a0); c01 = vfmaq_n_f64(c01, b1, a0);
                            c02 = vfmaq_n_f64(c02, b2, a0); c03 = vfmaq_n_f64(c03, b3, a0);

                            const double a1 = u_r1[d];
                            c10 = vfmaq_n_f64(c10, b0, a1); c11 = vfmaq_n_f64(c11, b1, a1);
                            c12 = vfmaq_n_f64(c12, b2, a1); c13 = vfmaq_n_f64(c13, b3, a1);

                            const double a2 = u_r2[d];
                            c20 = vfmaq_n_f64(c20, b0, a2); c21 = vfmaq_n_f64(c21, b1, a2);
                            c22 = vfmaq_n_f64(c22, b2, a2); c23 = vfmaq_n_f64(c23, b3, a2);

                            const double a3 = u_r3[d];
                            c30 = vfmaq_n_f64(c30, b0, a3); c31 = vfmaq_n_f64(c31, b1, a3);
                            c32 = vfmaq_n_f64(c32, b2, a3); c33 = vfmaq_n_f64(c33, b3, a3);
                        }
                        double* o0 = out4.data() + static_cast<size_t>(0) * MR + rb * 8u;
                        double* o1 = out4.data() + static_cast<size_t>(1) * MR + rb * 8u;
                        double* o2 = out4.data() + static_cast<size_t>(2) * MR + rb * 8u;
                        double* o3 = out4.data() + static_cast<size_t>(3) * MR + rb * 8u;
                        vst1q_f64(o0 + 0u, c00); vst1q_f64(o0 + 2u, c01); vst1q_f64(o0 + 4u, c02); vst1q_f64(o0 + 6u, c03);
                        vst1q_f64(o1 + 0u, c10); vst1q_f64(o1 + 2u, c11); vst1q_f64(o1 + 4u, c12); vst1q_f64(o1 + 6u, c13);
                        vst1q_f64(o2 + 0u, c20); vst1q_f64(o2 + 2u, c21); vst1q_f64(o2 + 4u, c22); vst1q_f64(o2 + 6u, c23);
                        vst1q_f64(o3 + 0u, c30); vst1q_f64(o3 + 2u, c31); vst1q_f64(o3 + 4u, c32); vst1q_f64(o3 + 6u, c33);
                    }
                } else
#endif
                {
                    std::fill(out4.begin(), out4.begin() + static_cast<size_t>(br) * MR, 0.0);
                    for (uint32_t bi = 0; bi < br; ++bi) {
                        const double* ud = u4.data() + static_cast<size_t>(bi) * H;
                        double* od = out4.data() + static_cast<size_t>(bi) * MR;
                        for (uint32_t rb = 0; rb < mr_blks; ++rb) {
                            const double* hp = h_packed.data() + static_cast<size_t>(rb) * H * 8u;
                            for (uint32_t d = 0; d < H; ++d) {
                                const double a = ud[d];
                                for (uint32_t j = 0; j < 8u; ++j) {
                                    od[rb * 8u + j] += a * hp[d * 8u + j];
                                }
                            }
                        }
                    }
                }

                /* Quantize folded F64 rows -> Q8_0 (gs=64) */
                for (uint32_t bi = 0; bi < br; ++bi) {
                    const uint32_t row_idx = s + bi;
                    const double* od = out4.data() + static_cast<size_t>(bi) * MR;
                    for (uint32_t k = 0; k < MR; ++k) {
                        const float fv = static_cast<float>(od[k]);
                        if (!std::isfinite(fv)) {
                            any_bad.store(true, std::memory_order_relaxed);
                            return;
                        }
                        frow[k] = fv;
                    }
                    int8_t*   fw = f_w_ptr + static_cast<size_t>(row_idx) * MR;
                    uint16_t* fs = f_s_ptr + static_cast<size_t>(row_idx) * f_grp_per_row;
                    for (uint32_t g = 0; g < f_grp_per_row; ++g) {
                        quantize_group64_f32_to_q8(frow.data() + g * 64u, fw + g * 64u, fs + g);
                    }
                }
            }
        }
    };

    std::vector<std::thread> ths;
    for (uint32_t t = 1; t < num_threads; ++t) ths.emplace_back(fold_worker);
    fold_worker();
    for (auto& th : ths) if (th.joinable()) th.join();

    if (any_bad.load(std::memory_order_relaxed)) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: NaN/Inf encountered during FR-Spec F64 fold\n");
        return false;
    }

    return pwrite_exact(out_fd, u_q4_w.data(), u_q4_w.size(), fr.unembed_q4a.offset) &&
           pwrite_exact(out_fd, u_q4_s.data(), u_q4_s.size(), fr.unembed_q4a.offset + u_wb) &&
           pwrite_exact(out_fd, u_q4_b.data(), u_q4_b.size(), fr.unembed_q4a.offset + u_wb + u_sb) &&
           pwrite_exact(out_fd, f_q8_w.data(), f_q8_w.size(), fr.fold_weight_q8.offset) &&
           pwrite_exact(out_fd, f_q8_s.data(), f_q8_s.size(), fr.fold_weight_q8.offset + f_wb) &&
           pwrite_exact(out_fd, f_bias.data(), f_bias.size() * sizeof(float), fr.fold_bias_f32.offset);
}

static std::string json_escape_str(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8U);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20U) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

static std::string format_manifest_source_json(const std::string& raw, bool force_local_basename) {
    const bool is_url = (raw.rfind("http://", 0) == 0 || raw.rfind("https://", 0) == 0);
    bool strip_to_basename = force_local_basename;
    if (!strip_to_basename && !is_url) {
        if (!raw.empty() && (raw[0] == '/' || raw[0] == '~' || raw[0] == '.')) {
            strip_to_basename = true;
        } else {
            size_t slashes = 0U;
            for (char c : raw) {
                if (c == '/') ++slashes;
            }
            if (slashes > 1U) strip_to_basename = true;
        }
    }
    std::string label = raw;
    if (strip_to_basename) {
        while (!label.empty() && label.back() == '/') label.pop_back();
        const size_t pos = label.find_last_of('/');
        if (pos != std::string::npos) label = label.substr(pos + 1U);
        if (label.empty() || label == "." || label == "..") label = "local";
    }
    return json_escape_str(label);
}

} /* anonymous namespace */

/* ============================================================================
 * 7. Public C ABI Implementations (`include/strata_inkling2.h`)
 * ============================================================================ */

extern "C" {

int strata_inkling2_convert_hf(const StrataInkling2ConvertConfig* cfg_in) {
    auto wall_t0 = std::chrono::steady_clock::now();

    const std::string hf_dir = (cfg_in && cfg_in->hf_dir && cfg_in->hf_dir[0] != '\0')
                                   ? std::string(cfg_in->hf_dir)
                                   : std::string(STRATA_INKLING2_DEFAULT_HF_DIR);
    const std::string out_dir = (cfg_in && cfg_in->bundle_dir && cfg_in->bundle_dir[0] != '\0')
                                    ? std::string(cfg_in->bundle_dir)
                                    : std::string(STRATA_INKLING2_DEFAULT_BUNDLE_DIR);
    uint32_t num_threads = (cfg_in && cfg_in->num_threads > 0u) ? cfg_in->num_threads : 8u;
    if (num_threads > 18u) num_threads = 18u;

    if (!mkdirs(out_dir)) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: cannot create bundle dir '%s'\n", out_dir.c_str());
        return -1;
    }

    /* 1. Parse `config.json` (+ optional `generation_config.json`) */
    auto hdr_ptr = std::make_unique<StrataInkling2Header>();
    std::memset(hdr_ptr.get(), 0, sizeof(StrataInkling2Header));
    StrataInkling2Header& hdr = *hdr_ptr;

    uint32_t phys_is_local[STRATA_INKLING2_MAX_PHYS_LAYERS] = {};
    uint32_t mtp_is_local[STRATA_INKLING2_MAX_MTP_LAYERS]   = {};
    bool use_gate_bias = true;
    if (!parse_inkling2_configs(hf_dir, hdr.cfg, hdr.mtp, phys_is_local, mtp_is_local, &use_gate_bias)) {
        return -1;
    }

    /* 2. Discover backbone safetensors shard(s) and optional `mtp.safetensors` */
    std::vector<std::string> shard_paths;
    {
        std::string idx_raw;
        if (read_whole_file(hf_dir + "/model.safetensors.index.json", idx_raw)) {
            SmallJson idx_json;
            if (!parse_small_json(idx_raw, idx_json) || idx_json.type != SmallJson::J_OBJ) {
                std::fprintf(stderr, "[inkling2-conv] ERROR: invalid model.safetensors.index.json\n");
                return -1;
            }
            if (const SmallJson* wm = idx_json.get("weight_map"); wm && wm->type == SmallJson::J_OBJ) {
                auto is_safe_shard_basename = [](const std::string& name) -> bool {
                    if (name.empty() || name.size() > 255U || name == "." || name == ".." || name[0] == '-') return false;
                    for (unsigned char c : name) {
                        if (c == '/' || c == '\\' || c == '[' || c == ']' || c == '{' || c == '}' ||
                            c < 0x20U || c == 0x7FU) return false;
                    }
                    return true;
                };
                std::set<std::string> uniq;
                for (const auto& kv : wm->obj) {
                    if (kv.second.type == SmallJson::J_STR && !kv.second.s_val.empty()) {
                        if (!is_safe_shard_basename(kv.second.s_val)) {
                            std::fprintf(stderr, "[inkling2-conv] ERROR: unsafe shard filename '%s' in index\n",
                                         kv.second.s_val.c_str());
                            return -1;
                        }
                        uniq.insert(kv.second.s_val);
                    }
                }
                for (const auto& s : uniq) {
                    shard_paths.push_back(hf_dir + "/" + s);
                }
            }
        }
        if (shard_paths.empty()) {
            if (file_exists_nonempty(hf_dir + "/model-00001-of-00001.safetensors")) {
                shard_paths.push_back(hf_dir + "/model-00001-of-00001.safetensors");
            } else if (file_exists_nonempty(hf_dir + "/model.safetensors")) {
                shard_paths.push_back(hf_dir + "/model.safetensors");
            } else {
                std::fprintf(stderr, "[inkling2-conv] ERROR: no backbone .safetensors found in '%s'\n",
                             hf_dir.c_str());
                return -1;
            }
        }
    }

    const std::string mtp_path = hf_dir + "/mtp.safetensors";
    uint64_t mtp_file_bytes = 0ULL;
    const bool has_mtp = file_exists_nonempty(mtp_path, &mtp_file_bytes);

    std::vector<int> open_fds;
    auto close_all_fds = [&]() {
        for (int fd : open_fds) {
            if (fd >= 0) close(fd);
        }
        open_fds.clear();
    };

    std::unordered_map<std::string, SafetensorEntry> backbone_map;
    std::unordered_map<std::string, SafetensorEntry> mtp_map;
    uint32_t vision_skipped = 0u;
    uint64_t total_hf_bytes = 0ULL;

    for (const std::string& sp : shard_paths) {
        uint64_t sz = 0ULL;
        if (!file_exists_nonempty(sp, &sz)) {
            std::fprintf(stderr, "[inkling2-conv] ERROR: missing shard '%s'\n", sp.c_str());
            close_all_fds();
            return -1;
        }
        int fd = open(sp.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "[inkling2-conv] ERROR: cannot open '%s'\n", sp.c_str());
            close_all_fds();
            return -1;
        }
#if defined(F_NOCACHE)
        (void)fcntl(fd, F_NOCACHE, 1);
#endif
        open_fds.push_back(fd);
        total_hf_bytes += sz;
        if (!parse_safetensors_file(fd, sz, false, backbone_map, &vision_skipped)) {
            std::fprintf(stderr, "[inkling2-conv] ERROR: corrupt safetensors header in '%s'\n", sp.c_str());
            close_all_fds();
            return -1;
        }
    }

    if (has_mtp) {
        int mfd = open(mtp_path.c_str(), O_RDONLY);
        if (mfd < 0) {
            std::fprintf(stderr, "[inkling2-conv] ERROR: cannot open '%s'\n", mtp_path.c_str());
            close_all_fds();
            return -1;
        }
#if defined(F_NOCACHE)
        (void)fcntl(mfd, F_NOCACHE, 1);
#endif
        open_fds.push_back(mfd);
        total_hf_bytes += mtp_file_bytes;
        if (!parse_safetensors_file(mfd, mtp_file_bytes, true, mtp_map, nullptr)) {
            std::fprintf(stderr, "[inkling2-conv] ERROR: corrupt safetensors header in '%s'\n", mtp_path.c_str());
            close_all_fds();
            return -1;
        }
    }

    /* 3. Detect per-layer expert quantization dtype (NVFP4 iff `w13_weight_amax` exists, else MXFP8) */
    uint32_t layer_expert_dtype[STRATA_INKLING2_MAX_PHYS_LAYERS] = {};
    uint32_t num_nvfp4_layers = 0u;
    uint32_t num_mxfp8_layers = 0u;
    for (uint32_t i = 0; i < hdr.cfg.num_phys_layers; ++i) {
        std::string amax_key = "model.layers." + std::to_string(i) + ".mlp.experts.w13_weight_amax";
        if (backbone_map.find(amax_key) != backbone_map.end()) {
            layer_expert_dtype[i] = STRATA_INKLING2_DTYPE_NVFP4;
            ++num_nvfp4_layers;
        } else {
            layer_expert_dtype[i] = STRATA_INKLING2_DTYPE_MXFP8;
            ++num_mxfp8_layers;
        }
    }

    /* 4. Plan deterministic 16 KiB-aligned UMA layout */
    plan_uma_layout(hdr, phys_is_local, layer_expert_dtype, use_gate_bias, has_mtp, mtp_is_local);

    /* 5. Build declarative conversion task list covering 100% of backbone + MTP tensors */
    std::vector<ConvertTask> tasks;
    tasks.reserve(4096);

    const StrataInkling2Config& c = hdr.cfg;
    const uint32_t H     = c.hidden_size;
    const uint32_t NH    = c.num_attention_heads;
    const uint32_t NKV   = c.num_key_value_heads;
    const uint32_t HD    = c.head_dim;
    const uint32_t DR    = c.d_rel;
    const uint32_t KS    = c.sconv_kernel_size;
    const uint32_t E     = c.num_experts;
    const uint32_t QDIM  = NH * HD;
    const uint32_t KVDIM = NKV * HD;

    bool plan_ok = true;
    plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, "model.embed.weight", hdr.embed_tokens, {c.vocab_size_padded, H}, false, tasks);
    plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, "model.embed_norm.weight", hdr.embed_norm, {H}, false, tasks);
    plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, "model.norm.weight", hdr.final_norm, {H}, false, tasks);
    plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, "model.attn_res.output_query", hdr.output_res_query, {H}, false, tasks);
    plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, "model.unembed.weight", hdr.unembed, {c.vocab_size_padded, H}, true, tasks);

    if (c.loop_num_passes > 1u) {
        const uint32_t n_cp = 2u * (c.loop_core_end - c.loop_core_start);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, "model.middle_cycle_cross_pass_queries", hdr.cross_pass_queries, {n_cp, H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, "model.middle_cycle_cross_pass_kv_queries", hdr.cross_pass_kv_queries, {n_cp, H}, false, tasks);
    }

    static const char* kCtrlModNames[4] = {"attn", "k", "v", "mlp"};

    for (uint32_t i = 0; i < c.num_phys_layers && plan_ok; ++i) {
        const StrataInkling2LayerDesc& ld = hdr.layers[i];
        const std::string lp = "model.layers." + std::to_string(i) + ".";
        const bool is_local  = (ld.is_local_swa != 0u);
        const uint32_t extent = is_local ? c.sliding_window_size : c.rel_extent;

        if (i > 0u) {
            const std::string aq_name = "model.attn_res.attn_queries." + std::to_string(i - 1u);
            if (is_local) {
                plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, aq_name, ld.attn_res_query, {H}, false, tasks);
            } else {
                plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, aq_name, ld.attn_res_query, {3u, H}, false, tasks);
            }
        }
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, "model.attn_res.mlp_queries." + std::to_string(i), ld.mlp_res_query, {H}, false, tasks);

        if (is_local) {
            plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn_norm.weight", ld.attn_norm, {H}, false, tasks);
        } else {
            plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "q_input_norm.weight", ld.q_input_norm, {H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "k_input_norm.weight", ld.k_input_norm, {H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "v_input_norm.weight", ld.v_input_norm, {H}, false, tasks);
        }
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "mlp_norm.weight", ld.mlp_norm, {H}, false, tasks);

        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.wq_du.weight", ld.wq, {QDIM, H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.wk_dv.weight", ld.wk, {KVDIM, H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.wv_dv.weight", ld.wv, {KVDIM, H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.wo_ud.weight", ld.wo, {H, QDIM}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.gate_attn.weight", ld.gate_attn, {QDIM, H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.wr_du.weight", ld.wr, {NH * DR, H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.attn_sink_proj.weight", ld.attn_sink_proj, {NH, H}, false, tasks);

        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.q_l2_clip.scale", ld.q_l2_clip_scale, {NH, HD}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.rel_l2_clip.scale", ld.rel_l2_clip_scale, {NH, DR}, false, tasks);
        plan_ok = plan_ok && enqueue_transpose_2d_f32(backbone_map, lp + "attn.rel_logits_proj.proj", ld.rel_logits_proj, DR, extent, tasks);

        if (is_local) {
            plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.k_sconv.weight", ld.k_sconv_w, {KVDIM, 1u, KS}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.v_sconv.weight", ld.v_sconv_w, {KVDIM, 1u, KS}, false, tasks);
        } else {
            /* Global `static_w` [4, 1024] -> transpose to F32 [1024, 4] preserving LAG-FIRST order (DO NOT flip!) */
            plan_ok = plan_ok && enqueue_transpose_2d_f32(backbone_map, lp + "attn.k_sconv.static_w", ld.k_sconv_w, KS, KVDIM, tasks);
            plan_ok = plan_ok && enqueue_transpose_2d_f32(backbone_map, lp + "attn.v_sconv.static_w", ld.v_sconv_w, KS, KVDIM, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.k_sconv.weight_proj", ld.k_sconv_proj, {NKV * KS, H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn.v_sconv.weight_proj", ld.v_sconv_proj, {NKV * KS, H}, false, tasks);
        }
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "attn_sconv.weight", ld.attn_sconv_w, {H, 1u, KS}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "mlp_sconv.weight", ld.mlp_sconv_w, {H, 1u, KS}, false, tasks);

        plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "mlp.gate.weight", ld.moe_gate_weight, {E, H}, false, tasks);
        if (use_gate_bias) {
            plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, lp + "mlp.gate.bias", ld.moe_gate_bias, {E}, false, tasks);
        }
        plan_ok = plan_ok && enqueue_expert_matrix_tasks(backbone_map, lp + "mlp.experts.w13", ld.experts_w13, true, tasks);
        plan_ok = plan_ok && enqueue_expert_matrix_tasks(backbone_map, lp + "mlp.experts.w2",  ld.experts_w2,  false, tasks);

        if (ld.in_loop_core) {
            for (uint32_t m = 0; m < 4u && plan_ok; ++m) {
                const std::string cp = lp + "controllers." + kCtrlModNames[m] + ".";
                plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, cp + "down_weight",   ld.controllers[m].down_weight,   {c.loop_gate_rank, 2u * H}, false, tasks);
                plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, cp + "gain_weight",   ld.controllers[m].gain_weight,   {H, c.loop_gate_rank}, false, tasks);
                plan_ok = plan_ok && enqueue_slice_tasks(backbone_map, cp + "forget_weight", ld.controllers[m].forget_weight, {H, c.loop_gate_rank}, false, tasks);
            }
        }
    }

    if (has_mtp && plan_ok) {
        const StrataInkling2MtpDesc& mtp = hdr.mtp;
        const uint32_t MR = mtp.markov_rank;
        const uint32_t MI = mtp.intermediate_size;

        /* `fuse_weight` [4, 4] -> row-wise softmax F32 [4, 4] */
        auto it_fw = mtp_map.find("fuse_weight");
        if (it_fw == mtp_map.end() ||
            !check_shape_eq(it_fw->second.shape, {mtp.num_layers + 1u, STRATA_INKLING2_MAX_MTP_TAPS})) {
            std::fprintf(stderr, "[inkling2-conv] ERROR: missing or invalid MTP 'fuse_weight'\n");
            plan_ok = false;
        } else {
            it_fw->second.used = true;
            ConvertTask tf;
            tf.kind       = TASK_FUSE_SOFTMAX_F32;
            tf.src_fd     = it_fw->second.shard_fd;
            tf.src_dtype  = it_fw->second.dtype;
            tf.src_off    = it_fw->second.abs_offset;
            tf.dst_off    = mtp.fuse_alpha.offset;
            tf.elem_count = mtp.fuse_alpha.elem_count;
            tf.dim0       = mtp.num_layers + 1u;
            tf.dim1       = STRATA_INKLING2_MAX_MTP_TAPS;
            tf.debug_name = "mtp.fuse_weight";
            tasks.push_back(std::move(tf));
        }

        plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, "mask_embed", mtp.mask_embed, {H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, "norm.weight", mtp.final_norm, {H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, "markov_head.dflare_norm.weight", mtp.markov_dflare_norm, {H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, "markov_head.markov_w1.weight", mtp.markov_w1, {c.vocab_size_padded, MR}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, "markov_head.rnn_proj.weight", mtp.markov_rnn_weight, {3u * MR, 2u * MR + 2u * H}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, "markov_head.rnn_proj.bias", mtp.markov_rnn_bias, {3u * MR}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, "markov_head.hidden_out.weight", mtp.markov_hidden_out, {H, MR}, false, tasks);
        plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, "markov_head.hidden_out.bias", mtp.markov_hidden_bias, {H}, false, tasks);

        for (uint32_t l = 0; l < mtp.num_layers && plan_ok; ++l) {
            const StrataInkling2MtpLayerDesc& ml = mtp.layers[l];
            const std::string mp = "layers." + std::to_string(l) + ".";
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "dflare_norm.weight", ml.dflare_norm, {H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "input_layernorm.weight", ml.input_layernorm, {H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "post_attention_layernorm.weight", ml.post_attn_norm, {H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "self_attn.q_norm.weight", ml.q_norm, {HD}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "self_attn.k_norm.weight", ml.k_norm, {HD}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "self_attn.k_norm_ctx.weight", ml.k_norm_ctx, {HD}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "self_attn.q_proj.weight", ml.q_proj, {QDIM, H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "self_attn.k_proj.weight", ml.k_proj, {KVDIM, H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "self_attn.v_proj.weight", ml.v_proj, {KVDIM, H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "self_attn.k_proj_ctx.weight", ml.k_proj_ctx, {KVDIM, H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "self_attn.v_proj_ctx.weight", ml.v_proj_ctx, {KVDIM, H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "self_attn.o_proj.weight", ml.o_proj, {H, QDIM}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "mlp.gate_proj.weight", ml.gate_proj, {MI, H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "mlp.up_proj.weight", ml.up_proj, {MI, H}, false, tasks);
            plan_ok = plan_ok && enqueue_slice_tasks(mtp_map, mp + "mlp.down_proj.weight", ml.down_proj, {H, MI}, false, tasks);
        }
    }

    if (!plan_ok) {
        close_all_fds();
        return -1;
    }

    /* Check for any unexpected leftover non-visual tensors */
    for (const auto& kv : backbone_map) {
        if (!kv.second.used) {
            std::fprintf(stderr, "[inkling2-conv] WARNING: unused backbone tensor '%s'\n", kv.first.c_str());
        }
    }
    for (const auto& kv : mtp_map) {
        if (!kv.second.used) {
            std::fprintf(stderr, "[inkling2-conv] WARNING: unused MTP tensor '%s'\n", kv.first.c_str());
        }
    }

    /* Sort tasks by (src_fd, src_off) for sequential NVMe read locality */
    std::sort(tasks.begin(), tasks.end(), [](const ConvertTask& a, const ConvertTask& b) {
        if (a.src_fd != b.src_fd) return a.src_fd < b.src_fd;
        return a.src_off < b.src_off;
    });

    /* 6. Create & preallocate atomic temp output `inkling2_uma.strata.tmp` */
    const std::string uma_path     = out_dir + "/" + STRATA_INKLING2_UMA_FILENAME;
    const std::string uma_tmp_path = uma_path + ".tmp";
    unlink(uma_tmp_path.c_str());

    int out_fd = open(uma_tmp_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (out_fd < 0) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: cannot open '%s' (%s)\n",
                     uma_tmp_path.c_str(), std::strerror(errno));
        close_all_fds();
        return -1;
    }
#if defined(F_NOCACHE)
    (void)fcntl(out_fd, F_NOCACHE, 1);
#endif
    try_preallocate_apfs(out_fd, hdr.total_file_bytes);
    if (ftruncate(out_fd, static_cast<off_t>(hdr.total_file_bytes)) != 0) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: ftruncate(%llu) failed (%s)\n",
                     static_cast<unsigned long long>(hdr.total_file_bytes), std::strerror(errno));
        close(out_fd);
        unlink(uma_tmp_path.c_str());
        close_all_fds();
        return -1;
    }

    std::fprintf(stdout,
                 "[inkling2-conv] Converting '%s' -> '%s'\n"
                 "  Layout: %u phys / %u logical (%u SWA + %u Global) | Experts: %u NVFP4 + %u MXFP8 | MTP: %s\n"
                 "  Target: %.3f GiB (%llu B, 16 KiB aligned) | Tasks: %zu | Threads: %u\n",
                 hf_dir.c_str(), out_dir.c_str(),
                 hdr.cfg.num_phys_layers, hdr.cfg.num_logical_layers,
                 hdr.cfg.num_local_visits, hdr.cfg.num_global_visits,
                 num_nvfp4_layers, num_mxfp8_layers,
                 has_mtp ? "ENABLED (3L DSpark Block-8)" : "DISABLED",
                 static_cast<double>(hdr.total_file_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<unsigned long long>(hdr.total_file_bytes),
                 tasks.size(), num_threads);
    std::fflush(stdout);

    /* 7. Execute all conversion tasks concurrently across worker threads */
    std::atomic<size_t> next_task{0};
    std::atomic<bool>   any_failed{false};

    auto worker_fn = [&]() {
        std::vector<uint8_t> src_buf;
        std::vector<uint8_t> dst_buf;
        src_buf.reserve(8ULL * 1024ULL * 1024ULL);
        dst_buf.reserve(8ULL * 1024ULL * 1024ULL);

        while (!any_failed.load(std::memory_order_relaxed)) {
            size_t idx = next_task.fetch_add(1, std::memory_order_relaxed);
            if (idx >= tasks.size()) break;
            if (!execute_convert_task(out_fd, tasks[idx], src_buf, dst_buf)) {
                any_failed.store(true, std::memory_order_relaxed);
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    const uint32_t spawned = (num_threads > 1u) ? (num_threads - 1u) : 0u;
    threads.reserve(spawned);
    for (uint32_t t = 0; t < spawned; ++t) {
        threads.emplace_back(worker_fn);
    }
    worker_fn();
    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }

    close_all_fds();

    if (any_failed.load(std::memory_order_relaxed)) {
        close(out_fd);
        unlink(uma_tmp_path.c_str());
        return -1;
    }

    /* 8. Zero 128 KiB header block, write header at offset 0, fsync, and atomic rename */
    if (!pwrite_zeros(out_fd, 0, STRATA_INKLING2_HEADER_PADDED_BYTES) ||
        !pwrite_exact(out_fd, &hdr, sizeof(hdr), 0) ||
        fsync(out_fd) != 0) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: failed writing header / fsync\n");
        close(out_fd);
        unlink(uma_tmp_path.c_str());
        return -1;
    }
    close(out_fd);

    if (rename(uma_tmp_path.c_str(), uma_path.c_str()) != 0) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: atomic rename failed (%s)\n", std::strerror(errno));
        unlink(uma_tmp_path.c_str());
        return -1;
    }

    /* 9. Copy tokenizer & config files to bundle directory */
    if (!copy_file_if_exists(hf_dir + "/config.json", out_dir + "/config.json", true) ||
        !copy_file_if_exists(hf_dir + "/tokenizer.json", out_dir + "/tokenizer.json", true)) {
        std::fprintf(stderr, "[inkling2-conv] ERROR: failed copying config.json / tokenizer.json\n");
        return -1;
    }
    (void)copy_file_if_exists(hf_dir + "/tokenizer_config.json", out_dir + "/tokenizer_config.json", false);
    (void)copy_file_if_exists(hf_dir + "/chat_template.jinja", out_dir + "/chat_template.jinja", false);
    (void)copy_file_if_exists(hf_dir + "/generation_config.json", out_dir + "/generation_config.json", false);
    (void)copy_file_if_exists(hf_dir + "/special_tokens_map.json", out_dir + "/special_tokens_map.json", false);

    const double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_t0).count();
    const double peak_rss = current_peak_rss_gib();

    /* Write human-readable manifest.json */
    {
        FILE* mf = std::fopen((out_dir + "/manifest.json").c_str(), "wb");
        if (mf) {
            const std::string safe_src = format_manifest_source_json(hf_dir, true);
            std::fprintf(mf,
                "{\n"
                "  \"format\": \"STRATA_INKLING2_V1\",\n"
                "  \"architecture\": \"Inkling2LoopForConditionalGeneration\",\n"
                "  \"source\": \"%s\",\n"
                "  \"num_phys_layers\": %u,\n"
                "  \"num_logical_layers\": %u,\n"
                "  \"num_local_visits\": %u,\n"
                "  \"num_global_visits\": %u,\n"
                "  \"hidden_size\": %u,\n"
                "  \"vocab_size_padded\": %u,\n"
                "  \"vocab_size_unpadded\": %u,\n"
                "  \"num_nvfp4_layers\": %u,\n"
                "  \"num_mxfp8_layers\": %u,\n"
                "  \"mtp_enabled\": %s,\n"
                "  \"hf_safetensors_bytes\": %llu,\n"
                "  \"uma_file_bytes\": %llu,\n"
                "  \"active_bytes_per_tok\": %llu,\n"
                "  \"mtp_active_bytes_per_blk\": %llu,\n"
                "  \"backbone_tensors_converted\": %zu,\n"
                "  \"mtp_tensors_converted\": %zu,\n"
                "  \"vision_tensors_skipped\": %u,\n"
                "  \"elapsed_sec\": %.3f,\n"
                "  \"peak_rss_gib\": %.4f\n"
                "}\n",
                safe_src.c_str(),
                hdr.cfg.num_phys_layers, hdr.cfg.num_logical_layers,
                hdr.cfg.num_local_visits, hdr.cfg.num_global_visits,
                hdr.cfg.hidden_size,
                hdr.cfg.vocab_size_padded, hdr.cfg.vocab_size_unpadded,
                num_nvfp4_layers, num_mxfp8_layers,
                has_mtp ? "true" : "false",
                static_cast<unsigned long long>(total_hf_bytes),
                static_cast<unsigned long long>(hdr.total_file_bytes),
                static_cast<unsigned long long>(hdr.active_bytes_per_tok),
                static_cast<unsigned long long>(hdr.mtp_active_bytes_per_blk),
                backbone_map.size(), mtp_map.size(), vision_skipped,
                elapsed_sec, peak_rss);
            std::fclose(mf);
        }
    }

    std::fprintf(stdout,
                 "[inkling2-conv] COMPLETE in %.2f s (%.2f GB/s): uma=%.3f GiB | active=%.3f GB/tok | "
                 "mtp_blk=%.3f GB | backbone=%zu + mtp=%zu (vision skipped=%u) | Peak RSS=%.3f GiB\n",
                 elapsed_sec,
                 (elapsed_sec > 1e-6) ? (static_cast<double>(hdr.total_file_bytes) / (1e9 * elapsed_sec)) : 0.0,
                 static_cast<double>(hdr.total_file_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(hdr.active_bytes_per_tok) / 1e9,
                 static_cast<double>(hdr.mtp_active_bytes_per_blk) / 1e9,
                 backbone_map.size(), mtp_map.size(), vision_skipped, peak_rss);
    std::fflush(stdout);
    return 0;
}

int strata_inkling2_repack_fast(const StrataInkling2RepackConfig* cfg_in) {
    auto wall_t0 = std::chrono::steady_clock::now();

    const std::string src_dir = (cfg_in && cfg_in->src_bundle_dir && cfg_in->src_bundle_dir[0] != '\0')
                                    ? std::string(cfg_in->src_bundle_dir)
                                    : std::string(STRATA_INKLING2_DEFAULT_BUNDLE_DIR);
    const std::string dst_dir = (cfg_in && cfg_in->dst_bundle_dir && cfg_in->dst_bundle_dir[0] != '\0')
                                    ? std::string(cfg_in->dst_bundle_dir)
                                    : std::string(STRATA_INKLING2_DEFAULT_FAST_BUNDLE_DIR);
    const bool q4_unembed = cfg_in ? cfg_in->q4_unembed : false;
    uint32_t mtp_lin_dt   = STRATA_INKLING2_DTYPE_Q4_A; /* Turbo default: 4-bit Q4_A 3-layer DSpark drafter */
    if (cfg_in) {
        if (cfg_in->mtp_dtype == STRATA_INKLING2_DTYPE_Q8_0) {
            mtp_lin_dt = STRATA_INKLING2_DTYPE_Q8_0;
        } else if (cfg_in->mtp_dtype == STRATA_INKLING2_DTYPE_F16) {
            mtp_lin_dt = STRATA_INKLING2_DTYPE_F16;
        } else if (cfg_in->mtp_dtype == STRATA_INKLING2_DTYPE_Q4_A ||
                   cfg_in->mtp_dtype == STRATA_INKLING2_DTYPE_NONE) {
            mtp_lin_dt = STRATA_INKLING2_DTYPE_Q4_A;
        } else {
            std::fprintf(stderr,
                         "[inkling2-repack] ERROR: unsupported mtp_dtype=%u (expected NONE/Q4_A/Q8_0/F16)\n",
                         cfg_in->mtp_dtype);
            return -2;
        }
    }
    uint32_t num_threads  = (cfg_in && cfg_in->num_threads > 0u) ? cfg_in->num_threads : 8u;
    if (num_threads > 18u) num_threads = 18u;

    const std::string src_uma_path = src_dir + "/" + STRATA_INKLING2_UMA_FILENAME;
    const std::string dst_uma_path = dst_dir + "/" + STRATA_INKLING2_UMA_FILENAME;

    if (paths_refer_to_same_entry(src_dir, dst_dir) ||
        paths_refer_to_same_entry(src_uma_path, dst_uma_path)) {
        std::fprintf(stderr,
                     "[inkling2-repack] ERROR: src and dst directories must be distinct ('%s' == '%s')\n",
                     src_dir.c_str(), dst_dir.c_str());
        return -2;
    }

    int src_fd = open(src_uma_path.c_str(), O_RDONLY);
    if (src_fd < 0) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: cannot open source '%s' (%s)\n",
                     src_uma_path.c_str(), std::strerror(errno));
        return -1;
    }
#if defined(F_NOCACHE)
    (void)fcntl(src_fd, F_NOCACHE, 1);
#endif

    struct stat src_st;
    if (fstat(src_fd, &src_st) != 0 ||
        static_cast<uint64_t>(src_st.st_size) < STRATA_INKLING2_HEADER_PADDED_BYTES ||
        (static_cast<uint64_t>(src_st.st_size) % STRATA_INKLING2_PAGE_BYTES) != 0ULL) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: invalid source UMA file size on '%s'\n",
                     src_uma_path.c_str());
        close(src_fd);
        return -1;
    }
    const uint64_t src_file_bytes = static_cast<uint64_t>(src_st.st_size);

    auto src_hdr_ptr = std::make_unique<StrataInkling2Header>();
    if (!pread_exact(src_fd, src_hdr_ptr.get(), sizeof(StrataInkling2Header), 0)) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: failed reading source header\n");
        close(src_fd);
        return -1;
    }
    StrataInkling2Header& src_hdr = *src_hdr_ptr;
    if (src_hdr.version == STRATA_INKLING2_FORMAT_VERSION_V1) {
        std::memset(&src_hdr.mtp.frspec, 0, sizeof(src_hdr.mtp.frspec));
    }
    if (!validate_header(src_hdr, src_file_bytes)) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: source header validation failed on '%s'\n",
                     src_uma_path.c_str());
        close(src_fd);
        return -1;
    }
    if (src_hdr.unembed.dtype != STRATA_INKLING2_DTYPE_F16 ||
        src_hdr.embed_tokens.dtype != STRATA_INKLING2_DTYPE_F16) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: source bundle must be FP16 base bundle (got embed=%s unembed=%s)\n",
                     dtype_name(src_hdr.embed_tokens.dtype), dtype_name(src_hdr.unembed.dtype));
        close(src_fd);
        return -1;
    }

    if (!mkdirs(dst_dir)) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: cannot create dst dir '%s'\n", dst_dir.c_str());
        close(src_fd);
        return -1;
    }

    Inkling2WoLoraAdapter wo_lora{};
    const std::string lora_dir = (cfg_in && cfg_in->lora_dir && cfg_in->lora_dir[0] != '\0')
        ? std::string(cfg_in->lora_dir) : std::string();
    const float lora_scale = (cfg_in && cfg_in->lora_scale > 0.0f) ? cfg_in->lora_scale : 1.0f;
    if (!lora_dir.empty()) {
        if (!load_inkling2_wo_lora(lora_dir, lora_scale, src_hdr, wo_lora)) {
            close(src_fd);
            return -1;
        }
        std::fprintf(stdout,
                     "[inkling2-repack] Loaded wo LoRA '%s': %u layers, rank=%u, effective_scale=%.4f\n",
                     lora_dir.c_str(), wo_lora.num_layers, wo_lora.rank, wo_lora.effective_scale);
    }

    auto dst_hdr_ptr = std::make_unique<StrataInkling2Header>();
    StrataInkling2Header& dst_hdr = *dst_hdr_ptr;
    plan_repack_layout(src_hdr, q4_unembed, mtp_lin_dt, dst_hdr);

    /* Enqueue all repack tasks covering 100% of backbone + MTP (FR-Spec runs in Phase 2) */
    std::vector<RepackTask> tasks;
    tasks.reserve(4096);
    bool plan_ok = true;

    plan_ok = plan_ok && enqueue_repack_slice(src_hdr.embed_tokens,          dst_hdr.embed_tokens,          "embed_tokens",          tasks);
    plan_ok = plan_ok && enqueue_repack_slice(src_hdr.embed_norm,            dst_hdr.embed_norm,            "embed_norm",            tasks);
    plan_ok = plan_ok && enqueue_repack_slice(src_hdr.final_norm,            dst_hdr.final_norm,            "final_norm",            tasks);
    plan_ok = plan_ok && enqueue_repack_slice(src_hdr.output_res_query,      dst_hdr.output_res_query,      "output_res_query",      tasks);
    plan_ok = plan_ok && enqueue_repack_slice(src_hdr.unembed,               dst_hdr.unembed,               "unembed",               tasks);
    plan_ok = plan_ok && enqueue_repack_slice(src_hdr.cross_pass_queries,    dst_hdr.cross_pass_queries,    "cross_pass_queries",    tasks);
    plan_ok = plan_ok && enqueue_repack_slice(src_hdr.cross_pass_kv_queries, dst_hdr.cross_pass_kv_queries, "cross_pass_kv_queries", tasks);

    for (uint32_t i = 0; i < src_hdr.cfg.num_phys_layers && plan_ok; ++i) {
        const StrataInkling2LayerDesc& sl = src_hdr.layers[i];
        const StrataInkling2LayerDesc& dl = dst_hdr.layers[i];
        const std::string lp = "L" + std::to_string(i) + ".";

        plan_ok = plan_ok && enqueue_repack_slice(sl.attn_res_query,    dl.attn_res_query,    lp + "attn_res_query",    tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.mlp_res_query,     dl.mlp_res_query,     lp + "mlp_res_query",     tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.attn_norm,         dl.attn_norm,         lp + "attn_norm",         tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.q_input_norm,      dl.q_input_norm,      lp + "q_input_norm",      tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.k_input_norm,      dl.k_input_norm,      lp + "k_input_norm",      tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.v_input_norm,      dl.v_input_norm,      lp + "v_input_norm",      tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.mlp_norm,          dl.mlp_norm,          lp + "mlp_norm",          tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.wq,                dl.wq,                lp + "wq",                tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.wk,                dl.wk,                lp + "wk",                tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.wv,                dl.wv,                lp + "wv",                tasks);
        {
            const size_t t_before = tasks.size();
            plan_ok = plan_ok && enqueue_repack_slice(sl.wo,            dl.wo,                lp + "wo",                tasks);
            if (plan_ok && wo_lora.loaded && wo_lora.has_layer[i]) {
                for (size_t ti = t_before; ti < tasks.size(); ++ti) {
                    tasks[ti].lora_layer = static_cast<int32_t>(i);
                }
            }
        }
        plan_ok = plan_ok && enqueue_repack_slice(sl.gate_attn,         dl.gate_attn,         lp + "gate_attn",         tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.wr,                dl.wr,                lp + "wr",                tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.attn_sink_proj,    dl.attn_sink_proj,    lp + "attn_sink_proj",    tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.q_l2_clip_scale,   dl.q_l2_clip_scale,   lp + "q_l2_clip_scale",   tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.rel_l2_clip_scale, dl.rel_l2_clip_scale, lp + "rel_l2_clip_scale", tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.rel_logits_proj,   dl.rel_logits_proj,   lp + "rel_logits_proj",   tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.k_sconv_w,         dl.k_sconv_w,         lp + "k_sconv_w",         tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.v_sconv_w,         dl.v_sconv_w,         lp + "v_sconv_w",         tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.k_sconv_proj,      dl.k_sconv_proj,      lp + "k_sconv_proj",      tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.v_sconv_proj,      dl.v_sconv_proj,      lp + "v_sconv_proj",      tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.attn_sconv_w,      dl.attn_sconv_w,      lp + "attn_sconv_w",      tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.mlp_sconv_w,       dl.mlp_sconv_w,       lp + "mlp_sconv_w",       tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.moe_gate_weight,   dl.moe_gate_weight,   lp + "moe_gate_weight",   tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sl.moe_gate_bias,     dl.moe_gate_bias,     lp + "moe_gate_bias",     tasks);
        plan_ok = plan_ok && enqueue_repack_experts(sl.experts_w13,     dl.experts_w13,       lp + "experts_w13",       tasks);
        plan_ok = plan_ok && enqueue_repack_experts(sl.experts_w2,      dl.experts_w2,        lp + "experts_w2",        tasks);

        if (sl.in_loop_core) {
            for (uint32_t m = 0; m < 4u && plan_ok; ++m) {
                const std::string cp = lp + "ctrl" + std::to_string(m) + ".";
                plan_ok = plan_ok && enqueue_repack_slice(sl.controllers[m].down_weight,   dl.controllers[m].down_weight,   cp + "down",   tasks);
                plan_ok = plan_ok && enqueue_repack_slice(sl.controllers[m].gain_weight,   dl.controllers[m].gain_weight,   cp + "gain",   tasks);
                plan_ok = plan_ok && enqueue_repack_slice(sl.controllers[m].forget_weight, dl.controllers[m].forget_weight, cp + "forget", tasks);
            }
        }
    }

    if (src_hdr.mtp.enabled && plan_ok) {
        const StrataInkling2MtpDesc& sm = src_hdr.mtp;
        const StrataInkling2MtpDesc& dm = dst_hdr.mtp;
        plan_ok = plan_ok && enqueue_repack_slice(sm.fuse_alpha,         dm.fuse_alpha,         "mtp.fuse_alpha",         tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sm.mask_embed,         dm.mask_embed,         "mtp.mask_embed",         tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sm.final_norm,         dm.final_norm,         "mtp.final_norm",         tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sm.markov_dflare_norm, dm.markov_dflare_norm, "mtp.markov_dflare_norm", tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sm.markov_w1,          dm.markov_w1,          "mtp.markov_w1",          tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sm.markov_rnn_weight,  dm.markov_rnn_weight,  "mtp.markov_rnn_weight",  tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sm.markov_rnn_bias,    dm.markov_rnn_bias,    "mtp.markov_rnn_bias",    tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sm.markov_hidden_out,  dm.markov_hidden_out,  "mtp.markov_hidden_out",  tasks);
        plan_ok = plan_ok && enqueue_repack_slice(sm.markov_hidden_bias, dm.markov_hidden_bias, "mtp.markov_hidden_bias", tasks);

        for (uint32_t l = 0; l < sm.num_layers && plan_ok; ++l) {
            const StrataInkling2MtpLayerDesc& sml = sm.layers[l];
            const StrataInkling2MtpLayerDesc& dml = dm.layers[l];
            const std::string mp = "mtp.L" + std::to_string(l) + ".";
            plan_ok = plan_ok && enqueue_repack_slice(sml.dflare_norm,     dml.dflare_norm,     mp + "dflare_norm",     tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.input_layernorm, dml.input_layernorm, mp + "input_layernorm", tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.post_attn_norm,  dml.post_attn_norm,  mp + "post_attn_norm",  tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.q_norm,          dml.q_norm,          mp + "q_norm",          tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.k_norm,          dml.k_norm,          mp + "k_norm",          tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.k_norm_ctx,      dml.k_norm_ctx,      mp + "k_norm_ctx",      tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.q_proj,          dml.q_proj,          mp + "q_proj",          tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.k_proj,          dml.k_proj,          mp + "k_proj",          tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.v_proj,          dml.v_proj,          mp + "v_proj",          tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.k_proj_ctx,      dml.k_proj_ctx,      mp + "k_proj_ctx",      tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.v_proj_ctx,      dml.v_proj_ctx,      mp + "v_proj_ctx",      tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.o_proj,          dml.o_proj,          mp + "o_proj",          tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.gate_proj,       dml.gate_proj,       mp + "gate_proj",       tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.up_proj,         dml.up_proj,         mp + "up_proj",         tasks);
            plan_ok = plan_ok && enqueue_repack_slice(sml.down_proj,       dml.down_proj,       mp + "down_proj",       tasks);
        }
    }

    if (!plan_ok) {
        close(src_fd);
        return -1;
    }

    std::sort(tasks.begin(), tasks.end(), [](const RepackTask& a, const RepackTask& b) {
        return a.src_off < b.src_off;
    });

    const std::string dst_tmp_path = dst_uma_path + ".tmp";
    unlink(dst_tmp_path.c_str());

    int out_fd = open(dst_tmp_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (out_fd < 0) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: cannot open '%s' (%s)\n",
                     dst_tmp_path.c_str(), std::strerror(errno));
        close(src_fd);
        return -1;
    }
#if defined(F_NOCACHE)
    (void)fcntl(out_fd, F_NOCACHE, 1);
#endif
    try_preallocate_apfs(out_fd, dst_hdr.total_file_bytes);
    if (ftruncate(out_fd, static_cast<off_t>(dst_hdr.total_file_bytes)) != 0) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: ftruncate(%llu) failed (%s)\n",
                     static_cast<unsigned long long>(dst_hdr.total_file_bytes), std::strerror(errno));
        close(out_fd);
        unlink(dst_tmp_path.c_str());
        close(src_fd);
        return -1;
    }

    std::fprintf(stdout,
                 "[inkling2-repack] Repacking '%s' -> '%s' (%s)\n"
                 "  Quant   : embed=Q4_A, attn=Q4_A, wr=Q8_0, unembed=%s, ctrl=Q8_0, L0..23=NVFP4, MTP=%s (rnn=%s), FR-Spec=%u\n"
                 "  Target  : %.3f GiB (from %.3f GiB) | Active: %.3f GB/tok Main, %.3f GB/blk MTP | Tasks: %zu | Threads: %u\n",
                 src_dir.c_str(), dst_dir.c_str(),
                 q4_unembed ? "Cfg C" : "Cfg B++",
                 dtype_name(dst_hdr.unembed.dtype),
                 dst_hdr.mtp.enabled ? dtype_name(dst_hdr.mtp.layers[0].q_proj.dtype) : "NONE",
                 dst_hdr.mtp.enabled ? dtype_name(dst_hdr.mtp.markov_rnn_weight.dtype) : "NONE",
                 dst_hdr.mtp.frspec.enabled ? dst_hdr.mtp.frspec.shortlist_size : 0u,
                 static_cast<double>(dst_hdr.total_file_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(src_file_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(dst_hdr.active_bytes_per_tok) / 1e9,
                 static_cast<double>(dst_hdr.mtp_active_bytes_per_blk) / 1e9,
                 tasks.size(), num_threads);
    std::fflush(stdout);

    /* Phase 1: Parallel backbone + experts + MTP repack */
    std::atomic<size_t> next_task{0};
    std::atomic<bool>   any_failed{false};

    auto worker_fn = [&]() {
        std::vector<uint8_t> buf_in, buf_in2, buf_w, buf_s, buf_b;
        buf_in.reserve(24ULL * 1024ULL * 1024ULL);
        buf_w.reserve(16ULL * 1024ULL * 1024ULL);
        buf_s.reserve(2ULL * 1024ULL * 1024ULL);
        buf_b.reserve(2ULL * 1024ULL * 1024ULL);

        while (!any_failed.load(std::memory_order_relaxed)) {
            size_t idx = next_task.fetch_add(1, std::memory_order_relaxed);
            if (idx >= tasks.size()) break;
            if (!execute_repack_task(src_fd, out_fd, tasks[idx], wo_lora.loaded ? &wo_lora : nullptr,
                                     buf_in, buf_in2, buf_w, buf_s, buf_b)) {
                any_failed.store(true, std::memory_order_relaxed);
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    const uint32_t spawned = (num_threads > 1u) ? (num_threads - 1u) : 0u;
    threads.reserve(spawned);
    for (uint32_t t = 0; t < spawned; ++t) threads.emplace_back(worker_fn);
    worker_fn();
    for (auto& th : threads) if (th.joinable()) th.join();

    /* Phase 2: Build 32K weight-intrinsic FR-Spec folded Markov tables */
    if (!any_failed.load(std::memory_order_relaxed)) {
        if (!build_frspec_tables(src_fd, out_fd, src_hdr, dst_hdr, num_threads)) {
            any_failed.store(true, std::memory_order_relaxed);
        }
    }

    close(src_fd);

    if (any_failed.load(std::memory_order_relaxed)) {
        close(out_fd);
        unlink(dst_tmp_path.c_str());
        return -1;
    }

    /* Validate header, write zeroed 128 KiB header block, fsync, atomic rename */
    if (!validate_header(dst_hdr, dst_hdr.total_file_bytes) ||
        !pwrite_zeros(out_fd, 0, STRATA_INKLING2_HEADER_PADDED_BYTES) ||
        !pwrite_exact(out_fd, &dst_hdr, sizeof(dst_hdr), 0) ||
        fsync(out_fd) != 0) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: failed writing header / fsync\n");
        close(out_fd);
        unlink(dst_tmp_path.c_str());
        return -1;
    }
    close(out_fd);

    if (rename(dst_tmp_path.c_str(), dst_uma_path.c_str()) != 0) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: atomic rename failed (%s)\n", std::strerror(errno));
        unlink(dst_tmp_path.c_str());
        return -1;
    }

    /* Copy tokenizer and configs from source bundle */
    if (!copy_file_if_exists(src_dir + "/tokenizer.json", dst_dir + "/tokenizer.json", false) ||
        !copy_file_if_exists(src_dir + "/config.json",    dst_dir + "/config.json",    false)) {
        std::fprintf(stderr, "[inkling2-repack] ERROR: failed copying tokenizer.json / config.json\n");
        return -1;
    }
    (void)copy_file_if_exists(src_dir + "/tokenizer_config.json",   dst_dir + "/tokenizer_config.json",   false);
    (void)copy_file_if_exists(src_dir + "/chat_template.jinja",     dst_dir + "/chat_template.jinja",     false);
    (void)copy_file_if_exists(src_dir + "/generation_config.json",  dst_dir + "/generation_config.json",  false);
    (void)copy_file_if_exists(src_dir + "/special_tokens_map.json", dst_dir + "/special_tokens_map.json", false);

    const double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_t0).count();
    const double peak_rss = current_peak_rss_gib();

    {
        FILE* mf = std::fopen((dst_dir + "/manifest.json").c_str(), "wb");
        if (mf) {
            const std::string safe_src = format_manifest_source_json(src_dir, true);
            std::fprintf(mf,
                "{\n"
                "  \"format\": \"STRATA_INKLING2_V2_FAST\",\n"
                "  \"profile\": \"%s\",\n"
                "  \"source\": \"%s\",\n"
                "  \"embed_dtype\": \"%s\",\n"
                "  \"attn_dtype\": \"%s\",\n"
                "  \"wr_dtype\": \"%s\",\n"
                "  \"unembed_dtype\": \"%s\",\n"
                "  \"ctrl_dtype\": \"%s\",\n"
                "  \"experts_dtype\": \"NVFP4\",\n"
                "  \"mtp_dtype\": \"%s\",\n"
                "  \"mtp_rnn_dtype\": \"%s\",\n"
                "  \"frspec_shortlist_size\": %u,\n"
                "  \"uma_file_bytes\": %llu,\n"
                "  \"active_bytes_per_tok\": %llu,\n"
                "  \"mtp_active_bytes_per_blk\": %llu,\n"
                "  \"elapsed_sec\": %.3f,\n"
                "  \"peak_rss_gib\": %.4f\n"
                "}\n",
                q4_unembed ? "Cfg_C" : "Cfg_B++",
                safe_src.c_str(),
                dtype_name(dst_hdr.embed_tokens.dtype),
                dtype_name(dst_hdr.layers[0].wq.dtype),
                dtype_name(dst_hdr.layers[0].wr.dtype),
                dtype_name(dst_hdr.unembed.dtype),
                (dst_hdr.cfg.loop_core_start < dst_hdr.cfg.num_phys_layers)
                    ? dtype_name(dst_hdr.layers[dst_hdr.cfg.loop_core_start].controllers[0].down_weight.dtype)
                    : "NONE",
                dst_hdr.mtp.enabled ? dtype_name(dst_hdr.mtp.layers[0].q_proj.dtype) : "NONE",
                dst_hdr.mtp.enabled ? dtype_name(dst_hdr.mtp.markov_rnn_weight.dtype) : "NONE",
                dst_hdr.mtp.frspec.enabled ? dst_hdr.mtp.frspec.shortlist_size : 0u,
                static_cast<unsigned long long>(dst_hdr.total_file_bytes),
                static_cast<unsigned long long>(dst_hdr.active_bytes_per_tok),
                static_cast<unsigned long long>(dst_hdr.mtp_active_bytes_per_blk),
                elapsed_sec, peak_rss);
            std::fclose(mf);
        }
    }

    std::fprintf(stdout,
                 "[inkling2-repack] COMPLETE in %.2f s (%.2f GB/s): uma=%.3f GiB | active=%.3f GB/tok | "
                 "mtp_blk=%.3f GB | FR-Spec=%u | Peak RSS=%.3f GiB\n",
                 elapsed_sec,
                 (elapsed_sec > 1e-6) ? (static_cast<double>(dst_hdr.total_file_bytes) / (1e9 * elapsed_sec)) : 0.0,
                 static_cast<double>(dst_hdr.total_file_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(dst_hdr.active_bytes_per_tok) / 1e9,
                 static_cast<double>(dst_hdr.mtp_active_bytes_per_blk) / 1e9,
                 dst_hdr.mtp.frspec.enabled ? dst_hdr.mtp.frspec.shortlist_size : 0u,
                 peak_rss);
    std::fflush(stdout);
    return 0;
}

int strata_inkling2_load_bundle(const char* bundle_dir, StrataInkling2Bundle* out_bundle) {
    if (!out_bundle) return -1;
    std::memset(out_bundle, 0, sizeof(*out_bundle));
    out_bundle->uma_fd = -1;

    const char* dir = (bundle_dir && bundle_dir[0] != '\0')
                          ? bundle_dir
                          : STRATA_INKLING2_DEFAULT_BUNDLE_DIR;
    std::snprintf(out_bundle->bundle_dir, sizeof(out_bundle->bundle_dir), "%s", dir);

    auto fail_cleanup = [&]() -> int {
        strata_inkling2_free_bundle(out_bundle);
        return -1;
    };

    const std::string uma_path = std::string(dir) + "/" + STRATA_INKLING2_UMA_FILENAME;
    out_bundle->uma_fd = open(uma_path.c_str(), O_RDONLY);
    if (out_bundle->uma_fd < 0) {
        std::fprintf(stderr, "[inkling2-load] ERROR: cannot open '%s'\n", uma_path.c_str());
        return fail_cleanup();
    }

    struct stat st;
    if (fstat(out_bundle->uma_fd, &st) != 0 ||
        static_cast<uint64_t>(st.st_size) < STRATA_INKLING2_HEADER_PADDED_BYTES ||
        (static_cast<uint64_t>(st.st_size) % STRATA_INKLING2_PAGE_BYTES) != 0ULL) {
        std::fprintf(stderr, "[inkling2-load] ERROR: invalid or unaligned size on '%s'\n", uma_path.c_str());
        return fail_cleanup();
    }

    out_bundle->uma_size = static_cast<size_t>(st.st_size);
    void* mapped = mmap(nullptr, out_bundle->uma_size, PROT_READ, MAP_PRIVATE, out_bundle->uma_fd, 0);
    if (mapped == MAP_FAILED) {
        std::fprintf(stderr, "[inkling2-load] ERROR: mmap failed on '%s' (%s)\n",
                     uma_path.c_str(), std::strerror(errno));
        return fail_cleanup();
    }
    out_bundle->uma_base = mapped;
    if ((reinterpret_cast<uintptr_t>(out_bundle->uma_base) % STRATA_INKLING2_PAGE_BYTES) != 0u) {
        std::fprintf(stderr, "[inkling2-load] ERROR: mmap base not 16 KiB aligned\n");
        return fail_cleanup();
    }

    std::memcpy(&out_bundle->hdr, out_bundle->uma_base, sizeof(StrataInkling2Header));
    if (out_bundle->hdr.version == STRATA_INKLING2_FORMAT_VERSION_V1) {
        std::memset(&out_bundle->hdr.mtp.frspec, 0, sizeof(out_bundle->hdr.mtp.frspec));
    }
    if (!validate_header(out_bundle->hdr, static_cast<uint64_t>(out_bundle->uma_size))) {
        std::fprintf(stderr, "[inkling2-load] ERROR: header/alignment validation failed on '%s'\n",
                     uma_path.c_str());
        return fail_cleanup();
    }

    const std::string tok_path = std::string(dir) + "/tokenizer.json";
    if (file_exists_nonempty(tok_path)) {
        out_bundle->tokenizer = strata_inkling2_tokenizer_load(tok_path.c_str());
        if (!out_bundle->tokenizer) {
            std::fprintf(stderr, "[inkling2-load] ERROR: failed to parse '%s'\n", tok_path.c_str());
            return fail_cleanup();
        }
    }

    return 0;
}

void strata_inkling2_free_bundle(StrataInkling2Bundle* bundle) {
    if (!bundle) return;
    if (bundle->tokenizer) {
        strata_inkling2_tokenizer_free(bundle->tokenizer);
        bundle->tokenizer = nullptr;
    }
    if (bundle->uma_base && bundle->uma_size > 0) {
        munmap(bundle->uma_base, bundle->uma_size);
    }
    if (bundle->uma_fd >= 0) {
        close(bundle->uma_fd);
    }
    std::memset(bundle, 0, sizeof(*bundle));
    bundle->uma_fd = -1;
}

int strata_inkling2_inspect_bundle(const char* bundle_dir) {
    StrataInkling2Bundle bdl;
    if (strata_inkling2_load_bundle(bundle_dir, &bdl) != 0) {
        return -1;
    }

    const StrataInkling2Header& h = bdl.hdr;
    const StrataInkling2Config& c = h.cfg;
    const double kGiB = 1024.0 * 1024.0 * 1024.0;
    const double kMiB = 1024.0 * 1024.0;
    const bool page_aligned = ((bdl.uma_size % STRATA_INKLING2_PAGE_BYTES) == 0u) &&
                              ((reinterpret_cast<uintptr_t>(bdl.uma_base) % STRATA_INKLING2_PAGE_BYTES) == 0u);

    uint32_t n_nvfp4 = 0u, n_mxfp8 = 0u, n_local_phys = 0u, n_global_phys = 0u;
    for (uint32_t i = 0; i < c.num_phys_layers; ++i) {
        if (h.layers[i].expert_dtype == STRATA_INKLING2_DTYPE_NVFP4) ++n_nvfp4;
        else                                                         ++n_mxfp8;
        if (h.layers[i].is_local_swa) ++n_local_phys;
        else                          ++n_global_phys;
    }

    const char* ctrl_dt = "NONE";
    if (c.loop_num_passes > 1u && c.loop_core_start < c.num_phys_layers) {
        ctrl_dt = dtype_name(h.layers[c.loop_core_start].controllers[0].down_weight.dtype);
    }

    std::printf("====================================================================================================\n");
    std::printf("  STRATA-M5 v2.0 — Inkling-2-Mini Zero-Copy UMA Bundle Inspection (format v%u)\n", h.version);
    std::printf("====================================================================================================\n");
    std::printf("  Bundle Directory  : %s\n", bdl.bundle_dir);
    std::printf("  UMA File          : %s (%.3f GiB / %llu B, 16 KiB aligned=%s)\n",
                STRATA_INKLING2_UMA_FILENAME,
                static_cast<double>(bdl.uma_size) / kGiB,
                static_cast<unsigned long long>(bdl.uma_size),
                page_aligned ? "YES" : "NO");
    std::printf("  Architecture      : inkling2_loop_text | %u phys (%u SWA + %u Global) -> %u logical (%u SWA + %u Global)\n",
                c.num_phys_layers, n_local_phys, n_global_phys,
                c.num_logical_layers, c.num_local_visits, c.num_global_visits);
    std::printf("  Looped Core       : layers [%u..%u) x %u passes (gate_rank=%u) | AttnRes block=%u | sconv K=%u\n",
                c.loop_core_start, c.loop_core_end, c.loop_num_passes, c.loop_gate_rank,
                c.attn_res_block_size, c.sconv_kernel_size);
    std::printf("  Attention & Geom  : d=%u, H=%u/%u, d_head=%u, d_rel=%u, rel_extent=%u, SWA=%u, softcap=%.1f\n",
                c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                c.d_rel, c.rel_extent, c.sliding_window_size, c.final_logit_softcapping);
    std::printf("  Dense Precision   : embed=%s, attn=%s, wr=%s, unembed=%s, ctrl=%s, sink=%s\n",
                dtype_name(h.embed_tokens.dtype),
                dtype_name(h.layers[0].wq.dtype),
                dtype_name(h.layers[0].wr.dtype),
                dtype_name(h.unembed.dtype),
                ctrl_dt,
                dtype_name(h.layers[0].attn_sink_proj.dtype));
    std::printf("  Sparse MoE        : E=%u top-%u, d_ff=%u, route_scale=%.1f | %u NVFP4 + %u MXFP8 layers\n",
                c.num_experts, c.num_experts_per_tok, c.intermediate_size, c.route_scale,
                n_nvfp4, n_mxfp8);
    std::printf("  Vocab & Tokenizer : unpadded=%u, padded=%u, eos=%u, pad=%u | tokenizer_vocab=%u\n",
                c.vocab_size_unpadded, c.vocab_size_padded, c.eos_token_id, c.pad_token_id,
                strata_inkling2_tokenizer_vocab_size(bdl.tokenizer));
    if (h.mtp.enabled) {
        std::printf("  DSpark MTP        : ENABLED (%u layers [%s], rnn=%s, block=%u, rank=%u, d_ff=%u, SWA=%u, taps={%u,%u,%u,%u}, %.3f MiB)\n",
                    h.mtp.num_layers,
                    dtype_name(h.mtp.layers[0].q_proj.dtype),
                    dtype_name(h.mtp.markov_rnn_weight.dtype),
                    h.mtp.block_size, h.mtp.markov_rank,
                    h.mtp.intermediate_size, h.mtp.sliding_window,
                    h.mtp.target_layer_ids[0], h.mtp.target_layer_ids[1],
                    h.mtp.target_layer_ids[2], h.mtp.target_layer_ids[3],
                    static_cast<double>(h.mtp.mtp_page_bytes) / kMiB);
        if (h.mtp.frspec.enabled) {
            const StrataInkling2FrSpecDesc& fr = h.mtp.frspec;
            std::printf("  FR-Spec Head      : ENABLED (shortlist=%u, unembed=%s [%.2f MiB], fold_w=%s [%.2f MiB], fold_b=%s [%.2f MiB])\n",
                        fr.shortlist_size,
                        dtype_name(fr.unembed_q4a.dtype),    static_cast<double>(fr.unembed_q4a.bytes) / kMiB,
                        dtype_name(fr.fold_weight_q8.dtype), static_cast<double>(fr.fold_weight_q8.bytes) / kMiB,
                        dtype_name(fr.fold_bias_f32.dtype),  static_cast<double>(fr.fold_bias_f32.bytes) / kMiB);
        } else {
            std::printf("  FR-Spec Head      : DISABLED\n");
        }
    } else {
        std::printf("  DSpark MTP        : DISABLED\n");
    }
    std::printf("  Active Traffic    : %.3f GB/tok Main | %.3f GB/blk MTP\n",
                static_cast<double>(h.active_bytes_per_tok) / 1e9,
                static_cast<double>(h.mtp_active_bytes_per_blk) / 1e9);
    std::printf("====================================================================================================\n");

    strata_inkling2_free_bundle(&bdl);
    return page_aligned ? 0 : -1;
}

StrataInkling2Tokenizer* strata_inkling2_tokenizer_load(const char* tokenizer_json_path) {
    if (!tokenizer_json_path) return nullptr;
    std::string json_str;
    if (!read_whole_file(tokenizer_json_path, json_str) || json_str.empty()) {
        std::fprintf(stderr, "[inkling2-tok] ERROR: failed to read '%s'\n", tokenizer_json_path);
        return nullptr;
    }

    std::string b2u_utf8[256];
    int16_t u2b_cp[512];
    build_gpt2_byte_unicode_tables(b2u_utf8, u2b_cp);

    std::unordered_map<std::string, uint32_t> str_to_id;
    str_to_id.reserve(262144);

    struct RawMerge {
        std::string a;
        std::string b;
    };
    std::vector<RawMerge> raw_merges;
    raw_merges.reserve(450000);
    std::vector<AddedTokenEntry> added_list;
    bool ignore_merges = true;

    JsonCursor cur(json_str.data(), json_str.size());
    if (!cur.consume('{')) return nullptr;

    std::string top_key, sub_key, k_str, v_str;
    uint32_t max_id_seen = 0;

    if (!cur.consume('}')) {
        while (true) {
            if (!cur.parse_string(top_key)) return nullptr;
            if (!cur.consume(':')) return nullptr;

            if (top_key == "added_tokens") {
                if (!cur.consume('[')) return nullptr;
                if (!cur.consume(']')) {
                    while (true) {
                        if (!cur.consume('{')) return nullptr;
                        AddedTokenEntry at{"", 0, false};
                        if (!cur.consume('}')) {
                            while (true) {
                                if (!cur.parse_string(sub_key)) return nullptr;
                                if (!cur.consume(':')) return nullptr;
                                if (sub_key == "id") {
                                    uint64_t id_u64 = 0;
                                    if (!cur.parse_uint64(id_u64) || id_u64 >= (1ULL << 22)) return nullptr;
                                    at.id = static_cast<uint32_t>(id_u64);
                                } else if (sub_key == "content") {
                                    if (!cur.parse_string(at.content)) return nullptr;
                                } else if (sub_key == "special") {
                                    if (!cur.parse_bool(at.special)) return nullptr;
                                } else {
                                    if (!cur.skip_value()) return nullptr;
                                }
                                if (cur.consume('}')) break;
                                if (!cur.consume(',')) return nullptr;
                            }
                        }
                        if (!at.content.empty()) {
                            max_id_seen = std::max(max_id_seen, at.id);
                            added_list.push_back(std::move(at));
                        }
                        if (cur.consume(']')) break;
                        if (!cur.consume(',')) return nullptr;
                    }
                }
            } else if (top_key == "model") {
                if (!cur.consume('{')) return nullptr;
                if (!cur.consume('}')) {
                    while (true) {
                        if (!cur.parse_string(sub_key)) return nullptr;
                        if (!cur.consume(':')) return nullptr;
                        if (sub_key == "ignore_merges") {
                            if (!cur.parse_bool(ignore_merges)) return nullptr;
                        } else if (sub_key == "vocab") {
                            if (!cur.consume('{')) return nullptr;
                            if (!cur.consume('}')) {
                                while (true) {
                                    if (!cur.parse_string(k_str)) return nullptr;
                                    if (!cur.consume(':')) return nullptr;
                                    uint64_t tid = 0;
                                    if (!cur.parse_uint64(tid) || tid >= (1ULL << 22)) return nullptr;
                                    uint32_t id32 = static_cast<uint32_t>(tid);
                                    max_id_seen = std::max(max_id_seen, id32);
                                    str_to_id.emplace(k_str, id32);
                                    if (cur.consume('}')) break;
                                    if (!cur.consume(',')) return nullptr;
                                }
                            }
                        } else if (sub_key == "merges") {
                            if (!cur.consume('[')) return nullptr;
                            if (!cur.consume(']')) {
                                while (true) {
                                    char pk = cur.peek();
                                    if (pk == '[') {
                                        cur.consume('[');
                                        RawMerge rm;
                                        if (!cur.parse_string(rm.a)) return nullptr;
                                        if (!cur.consume(',')) return nullptr;
                                        if (!cur.parse_string(rm.b)) return nullptr;
                                        if (!cur.consume(']')) return nullptr;
                                        raw_merges.push_back(std::move(rm));
                                    } else if (pk == '"') {
                                        if (!cur.parse_string(v_str)) return nullptr;
                                        size_t sp = v_str.find(' ');
                                        if (sp != std::string::npos) {
                                            raw_merges.push_back({v_str.substr(0, sp), v_str.substr(sp + 1)});
                                        }
                                    } else {
                                        return nullptr;
                                    }
                                    if (cur.consume(']')) break;
                                    if (!cur.consume(',')) return nullptr;
                                }
                            }
                        } else {
                            if (!cur.skip_value()) return nullptr;
                        }
                        if (cur.consume('}')) break;
                        if (!cur.consume(',')) return nullptr;
                    }
                }
            } else {
                if (!cur.skip_value()) return nullptr;
            }

            if (cur.consume('}')) break;
            if (!cur.consume(',')) return nullptr;
        }
    }

    if (str_to_id.empty() || max_id_seen >= (1U << 22)) return nullptr;

    auto* tok = new StrataInkling2Tokenizer();
    tok->vocab_size    = max_id_seen + 1u;
    tok->ignore_merges = ignore_merges;
    tok->id_to_raw_bytes.assign(tok->vocab_size, std::string());
    tok->id_is_special.assign(tok->vocab_size, 0u);
    tok->raw_piece_to_id.reserve(str_to_id.size() + 64u);

    std::string raw_tmp;
    for (const auto& kv : str_to_id) {
        if (kv.second >= tok->vocab_size) {
            delete tok;
            return nullptr;
        }
        if (decode_bpe_vocab_key_to_bytes(kv.first, u2b_cp, raw_tmp)) {
            tok->id_to_raw_bytes[kv.second] = raw_tmp;
            tok->raw_piece_to_id.emplace(raw_tmp, kv.second);
        } else {
            tok->id_to_raw_bytes[kv.second] = kv.first;
        }
    }

    for (int b = 0; b < 256; ++b) {
        auto it = str_to_id.find(b2u_utf8[b]);
        if (it == str_to_id.end()) {
            delete tok;
            return nullptr;
        }
        tok->byte_token_ids[b] = it->second;
    }

    for (const auto& at : added_list) {
        if (at.id < tok->vocab_size) {
            tok->id_to_raw_bytes[at.id] = at.content;
            tok->id_is_special[at.id]   = at.special ? 1u : 0u;
        }
    }
    tok->added_sorted = std::move(added_list);
    std::sort(tok->added_sorted.begin(), tok->added_sorted.end(),
              [](const AddedTokenEntry& x, const AddedTokenEntry& y) {
                  if (x.content.size() != y.content.size()) return x.content.size() > y.content.size();
                  return x.id < y.id;
              });
    for (uint32_t idx = 0; idx < tok->added_sorted.size(); ++idx) {
        uint8_t b0 = static_cast<uint8_t>(tok->added_sorted[idx].content[0]);
        tok->added_by_b0[b0].push_back(idx);
    }

    uint64_t ht_cap  = 1024;
    uint32_t ht_bits = 10;
    while (ht_cap < raw_merges.size() * 2ULL + 64ULL) {
        ht_cap <<= 1;
        ++ht_bits;
    }
    tok->merge_ht.assign(static_cast<size_t>(ht_cap),
                         MergeSlot{std::numeric_limits<uint64_t>::max(), 0u, 0u});
    tok->merge_mask  = ht_cap - 1ULL;
    tok->merge_shift = 64u - ht_bits;

    std::string combined;
    for (uint32_t rank = 0; rank < raw_merges.size(); ++rank) {
        const auto& rm = raw_merges[rank];
        auto it_a = str_to_id.find(rm.a);
        auto it_b = str_to_id.find(rm.b);
        if (it_a == str_to_id.end() || it_b == str_to_id.end()) continue;
        combined.assign(rm.a);
        combined.append(rm.b);
        auto it_m = str_to_id.find(combined);
        if (it_m == str_to_id.end()) continue;
        tok->insert_merge(it_a->second, it_b->second, rank, it_m->second);
    }

    return tok;
}

void strata_inkling2_tokenizer_free(StrataInkling2Tokenizer* tok) {
    delete tok;
}

uint32_t strata_inkling2_tokenizer_vocab_size(const StrataInkling2Tokenizer* tok) {
    return tok ? tok->vocab_size : 0u;
}

int strata_inkling2_encode(
    const StrataInkling2Tokenizer* tok,
    const char*                    text,
    size_t                         text_len,
    bool                           parse_special,
    uint32_t*                      out_ids,
    uint32_t                       out_cap,
    uint32_t*                      out_len
) {
    if (!tok || !out_len) return -1;
    if (!text) {
        if (text_len == 0) {
            *out_len = 0;
            return 0;
        }
        return -1;
    }
    if (text_len == static_cast<size_t>(-1)) {
        text_len = std::strlen(text);
    }
    *out_len = 0;
    if (text_len == 0) return 0;

    std::vector<uint32_t> produced;
    produced.reserve(text_len / 2 + 16);
    std::vector<std::pair<uint32_t, uint32_t>> pieces;

    auto encode_plain_span = [&](const char* span_ptr, size_t span_len) {
        if (span_len == 0) return;
        o200k_pretokenize_span(span_ptr, span_len, pieces);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(span_ptr);
        for (const auto& pc : pieces) {
            bpe_encode_piece(tok, base + pc.first, pc.second, produced);
        }
    };

    if (!parse_special || tok->added_sorted.empty()) {
        encode_plain_span(text, text_len);
    } else {
        size_t span_start = 0;
        size_t pos = 0;
        while (pos < text_len) {
            uint8_t b0 = static_cast<uint8_t>(text[pos]);
            const auto& candidates = tok->added_by_b0[b0];
            int32_t matched_idx = -1;
            for (uint32_t c_idx : candidates) {
                const AddedTokenEntry& at = tok->added_sorted[c_idx];
                size_t mlen = at.content.size();
                if (pos + mlen <= text_len &&
                    std::memcmp(text + pos, at.content.data(), mlen) == 0) {
                    matched_idx = static_cast<int32_t>(c_idx);
                    break;
                }
            }
            if (matched_idx >= 0) {
                if (pos > span_start) {
                    encode_plain_span(text + span_start, pos - span_start);
                }
                const AddedTokenEntry& at = tok->added_sorted[static_cast<size_t>(matched_idx)];
                produced.push_back(at.id);
                pos += at.content.size();
                span_start = pos;
            } else {
                ++pos;
            }
        }
        if (span_start < text_len) {
            encode_plain_span(text + span_start, text_len - span_start);
        }
    }

    *out_len = static_cast<uint32_t>(produced.size());
    if (!out_ids) return 0;
    if (produced.size() > out_cap) return -2;
    if (!produced.empty()) {
        std::memcpy(out_ids, produced.data(), produced.size() * sizeof(uint32_t));
    }
    return 0;
}

int strata_inkling2_decode(
    const StrataInkling2Tokenizer* tok,
    const uint32_t*                ids,
    uint32_t                       num_ids,
    bool                           skip_special,
    char*                          out_utf8,
    size_t                         out_cap,
    size_t*                        out_len
) {
    if (!tok || (!ids && num_ids > 0)) return -1;
    std::string acc;
    acc.reserve(static_cast<size_t>(num_ids) * 4 + 16);
    for (uint32_t i = 0; i < num_ids; ++i) {
        uint32_t tid = ids[i];
        if (tid >= tok->vocab_size) continue;
        if (skip_special && tok->id_is_special[tid]) continue;
        acc.append(tok->id_to_raw_bytes[tid]);
    }
    if (out_len) *out_len = acc.size();
    if (!out_utf8) return 0;
    if (acc.size() + 1 > out_cap) return -2;
    std::memcpy(out_utf8, acc.data(), acc.size() + 1);
    return 0;
}

void strata_inkling2_stream_decoder_init(StrataInkling2StreamDecoder* dec) {
    if (!dec) return;
    std::memset(dec, 0, sizeof(*dec));
}

int strata_inkling2_stream_decode_step(
    const StrataInkling2Tokenizer* tok,
    StrataInkling2StreamDecoder*   dec,
    uint32_t                       token_id,
    bool                           skip_special,
    char*                          out_utf8,
    size_t                         out_cap,
    size_t*                        out_len
) {
    if (!tok || !dec || !out_utf8 || out_cap == 0) return -1;
    out_utf8[0] = '\0';
    if (out_len) *out_len = 0;

    if (token_id >= tok->vocab_size) return 0;
    if (skip_special && tok->id_is_special[token_id]) return 0;

    const std::string& raw = tok->id_to_raw_bytes[token_id];
    if (raw.empty() && dec->pending_len == 0) return 0;

    std::string combined;
    const size_t pend = std::min<size_t>(dec->pending_len, sizeof(dec->pending));
    combined.reserve(pend + raw.size());
    if (pend > 0) {
        combined.append(reinterpret_cast<const char*>(dec->pending), pend);
    }
    combined.append(raw);

    const uint8_t* u = reinterpret_cast<const uint8_t*>(combined.data());
    const size_t n   = combined.size();
    size_t hold = 0;
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        uint8_t b = u[n - back];
        if ((b & 0xC0u) == 0x80u) continue;
        size_t need = 1;
        if ((b & 0xE0u) == 0xC0u)      need = 2;
        else if ((b & 0xF0u) == 0xE0u) need = 3;
        else if ((b & 0xF8u) == 0xF0u) need = 4;
        if (need > back) hold = back;
        break;
    }

    const size_t emit_len = n - hold;
    if (emit_len + 1 > out_cap) {
        if (out_len) *out_len = emit_len;
        return -2;
    }

    if (hold > 0) {
        std::memcpy(dec->pending, u + emit_len, hold);
    }
    dec->pending_len = static_cast<uint8_t>(hold);

    if (emit_len > 0) {
        std::memcpy(out_utf8, combined.data(), emit_len);
    }
    out_utf8[emit_len] = '\0';
    if (out_len) *out_len = emit_len;
    return 0;
}

} /* extern "C" */

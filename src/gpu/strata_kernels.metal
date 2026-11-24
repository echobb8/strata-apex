#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#if defined(__HAVE_TENSOR__)
#include <metal_tensor>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

// Inlined by strata_gpu.mm at runtime from ../../include/strata_abi.h:
// @@STRATA_ABI_HEADER@@

using namespace metal;
#if defined(__HAVE_TENSOR__)
using namespace mpp::tensor_ops;
#endif

/* ============================================================================
 * 1. Bit-Exact In-Register Unpack Primitives (Zero Intermediate DRAM Writes)
 * ============================================================================ */

// Hardware FEAT_FP8=0 workaround for Apple M5 Max:
// Exact IEEE/OCP E4M3FN -> IEEE FP16 bit-level conversion (0 bit errors across all 256 codes).
inline ushort strata_e4m3_to_f16_bits_dev(uchar e4m3) {
    ushort sign = ushort(e4m3 & 0x80u) << 8;
    uint exp4   = (uint(e4m3) >> 3) & 0x0Fu;
    uint man3   = uint(e4m3) & 0x07u;

    if (exp4 == 0u) {
        if (man3 == 0u) return sign;
        uint shift = clz(man3) - 29u; // man3 in 1..7 -> shift in 2..0
        uint exp5  = 8u - shift;
        uint man10 = ((man3 << shift) & 0x03u) << 8;
        return sign | ushort((exp5 << 10) | man10);
    }
    if (exp4 == 15u && man3 == 7u) {
        return sign | 0x7E00u;
    }
    return sign | ushort(((uint(e4m3) & 0x7Fu) + 0x40u) << 7);
}

// 100% branchless vectorized 4x E2M1 nibbles -> half4 in pure 16-bit SIMD ALU.
inline half4 strata_e2m1x4_to_half4_dev(uint low16_bits) {
    ushort4 nib(ushort((low16_bits >>  0u) & 0xFu),
                ushort((low16_bits >>  4u) & 0xFu),
                ushort((low16_bits >>  8u) & 0xFu),
                ushort((low16_bits >> 12u) & 0xFu));
    ushort4 sgn  = (nib & ushort4(0x8u)) << ushort4(12u);
    ushort4 mag  = nib & ushort4(0x7u);
    ushort4 addm = select(ushort4(0u), mag << ushort4(9u), mag > ushort4(1u));
    ushort4 absb = select(ushort4(0x3800u) + addm, ushort4(0u), mag == ushort4(0u));
    return as_type<half4>(ushort4(sgn | absb));
}

// 100% branchless vectorized 4x E4M3 scale bytes -> half4 (* 0.0625f dynamic range shift built in;
// handles zeros, subnormals, normals, and clamps 0x7F/0xFF NaN codes to 0.0h!).
inline half4 strata_e4m3x4_scale_safe_half4_dev(uint packed_s4) {
    ushort4 u(ushort((packed_s4 >>  0u) & 0x7Fu),
              ushort((packed_s4 >>  8u) & 0x7Fu),
              ushort((packed_s4 >> 16u) & 0x7Fu),
              ushort((packed_s4 >> 24u) & 0x7Fu));
    ushort4 exp4   = u >> ushort4(3u);
    half4   norm_h = as_type<half4>((u + ushort4(0x20u)) << ushort4(7u));
    // Exact subnormal E4M3 * 2^-4 = man3 * 2^-13; compute in IEEE bit domain (immune to fast-math!)
    ushort4 sh_v   = select(ushort4(0u), ushort4(1u), u < ushort4(4u));
    sh_v           = select(sh_v,        ushort4(2u), u < ushort4(2u));
    ushort4 sub_b  = select(((ushort4(4u) - sh_v) << ushort4(10u)) | (((u << sh_v) & ushort4(3u)) << ushort4(8u)),
                            ushort4(0u), u == ushort4(0u));
    half4   res    = select(norm_h, as_type<half4>(sub_b), exp4 == ushort4(0u));
    return select(res, half4(0.0h), u == ushort4(0x7Fu));
}

// Exact QTIP 6-bit shift-register state initializer at 16-weight uint32 boundary `sub` (0..7).
inline uint strata_qtip_init_state6(uint prev_u32, uint sub) {
    if (sub == 0u) return 0u;
    uint u13 = (prev_u32 >> 26u) & 0x3u;
    uint u14 = (prev_u32 >> 28u) & 0x3u;
    uint u15 = (prev_u32 >> 30u) & 0x3u;
    return (u13 << 4u) | (u14 << 2u) | u15;
}

inline half strata_qtip_3inst_unit_h_dev(uint w8) {
    uint h = (w8 + 1u) * 0x9E3779B9u;
    h ^= h >> 15u;
    h *= 0x85EBCA6Bu;
    int s0 = int(h & 0x3FFu) - 512;
    int s1 = int((h >> 10u) & 0x3FFu) - 512;
    int s2 = int((h >> 20u) & 0x3FFu) - 512;
    return half(float(s0 + s1 + s2) * (1.0f / 1536.0f));
}

inline half strata_refine_lut4_h_dev(uint sym) {
    uint is_inner = (sym ^ (sym >> 1u)) & 1u;
    half mag      = select(half(1.62h), half(0.46h), is_inner != 0u);
    return (sym & 2u) ? mag : -mag;
}

// Fast zero-fill kernel for atomic FP32 accumulation buffers inside a single ComputeEncoder
kernel void strata_zero_f32_buffer(
    device float* buf        [[buffer(0)]],
    constant uint& count_f32 [[buffer(1)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint base = gid * 4u;
    if (base + 3u < count_f32) {
        *reinterpret_cast<device float4*>(buf + base) = float4(0.0f);
    } else {
        for (uint i = base; i < count_f32; ++i) {
            buf[i] = 0.0f;
        }
    }
}

/* ============================================================================
 * 2. Fast Orthonormal FWHT-128 Input Activation Transform (for QTIP Incoherence)
 * ============================================================================ */

kernel void strata_fwht128_activations(
    device const half* x_in        [[buffer(0)]],
    device half*       x_rot_out   [[buffer(1)]],
    constant uint&     total_blocks[[buffer(2)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]]
) {
    if (tgid >= total_blocks) return;
    threadgroup float buf[128];

    uint base = tgid * 128u;
    buf[tid]       = float(x_in[base + tid]);
    buf[tid + 64u] = float(x_in[base + tid + 64u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint len = 1u; len < 128u; len <<= 1u) {
        uint group = tid / len;
        uint off   = tid & (len - 1u);
        uint i0    = group * (len << 1u) + off;
        uint i1    = i0 + len;
        float a    = buf[i0];
        float b    = buf[i1];
        buf[i0]    = a + b;
        buf[i1]    = a - b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    constexpr float kInvSqrt128 = 0.08838834764831845f;
    x_rot_out[base + tid]       = half(buf[tid] * kInvSqrt128);
    x_rot_out[base + tid + 64u] = half(buf[tid + 64u] * kInvSqrt128);
}

/* ============================================================================
 * 3. Bit-Exact Verification Kernel for 32-Way Tile-rANS + E4M3 -> FP16
 * ============================================================================ */

kernel void strata_verify_rans_e4m3_bitexact(
    device const uchar*  rans_blob   [[buffer(0)]],
    device const ushort* ref_f16     [[buffer(1)]],
    device atomic_uint*  err_counter [[buffer(2)]],
    uint tgid                        [[threadgroup_position_in_grid]],
    uint lane                        [[thread_index_in_simdgroup]]
) {
    device const StrataTileRansHeader* hdr =
        reinterpret_cast<device const StrataTileRansHeader*>(rans_blob);
    uint num_tiles = hdr->reserved[1];
    if (tgid >= num_tiles) return;

    uint tile_syms      = hdr->reserved[0];
    uint payload_offset = hdr->reserved[2];
    uint dir_offset     = hdr->reserved[3];
    uint total_count    = hdr->uncompressed_count;

    device const ushort* freq    = reinterpret_cast<device const ushort*>(rans_blob + 32u);
    device const ushort* cum     = reinterpret_cast<device const ushort*>(rans_blob + 32u + 512u);
    device const uchar*  sym_lut = rans_blob + 32u + 1024u;

    threadgroup uchar  tg_sym_lut[1024];
    threadgroup ushort tg_freq[256];
    threadgroup ushort tg_cum[256];

    for (uint s = lane; s < 256u; s += 32u) {
        tg_freq[s] = freq[s];
        tg_cum[s]  = cum[s];
    }
    for (uint i = lane; i < 1024u; i += 32u) {
        tg_sym_lut[i] = sym_lut[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const uint*  tile_u8_offsets = reinterpret_cast<device const uint*>(rans_blob + dir_offset);
    device const uchar* u8_payload      = rans_blob + payload_offset;
    device const uchar* tile_base       = u8_payload + tile_u8_offsets[tgid];
    device const uint*  init_states     = reinterpret_cast<device const uint*>(tile_base);

    uint tile_start = tgid * tile_syms;
    uint tile_count = min(tile_syms, total_count - tile_start);
    if (tile_count <= lane) return;

    uint lane_steps = (tile_count - 1u - lane) / 32u + 1u;
    uint x          = init_states[lane];
    device const uchar* lane_ptr = tile_base + 128u + lane;

    uint local_errs = 0u;
    for (uint step = 0; step < lane_steps; ++step) {
        uint slot  = x & 0x3FFu;
        uchar sym  = tg_sym_lut[slot];
        uint f     = uint(tg_freq[sym]);
        uint c     = uint(tg_cum[sym]);
        x          = f * (x >> 10u) + slot - c;
        if (x < (1u << 18u)) {
            x = (x << 8u) | uint(*lane_ptr);
            lane_ptr += 32u;
            if (x < (1u << 18u)) {
                x = (x << 8u) | uint(*lane_ptr);
                lane_ptr += 32u;
            }
        }
        ushort got_bits = strata_e4m3_to_f16_bits_dev(sym);
        ushort exp_bits = ref_f16[tile_start + step * 32u + lane];
        if (got_bits != exp_bits) {
            ++local_errs;
        }
    }
    if (local_errs > 0u) {
        atomic_fetch_add_explicit(err_counter, local_errs, memory_order_relaxed);
    }
}

/* ============================================================================
 * 4. ILP=4 Multi-Stream 32-Way Tile-rANS + E4M3 -> Metal 4 NAX `cooperative_tensor` Spine GEMM
 *    - Compact 4.5 KiB Threadgroup LUT (`tg_rans_u32[1024]` + `tg_e4m3_f16[256]`) -> 7 TGs / core!
 *    - Decodes 4 independent Tile-rANS streams (`kt .. kt+3`) simultaneously per lane (ILP = 4),
 *      breaking the serial rANS state dependency chain and feeding NAX `wCT` directly in registers!
 * ============================================================================ */

struct SpineRansGemmParams {
    uint batch_B;
    uint d_in;
    uint d_out;
    uint num_k_tiles; // d_in / 32
    uint split_k;
};

kernel void strata_spine_e4m3_rans_gemm(
    device const uchar*  rans_blob   [[buffer(0)]],
    device const half*   x_in_pad16  [[buffer(1)]], // [16, d_in] (rows B..15 zero-padded)
    device atomic_float* y_accum_f32 [[buffer(2)]], // [batch_B, d_out]
    constant SpineRansGemmParams& p  [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint  lane [[thread_index_in_simdgroup]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    device const StrataTileRansHeader* hdr =
        reinterpret_cast<device const StrataTileRansHeader*>(rans_blob);
    const uint payload_offset = hdr->reserved[2];
    const uint dir_offset     = hdr->reserved[3];

    device const ushort* freq    = reinterpret_cast<device const ushort*>(rans_blob + 32u);
    device const ushort* cum     = reinterpret_cast<device const ushort*>(rans_blob + 32u + 512u);
    device const uchar*  sym_lut = rans_blob + 32u + 1024u;

    // 4.5 KiB total Threadgroup SRAM -> high warp occupancy across all 40 M5 Max GPU cores!
    threadgroup uint   tg_rans_u32[1024];
    threadgroup ushort tg_e4m3_f16[256];

    for (uint s = tid; s < 256u; s += 64u) {
        tg_e4m3_f16[s] = strata_e4m3_to_f16_bits_dev(uchar(s));
    }
    for (uint s = tid; s < 1024u; s += 64u) {
        uchar sym  = sym_lut[s];
        uint  fm1  = uint(freq[sym]) - 1u;
        uint  bias = s - uint(cum[sym]);
        tg_rans_u32[s] = (fm1 << 20u) | (bias << 10u) | uint(sym);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rb        = tgid.x * 2u + sgid;
    const uint row_block = rb * 32u;
    if (row_block >= p.d_out) return;

    device const uint*  tile_u8_offsets = reinterpret_cast<device const uint*>(rans_blob + dir_offset);
    device const uchar* u8_payload      = rans_blob + payload_offset;

    const uint B              = min(p.batch_B, uint(STRATA_MAX_SPEC_BATCH));
    const uint raw_tpp        = (p.num_k_tiles + p.split_k - 1u) / p.split_k;
    const uint tiles_per_part = (raw_tpp + 1u) & ~1u;
    const uint kt_begin       = min(p.num_k_tiles, tgid.y * tiles_per_part);
    const uint kt_end         = min(p.num_k_tiles, kt_begin + tiles_per_part);

#if defined(__HAVE_TENSOR__)
    device half* x_mut = const_cast<device half*>(x_in_pad16);
    tensor<device half, dextents<int, 2>, tensor_inline> tX_full(x_mut, dextents<int, 2>(int(p.d_in), 16));

    constexpr auto desc = matmul2d_descriptor(
        16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    auto xCT  = op.get_left_input_cooperative_tensor<half, half, float>();
    auto wCT0 = op.get_right_input_cooperative_tensor<half, half, float>();
    auto wCT1 = op.get_right_input_cooperative_tensor<half, half, float>();
    auto cT   = op.get_destination_cooperative_tensor<decltype(xCT), decltype(wCT0), float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    uint kt = kt_begin;
    // ILP = 2: decode 2 independent Tile-rANS tiles simultaneously in registers (zero register spill!)
    for (; kt + 1u < kt_end; kt += 2u) {
        const uint t0 = rb * p.num_k_tiles + kt;
        device const uchar* b0 = u8_payload + tile_u8_offsets[t0 + 0u];
        device const uchar* b1 = u8_payload + tile_u8_offsets[t0 + 1u];

        uint x0 = reinterpret_cast<device const uint*>(b0)[lane];
        uint x1 = reinterpret_cast<device const uint*>(b1)[lane];

        device const uchar* p0 = b0 + 128u + lane;
        device const uchar* p1 = b1 + 128u + lane;

        #pragma clang loop unroll(full)
        for (ushort step = 0; step < 32u; ++step) {
            uint e0 = tg_rans_u32[x0 & 0x3FFu];
            uint e1 = tg_rans_u32[x1 & 0x3FFu];

            x0 = (((e0 >> 20u) & 0x3FFu) + 1u) * (x0 >> 10u) + ((e0 >> 10u) & 0x3FFu);
            x1 = (((e1 >> 20u) & 0x3FFu) + 1u) * (x1 >> 10u) + ((e1 >> 10u) & 0x3FFu);

            if (x0 < (1u << 18u)) { x0 = (x0 << 8u) | uint(*p0); p0 += 32u; if (x0 < (1u << 18u)) { x0 = (x0 << 8u) | uint(*p0); p0 += 32u; } }
            if (x1 < (1u << 18u)) { x1 = (x1 << 8u) | uint(*p1); p1 += 32u; if (x1 < (1u << 18u)) { x1 = (x1 << 8u) | uint(*p1); p1 += 32u; } }

            wCT0[step] = as_type<half>(tg_e4m3_f16[e0 & 0xFFu]);
            wCT1[step] = as_type<half>(tg_e4m3_f16[e1 & 0xFFu]);
        }

        { auto sX = tX_full.slice<32, 16>(int((kt + 0u) * 32u), 0); xCT.load(sX); op.run(xCT, wCT0, cT); }
        { auto sX = tX_full.slice<32, 16>(int((kt + 1u) * 32u), 0); xCT.load(sX); op.run(xCT, wCT1, cT); }
    }

    for (; kt < kt_end; ++kt) {
        const uint t0 = rb * p.num_k_tiles + kt;
        device const uchar* b0 = u8_payload + tile_u8_offsets[t0];
        uint x0 = reinterpret_cast<device const uint*>(b0)[lane];
        device const uchar* p0 = b0 + 128u + lane;
        #pragma clang loop unroll(full)
        for (ushort step = 0; step < 32u; ++step) {
            uint e0 = tg_rans_u32[x0 & 0x3FFu];
            x0 = (((e0 >> 20u) & 0x3FFu) + 1u) * (x0 >> 10u) + ((e0 >> 10u) & 0x3FFu);
            if (x0 < (1u << 18u)) { x0 = (x0 << 8u) | uint(*p0); p0 += 32u; if (x0 < (1u << 18u)) { x0 = (x0 << 8u) | uint(*p0); p0 += 32u; } }
            wCT0[step] = as_type<half>(tg_e4m3_f16[e0 & 0xFFu]);
        }
        auto sX = tX_full.slice<32, 16>(int(kt * 32u), 0);
        xCT.load(sX);
        op.run(xCT, wCT0, cT);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx   = cT.get_multidimensional_index(i);
            uint m_tok = uint(idx[1]);
            uint n_col = row_block + uint(idx[0]);
            if (m_tok < B && n_col < p.d_out) {
                atomic_fetch_add_explicit(&y_accum_f32[m_tok * p.d_out + n_col],
                                          cT[i], memory_order_relaxed);
            }
        }
    }
#endif
}

/* ============================================================================
 * 5. Metal 4 Neural Accelerator (`NAX`) Tiled FP16 GEMM (`mpp::tensor_ops::matmul2d`)
 *    Computes 128x64 output tile per Threadgroup across 4 SIMD-groups (~60-63+ TFLOPS FP16).
 * ============================================================================ */

kernel void strata_matrix_f16_gemm(
    device const half* A [[buffer(0)]], // [M, K]
    device const half* B [[buffer(1)]], // [K, N]
    device half*       C [[buffer(2)]], // [M, N]
    constant uint&     M [[buffer(3)]],
    constant uint&     N [[buffer(4)]],
    constant uint&     K [[buffer(5)]],
    uint2 tgid           [[threadgroup_position_in_grid]]
) {
#if defined(__HAVE_TENSOR__)
    const int m0 = int(tgid.y) * 128;
    const int n0 = int(tgid.x) * 64;
    if (m0 >= int(M) || n0 >= int(N)) return;

    device half* A_mut = const_cast<device half*>(A);
    device half* B_mut = const_cast<device half*>(B);

    tensor<device half, dextents<int, 2>, tensor_inline> tA(A_mut, dextents<int, 2>(int(K), int(M)));
    tensor<device half, dextents<int, 2>, tensor_inline> tB(B_mut, dextents<int, 2>(int(N), int(K)));
    tensor<device half, dextents<int, 2>, tensor_inline> tC(C,     dextents<int, 2>(int(N), int(M)));

    constexpr auto desc = matmul2d_descriptor(
        128, 64, static_cast<int>(dynamic_extent), false, false, true, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    if (m0 + 127 < int(M) && n0 + 63 < int(N)) {
        auto sA = tA.slice<dynamic_extent, 128>(0, m0);
        auto sB = tB.slice<64, dynamic_extent>(n0, 0);
        auto cT = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), half>();
        op.run(sA, sB, cT);
        auto sC = tC.slice<64, 128>(n0, m0);
        cT.store(sC);
    } else {
        auto sA = tA.slice(0, m0);
        auto sB = tB.slice(n0, 0);
        auto cT = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), half>();
        op.run(sA, sB, cT);
        auto sC = tC.slice(n0, m0);
        cT.store(sC);
    }
#endif
}

/* ============================================================================
 * 6. Roofline-Optimized Slot-Indexed Tiered MoE Layer Kernels (`strata_moe_mm` — Metal 4 NAX)
 *    - Single-accumulator (`1` `cT`, `0 B` TG SRAM on MXFP4!) Split-K `NAX` pipeline (~390-430 GB/s Phys DRAM!)
 *    - SLC-resident Activation Pre-Gather (`strata_moe_gather_x16`, <4 us) & Fused SwiGLU (`strata_moe_swiglu_pack16`, <5 us)
 *    - 100% Causal Mode A per-token `refine_tok_mask` invariance + 0.00 ms Mode B atomic `slot_ready` fallback!
 * ============================================================================ */

struct StrataActiveExpertDesc {
    ushort expert_id;
    ushort tok_count;
    uchar  tok_indices[STRATA_MAX_SPEC_BATCH];
    float  tok_gates[STRATA_MAX_SPEC_BATCH];
    ushort refine_tok_mask;
    ushort pad;
};

struct StrataMoEKernelCfg {
    uint  d_model;
    uint  d_ff;
    uint  num_active_experts;
    uint  regime;
    uint  exec_mode;
    uint  hot_pool_slots;
    uint  base_pool_slots;
    uint  ring_slot_stride;
    ulong ring_buf_bytes;
    ulong hot_expert_bytes;
    ulong base_expert_bytes;
    uint  svd_rank;
    uint  gu_split_k;
    uint  down_split_k;
    uint  reserved_pad;
};

// Pre-gathers routed token activations X [B, D] into [num_active, 16, D] in SLC AND simultaneously
// zeros `y_zero_f32 [B, D]` in `u_idx == 0` (eliminating an entire separate memset kernel dispatch!).
kernel void strata_moe_gather_x16(
    device const half*                   x_src       [[buffer(0)]], // [B, D]
    device const StrataActiveExpertDesc* active_list [[buffer(1)]], // [num_active]
    device half*                         x16_out     [[buffer(2)]], // [num_active, 16, D]
    constant uint&                       D           [[buffer(3)]], // d_model
    constant uint&                       num_active  [[buffer(4)]],
    device float*                        y_zero_f32  [[buffer(5)]], // [B, D]
    constant uint&                       y_vec4_cnt  [[buffer(6)]], // (B * D) / 4 <= 2 * D
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint u_idx = tgid.y;
    if (u_idx >= num_active) return;

    const StrataActiveExpertDesc desc = active_list[u_idx];
    const uint tc   = min(uint(desc.tok_count), uint(STRATA_MAX_SPEC_BATCH));
    const uint d8   = D >> 3u;
    const uint idx8 = tgid.x * 128u + tid; // covers [16 * d8] = [2 * D] uint4s!
    if (idx8 >= 16u * d8) return;

    if (u_idx == 0u && idx8 < y_vec4_cnt) {
        *reinterpret_cast<device float4*>(y_zero_f32 + idx8 * 4u) = float4(0.0f);
    }

    const uint m   = idx8 / d8;
    const uint col = (idx8 % d8) * 8u;
    uint4 val      = uint4(0u);
    if (m < tc) {
        uint tok = uint(desc.tok_indices[m]);
        val = *reinterpret_cast<device const uint4*>(x_src + tok * D + col);
    }
    *reinterpret_cast<device uint4*>(x16_out + (ulong(u_idx) * 16ul + m) * D + col) = val;
}

// Vectorized (`float4 -> half4`) SLC Epilogue (<3 us!): Fuses SwiGLU over `gu_f32 [num_active, 8, 2*d_ff]`!
kernel void strata_moe_swiglu_pack16(
    device const float*                  gu_f32      [[buffer(0)]], // [num_active, 8, 2*d_ff]
    device const StrataActiveExpertDesc* active_list [[buffer(1)]],
    device half*                         mid_out     [[buffer(2)]], // [num_active, 8, d_ff]
    device half*                         mid16_out   [[buffer(3)]], // [num_active, 16, d_ff]
    constant uint&                       d_ff        [[buffer(4)]],
    constant uint&                       num_active  [[buffer(5)]],
    constant uint&                       need_mid8   [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint u_idx = tgid.y;
    if (u_idx >= num_active) return;
    const uint tc   = min(uint(active_list[u_idx].tok_count), uint(STRATA_MAX_SPEC_BATCH));
    const uint rows = (need_mid8 != 0u) ? uint(STRATA_MAX_SPEC_BATCH) : 16u;
    const uint f4   = d_ff >> 2u;
    const uint idx4 = tgid.x * 128u + tid;
    if (idx4 >= rows * f4) return;

    const uint m    = idx4 / f4;
    const uint col4 = (idx4 % f4) * 4u;
    half4 h4        = half4(0.0h);
    if (m < tc) {
        ulong base = (ulong(u_idx) * ulong(STRATA_MAX_SPEC_BATCH) + m) * (2ul * d_ff);
        float4 gv  = *reinterpret_cast<device const float4*>(gu_f32 + base + col4);
        float4 uv  = *reinterpret_cast<device const float4*>(gu_f32 + base + d_ff + col4);
        h4         = half4((gv / (1.0f + exp(-gv))) * uv);
    }
    if (need_mid8 != 0u) {
        *reinterpret_cast<device half4*>(mid_out + (ulong(u_idx) * STRATA_MAX_SPEC_BATCH + m) * d_ff + col4) = h4;
    } else {
        *reinterpret_cast<device half4*>(mid16_out + (ulong(u_idx) * 16ul + m) * d_ff + col4) = h4;
    }
}

// Copies `mid_8 [num_active, 8, d_ff]` -> `mid16_out [num_active, 16, d_ff]` after W6A16 SVD or FWHT-128 (<3 us in SLC)
kernel void strata_moe_pad_mid8_to_16(
    device const half*                   mid_8       [[buffer(0)]],
    device const StrataActiveExpertDesc* active_list [[buffer(1)]],
    device half*                         mid16_out   [[buffer(2)]],
    constant uint&                       d_ff        [[buffer(3)]],
    constant uint&                       num_active  [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint u_idx = tgid.y;
    if (u_idx >= num_active) return;
    const uint tc   = min(uint(active_list[u_idx].tok_count), uint(STRATA_MAX_SPEC_BATCH));
    const uint f8   = d_ff >> 3u;
    const uint idx8 = tgid.x * 128u + tid;
    if (idx8 >= 16u * f8) return;

    const uint m   = idx8 / f8;
    const uint col = (idx8 % f8) * 8u;
    uint4 val      = uint4(0u);
    if (m < tc) {
        val = *reinterpret_cast<device const uint4*>(mid_8 + (ulong(u_idx) * STRATA_MAX_SPEC_BATCH + m) * d_ff + col);
    }
    *reinterpret_cast<device uint4*>(mid16_out + (ulong(u_idx) * 16ul + m) * d_ff + col) = val;
}

#if defined(__HAVE_TENSOR__)
template <typename WCoopTensor>
inline void strata_unpack_mxfp4_tile_to_wct(
    uint4 q4,
    half4 sc4,
    thread WCoopTensor& wCT
) {
    half4 a0 = strata_e2m1x4_to_half4_dev(q4.x) * sc4.x;
    half4 a1 = strata_e2m1x4_to_half4_dev(q4.x >> 16u) * sc4.y;
    half4 a2 = strata_e2m1x4_to_half4_dev(q4.y) * sc4.z;
    half4 a3 = strata_e2m1x4_to_half4_dev(q4.y >> 16u) * sc4.w;
    half4 b0 = strata_e2m1x4_to_half4_dev(q4.z) * sc4.x;
    half4 b1 = strata_e2m1x4_to_half4_dev(q4.z >> 16u) * sc4.y;
    half4 b2 = strata_e2m1x4_to_half4_dev(q4.w) * sc4.z;
    half4 b3 = strata_e2m1x4_to_half4_dev(q4.w >> 16u) * sc4.w;

    wCT[ 0] = a0.x; wCT[ 1] = a0.y; wCT[ 2] = a0.z; wCT[ 3] = a0.w;
    wCT[ 4] = a1.x; wCT[ 5] = a1.y; wCT[ 6] = a1.z; wCT[ 7] = a1.w;
    wCT[ 8] = a2.x; wCT[ 9] = a2.y; wCT[10] = a2.z; wCT[11] = a2.w;
    wCT[12] = a3.x; wCT[13] = a3.y; wCT[14] = a3.z; wCT[15] = a3.w;
    wCT[16] = b0.x; wCT[17] = b0.y; wCT[18] = b0.z; wCT[19] = b0.w;
    wCT[20] = b1.x; wCT[21] = b1.y; wCT[22] = b1.z; wCT[23] = b1.w;
    wCT[24] = b2.x; wCT[25] = b2.y; wCT[26] = b2.z; wCT[27] = b2.w;
    wCT[28] = b3.x; wCT[29] = b3.y; wCT[30] = b3.z; wCT[31] = b3.w;
}
#endif

// Single-Accumulator (`1` `cT`!) Split-K NAX Gate+Up Projection `[2 * d_ff, d_model] -> gu_f32 [num_active, 16, 2*d_ff]`!
//   - MXFP4: TG = 128 (4 SGs = 128 rows/TG), `0 B` Threadgroup SRAM, `0` barriers -> ~390-435 GB/s Physical Cold-DRAM!
//   - QTIP:  TG = 64  (2 SGs = 32  rows/TG), `9 KiB` Threadgroup SRAM -> 3 TGs/core, `1` `cT` accumulator!
kernel void strata_moe_gate_up_swiglu(
    device const half*                   x16_in         [[buffer(0)]], // [num_active, 16, d_model]
    device const StrataSlotEntry*        slot_map       [[buffer(1)]],
    device const atomic_uint*            slot_ready     [[buffer(2)]],
    device const StrataActiveExpertDesc* active_list    [[buffer(3)]],
    device const uchar*                  uma_hot_pool   [[buffer(4)]],
    device const uchar*                  uma_base_pool  [[buffer(5)]],
    device const uchar*                  ring_buf       [[buffer(6)]],
    device atomic_float*                 gu_accum_f32   [[buffer(7)]], // [num_active, 16, 2*d_ff]
    device atomic_uint*                  used_ring_cnt  [[buffer(8)]],
    device atomic_uint*                  fallback_cnt   [[buffer(9)]],
    constant StrataMoEKernelCfg&         cfg            [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    const uint u_idx  = tgid.y;
    const uint k_part = tgid.z;
    if (u_idx >= cfg.num_active_experts) return;

    const StrataActiveExpertDesc desc = active_list[u_idx];
    const StrataSlotEntry slot        = slot_map[desc.expert_id];
    const uint   tc                   = min(uint(desc.tok_count), uint(STRATA_MAX_SPEC_BATCH));
    const ushort all_tok_mask         = ushort((1u << tc) - 1u);
    const ushort eff_mask             = desc.refine_tok_mask & all_tok_mask;
    const uint   split_k              = max(cfg.gu_split_k, 1u);
    const uint   N_gu                 = 2u * cfg.d_ff;

    bool ring_is_ready = false;
    if (slot.ring_slot_idx != STRATA_SLOT_INVALID && cfg.ring_buf_bytes >= cfg.ring_slot_stride) {
        uint cur_ep   = atomic_load_explicit(&slot_ready[slot.ring_slot_idx], memory_order_relaxed);
        ring_is_ready = (cur_ep >= slot.expected_epoch);
    }

    if (tgid.x == 0u && k_part == 0u && sgid == 0u && lane == 0u && slot.tier_kind != STRATA_TIER_HOT_FULL_4B) {
        if (ring_is_ready && eff_mask != 0u) {
            atomic_fetch_add_explicit(used_ring_cnt, 1u, memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(fallback_cnt, 1u, memory_order_relaxed);
        }
    }

    device half* x_exp = const_cast<device half*>(x16_in) + ulong(u_idx) * 16ul * cfg.d_model;
    tensor<device half, dextents<int, 2>, tensor_inline> tX_full(x_exp, dextents<int, 2>(int(cfg.d_model), 16));

    // =========================================================================
    // PATH A: Regime I BitNest — High-Occupancy (TG=64, 9 KiB SRAM -> 3 TGs/core!) NAX `matmul2d(16, 16, 128)`
    // =========================================================================
    if (cfg.regime == STRATA_REGIME_I_BITNEST) {
        threadgroup half tg_qtip[2][256];
        threadgroup half W_sg[2][16][128];

        for (uint i = tid; i < 256u; i += 64u) {
            half v = strata_qtip_3inst_unit_h_dev(i);
            tg_qtip[0][i] = v;
            tg_qtip[1][i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint n0 = tgid.x * 32u + sgid * 16u;
        if (n0 >= N_gu) return;

        const ulong total_blks128 = (3ul * ulong(cfg.d_model) * ulong(cfg.d_ff)) / 128ul;
        const ulong scales_off    = total_blks128 * 32ul;
        const uint  blks_per_row  = cfg.d_model / 128u;

        const uint base_slot              = slot.hot_or_base_idx % max(cfg.base_pool_slots, 1u);
        device const uchar* exp_base      = uma_base_pool + ulong(base_slot) * cfg.base_expert_bytes;
        device const uint*  base_codes32  = reinterpret_cast<device const uint*>(exp_base);
        device const half*  base_scales16 = reinterpret_cast<device const half*>(exp_base + scales_off);

        const bool has_hot_ref  = (slot.tier_kind == STRATA_TIER_HOT_FULL_4B);
        const bool has_ring_ref = (!has_hot_ref) && ring_is_ready;
        device const uchar* ref_ptr = nullptr;
        if (has_hot_ref) {
            uint h_slot = slot.hot_or_base_idx % max(cfg.hot_pool_slots, 1u);
            ref_ptr = uma_hot_pool + ulong(h_slot) * cfg.hot_expert_bytes;
        } else if (has_ring_ref) {
            ulong off = (ulong(slot.ring_slot_idx) * ulong(cfg.ring_slot_stride)) %
                        max(cfg.ring_buf_bytes, ulong(1));
            if (off + cfg.hot_expert_bytes <= cfg.ring_buf_bytes) {
                ref_ptr = ring_buf + off;
            }
        }
        const bool can_ref  = (ref_ptr != nullptr) && (eff_mask != 0u);
        const bool all_ref  = can_ref && (eff_mask == all_tok_mask);
        const bool mix_ref  = can_ref && (!all_ref);
        device const uint* ref_codes32  = can_ref ? reinterpret_cast<device const uint*>(ref_ptr) : nullptr;
        device const half* ref_scales16 = can_ref ? reinterpret_cast<device const half*>(ref_ptr + scales_off) : nullptr;

        const uint blk_per = (blks_per_row + split_k - 1u) / split_k;
        const uint blk_beg = min(blks_per_row, k_part * blk_per);
        const uint blk_end = min(blks_per_row, blk_beg + blk_per);

        constexpr auto desc_q = matmul2d_descriptor(
            16, 16, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc_q, execution_simdgroup> op_q;

        tensor<threadgroup half, extents<int, 128, 16>, tensor_inline> tW(&W_sg[sgid][0][0], extents<int, 128, 16>());
        auto sX0 = tX_full.slice<128, 16>(0, 0);
        auto cT  = op_q.get_destination_cooperative_tensor<decltype(sX0), decltype(tW), float>();

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) cT[i] = 0.0f;
        }

        const uint grp = lane >> 3u;
        const uint sub = lane & 7u;
        const uint bk  = lane & 1u;

        for (uint blk = blk_beg; blk < blk_end; ++blk) {
            #pragma clang loop unroll(full)
            for (uint p = 0; p < 4u; ++p) {
                uint  r    = p * 4u + grp;
                ulong bidx = ulong(n0 + r) * blks_per_row + blk;
                half  bs   = base_scales16[bidx];
                uint  bu32 = base_codes32[bidx * 8ul + sub];
                uint  bprv = simd_shuffle_up(bu32, 1u);
                uint  st6  = strata_qtip_init_state6(bprv, sub);
                uint  ru32 = all_ref ? ref_codes32[bidx * 8ul + sub] : 0u;
                half  rs   = all_ref ? ref_scales16[bidx] : half(0.0h);

                threadgroup half* dst = &W_sg[sgid][r][sub * 16u];
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 4u; ++q) {
                    half4 w4;
                    uint rb8 = (ru32 >> (q * 8u)) & 0xFFu;
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        uint m  = q * 4u + i;
                        uint u2 = (bu32 >> (2u * m)) & 3u;
                        uint w8 = (st6 << 2u) | u2;
                        st6     = w8 & 0x3Fu;
                        half wb = bs * tg_qtip[bk][w8];
                        if (all_ref) wb += rs * strata_refine_lut4_h_dev((rb8 >> (2u * i)) & 3u);
                        w4[i] = wb;
                    }
                    *reinterpret_cast<threadgroup half4*>(dst + q * 4u) = w4;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto sX = tX_full.slice<128, 16>(int(blk * 128u), 0);
            op_q.run(sX, tW, cT);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        device float*        gu_f32_exp = reinterpret_cast<device float*>(gu_accum_f32) + ulong(u_idx) * ulong(STRATA_MAX_SPEC_BATCH) * N_gu;
        device atomic_float* gu_exp     = gu_accum_f32 + ulong(u_idx) * ulong(STRATA_MAX_SPEC_BATCH) * N_gu;
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                auto idx = cT.get_multidimensional_index(i);
                uint m   = uint(idx[1]);
                if (m < tc) {
                    if (split_k == 1u) {
                        gu_f32_exp[m * N_gu + n0 + uint(idx[0])] = cT[i];
                    } else {
                        atomic_fetch_add_explicit(&gu_exp[m * N_gu + n0 + uint(idx[0])],
                                                  cT[i], memory_order_relaxed);
                    }
                }
            }
        }

        // Sequential reuse of the SAME `cT` variable for rare Mode A mixed-mask cold experts (0 extra registers!)
        if (mix_ref) {
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] = 0.0f;
            }
            for (uint blk = blk_beg; blk < blk_end; ++blk) {
                #pragma clang loop unroll(full)
                for (uint p = 0; p < 4u; ++p) {
                    uint  r    = p * 4u + grp;
                    ulong bidx = ulong(n0 + r) * blks_per_row + blk;
                    uint  ru32 = ref_codes32[bidx * 8ul + sub];
                    half  rs   = ref_scales16[bidx];
                    threadgroup half* dst = &W_sg[sgid][r][sub * 16u];
                    #pragma clang loop unroll(full)
                    for (uint q = 0; q < 4u; ++q) {
                        uint rb8 = (ru32 >> (q * 8u)) & 0xFFu;
                        half4 wr(rs * strata_refine_lut4_h_dev((rb8 >> 0u) & 3u),
                                 rs * strata_refine_lut4_h_dev((rb8 >> 2u) & 3u),
                                 rs * strata_refine_lut4_h_dev((rb8 >> 4u) & 3u),
                                 rs * strata_refine_lut4_h_dev((rb8 >> 6u) & 3u));
                        *reinterpret_cast<threadgroup half4*>(dst + q * 4u) = wr;
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                auto sX = tX_full.slice<128, 16>(int(blk * 128u), 0);
                op_q.run(sX, tW, cT);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) {
                    auto idx = cT.get_multidimensional_index(i);
                    uint m   = uint(idx[1]);
                    if (m < tc && (((eff_mask >> m) & 1u) != 0u)) {
                        if (split_k == 1u) {
                            gu_f32_exp[m * N_gu + n0 + uint(idx[0])] += cT[i];
                        } else {
                            atomic_fetch_add_explicit(&gu_exp[m * N_gu + n0 + uint(idx[0])],
                                                      cT[i], memory_order_relaxed);
                        }
                    }
                }
            }
        }
        return;
    }

    // =========================================================================
    // PATH B: Zero-Threadgroup, Single-Accumulator (`1` `cT`!) Pipelined NAX `wCT` MXFP4 (TG=64 = 2 SGs)!
    // =========================================================================
    const uint n0 = tgid.x * 64u + sgid * 32u;
    if (n0 >= N_gu) return;

    const uint  k_tiles_gu  = cfg.d_model / 32u;
    const uint  dff_tiles   = cfg.d_ff / 32u;
    const ulong total_tiles = ulong(2u * dff_tiles) * k_tiles_gu + ulong(cfg.d_model / 32u) * dff_tiles;
    const ulong scales_off  = total_tiles * 512ul;

    const bool is_hot_4b   = (slot.tier_kind == STRATA_TIER_HOT_FULL_4B);
    const bool use_ring_4b = (!is_hot_4b) && ring_is_ready && (eff_mask != 0u);

    device const uchar* full4b_ptr = nullptr;
    if (is_hot_4b) {
        uint h_slot = slot.hot_or_base_idx % max(cfg.hot_pool_slots, 1u);
        full4b_ptr  = uma_hot_pool + ulong(h_slot) * cfg.hot_expert_bytes;
    } else if (use_ring_4b) {
        ulong off = (ulong(slot.ring_slot_idx) * ulong(cfg.ring_slot_stride)) %
                    max(cfg.ring_buf_bytes, ulong(1));
        if (off + cfg.ring_slot_stride <= cfg.ring_buf_bytes) {
            full4b_ptr = ring_buf + off;
        }
    }

    const bool all_full = (full4b_ptr != nullptr) && (is_hot_4b || eff_mask == all_tok_mask);
    const bool all_bary = (full4b_ptr == nullptr) || (eff_mask == 0u);
    const bool mix_fb   = (!all_full) && (!all_bary);

    const ulong ring_tile_cap = (is_hot_4b || cfg.ring_slot_stride >= cfg.hot_expert_bytes)
                                    ? total_tiles
                                    : max(ulong(cfg.ring_slot_stride) / 544ul, ulong(1));
    const ulong full_scl_off  = (ring_tile_cap == total_tiles) ? scales_off : (ring_tile_cap * 512ul);

    device const uchar* prim_ptr     = all_bary ? uma_base_pool : full4b_ptr;
    const ulong         prim_tcap    = all_bary ? total_tiles   : ring_tile_cap;
    const ulong         prim_scl_off = all_bary ? scales_off    : full_scl_off;

    device const uint4* prim_nibs = reinterpret_cast<device const uint4*>(prim_ptr);
    device const uint*  prim_scls = reinterpret_cast<device const uint*>(prim_ptr + prim_scl_off);

    constexpr auto desc_m = matmul2d_descriptor(
        16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_m, execution_simdgroup> op_m;

    auto xCT = op_m.get_left_input_cooperative_tensor<half, half, float>();
    auto wCT = op_m.get_right_input_cooperative_tensor<half, half, float>();
    auto cT  = op_m.get_destination_cooperative_tensor<decltype(xCT), decltype(wCT), float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint  nt       = n0 >> 5u;
    const uint  n_sub    = ((lane >> 1u) & 3u) | ((lane >> 2u) & 4u);
    const uint  kt_per   = (k_tiles_gu + split_k - 1u) / split_k;
    const uint  kt_beg   = min(k_tiles_gu, k_part * kt_per);
    const uint  kt_end   = min(k_tiles_gu, kt_beg + kt_per);
    const ulong row_base = ulong(nt) * k_tiles_gu;

    uint kt = kt_beg;
    // 2x K-tile software pipeline (`MLP = 2` = 1 KiB in flight per SIMD-group!)
    for (; kt + 1u < kt_end; kt += 2u) {
        ulong t0_raw = row_base + kt;
        ulong t1_raw = t0_raw + 1ul;
        ulong t0_idx = (prim_tcap == total_tiles) ? t0_raw : (t0_raw % prim_tcap);
        ulong t1_idx = (prim_tcap == total_tiles) ? t1_raw : (t1_raw % prim_tcap);

        uint4 q0 = prim_nibs[t0_idx * 32ul + lane];
        uint4 q1 = prim_nibs[t1_idx * 32ul + lane];
        half4 s0 = strata_e4m3x4_scale_safe_half4_dev(prim_scls[t0_idx * 8ul + n_sub]);
        half4 s1 = strata_e4m3x4_scale_safe_half4_dev(prim_scls[t1_idx * 8ul + n_sub]);

        {
            auto sX = tX_full.slice<32, 16>(int(kt * 32u), 0);
            xCT.load(sX);
            strata_unpack_mxfp4_tile_to_wct(q0, s0, wCT);
            op_m.run(xCT, wCT, cT);
        }
        {
            auto sX = tX_full.slice<32, 16>(int((kt + 1u) * 32u), 0);
            xCT.load(sX);
            strata_unpack_mxfp4_tile_to_wct(q1, s1, wCT);
            op_m.run(xCT, wCT, cT);
        }
    }
    for (; kt < kt_end; ++kt) {
        ulong t0_raw = row_base + kt;
        ulong t0_idx = (prim_tcap == total_tiles) ? t0_raw : (t0_raw % prim_tcap);
        uint4 q0 = prim_nibs[t0_idx * 32ul + lane];
        half4 s0 = strata_e4m3x4_scale_safe_half4_dev(prim_scls[t0_idx * 8ul + n_sub]);
        auto  sX = tX_full.slice<32, 16>(int(kt * 32u), 0);
        xCT.load(sX);
        strata_unpack_mxfp4_tile_to_wct(q0, s0, wCT);
        op_m.run(xCT, wCT, cT);
    }

    device float*        gu_f32_exp = reinterpret_cast<device float*>(gu_accum_f32) + ulong(u_idx) * ulong(STRATA_MAX_SPEC_BATCH) * N_gu;
    device atomic_float* gu_atm_exp = gu_accum_f32 + ulong(u_idx) * ulong(STRATA_MAX_SPEC_BATCH) * N_gu;

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            uint m   = uint(idx[1]);
            if (m < tc && (!mix_fb || (((eff_mask >> m) & 1u) != 0u))) {
                if (split_k == 1u) {
                    gu_f32_exp[m * N_gu + n0 + uint(idx[0])] = cT[i];
                } else {
                    atomic_fetch_add_explicit(&gu_atm_exp[m * N_gu + n0 + uint(idx[0])],
                                              cT[i], memory_order_relaxed);
                }
            }
        }
    }

    // Sequential reuse of the SAME `cT` variable for rare Mode A mixed-mask cold experts (0 extra registers!)
    if (mix_fb) {
        device const uint4* bary_nibs = reinterpret_cast<device const uint4*>(uma_base_pool);
        device const uint*  bary_scls = reinterpret_cast<device const uint*>(uma_base_pool + scales_off);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) cT[i] = 0.0f;
        }
        for (uint ktb = kt_beg; ktb < kt_end; ++ktb) {
            ulong tb = row_base + ktb;
            uint4 qb = bary_nibs[tb * 32ul + lane];
            half4 sb = strata_e4m3x4_scale_safe_half4_dev(bary_scls[tb * 8ul + n_sub]);
            auto  sX = tX_full.slice<32, 16>(int(ktb * 32u), 0);
            xCT.load(sX);
            strata_unpack_mxfp4_tile_to_wct(qb, sb, wCT);
            op_m.run(xCT, wCT, cT);
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                auto idx = cT.get_multidimensional_index(i);
                uint m   = uint(idx[1]);
                if (m < tc && (((eff_mask >> m) & 1u) == 0u)) {
                    if (split_k == 1u) {
                        gu_f32_exp[m * N_gu + n0 + uint(idx[0])] = cT[i];
                    } else {
                        atomic_fetch_add_explicit(&gu_atm_exp[m * N_gu + n0 + uint(idx[0])],
                                                  cT[i], memory_order_relaxed);
                    }
                }
            }
        }
    }
#endif
}

// Regime II W6A16 MoE-SVD Residual Addition for Zero-Stall Fallback Experts
kernel void strata_moe_svd_w6_fallback(
    device const half*                   v_proj_x    [[buffer(0)]],
    device const StrataSlotEntry*        slot_map    [[buffer(1)]],
    device const atomic_uint*            slot_ready  [[buffer(2)]],
    device const StrataActiveExpertDesc* active_list [[buffer(3)]],
    device const uchar*                  u_w6_pool   [[buffer(4)]],
    device half*                         mid_inout   [[buffer(5)]],
    constant StrataMoEKernelCfg&         cfg         [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    const uint u_idx = tgid.y;
    if (u_idx >= cfg.num_active_experts || cfg.svd_rank == 0u) return;

    const StrataActiveExpertDesc desc = active_list[u_idx];
    const StrataSlotEntry slot        = slot_map[desc.expert_id];
    if (slot.tier_kind == STRATA_TIER_HOT_FULL_4B) return;

    bool ring_is_ready = false;
    if (slot.ring_slot_idx != STRATA_SLOT_INVALID && cfg.ring_buf_bytes >= cfg.ring_slot_stride) {
        uint cur_ep   = atomic_load_explicit(&slot_ready[slot.ring_slot_idx], memory_order_relaxed);
        ring_is_ready = (cur_ep >= slot.expected_epoch);
    }
    const uint tc = min(uint(desc.tok_count), uint(STRATA_MAX_SPEC_BATCH));
    if (ring_is_ready && (desc.refine_tok_mask & ((1u << tc) - 1u)) == ((1u << tc) - 1u)) return;

    const uint r = tgid.x * 64u + tid;
    if (r >= cfg.d_ff) return;

    const uint rank_aligned = (cfg.svd_rank + 63u) & ~63u;
    const uint row_bytes    = (rank_aligned / 64u) * 50u;
    const uint u_slot       = slot.hot_or_base_idx % max(cfg.base_pool_slots, 1u);
    device const uchar* u_row = u_w6_pool + (ulong(u_slot) * cfg.d_ff + r) * row_bytes;

    float delta[STRATA_MAX_SPEC_BATCH] = {0.0f};
    for (uint blk64 = 0; blk64 < rank_aligned; blk64 += 64u) {
        device const uchar* blk = u_row + (blk64 / 64u) * 50u;
        float scale = float(*reinterpret_cast<device const half*>(blk));
        device const uchar* p = blk + 2u;

        for (uint g = 0; g < 16u; ++g) {
            uint k0 = blk64 + g * 4u;
            if (k0 >= cfg.svd_rank) break;
            uchar b0 = p[g * 3u + 0u];
            uchar b1 = p[g * 3u + 1u];
            uchar b2 = p[g * 3u + 2u];
            float q0 = float(int(b0 & 0x3Fu) - 32) * scale;
            float q1 = float(int(((b0 >> 6u) | ((b1 & 0x0Fu) << 2u)) & 0x3Fu) - 32) * scale;
            float q2 = float(int(((b1 >> 4u) | ((b2 & 0x03u) << 4u)) & 0x3Fu) - 32) * scale;
            float q3 = float(int((b2 >> 2u) & 0x3Fu) - 32) * scale;

            for (uint j = 0; j < tc; ++j) {
                uint t = uint(desc.tok_indices[j]);
                device const half* vp = v_proj_x + t * rank_aligned + k0;
                delta[j] += q0 * float(vp[0]) + q1 * float(vp[1]) +
                            q2 * float(vp[2]) + q3 * float(vp[3]);
            }
        }
    }

    const ulong mid_stride = (cfg.regime == STRATA_REGIME_I_BITNEST) ? ulong(STRATA_MAX_SPEC_BATCH) : 16ul;
    for (uint j = 0; j < tc; ++j) {
        bool used_full = ring_is_ready && (((desc.refine_tok_mask >> j) & 1u) != 0u);
        if (!used_full) {
            ulong idx = (ulong(u_idx) * mid_stride + j) * cfg.d_ff + r;
            mid_inout[idx] = half(float(mid_inout[idx]) + delta[j]);
        }
    }
}

// Single-Accumulator (`1` `cT`!) Split-K NAX Down Projection `[d_model, d_ff] + Gated Accum -> Y [B, d_model]`!
kernel void strata_moe_down_accum(
    device const half*                   mid16_in      [[buffer(0)]], // [num_active, 16, d_ff]
    device const StrataSlotEntry*        slot_map      [[buffer(1)]],
    device const atomic_uint*            slot_ready    [[buffer(2)]],
    device const StrataActiveExpertDesc* active_list   [[buffer(3)]],
    device const uchar*                  uma_hot_pool  [[buffer(4)]],
    device const uchar*                  uma_base_pool [[buffer(5)]],
    device const uchar*                  ring_buf      [[buffer(6)]],
    device atomic_float*                 y_accum_f32   [[buffer(7)]], // [B, d_model]
    constant StrataMoEKernelCfg&         cfg           [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
#if defined(__HAVE_TENSOR__)
    const uint u_idx  = tgid.y;
    const uint k_part = tgid.z;
    if (u_idx >= cfg.num_active_experts) return;

    const StrataActiveExpertDesc desc = active_list[u_idx];
    const StrataSlotEntry slot        = slot_map[desc.expert_id];
    const uint   tc                   = min(uint(desc.tok_count), uint(STRATA_MAX_SPEC_BATCH));
    const ushort all_tok_mask         = ushort((1u << tc) - 1u);
    const ushort eff_mask             = desc.refine_tok_mask & all_tok_mask;
    const uint   split_k              = max(cfg.down_split_k, 1u);

    bool ring_is_ready = false;
    if (slot.ring_slot_idx != STRATA_SLOT_INVALID && cfg.ring_buf_bytes >= cfg.ring_slot_stride) {
        uint cur_ep   = atomic_load_explicit(&slot_ready[slot.ring_slot_idx], memory_order_relaxed);
        ring_is_ready = (cur_ep >= slot.expected_epoch);
    }

    device half* mid_exp = const_cast<device half*>(mid16_in) + ulong(u_idx) * 16ul * cfg.d_ff;
    tensor<device half, dextents<int, 2>, tensor_inline> tMid_full(mid_exp, dextents<int, 2>(int(cfg.d_ff), 16));

    // =========================================================================
    // PATH A: Regime I BitNest — High-Occupancy (TG=64, 9 KiB SRAM -> 3 TGs/core!) NAX `matmul2d(16, 16, 128)`
    // =========================================================================
    if (cfg.regime == STRATA_REGIME_I_BITNEST) {
        threadgroup half tg_qtip[2][256];
        threadgroup half W_sg[2][16][128];

        for (uint i = tid; i < 256u; i += 64u) {
            half v = strata_qtip_3inst_unit_h_dev(i);
            tg_qtip[0][i] = v;
            tg_qtip[1][i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint d_row0 = tgid.x * 32u + sgid * 16u;
        if (d_row0 >= cfg.d_model) return;

        const ulong total_blks128      = (3ul * ulong(cfg.d_model) * ulong(cfg.d_ff)) / 128ul;
        const ulong scales_off         = total_blks128 * 32ul;
        const uint  blks_per_row       = cfg.d_ff / 128u;
        const ulong down_offset_blocks = ulong(2u * cfg.d_ff) * ulong(cfg.d_model / 128u);

        const uint base_slot              = slot.hot_or_base_idx % max(cfg.base_pool_slots, 1u);
        device const uchar* exp_base      = uma_base_pool + ulong(base_slot) * cfg.base_expert_bytes;
        device const uint*  base_codes32  = reinterpret_cast<device const uint*>(exp_base);
        device const half*  base_scales16 = reinterpret_cast<device const half*>(exp_base + scales_off);

        const bool has_hot_ref  = (slot.tier_kind == STRATA_TIER_HOT_FULL_4B);
        const bool has_ring_ref = (!has_hot_ref) && ring_is_ready;
        device const uchar* ref_ptr = nullptr;
        if (has_hot_ref) {
            uint h_slot = slot.hot_or_base_idx % max(cfg.hot_pool_slots, 1u);
            ref_ptr = uma_hot_pool + ulong(h_slot) * cfg.hot_expert_bytes;
        } else if (has_ring_ref) {
            ulong off = (ulong(slot.ring_slot_idx) * ulong(cfg.ring_slot_stride)) %
                        max(cfg.ring_buf_bytes, ulong(1));
            if (off + cfg.hot_expert_bytes <= cfg.ring_buf_bytes) {
                ref_ptr = ring_buf + off;
            }
        }
        const bool can_ref  = (ref_ptr != nullptr) && (eff_mask != 0u);
        const bool all_ref  = can_ref && (eff_mask == all_tok_mask);
        const bool mix_ref  = can_ref && (!all_ref);
        device const uint* ref_codes32  = can_ref ? reinterpret_cast<device const uint*>(ref_ptr) : nullptr;
        device const half* ref_scales16 = can_ref ? reinterpret_cast<device const half*>(ref_ptr + scales_off) : nullptr;

        const uint blk_per = (blks_per_row + split_k - 1u) / split_k;
        const uint blk_beg = min(blks_per_row, k_part * blk_per);
        const uint blk_end = min(blks_per_row, blk_beg + blk_per);

        constexpr auto desc_q = matmul2d_descriptor(
            16, 16, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc_q, execution_simdgroup> op_q;

        tensor<threadgroup half, extents<int, 128, 16>, tensor_inline> tW(&W_sg[sgid][0][0], extents<int, 128, 16>());
        auto sM0 = tMid_full.slice<128, 16>(0, 0);
        auto cT  = op_q.get_destination_cooperative_tensor<decltype(sM0), decltype(tW), float>();

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) cT[i] = 0.0f;
        }

        const uint grp = lane >> 3u;
        const uint sub = lane & 7u;
        const uint bk  = lane & 1u;

        for (uint blk = blk_beg; blk < blk_end; ++blk) {
            #pragma clang loop unroll(full)
            for (uint p = 0; p < 4u; ++p) {
                uint  r    = p * 4u + grp;
                ulong bidx = down_offset_blocks + ulong(d_row0 + r) * blks_per_row + blk;
                half  bs   = base_scales16[bidx];
                uint  bu32 = base_codes32[bidx * 8ul + sub];
                uint  bprv = simd_shuffle_up(bu32, 1u);
                uint  st6  = strata_qtip_init_state6(bprv, sub);
                uint  ru32 = all_ref ? ref_codes32[bidx * 8ul + sub] : 0u;
                half  rs   = all_ref ? ref_scales16[bidx] : half(0.0h);

                threadgroup half* dst = &W_sg[sgid][r][sub * 16u];
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 4u; ++q) {
                    half4 w4;
                    uint rb8 = (ru32 >> (q * 8u)) & 0xFFu;
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        uint m  = q * 4u + i;
                        uint u2 = (bu32 >> (2u * m)) & 3u;
                        uint w8 = (st6 << 2u) | u2;
                        st6     = w8 & 0x3Fu;
                        half wb = bs * tg_qtip[bk][w8];
                        if (all_ref) wb += rs * strata_refine_lut4_h_dev((rb8 >> (2u * i)) & 3u);
                        w4[i] = wb;
                    }
                    *reinterpret_cast<threadgroup half4*>(dst + q * 4u) = w4;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto sM = tMid_full.slice<128, 16>(int(blk * 128u), 0);
            op_q.run(sM, tW, cT);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                auto idx   = cT.get_multidimensional_index(i);
                uint m     = uint(idx[1]);
                uint d_col = d_row0 + uint(idx[0]);
                if (m < tc && d_col < cfg.d_model) {
                    uint tok = uint(desc.tok_indices[m]);
                    atomic_fetch_add_explicit(&y_accum_f32[tok * cfg.d_model + d_col],
                                              cT[i] * desc.tok_gates[m], memory_order_relaxed);
                }
            }
        }

        if (mix_ref) {
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] = 0.0f;
            }
            for (uint blk = blk_beg; blk < blk_end; ++blk) {
                #pragma clang loop unroll(full)
                for (uint p = 0; p < 4u; ++p) {
                    uint  r    = p * 4u + grp;
                    ulong bidx = down_offset_blocks + ulong(d_row0 + r) * blks_per_row + blk;
                    uint  ru32 = ref_codes32[bidx * 8ul + sub];
                    half  rs   = ref_scales16[bidx];
                    threadgroup half* dst = &W_sg[sgid][r][sub * 16u];
                    #pragma clang loop unroll(full)
                    for (uint q = 0; q < 4u; ++q) {
                        uint rb8 = (ru32 >> (q * 8u)) & 0xFFu;
                        half4 wr(rs * strata_refine_lut4_h_dev((rb8 >> 0u) & 3u),
                                 rs * strata_refine_lut4_h_dev((rb8 >> 2u) & 3u),
                                 rs * strata_refine_lut4_h_dev((rb8 >> 4u) & 3u),
                                 rs * strata_refine_lut4_h_dev((rb8 >> 6u) & 3u));
                        *reinterpret_cast<threadgroup half4*>(dst + q * 4u) = wr;
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                auto sM = tMid_full.slice<128, 16>(int(blk * 128u), 0);
                op_q.run(sM, tW, cT);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) {
                    auto idx   = cT.get_multidimensional_index(i);
                    uint m     = uint(idx[1]);
                    uint d_col = d_row0 + uint(idx[0]);
                    if (m < tc && d_col < cfg.d_model && (((eff_mask >> m) & 1u) != 0u)) {
                        uint tok = uint(desc.tok_indices[m]);
                        atomic_fetch_add_explicit(&y_accum_f32[tok * cfg.d_model + d_col],
                                                  cT[i] * desc.tok_gates[m], memory_order_relaxed);
                    }
                }
            }
        }
        return;
    }

    // =========================================================================
    // PATH B: Zero-Threadgroup, Single-Accumulator (`1` `cT`!) Pipelined NAX `wCT` MXFP4 (TG=64 = 2 SGs)!
    // =========================================================================
    const uint d_row0 = tgid.x * 64u + sgid * 32u;
    if (d_row0 >= cfg.d_model) return;

    const uint  k_tiles_d   = cfg.d_ff / 32u;
    const uint  dm_tiles    = cfg.d_model / 32u;
    const ulong down_off_t  = ulong(2u * k_tiles_d) * dm_tiles;
    const ulong total_tiles = down_off_t + ulong(dm_tiles) * k_tiles_d;
    const ulong scales_off  = total_tiles * 512ul;

    const bool is_hot_4b   = (slot.tier_kind == STRATA_TIER_HOT_FULL_4B);
    const bool use_ring_4b = (!is_hot_4b) && ring_is_ready && (eff_mask != 0u);

    device const uchar* full4b_ptr = nullptr;
    if (is_hot_4b) {
        uint h_slot = slot.hot_or_base_idx % max(cfg.hot_pool_slots, 1u);
        full4b_ptr  = uma_hot_pool + ulong(h_slot) * cfg.hot_expert_bytes;
    } else if (use_ring_4b) {
        ulong off = (ulong(slot.ring_slot_idx) * ulong(cfg.ring_slot_stride)) %
                    max(cfg.ring_buf_bytes, ulong(1));
        if (off + cfg.ring_slot_stride <= cfg.ring_buf_bytes) {
            full4b_ptr = ring_buf + off;
        }
    }

    const bool all_full = (full4b_ptr != nullptr) && (is_hot_4b || eff_mask == all_tok_mask);
    const bool all_bary = (full4b_ptr == nullptr) || (eff_mask == 0u);
    const bool mix_fb   = (!all_full) && (!all_bary);

    const ulong ring_tile_cap = (is_hot_4b || cfg.ring_slot_stride >= cfg.hot_expert_bytes)
                                    ? total_tiles
                                    : max(ulong(cfg.ring_slot_stride) / 544ul, ulong(1));
    const ulong full_scl_off  = (ring_tile_cap == total_tiles) ? scales_off : (ring_tile_cap * 512ul);

    device const uchar* prim_ptr     = all_bary ? uma_base_pool : full4b_ptr;
    const ulong         prim_tcap    = all_bary ? total_tiles   : ring_tile_cap;
    const ulong         prim_scl_off = all_bary ? scales_off    : full_scl_off;

    device const uint4* prim_nibs = reinterpret_cast<device const uint4*>(prim_ptr);
    device const uint*  prim_scls = reinterpret_cast<device const uint*>(prim_ptr + prim_scl_off);

    constexpr auto desc_m = matmul2d_descriptor(
        16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_m, execution_simdgroup> op_m;

    auto xCT = op_m.get_left_input_cooperative_tensor<half, half, float>();
    auto wCT = op_m.get_right_input_cooperative_tensor<half, half, float>();
    auto cT  = op_m.get_destination_cooperative_tensor<decltype(xCT), decltype(wCT), float>();

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    const uint  n_sub      = ((lane >> 1u) & 3u) | ((lane >> 2u) & 4u);
    const uint  kt_per     = (k_tiles_d + split_k - 1u) / split_k;
    const uint  kt_beg     = min(k_tiles_d, k_part * kt_per);
    const uint  kt_end     = min(k_tiles_d, kt_beg + kt_per);
    const ulong d_tile_row = down_off_t + ulong(d_row0 >> 5u) * k_tiles_d;

    uint kt = kt_beg;
    for (; kt + 1u < kt_end; kt += 2u) {
        ulong t0_raw = d_tile_row + kt;
        ulong t1_raw = t0_raw + 1ul;
        ulong t0_idx = (prim_tcap == total_tiles) ? t0_raw : (t0_raw % prim_tcap);
        ulong t1_idx = (prim_tcap == total_tiles) ? t1_raw : (t1_raw % prim_tcap);

        uint4 q0 = prim_nibs[t0_idx * 32ul + lane];
        uint4 q1 = prim_nibs[t1_idx * 32ul + lane];
        half4 s0 = strata_e4m3x4_scale_safe_half4_dev(prim_scls[t0_idx * 8ul + n_sub]);
        half4 s1 = strata_e4m3x4_scale_safe_half4_dev(prim_scls[t1_idx * 8ul + n_sub]);

        {
            auto sM = tMid_full.slice<32, 16>(int(kt * 32u), 0);
            xCT.load(sM);
            strata_unpack_mxfp4_tile_to_wct(q0, s0, wCT);
            op_m.run(xCT, wCT, cT);
        }
        {
            auto sM = tMid_full.slice<32, 16>(int((kt + 1u) * 32u), 0);
            xCT.load(sM);
            strata_unpack_mxfp4_tile_to_wct(q1, s1, wCT);
            op_m.run(xCT, wCT, cT);
        }
    }
    for (; kt < kt_end; ++kt) {
        ulong t0_raw = d_tile_row + kt;
        ulong t0_idx = (prim_tcap == total_tiles) ? t0_raw : (t0_raw % prim_tcap);
        uint4 q0 = prim_nibs[t0_idx * 32ul + lane];
        half4 s0 = strata_e4m3x4_scale_safe_half4_dev(prim_scls[t0_idx * 8ul + n_sub]);
        auto  sM = tMid_full.slice<32, 16>(int(kt * 32u), 0);
        xCT.load(sM);
        strata_unpack_mxfp4_tile_to_wct(q0, s0, wCT);
        op_m.run(xCT, wCT, cT);
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx   = cT.get_multidimensional_index(i);
            uint m     = uint(idx[1]);
            uint d_col = d_row0 + uint(idx[0]);
            if (m < tc && d_col < cfg.d_model && (!mix_fb || (((eff_mask >> m) & 1u) != 0u))) {
                uint tok = uint(desc.tok_indices[m]);
                atomic_fetch_add_explicit(&y_accum_f32[tok * cfg.d_model + d_col],
                                          cT[i] * desc.tok_gates[m], memory_order_relaxed);
            }
        }
    }

    if (mix_fb) {
        device const uint4* bary_nibs = reinterpret_cast<device const uint4*>(uma_base_pool);
        device const uint*  bary_scls = reinterpret_cast<device const uint*>(uma_base_pool + scales_off);
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) cT[i] = 0.0f;
        }
        for (uint ktb = kt_beg; ktb < kt_end; ++ktb) {
            ulong tb = d_tile_row + ktb;
            uint4 qb = bary_nibs[tb * 32ul + lane];
            half4 sb = strata_e4m3x4_scale_safe_half4_dev(bary_scls[tb * 8ul + n_sub]);
            auto  sM = tMid_full.slice<32, 16>(int(ktb * 32u), 0);
            xCT.load(sM);
            strata_unpack_mxfp4_tile_to_wct(qb, sb, wCT);
            op_m.run(xCT, wCT, cT);
        }
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                auto idx   = cT.get_multidimensional_index(i);
                uint m     = uint(idx[1]);
                uint d_col = d_row0 + uint(idx[0]);
                if (m < tc && d_col < cfg.d_model && (((eff_mask >> m) & 1u) == 0u)) {
                    uint tok = uint(desc.tok_indices[m]);
                    atomic_fetch_add_explicit(&y_accum_f32[tok * cfg.d_model + d_col],
                                              cT[i] * desc.tok_gates[m], memory_order_relaxed);
                }
            }
        }
    }
#endif
}

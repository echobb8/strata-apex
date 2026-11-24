#ifndef STRATA_INKLING2_H
#define STRATA_INKLING2_H

/*
 * STRATA-M5 v2.0 — Native Apple M5 Max Metal 4 Engine ABI for `thinkingmachines/Inkling-2-Mini`
 *
 * Architecture (`inkling2_loop_text` + 3-Layer DSpark `rnn_hidden` MTP):
 *   - 24 physical layers (`d=2304`, 20 local SWA-1024 + 4 global full-attn `{5, 11, 17, 23}`)
 *   - Looped middle core (`core_start=6`, `core_end=18`, `num_passes=3`, `gate_rank=64`)
 *     => 48 logical layer visits per token (`6` prelude + `3 x 12` core + `6` coda: 40 SWA + 8 Global)
 *   - Block Attention Residuals (`n_layers_per_attn_res_block=6`, unweighted RMSNorm on K & V)
 *   - Relative-Position Bias Attention (`d_rel=16`, `rel_extent=1024`, NO RoPE on backbone!) +
 *     per-query attention sink + L2-clipped Q/K/R + unweighted V-RMSNorm + sigmoid gate +
 *     causal short-conv (kernel=4, static on SWA/attn/mlp, dynamic per-head taps on global K/V) +
 *     log-position query/rel scaling (`n_floor=8192`, `alpha=0.1`) on global layers
 *   - 128-Expert Top-8 Sparse MoE (`intermediate=1152`, `route_scale=8.0`, FP32 sigmoid router + bias,
 *     Softcapped SiTU SwiGLU: `4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)`)
 *   - 100% Native Mixed-Precision Zero-Copy Single-Buffer UMA (`~18.2 GiB` total, 16 KiB page-aligned):
 *       * Layers 0..19 MoE Experts  : Native NVFP4 (`E2M1`, block 1x16, `F8_E4M3` scale + `F32` `amax/denom`)
 *       * Layers 20..23 MoE Experts : Native MXFP8 (`F8_E4M3`, block 1x32, `UE8M0` scale)
 *       * Dense Projections / Embed / Unembed / MTP : IEEE FP16 (`half`)
 *       * Norms, Clips, Rel-Bias, SConv, AttnRes, Router, Bias : Lossless FP32 (`float`)
 *   - Native 3-Layer DSpark Block-8 (`K=1..7`) `rnn_hidden` Markov MTP Drafter (`target_layer_ids={5,23,41,48}`)
 *   - Final logit softcapping (`30.0`) over `unpadded_vocab_size = 200058` + 100% bit-exact sliding-window penalties
 *
 * Shared between C/C++/ObjC++ host code and Metal 4 MSL kernels via `// @@STRATA_INKLING2_HEADER@@`.
 */

#ifndef __METAL_VERSION__
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "strata_paths.h"
typedef uint8_t  sik_u8;
typedef uint16_t sik_u16;
typedef uint32_t sik_u32;
typedef uint64_t sik_u64;
#else
typedef uchar    sik_u8;
typedef ushort   sik_u16;
typedef uint     sik_u32;
typedef ulong    sik_u64;
#endif

/* ============================================================================
 * 1. Format Constants & Canonical Inkling-2-Mini Special Token IDs
 * ============================================================================ */

#define STRATA_INKLING2_MAGIC_U64              0x304B4E49354D5453ULL /* "STM5INK0" */
#define STRATA_INKLING2_FORMAT_VERSION_V1      1U
#define STRATA_INKLING2_FORMAT_VERSION         2U
#define STRATA_INKLING2_PAGE_BYTES             16384ULL              /* 16 KiB Apple Silicon VM page */
#define STRATA_INKLING2_HEADER_PADDED_BYTES    131072U               /* 8 * 16 KiB = 128 KiB */
#define STRATA_INKLING2_TENSOR_ALIGN           256ULL                /* 256 B Metal buffer offset align */
#define STRATA_INKLING2_QUANT_GROUP_SIZE       64U                   /* Group size for Q8_0 and Q4_A */
#define STRATA_INKLING2_FRSPEC_SHORTLIST_SIZE  32768U                /* 32K weight-intrinsic FR-Spec shortlist */

#define STRATA_INKLING2_MAX_PHYS_LAYERS        32U                   /* 24 physical layers */
#define STRATA_INKLING2_MAX_LOGICAL_LAYERS     64U                   /* 48 logical layer visits */
#define STRATA_INKLING2_MAX_MTP_LAYERS         8U                    /* 3 DSpark MTP layers */
#define STRATA_INKLING2_MAX_MTP_TAPS           4U                    /* target_layer_ids = {5,23,41,48} */
#define STRATA_INKLING2_MAX_SPEC_DRAFT_K       7U                    /* block_size=8 => up to 7 drafted toks */
#define STRATA_INKLING2_DEFAULT_MAX_SEQ_LEN    65536U                /* 64K default (supports up to 1048576 / 1M) */
#define STRATA_INKLING2_MAX_SEQ_LEN_CAP        1048576U              /* Native 1M context (`model_max_length`) */
#define STRATA_INKLING2_MAX_BATCH_SIZE         64U                   /* Max concurrent continuous-batch streams */

/* Tensor dtypes in `inkling2_uma.strata` */
#define STRATA_INKLING2_DTYPE_NONE             0U
#define STRATA_INKLING2_DTYPE_F32              1U /* IEEE-754 float32 (or raw U32 4B/elem e.g. frspec.token_ids) */
#define STRATA_INKLING2_DTYPE_F16              2U /* IEEE-754 float16 */
#define STRATA_INKLING2_DTYPE_NVFP4            3U /* Packed 2x E2M1 nibbles/byte + F8_E4M3 block-16 scale + F32 expert scale */
#define STRATA_INKLING2_DTYPE_MXFP8            4U /* F8_E4M3 weight bytes + UE8M0 block-32 scale bytes */
#define STRATA_INKLING2_DTYPE_Q8_0             5U /* Planar symmetric Int8 gs=64: i8[R,C] + align256 + f16[R,C/64] */
#define STRATA_INKLING2_DTYPE_Q4_A             6U /* Planar affine Q4 gs=64: u32[R,C/8] + align256 + f16[R,C/64] scales + align256 + f16[R,C/64] biases */

/* Runtime Global KV Cache dtypes (`strata_inkling2_gpu_create_ex`; shared with Metal MSL) */
#define STRATA_INKLING2_KV_DTYPE_F16           0U /* Default lossless IEEE FP16 KV (32.00 KiB/tok across 8 global visits) */
#define STRATA_INKLING2_KV_DTYPE_Q8_0          1U /* Per-head symmetric int8[128] + planar half scale[n_kv, stride] (16.25 KiB/tok) */

/* Canonical Inkling-2-Mini special token IDs (`unpadded_vocab_size = 200058`, `padded = 201024`) */
#define STRATA_INKLING2_TOKEN_ENDOFTEXT        199999U /* `<|endoftext|>` (pad / secondary EOS) */
#define STRATA_INKLING2_TOKEN_MSG_USER         200000U /* `<|message_user|>` */
#define STRATA_INKLING2_TOKEN_MSG_MODEL        200001U /* `<|message_model|>` */
#define STRATA_INKLING2_TOKEN_MSG_SYSTEM       200002U /* `<|message_system|>` */
#define STRATA_INKLING2_TOKEN_MSG_TOOL         200003U /* `<|message_tool|>` */
#define STRATA_INKLING2_TOKEN_CONTENT_TEXT     200004U /* `<|content_text|>` */
#define STRATA_INKLING2_TOKEN_CONTENT_IMAGE    200005U /* `<|content_image|>` */
#define STRATA_INKLING2_TOKEN_EOS              200006U /* `<|content_model_end_sampling|>` (primary EOS) */
#define STRATA_INKLING2_TOKEN_CONTENT_THINK    200008U /* `<|content_thinking|>` */
#define STRATA_INKLING2_TOKEN_END_MSG          200010U /* `<|end_message|>` */
#define STRATA_INKLING2_TOKEN_TOOL_ERROR       200022U /* `<|content_tool_error|>` */
#define STRATA_INKLING2_TOKEN_CONTENT_XML      200024U /* `<|content_xml|>` */
#define STRATA_INKLING2_TOKEN_BOS              200028U /* `<|begin_of_text|>` */
#define STRATA_INKLING2_TOKEN_INVOKE_JSON      200049U /* `<|content_invoke_tool_json|>` */
#define STRATA_INKLING2_TOKEN_INVOKE_TEXT      200057U /* `<|content_invoke_tool_text|>` */

#define STRATA_INKLING2_UMA_FILENAME           "inkling2_uma.strata"
#define STRATA_INKLING2_DEFAULT_HF_REPO        "thinkingmachines/Inkling-2-Mini"
#ifndef __METAL_VERSION__
#define STRATA_INKLING2_DEFAULT_HF_DIR         strata_apex_bundle_path("inkling-2-mini-hf")
#define STRATA_INKLING2_DEFAULT_BUNDLE_DIR     strata_apex_bundle_path("inkling-2-mini")
#define STRATA_INKLING2_DEFAULT_FAST_BUNDLE_DIR strata_apex_bundle_path("inkling-2-mini-fast")
#endif

/* ============================================================================
 * 2. Page-Aligned On-Disk `.strata` Descriptors (POD, Exact 8-Byte Aligned)
 * ============================================================================ */

/* Dense 1D/2D/3D Tensor Slice Descriptor (`sizeof = 32` bytes, 256B-aligned offset)
 *
 * Sub-array layouts for quantized planar slices (`gs = 64`):
 *   - `STRATA_INKLING2_DTYPE_Q8_0` (`shape0 = rows`, `shape1 = cols`, `cols % 64 == 0`):
 *       * `offset + 0`                         : `int8_t   quants[rows, cols]`
 *       * `offset + align256(rows * cols)`     : `uint16_t scales_f16[rows, cols / 64]`
 *       * `w = float(q_i8) * float(scale_f16)`
 *       * `bytes = align256(rows * cols) + align256(rows * (cols / 64) * 2)`
 *   - `STRATA_INKLING2_DTYPE_Q4_A` (`shape0 = rows`, `shape1 = cols`, `cols % 64 == 0`):
 *       * `offset + 0`                         : `uint32_t quants[rows, cols / 8]` (nibble k = `(u32 >> (4*k)) & 0xF`)
 *       * `offset + align256(rows * cols / 2)` : `uint16_t scales_f16[rows, cols / 64]`
 *       * `+ align256(rows * (cols / 64) * 2)` : `uint16_t biases_f16[rows, cols / 64]`
 *       * `w = float(nibble) * float(scale_f16) + float(bias_f16)`
 *       * `bytes = align256(rows * (cols / 8) * 4) + 2 * align256(rows * (cols / 64) * 2)`
 */
typedef struct StrataInkling2TensorSlice {
    sik_u64 offset;
    sik_u64 bytes;
    sik_u32 elem_count;
    sik_u32 dtype;  /* STRATA_INKLING2_DTYPE_F32, F16, Q8_0, or Q4_A */
    sik_u32 shape0; /* Rows (outer dim) */
    sik_u32 shape1; /* Cols (inner dim, 1 for 1D) */
} StrataInkling2TensorSlice;

/*
 * Native 3D Stacked MoE Expert Matrix Descriptor (`sizeof = 72` bytes, 256B-aligned offsets):
 *   - For `w13` (de-interleaved at convert time!): `rows_per_expert = 2 * intermediate_size = 2304`
 *     where rows `[0 .. 1151]` are `gate` and rows `[1152 .. 2303]` are `up`, `cols = hidden_size = 2304`.
 *   - For `w2`: `rows_per_expert = hidden_size = 2304`, `cols = intermediate_size = 1152`.
 *   - When `dtype == STRATA_INKLING2_DTYPE_NVFP4`:
 *       * `weight_off`: `uint8_t [E, rows_per_expert, cols / 2]` (low nibble = col 2k, high nibble = col 2k+1, E2M1 LUT)
 *       * `scales_off`: `uint8_t [E, rows_per_expert, cols / 16]` (`float8_e4m3fn` block-16 scales)
 *       * `expert_scale_off`: `float [E]` (pre-divided `amax[e] / outer_scale_denominator`)
 *       * `block_size = 16`
 *   - When `dtype == STRATA_INKLING2_DTYPE_MXFP8`:
 *       * `weight_off`: `uint8_t [E, rows_per_expert, cols]` (`float8_e4m3fn` weights)
 *       * `scales_off`: `uint8_t [E, rows_per_expert, cols / 32]` (`UE8M0` exponent: `scale = 2^(s - 127)`)
 *       * `expert_scale_off`: `0` (unused)
 *       * `block_size = 32`
 */
typedef struct StrataInkling2ExpertMatrix {
    sik_u64 weight_off;
    sik_u64 weight_bytes;
    sik_u64 scales_off;
    sik_u64 scales_bytes;
    sik_u64 expert_scale_off;
    sik_u64 expert_scale_bytes;
    sik_u32 num_experts;
    sik_u32 rows_per_expert;
    sik_u32 cols;
    sik_u32 dtype;      /* NVFP4 or MXFP8 */
    sik_u32 block_size; /* 16 (NVFP4) or 32 (MXFP8) */
    sik_u32 reserved;
} StrataInkling2ExpertMatrix;

/* Looped Core Message Controller (`attn`, `k`, `v`, `mlp` -> `4` per core layer, `sizeof = 96` B) */
typedef struct StrataInkling2LoopCtrlDesc {
    StrataInkling2TensorSlice down_weight;   /* F16 [gate_rank=64, 2*D=4608] */
    StrataInkling2TensorSlice gain_weight;   /* F16 [D=2304, gate_rank=64]   */
    StrataInkling2TensorSlice forget_weight; /* F16 [D=2304, gate_rank=64]   */
} StrataInkling2LoopCtrlDesc;

/* Physical Decoder Layer Descriptor (`layers[0..23]` in `inkling2_uma.strata`) */
typedef struct StrataInkling2LayerDesc {
    sik_u32 phys_layer_idx; /* 0..23 */
    sik_u32 is_local_swa;   /* 1 => local SWA-1024; 0 => global full attention ({5, 11, 17, 23}) */
    sik_u32 in_loop_core;   /* 1 iff `core_start <= phys_layer_idx < core_end` (6..17) */
    sik_u32 expert_dtype;   /* NVFP4 (L0..19) or MXFP8 (L20..23) */
    sik_u64 layer_page_off;
    sik_u64 layer_page_bytes;

    /* Block Attention Residual Queries (F32):
     *   - `attn_res_query`: empty for layer 0; `[2304]` for local layers `>0`; `[3, 2304]` for global layers.
     *   - `mlp_res_query`:  `[2304]` for all layers 0..23. */
    StrataInkling2TensorSlice attn_res_query;
    StrataInkling2TensorSlice mlp_res_query;

    /* Input RMSNorms (F32 [2304]):
     *   - Local SWA layers: `attn_norm` populated (`q/k/v_input_norm` empty).
     *   - Global layers: `q_input_norm`, `k_input_norm`, `v_input_norm` populated (`attn_norm` empty). */
    StrataInkling2TensorSlice attn_norm;
    StrataInkling2TensorSlice q_input_norm;
    StrataInkling2TensorSlice k_input_norm;
    StrataInkling2TensorSlice v_input_norm;
    StrataInkling2TensorSlice mlp_norm;

    /* Self-Attention Linear Projections (F16) */
    StrataInkling2TensorSlice wq;             /* F16 [3072, 2304] (24 * 128) */
    StrataInkling2TensorSlice wk;             /* F16 [1024, 2304] (8 * 128)  */
    StrataInkling2TensorSlice wv;             /* F16 [1024, 2304] (8 * 128)  */
    StrataInkling2TensorSlice wo;             /* F16 [2304, 3072]            */
    StrataInkling2TensorSlice gate_attn;      /* F16 [3072, 2304]            */
    StrataInkling2TensorSlice wr;             /* F16 [384,  2304] (24 * 16)  */
    StrataInkling2TensorSlice attn_sink_proj; /* F16 [24,   2304]            */

    /* L2 Clips & Relative-Position Bias Bank (F32) */
    StrataInkling2TensorSlice q_l2_clip_scale;   /* F32 [24, 128]   */
    StrataInkling2TensorSlice rel_l2_clip_scale; /* F32 [24, 16]    */
    StrataInkling2TensorSlice rel_logits_proj;   /* F32 [1024, 16] (transposed from [16, 1024] for contiguous dot!) */

    /* Causal Short Convolutions (kernel_size = 4):
     *   - Local SWA: `k_sconv_w`, `v_sconv_w` are F32 `[1024, 4]` oldest-first (`[t-3, t-2, t-1, t]`).
     *   - Global:    `k_sconv_w`, `v_sconv_w` are F32 `[1024, 4]` LAG-FIRST (`lag 0 = current`, transposed from `[4, 1024]`),
     *                plus `k_sconv_proj`, `v_sconv_proj` F16 `[32, 2304]` (`8 kv_heads * 4 lags`).
     *   - Post-attn & Post-MLP (ALL layers): F32 `[2304, 4]` oldest-first (`[t-3, t-2, t-1, t]`). */
    StrataInkling2TensorSlice k_sconv_w;
    StrataInkling2TensorSlice v_sconv_w;
    StrataInkling2TensorSlice k_sconv_proj;
    StrataInkling2TensorSlice v_sconv_proj;
    StrataInkling2TensorSlice attn_sconv_w;
    StrataInkling2TensorSlice mlp_sconv_w;

    /* 128-Expert Top-8 Sparse MoE (Router F32, Experts Native NVFP4 / MXFP8) */
    StrataInkling2TensorSlice  moe_gate_weight; /* F32 [128, 2304] */
    StrataInkling2TensorSlice  moe_gate_bias;   /* F32 [128]       */
    StrataInkling2ExpertMatrix experts_w13;     /* De-interleaved [128, 2304, 2304] (gate[0..1151], up[1152..2303]) */
    StrataInkling2ExpertMatrix experts_w2;      /* [128, 2304, 1152] */

    /* Looped Core Controllers (`0=attn, 1=k, 2=v, 3=mlp`; populated iff `in_loop_core == 1`) */
    StrataInkling2LoopCtrlDesc controllers[4];
} StrataInkling2LayerDesc;

/* Single Layer of Native 3-Layer DSpark Block-8 MTP Drafter */
typedef struct StrataInkling2MtpLayerDesc {
    sik_u32 layer_idx;    /* 0..2 */
    sik_u32 is_local_swa; /* 1 for layers {0, 2}, 0 for layer {1} */
    StrataInkling2TensorSlice dflare_norm;      /* F32 [2304] */
    StrataInkling2TensorSlice input_layernorm;  /* F32 [2304] */
    StrataInkling2TensorSlice post_attn_norm;   /* F32 [2304] */
    StrataInkling2TensorSlice q_norm;           /* F32 [128]  */
    StrataInkling2TensorSlice k_norm;           /* F32 [128]  */
    StrataInkling2TensorSlice k_norm_ctx;       /* F32 [128]  */
    StrataInkling2TensorSlice q_proj;           /* F16/Q8_0/Q4_A [3072, 2304] */
    StrataInkling2TensorSlice k_proj;           /* F16/Q8_0/Q4_A [1024, 2304] */
    StrataInkling2TensorSlice v_proj;           /* F16/Q8_0/Q4_A [1024, 2304] */
    StrataInkling2TensorSlice k_proj_ctx;       /* F16/Q8_0/Q4_A [1024, 2304] */
    StrataInkling2TensorSlice v_proj_ctx;       /* F16/Q8_0/Q4_A [1024, 2304] */
    StrataInkling2TensorSlice o_proj;           /* F16/Q8_0/Q4_A [2304, 3072] */
    StrataInkling2TensorSlice gate_proj;        /* F16/Q8_0/Q4_A [9216, 2304] */
    StrataInkling2TensorSlice up_proj;          /* F16/Q8_0/Q4_A [9216, 2304] */
    StrataInkling2TensorSlice down_proj;        /* F16/Q8_0/Q4_A [2304, 9216] */
} StrataInkling2MtpLayerDesc;

/* Weight-Intrinsic `||markov_w1[v]||_2` 32K FR-Spec Folded Markov Head Descriptor (`sizeof = 144` B) */
typedef struct StrataInkling2FrSpecDesc {
    sik_u32 enabled;             /* 1 iff 32K FR-Spec tables present */
    sik_u32 shortlist_size;      /* e.g. 32768 (padded multiple of 256) */
    sik_u32 reserved0;
    sik_u32 reserved1;
    StrataInkling2TensorSlice token_ids;       /* U32 (dtype=F32/raw) [shortlist_size] mapping short_idx -> full vocab_id */
    StrataInkling2TensorSlice unembed_q4a;     /* Q4_A [shortlist_size, 2304] (~42.5 MB) */
    StrataInkling2TensorSlice fold_weight_q8;  /* Q8_0 [shortlist_size, 256]  (~8.7 MB, pre-folded unembed @ hidden_out in F64!) */
    StrataInkling2TensorSlice fold_bias_f32;   /* F32  [shortlist_size]       (~128 KB, pre-folded unembed @ hidden_bias in F64!) */
} StrataInkling2FrSpecDesc;

/* Native 3-Layer DSpark Block-8 `rnn_hidden` Markov MTP Drafter Descriptor */
typedef struct StrataInkling2MtpDesc {
    sik_u32 enabled;            /* 1 iff `mtp.safetensors` converted into bundle */
    sik_u32 num_layers;         /* 3 */
    sik_u32 intermediate_size;  /* 9216 */
    sik_u32 block_size;         /* 8 (anchor + 7 drafted tokens) */
    sik_u32 markov_rank;        /* 256 */
    sik_u32 sliding_window;     /* 1024 */
    float   rope_theta;         /* 1000000.0f */
    float   rms_norm_eps;       /* 1e-6f */
    sik_u32 target_layer_ids[STRATA_INKLING2_MAX_MTP_TAPS]; /* {5, 23, 41, 48} */
    sik_u64 mtp_page_off;
    sik_u64 mtp_page_bytes;

    StrataInkling2TensorSlice fuse_alpha;         /* F32 [4, 4] pre-softmaxed row-wise from `fuse_weight` */
    StrataInkling2TensorSlice mask_embed;         /* F16 [2304] */
    StrataInkling2TensorSlice final_norm;         /* F32 [2304] (`norm.weight`) */
    StrataInkling2TensorSlice markov_dflare_norm; /* F32 [2304] */
    StrataInkling2TensorSlice markov_w1;          /* F16 [201024, 256] */
    StrataInkling2TensorSlice markov_rnn_weight;  /* F16/Q8_0 [768, 5120] (`[state(256); w1(256); hidden(2304); f_rnn(2304)]`) */
    StrataInkling2TensorSlice markov_rnn_bias;    /* F32 [768] */
    StrataInkling2TensorSlice markov_hidden_out;  /* F16 [2304, 256] */
    StrataInkling2TensorSlice markov_hidden_bias; /* F32 [2304] */
    StrataInkling2MtpLayerDesc layers[STRATA_INKLING2_MAX_MTP_LAYERS];

    /* v2 Extension (appended at end of header for 100% binary v1 compatibility; zeroed on v1 load) */
    StrataInkling2FrSpecDesc  frspec;
} StrataInkling2MtpDesc;

/* Complete Model Hyperparameters stored in `inkling2_uma.strata` Header */
typedef struct StrataInkling2Config {
    sik_u32 num_phys_layers;             /* 24 */
    sik_u32 num_logical_layers;          /* 48 (6 + 3*12 + 6) */
    sik_u32 hidden_size;                 /* 2304 */
    sik_u32 vocab_size_padded;           /* 201024 */
    sik_u32 vocab_size_unpadded;         /* 200058 */
    sik_u32 num_attention_heads;         /* 24 */
    sik_u32 num_key_value_heads;         /* 8 */
    sik_u32 head_dim;                    /* 128 */
    sik_u32 d_rel;                       /* 16 */
    sik_u32 rel_extent;                  /* 1024 */
    sik_u32 sliding_window_size;         /* 1024 */
    sik_u32 sconv_kernel_size;           /* 4 */
    sik_u32 attn_res_block_size;         /* 6 */
    sik_u32 num_experts;                 /* 128 */
    sik_u32 num_experts_per_tok;         /* 8 */
    sik_u32 intermediate_size;           /* 1152 */
    sik_u32 loop_core_start;             /* 6 */
    sik_u32 loop_core_end;               /* 18 */
    sik_u32 loop_num_passes;             /* 3 */
    sik_u32 loop_gate_rank;              /* 64 */
    sik_u32 log_scaling_n_floor;         /* 8192 */
    sik_u32 max_position_embeddings;     /* 1048576 */
    sik_u32 eos_token_id;                /* 200006 (`<|content_model_end_sampling|>`) */
    sik_u32 pad_token_id;                /* 199999 (`<|endoftext|>`) */
    float   rms_norm_eps;                /* 1e-6f */
    float   qk_l2_clip_threshold;        /* 22.627417f */
    float   rel_l2_clip_threshold;       /* 8.0f */
    float   log_scaling_alpha;           /* 0.1f */
    float   mlp_gate_softcap;            /* 4.0f */
    float   mlp_up_softcap;              /* 25.0f */
    float   route_scale;                 /* 8.0f */
    float   final_logit_softcapping;     /* 30.0f */
    sik_u32 num_local_visits;            /* 40 */
    sik_u32 num_global_visits;           /* 8 */
    sik_u32 logical_to_phys[STRATA_INKLING2_MAX_LOGICAL_LAYERS]; /* visit `il` -> physical `0..23` */
    sik_u32 logical_pass_idx[STRATA_INKLING2_MAX_LOGICAL_LAYERS];/* 0 for prelude/coda/pass0; 1 or 2 for looped passes */
    sik_u32 logical_is_local[STRATA_INKLING2_MAX_LOGICAL_LAYERS];/* 1 => SWA-1024; 0 => Global */
    sik_u32 logical_kv_slot[STRATA_INKLING2_MAX_LOGICAL_LAYERS]; /* 0..39 for SWA ring; 0..7 for Global KV */
} StrataInkling2Config;

/* Header at Byte Offset 0 of `inkling2_uma.strata` (padded to 128 KiB) */
typedef struct StrataInkling2Header {
    sik_u64                   magic;
    sik_u32                   version;
    sik_u32                   header_padded_bytes;
    sik_u64                   total_file_bytes;
    sik_u64                   active_bytes_per_tok;     /* 48-visit + head M=1 physical DRAM read */
    sik_u64                   mtp_active_bytes_per_blk; /* 3-layer DSpark block-8 draft read */
    StrataInkling2Config      cfg;

    /* Global Backbone Tensors */
    StrataInkling2TensorSlice embed_tokens;              /* F16 [201024, 2304] */
    StrataInkling2TensorSlice embed_norm;                /* F32 [2304] */
    StrataInkling2TensorSlice final_norm;                /* F32 [2304] */
    StrataInkling2TensorSlice output_res_query;          /* F32 [2304] */
    StrataInkling2TensorSlice unembed;                   /* F16 [200058, 2304] (trimmed to unpadded vocab!) */
    StrataInkling2TensorSlice cross_pass_queries;        /* F32 [24, 2304] (`middle_cycle_cross_pass_queries`) */
    StrataInkling2TensorSlice cross_pass_kv_queries;     /* F32 [24, 2304] (`middle_cycle_cross_pass_kv_queries`) */

    /* 24 Physical Layers + Optional 3-Layer DSpark MTP */
    StrataInkling2LayerDesc   layers[STRATA_INKLING2_MAX_PHYS_LAYERS];
    StrataInkling2MtpDesc     mtp;
} StrataInkling2Header;

/* ============================================================================
 * 3. Host C/C++/ObjC++ Public API (Excluded during Metal MSL Compilation)
 * ============================================================================ */

#ifndef __METAL_VERSION__

#ifdef __cplusplus
static_assert(sizeof(StrataInkling2TensorSlice) == 32,
              "StrataInkling2TensorSlice ABI size must be 32 bytes");
static_assert(sizeof(StrataInkling2ExpertMatrix) == 72,
              "StrataInkling2ExpertMatrix ABI size must be 72 bytes");
static_assert(sizeof(StrataInkling2FrSpecDesc) == 144,
              "StrataInkling2FrSpecDesc ABI size must be 144 bytes");
static_assert(sizeof(StrataInkling2Header) <= STRATA_INKLING2_HEADER_PADDED_BYTES,
              "StrataInkling2Header must fit within 128 KiB padded header");
extern "C" {
#endif

static inline sik_u64 strata_inkling2_align256(sik_u64 v) {
    return (v + (STRATA_INKLING2_TENSOR_ALIGN - 1ULL)) & ~(STRATA_INKLING2_TENSOR_ALIGN - 1ULL);
}

static inline void strata_inkling2_q8_layout(
    sik_u32  rows,
    sik_u32  cols,
    sik_u64* w_bytes_out,
    sik_u64* s_bytes_out
) {
    const sik_u64 wb = strata_inkling2_align256((sik_u64)rows * (sik_u64)cols);
    const sik_u64 sb = strata_inkling2_align256(
        (sik_u64)rows * (sik_u64)(cols / STRATA_INKLING2_QUANT_GROUP_SIZE) * 2ULL);
    if (w_bytes_out) *w_bytes_out = wb;
    if (s_bytes_out) *s_bytes_out = sb;
}

static inline void strata_inkling2_q4a_layout(
    sik_u32  rows,
    sik_u32  cols,
    sik_u64* w_bytes_out,
    sik_u64* s_bytes_out,
    sik_u64* b_bytes_out
) {
    const sik_u64 wb = strata_inkling2_align256((sik_u64)rows * (sik_u64)(cols / 8U) * 4ULL);
    const sik_u64 sb = strata_inkling2_align256(
        (sik_u64)rows * (sik_u64)(cols / STRATA_INKLING2_QUANT_GROUP_SIZE) * 2ULL);
    if (w_bytes_out) *w_bytes_out = wb;
    if (s_bytes_out) *s_bytes_out = sb;
    if (b_bytes_out) *b_bytes_out = sb;
}

static inline void strata_inkling2_kv_q8_layer_layout(
    sik_u32  num_kv_heads,
    sik_u32  seq_stride,
    sik_u32  head_dim,
    sik_u64* data_bytes_out,
    sik_u64* scale_bytes_out
) {
    const sik_u64 db = strata_inkling2_align256(
        (sik_u64)num_kv_heads * (sik_u64)seq_stride * (sik_u64)head_dim);
    const sik_u64 sb = strata_inkling2_align256(
        (sik_u64)num_kv_heads * (sik_u64)seq_stride * 2ULL);
    if (data_bytes_out)  *data_bytes_out  = db;
    if (scale_bytes_out) *scale_bytes_out = sb;
}

typedef struct StrataInkling2Tokenizer StrataInkling2Tokenizer;

typedef struct StrataInkling2StreamDecoder {
    uint8_t pending[8];
    uint8_t pending_len;
} StrataInkling2StreamDecoder;

typedef struct StrataInkling2Bundle {
    char                     bundle_dir[512];
    int                      uma_fd;
    void*                    uma_base;
    size_t                   uma_size;
    StrataInkling2Header     hdr;
    StrataInkling2Tokenizer* tokenizer;
} StrataInkling2Bundle;

typedef struct StrataInkling2ConvertConfig {
    const char* hf_dir;      /* Local directory containing HF safetensors + config.json + tokenizer.json */
    const char* bundle_dir;  /* Output directory for `inkling2_uma.strata` + `tokenizer.json` */
    uint32_t    num_threads; /* Worker threads (default 8) */
} StrataInkling2ConvertConfig;

#ifndef __METAL_VERSION__
#define STRATA_INKLING2_DEFAULT_NOREFUSAL_BUNDLE_DIR strata_apex_bundle_path("inkling-2-mini-norefusal-fast")
#endif

typedef struct StrataInkling2RepackConfig {
    const char* src_bundle_dir;
    const char* dst_bundle_dir;
    bool        q4_unembed;   /* false => Cfg B++ (Q8_0 unembed, 3.07 GB/tok, cos=0.9978); true => Cfg C (Q4_A unembed, 2.85 GB/tok) */
    bool        q4_mtp;       /* Legacy flag; default repacker uses Q4_A 3L MTP unless mtp_dtype == STRATA_INKLING2_DTYPE_Q8_0 */
    uint32_t    num_threads;  /* default 8 */
    uint32_t    mtp_dtype;    /* 0 (default) or Q4_A => Q4_A 3L MTP (0.203 GB/blk); Q8_0 => Q8_0 3L MTP (0.326 GB/blk); F16 => FP16 */
    const char* lora_dir;     /* Optional PEFT LoRA dir (`adapter_config.json` + `adapter_model.safetensors`) targeting `attn.wo_ud` */
    float       lora_scale;   /* Optional multiplier applied to `lora_alpha / r` (<= 0 => 1.0f) */
} StrataInkling2RepackConfig;

typedef struct StrataInkling2SamplingParams {
    bool     greedy;             /* true => deterministic argmax (`temperature` ignored) */
    float    temperature;        /* e.g. 0.6f or 1.0f when `!greedy` */
    uint32_t top_k;              /* e.g. 20 (0 => disabled / default) */
    float    top_p;              /* e.g. 0.95f (<= 0 or >= 1 => disabled) */
    uint64_t rng_seed;           /* PRNG seed for Leviathan stochastic verification */
    float    presence_penalty;   /* OpenAI presence penalty [-2..2]; 0 => disabled */
    float    frequency_penalty;  /* OpenAI frequency penalty [-2..2]; 0 => disabled */
    float    repetition_penalty; /* HF multiplicative penalty (> 0; 0 or 1.0 => disabled) */
    uint32_t penalty_window;     /* 0 => full completion; >0 => sliding last max(N, 8) completion tokens */
} StrataInkling2SamplingParams;

typedef struct StrataInkling2GpuInfo {
    char     device_name[128];
    bool     zero_copy_mmap;
    double   load_time_ms;
    uint32_t max_seq_len;
    uint64_t uma_mapped_bytes;
    uint64_t active_bytes_per_token;
    uint64_t mtp_active_bytes_per_block;
    uint64_t swa_kv_bytes_allocated;
    uint64_t global_kv_bytes_allocated;
    uint64_t sconv_state_bytes;
    uint64_t workspace_bytes;
    bool     mtp_enabled;
    uint32_t kv_dtype;             /* STRATA_INKLING2_KV_DTYPE_F16 (0) or STRATA_INKLING2_KV_DTYPE_Q8_0 (1) */
    uint32_t int8_prefill_enabled;  /* 1 iff opt-in INT8 NAX prefill GEMMs enabled */
    uint32_t metal_cache_hit;       /* 1 iff MTLBinaryArchive warm cache hit at startup */
    double   pso_load_ms;           /* Wall-clock ms spent loading/compiling Metal PSOs */
    uint32_t bitplane_head_enabled; /* 1 iff 2-stage bitplane exact Q8_0 LM head active */
    uint32_t beam_tree_enabled;     /* 1 iff dynamic W=4 multi-branch DSpark beam tree active */
    uint32_t beam_tree_width;       /* Max beam tree branching width (4 when enabled, 1 when linear) */
} StrataInkling2GpuInfo;

typedef struct StrataInkling2ImageSpan {
    uint32_t     token_offset;   /* start index in the prefill token_ids[] array */
    uint32_t     num_tokens;     /* number of consecutive placeholder tokens to replace */
    const float* embeddings_f32; /* [num_tokens * hidden_size] (hidden_size=2304) */
} StrataInkling2ImageSpan;

typedef struct StrataInkling2PerfCounters {
    uint64_t decode_steps;
    uint64_t spec_verify_steps;
    uint64_t mtp_drafted_tokens;
    uint64_t mtp_accepted_tokens;
    uint64_t physical_dram_bytes_read;
} StrataInkling2PerfCounters;

typedef struct StrataInkling2GpuEngine  StrataInkling2GpuEngine;
typedef struct StrataInkling2Checkpoint StrataInkling2Checkpoint;
typedef struct StrataInkling2ContextSlot StrataInkling2ContextSlot;

/* ---- Converter, Fast Repacker & Bundle Loader (`src/model/strata_inkling2_convert.cpp`) ---- */

int  strata_inkling2_convert_hf(const StrataInkling2ConvertConfig* cfg);
int  strata_inkling2_repack_fast(const StrataInkling2RepackConfig* cfg);
int  strata_inkling2_load_bundle(const char* bundle_dir, StrataInkling2Bundle* out_bundle);
void strata_inkling2_free_bundle(StrataInkling2Bundle* bundle);
int  strata_inkling2_inspect_bundle(const char* bundle_dir);

/* ---- ByteLevel BPE Tokenizer (`src/model/strata_inkling2_convert.cpp`) ---- */

StrataInkling2Tokenizer* strata_inkling2_tokenizer_load(const char* tokenizer_json_path);
void     strata_inkling2_tokenizer_free(StrataInkling2Tokenizer* tok);
uint32_t strata_inkling2_tokenizer_vocab_size(const StrataInkling2Tokenizer* tok);

int strata_inkling2_encode(
    const StrataInkling2Tokenizer* tok,
    const char*                    text,
    size_t                         text_len,
    bool                           parse_special,
    uint32_t*                      out_ids,
    uint32_t                       out_cap,
    uint32_t*                      out_len
);

int strata_inkling2_decode(
    const StrataInkling2Tokenizer* tok,
    const uint32_t*                ids,
    uint32_t                       num_ids,
    bool                           skip_special,
    char*                          out_utf8,
    size_t                         out_cap,
    size_t*                        out_len
);

void strata_inkling2_stream_decoder_init(StrataInkling2StreamDecoder* dec);
int  strata_inkling2_stream_decode_step(
    const StrataInkling2Tokenizer* tok,
    StrataInkling2StreamDecoder*   dec,
    uint32_t                       token_id,
    bool                           skip_special,
    char*                          out_utf8,
    size_t                         out_cap,
    size_t*                        out_len
);

/* ---- Metal 4 GPU Engine (`src/gpu/strata_inkling2_gpu.mm`) ---- */

StrataInkling2GpuEngine* strata_inkling2_gpu_create(
    StrataInkling2Bundle* bundle,
    uint32_t              max_seq_len
);
StrataInkling2GpuEngine* strata_inkling2_gpu_create_ex(
    const StrataInkling2Bundle* bundle,
    uint32_t                    max_seq_len,
    uint32_t                    kv_dtype
);
void     strata_inkling2_gpu_destroy(StrataInkling2GpuEngine* eng);
void     strata_inkling2_gpu_reset(StrataInkling2GpuEngine* eng);
uint32_t strata_inkling2_gpu_seq_pos(const StrataInkling2GpuEngine* eng);
void     strata_inkling2_gpu_get_info(const StrataInkling2GpuEngine* eng, StrataInkling2GpuInfo* info_out);
void     strata_inkling2_gpu_get_counters(const StrataInkling2GpuEngine* eng, StrataInkling2PerfCounters* out);
void     strata_inkling2_gpu_reset_counters(StrataInkling2GpuEngine* eng);
void     strata_inkling2_gpu_set_adaptive_k(StrataInkling2GpuEngine* eng, bool enable);
bool     strata_inkling2_gpu_get_adaptive_k(const StrataInkling2GpuEngine* eng);
void     strata_inkling2_gpu_set_frspec(StrataInkling2GpuEngine* eng, bool enable);
bool     strata_inkling2_gpu_get_frspec(const StrataInkling2GpuEngine* eng);
void     strata_inkling2_gpu_set_bitplane_head(StrataInkling2GpuEngine* eng, bool enable);
bool     strata_inkling2_gpu_get_bitplane_head(const StrataInkling2GpuEngine* eng);
void     strata_inkling2_gpu_set_beam_tree(StrataInkling2GpuEngine* eng, bool enable);
bool     strata_inkling2_gpu_get_beam_tree(const StrataInkling2GpuEngine* eng);
void     strata_inkling2_gpu_set_int8_prefill(StrataInkling2GpuEngine* eng, bool enable);
int      strata_inkling2_gpu_load_alt_wo_bundle(StrataInkling2GpuEngine* eng, const char* alt_bundle_dir);
void     strata_inkling2_gpu_set_alt_wo(StrataInkling2GpuEngine* eng, bool enable);
bool     strata_inkling2_gpu_has_alt_wo(const StrataInkling2GpuEngine* eng);
uint64_t strata_inkling2_gpu_alt_wo_bytes(const StrataInkling2GpuEngine* eng);

void strata_inkling2_gpu_penalties_reset(StrataInkling2GpuEngine* eng);
void strata_inkling2_gpu_penalties_record(
    StrataInkling2GpuEngine* eng,
    const uint32_t*          tokens,
    uint32_t                 num_tokens
);

int strata_inkling2_gpu_prefill(
    StrataInkling2GpuEngine*            eng,
    const uint32_t*                     token_ids,
    uint32_t                            num_tokens,
    const StrataInkling2SamplingParams* sampling,
    uint32_t*                           next_token_out,
    float*                              last_logits_out
);

int strata_inkling2_gpu_prefill_with_images(
    StrataInkling2GpuEngine*            eng,
    const uint32_t*                     token_ids,
    uint32_t                            num_tokens,
    const StrataInkling2ImageSpan*      spans,
    uint32_t                            num_spans,
    const StrataInkling2SamplingParams* sampling,
    uint32_t*                           next_token_out,
    float*                              last_logits_out
);

int strata_inkling2_gpu_decode_step(
    StrataInkling2GpuEngine*            eng,
    uint32_t                            token_id,
    const StrataInkling2SamplingParams* sampling,
    uint32_t*                           next_token_out,
    float*                              logits_out
);

int strata_inkling2_gpu_spec_decode_step(
    StrataInkling2GpuEngine*            eng,
    uint32_t                            pending_token_id,
    uint32_t                            draft_k,
    const StrataInkling2SamplingParams* sampling,
    uint32_t*                           emitted_tokens_out,
    uint32_t*                           num_emitted_out,
    uint32_t*                           num_accepted_out
);

/* O(1) GPU Checkpoint & Multi-Slot Isolation for OpenAI Server Prefix Cache */
StrataInkling2Checkpoint* strata_inkling2_checkpoint_create(const StrataInkling2GpuEngine* eng);
void strata_inkling2_checkpoint_destroy(StrataInkling2Checkpoint* ckpt);
int  strata_inkling2_checkpoint_save(StrataInkling2GpuEngine* eng, StrataInkling2Checkpoint* ckpt);
int  strata_inkling2_checkpoint_restore(StrataInkling2GpuEngine* eng, const StrataInkling2Checkpoint* ckpt);

StrataInkling2ContextSlot* strata_inkling2_slot_create(StrataInkling2GpuEngine* eng);
void strata_inkling2_slot_destroy(StrataInkling2ContextSlot* slot);
void strata_inkling2_bind_slot(StrataInkling2GpuEngine* eng, StrataInkling2ContextSlot* slot);

/* Continuous Batching (`batch_size` in 1..64) */
typedef struct StrataInkling2BatchItem {
    StrataInkling2ContextSlot*          slot;           /* independent KV + SConv + pen_hist slot */
    uint32_t                            token_id;       /* incoming token to decode for this stream */
    bool                                use_alt_wo;     /* per-stream --no-refusal alt_wo selector */
    const StrataInkling2SamplingParams* sampling;       /* per-stream sampling/penalties/seed (nullptr => greedy) */
    uint32_t                            next_token_out; /* filled on return */
} StrataInkling2BatchItem;

/* Query / manage per-slot state cleanly without binding */
uint32_t strata_inkling2_slot_seq_pos(const StrataInkling2ContextSlot* slot);
bool     strata_inkling2_slot_mtp_synced(const StrataInkling2ContextSlot* slot);
size_t   strata_inkling2_slot_memory_bytes(const StrataInkling2GpuEngine* eng);

/* True GPU-batched multi-stream decode (`batch_size` in 1..64).
 * Advances each `items[b].slot` by 1 token independently using shared weight reads across all active streams!
 * Automatically partitions mixed `use_alt_wo` batches into <= 2 homogeneous GPU sub-batches (0 weight mixing).
 * Guarantees 100% bit-exact greedy equivalence to isolated solo `K=0` decode for every stream! */
int strata_inkling2_gpu_decode_batch(
    StrataInkling2GpuEngine* eng,
    StrataInkling2BatchItem* items,
    uint32_t                 batch_size
);

/* ---- CLI Entrypoints (`src/cli/strata_inkling2_bench.mm`) ---- */

int strata_inkling2_cmd_convert(int argc, char** argv);
int strata_inkling2_cmd_repack(int argc, char** argv);
int strata_inkling2_cmd_inspect(int argc, char** argv);
int strata_inkling2_cmd_run(int argc, char** argv);
int strata_inkling2_cmd_bench(int argc, char** argv);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !__METAL_VERSION__ */

#endif /* STRATA_INKLING2_H */

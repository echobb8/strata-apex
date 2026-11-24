#ifndef STRATA_QWEN_FLASH_H
#define STRATA_QWEN_FLASH_H

/*
 * STRATA-M5 v2.0 — Qwen/Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration / qwen4_exp_text)
 * Two-Part Zero-Copy 16 KiB UMA Bundle + F_NOCACHE NVMe PLE N-Gram Store + Native MTP Speculative ABI
 *
 * Target Hardware: Mac17,10 (Apple M5 Max: 40c Metal 4 NAX GPU, ~508 GB/s LPDDR5X, 128 GiB UMA,
 *                  [MTLDevice maxBufferLength] = 80.64 GiB, recommendedMaxWorkingSetSize = 107.52 GiB).
 *
 * ====================================================================================================
 * 100% VERIFIED ARCHITECTURE & EXACT EQUATIONS (`transformers` `modeling_qwen4_exp.py`,
 * `mlx_lm/models/qwen4_exp.py`, `sglang` `qwen4_exp_mtp.py`, and HF 131-shard BF16 headers):
 * ====================================================================================================
 *
 * 0. Checkpoint Inventory (`Qwen/Qwen3.8-Flash-Next`, 335.28 GiB BF16 across 131 shards, 1,658 tensors):
 *    - `model.visual.*`: 333 tensors (vision encoder — skipped for text engine).
 *    - `model.language_model.embed_tokens.weight`: BF16 `[248320, 2560]` (`tie_word_embeddings = false`).
 *    - `model.language_model.layers.0..47.*`: 48 hybrid decoder layers (`hidden_size = 2560`, `eps = 1e-6`):
 *        * 36 `linear_attention` layers (`l % 4 != 3`): `GatedDeltaNet` (`16` K-heads x `128`, `48` V-heads x `128`).
 *        * 12 `full_attention` layers (`l % 4 == 3`: `3, 7, ..., 47`): `QSA` (`24` Q-heads, `2` KV-heads, `head_dim = 256`,
 *          `partial_rotary_factor = 0.25` -> `64` rotary dims, `theta = 1e7`) + `QSAIndexer` (`budget = 2048`).
 *        * ALL 48 layers use `512`-expert SwiGLU MoE (`top_k = 10`, `moe_intermediate_size = 640`, renormalized
 *          softmax) + gated `shared_expert` (`640`) + 4-stream `Hyper-Connections` (`hc_count = 4, hc_lowrank = 320`).
 *        * Layer `1` ONLY (`ple_layer_ids = [2]`, 1-indexed -> `layer_idx == 1`, `ple_layer_index == 0`) has `PLE`
 *          (`51.2B` params across `128` shards `shard_0..127.weight` of `BF16 [2500012, 160]` = `320,001,536 x 160`).
 *    - `model.language_model.hyper_connection_mixer.*`: Final 4-stream -> 1-stream gated mixer (NO separate `final_norm`!).
 *    - `lm_head.weight`: Untied `BF16 [248320, 2560]`.
 *    - `mtp.*`: 31 tensors — Native 1-layer Multi-Token Prediction (`full_attention` QSA + `512`-expert MoE + mixer).
 *
 * 1. CRITICAL RMSNorm `+1.0` Folding Rule (Converter Contract):
 *    - In raw HF BF16 shards, EVERY `Qwen4ExpTextRMSNorm` / `GemmaRMSNorm` stores `w` around `0.0` (or shifted) and
 *      computes `y = rms_norm(x) * (1.0 + w)`, EXCEPT `layers.*.linear_attn.norm.weight` (`Qwen4ExpTextRMSNormGated`),
 *      which stores `w` around `1.0` directly (`y = rms_norm(x) * w * sigmoid(z)`).
 *    - **CONVERTER INVARIANT:** `@flash-stream` MUST add `+1.0f` when converting to `F32` for ALL of:
 *        * `attn_hyper_connection.hc_norm.weight` (`[10240]`), `mlp_hyper_connection.hc_norm.weight` (`[10240]`)
 *        * `hyper_connection_mixer.hc_norm.weight` (`[10240]`, both main model and `mtp.hyper_connection_mixer`)
 *        * `self_attn.q_norm.weight` (`[256]`), `self_attn.k_norm.weight` (`[256]`)
 *        * `self_attn.indexer.q_layernorm.weight` (`[128]`), `self_attn.indexer.k_layernorm.weight` (`[128]`)
 *        * `layers.1.ple.norm_key.weight` (`[10240]`), `norm_query.weight` (`[10240]`), `norm_conv.weight` (`[10240]`)
 *        * `mtp.pre_fc_norm_embedding.weight` (`[2560]`), `mtp.pre_fc_norm_hidden.weight` (`[10240]`), `mtp.layers.0.*` norms
 *      and MUST NOT add `+1.0f` to `layers.*.linear_attn.norm.weight` (`[128]`).
 *    - **RUNTIME INVARIANT:** `@flash-gpu` multiplies directly by `w_f32` in EVERY RMSNorm (no `+1.0f` at runtime).
 *    - **GROUPED vs UNGROUPED RMSNorm (`eps = 1e-6f`):**
 *        * `hc_norm [10240]` and `ple.{norm_key, norm_query, norm_conv} [10240]` use `group_size = 2560`:
 *          each stream `s in 0..3` computes its own `inv_rms_s = rsqrt(mean_{d=0..2559}(x[s,d]^2) + 1e-6f)`, then
 *          multiplies by the 10240-element `w[s, d]`.
 *        * `mtp.pre_fc_norm_hidden [10240]` is an UNGROUPED `GemmaRMSNorm(10240)` (`group_size = None`):
 *          computes ONE `inv_rms = rsqrt(mean_{i=0..10239}(x[i]^2) + 1e-6f)` across all `10240` dims!
 *
 * 2. 4-Stream Hyper-Connections (`hc_count = 4, D = 2560, hc_dim = 10240, hc_lowrank = 320`):
 *    - Entry after `e = embed_tokens(tok)` (`[2560]`): tile 4x -> `h[s, d] = e[d]` for `s = 0..3` (`[10240]` FP32).
 *    - Pre-Attn & Pre-MLP (`use_combine = true`):
 *        `h_norm = grouped_rms_norm_4x2560(h, hc_norm)`                        // [10240]
 *        `u      = silu((input_mix_weight_down @ h_norm) * 0.25f)`             // [320] (`/ hc_count` INSIDE silu!)
 *        `mix_w  = sigmoid(input_mix_weight_up @ u)`                           // [4, 2560]
 *        `x_in[d]= 0.25f * sum_{s=0..3} (mix_w[s, d] * h_norm[s, d])`          // [2560] (`.mean(dim=-2)`)
 *        `inj[s] = 2.0f * sigmoid(((block_inject_weight @ h_norm)[s]) * 0.25f)`// [4] (`/ hc_count` INSIDE sigmoid!)
 *    - Post-Block Residual Writeback (`y [2560]` = output of `linear_attn`, `self_attn`, or `mlp`):
 *        `h[s, d] = h[s, d] + inj[s] * y[d]`  (adds onto UN-NORMED `h`!).
 *    - Final `hyper_connection_mixer` (`use_combine = false`, both main model and MTP):
 *        Save `hc_hidden_states = h` (`[10240]`, passed to MTP!); then compute `h_norm`, `u`, `mix_w` as above and
 *        return `h_final[d] = 0.25f * sum_{s=0..3} (mix_w[s, d] * h_norm[s, d])` (`[2560]`) -> `lm_head(h_final)`.
 *
 * 3. Per-Layer Embedding (`PLE`, ONLY at `layer_idx == 1`, BEFORE `layers[1].attn_hyper_connection`):
 *    - `ple_eos_token_id = 248044` (`<|endoftext|>`). Maintain 2-token history `{prev2, prev1}` init to `{248044, 248044}`.
 *    - For current token `s0 = tok[t]`, `s1 = prev1`, `s2 = (s1 == 248044) ? 248044 : prev2`:
 *        `mix2 = (uint64_t)s0 * mult[0] ^ ((uint64_t)s1 * mult[1])`            // heads h = 0..7 (bigrams)
 *        `mix3 = mix2 ^ ((uint64_t)s2 * mult[2])`                              // heads h = 8..15 (trigrams)
 *        `row[h] = head_offsets[h] + ((h < 8 ? mix2 : mix3) % head_vocab_sizes[h])`
 *      Gather 16 rows of `[160]` from `ple_ngram.strata` -> `emb [2560]`; advance `prev2 = prev1; prev1 = s0`.
 *    - Gating & Dilated Depthwise Short Conv1d (`kernel = 4, dilation = ngram_size = 3` -> state len = `9`!):
 *        `k_norm = grouped_rms_norm_4x2560(key_proj(emb), ple.norm_key)`       // [4, 2560]
 *        `val    = value_proj(emb)`                                            // [2560]
 *        `q_norm = grouped_rms_norm_4x2560(h, ple.norm_query)`                 // [4, 2560]
 *        `dot_s  = (sum_{d=0..2559} k_norm[s,d] * q_norm[s,d]) / sqrt(2560.0f)`
 *        `gate_s = sqrt(max(fabs(dot_s), 1e-6f)) * sign(dot_s)`                // 0 if dot_s == 0
 *        `gv[s,d]= sigmoid(gate_s) * val[d]` -> flatten `gv [10240]`
 *        `gv_n   = grouped_rms_norm_4x2560(gv, ple.norm_conv)`                 // [10240]
 *        `conv_c = silu(w[c,0]*gv_n_{t-9}[c] + w[c,1]*gv_n_{t-6}[c] + w[c,2]*gv_n_{t-3}[c] + w[c,3]*gv_n_t[c])`
 *        `h      = h + gv + conv_c`                                            // [10240]
 *
 * 4. 36-Layer `linear_attn` (`GatedDeltaNet`, `l % 4 != 3`, `output_gate_type = "sigmoid"`):
 *    - Identical causal `conv1d(k=4, dilation=1) + silu`, `q[16,128], k[16,128], v[48,128]`, L2-norm `q,k` (`q /= sqrt(128)`),
 *      `hv -> hk = hv / 3`, `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`, FP32 `S[48, 128, 128]` update,
 *      EXCEPT step 7 uses **`sigmoid(z)`** (`output_gate_type = "sigmoid"`):
 *        `y[hv] = rms_norm_128(out[hv], linear_attn.norm.weight, 1e-6f) * sigmoid(z[hv])` -> `out_proj(flatten(y))`.
 *
 * 5. 12-Layer `self_attn` (`QSA` + `QSAIndexer`, `l % 4 == 3`, plus `mtp.layers.0`):
 *    - `q_gate = q_proj(x)` (`[12288] -> [24, 512]` -> `q[hq][256]`, `gate[hq][256]`); `k = k_proj(x) [2, 256]`, `v = v_proj(x) [2, 256]`.
 *    - Per-head `q_norm(256)`, `k_norm(256)`, NeoX half-split RoPE on first `64` of `256` dims (`theta = 1e7`).
 *    - `QSAIndexer` (`index_qk_proj [640, 2560]` -> `idx_q [4, 128]`, `raw_k [128]`, `compress_ratio = 4`, `budget = 2048`, `block_topk = 512`):
 *        * Append `raw_k` (`[128]`) to indexer cache. Whenever block `b` (`tokens 4b..4b+3`) completes:
 *          `block_k[b] = rope_64_of_128(rms_norm_128(mean_{j=0..3}(raw_k[4b+j]), k_layernorm), pos = 4*b)`.
 *        * For query at `q_pos` (`num_complete_blocks = (q_pos + 1) / 4`):
 *          If `num_complete_blocks <= 512` (`q_pos < 2051`): ALL causal tokens `0..q_pos` are visible (fast dense path!).
 *          Else (`q_pos >= 2051`): `q_idx[h] = rope_64_of_128(rms_norm_128(idx_q[h], q_layernorm), q_pos)` (`h=0..3`),
 *          `score[b] = (sum_{h=0..3} max(0.0f, dot(q_idx[h], block_k[b]))) / sqrt(128.0f)` for `b = 0..num_complete_blocks-1`.
 *          Select top-`512` blocks (`2048` tokens `{4b..4b+3}`) PLUS partial tail `[num_complete_blocks*4 .. q_pos]`.
 *    - Causal GQA (`24Q / 2KV = 12:1`, `scale = 0.0625f`) over selected tokens -> `o_proj(attn_out * sigmoid(gate))`.
 *
 * 6. 48-Layer `512`-Expert MoE (`top_k = 10`, `d_ff = 640`) + `shared_expert` (`640`):
 *    - Router: `logits = mlp.gate(x)` (`[512]` FP32), top-10 indices `idx[0..9]`, renormalized weights
 *      `w[i] = softmax_over_top10(logits[idx[0..9]])[i]`.
 *    - Experts (`gate_up_proj [512, 1280, 2560]` -> `gate [0..639], up [640..1279]`; `down_proj [512, 2560, 640]`):
 *      `moe_out = sum_{i=0..9} w[i] * down_proj[idx[i]](silu(gate) * up) + sigmoid(shared_expert_gate(x)) * shared_expert(x)`.
 *
 * 7. Native 1-Layer `MTP` (`mtp.*` — `residual_linear_shared` fusion):
 *    - To predict `x_{t+2}` given committed/drafted token `x_{t+1}` and pre-mixer 4-stream state `hc_prev[t]` (`[10240]`):
 *        `e_proj = fc_embedding(rms_norm_2560(embed_tokens(x_{t+1}), pre_fc_norm_embedding))`       // [2560]
 *        `h_norm = rms_norm_10240_ungrouped(hc_prev[t], pre_fc_norm_hidden)`                        // [10240] -> [4, 2560]
 *        `h_mtp[s] = e_proj + fc_hidden(h_norm[s])` for `s = 0..3`                                  // [4, 2560] = [10240]
 *      Run `mtp.layers.0` (at RoPE `pos = t`), save `mtp_hc_out = h_mtp_after_layer0` (`[10240]`, feeds next draft step
 *      `k+1` if `draft_k > 1`!), mix via `mtp.hyper_connection_mixer(mtp_hc_out)` -> `[2560]` -> `lm_head` -> `\hat{x}_{t+2}`!
 */

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef uint8_t  sqf_u8;
typedef uint16_t sqf_u16;
typedef uint32_t sqf_u32;
typedef uint64_t sqf_u64;
typedef int32_t  sqf_i32;
typedef int64_t  sqf_i64;
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef uint8_t  sqf_u8;
typedef uint16_t sqf_u16;
typedef uint32_t sqf_u32;
typedef uint64_t sqf_u64;
typedef int32_t  sqf_i32;
typedef int64_t  sqf_i64;
#endif

/* ============================================================================
 * 1. Alignment, Magic & Architecture Constants (Shared Host C++ & Metal 4 MSL)
 * ============================================================================ */

#define STRATA_QWEN_FLASH_PART0_MAGIC_U64        0x30504651354D5453ULL /* "STM5QFP0" */
#define STRATA_QWEN_FLASH_PART1_MAGIC_U64        0x31504651354D5453ULL /* "STM5QFP1" */
#define STRATA_QWEN_FLASH_PLE_MAGIC_U64          0x454C5051354D5453ULL /* "STM5QPLE" */
#define STRATA_QWEN_FLASH_FORMAT_VERSION         1U

#define STRATA_QWEN_FLASH_PAGE_BYTES             16384ULL              /* 16 KiB Apple Silicon VM/UAT page */
#define STRATA_QWEN_FLASH_HEADER_PADDED_BYTES    131072U               /* 8 * 16 KiB (128 KiB) for Part0 & Part1 */
#define STRATA_QWEN_FLASH_PLE_HEADER_BYTES       16384U                /* 1 * 16 KiB page for ple_ngram.strata */
#define STRATA_QWEN_FLASH_TENSOR_ALIGN           256ULL                /* 256 B Metal buffer offset alignment */

#define STRATA_QWEN_FLASH_MAX_LAYERS             64U
#define STRATA_QWEN_FLASH_MAX_PART_LAYERS        32U                   /* Each UMA file holds 24 layers (<= 32) */
#define STRATA_QWEN_FLASH_MAX_NGRAM_SIZE         4U
#define STRATA_QWEN_FLASH_MAX_NGRAM_HEADS        32U
#define STRATA_QWEN_FLASH_MAX_PLE_SHARDS         128U
#define STRATA_QWEN_FLASH_MAX_PLE_HEAD_DIM       512U                  /* Max PLE head_dim_per_ngram (8 groups of 64) */
#define STRATA_QWEN_FLASH_MAX_PLE_GROUPS         (STRATA_QWEN_FLASH_MAX_PLE_HEAD_DIM / STRATA_QWEN_FLASH_QUANT_GROUP_SIZE)
#define STRATA_QWEN_FLASH_MAX_SPEC_DRAFT_K       6U                    /* Supports draft_k = 0..6 (verify M <= 7) */
#define STRATA_QWEN_FLASH_DEFAULT_MAX_SEQ_LEN    263168U               /* 256K (262,144) + 1,024 decode headroom */
#define STRATA_QWEN_FLASH_MAX_SEQ_LEN_CAP        524288U               /* 512K max supported by sel_bits[4096] */

#define STRATA_QWEN_FLASH_QUANT_GROUP_SIZE       64U
#define STRATA_QWEN_FLASH_QUANT_BITS             4U

/* Layer types in 3:1 hybrid schedule */
#define STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN      0U /* GatedDeltaNet (`l % 4 != 3`, 36 layers) */
#define STRATA_QWEN_FLASH_LAYER_FULL_ATTN        1U /* QSA + Indexer (`l % 4 == 3`, 12 layers + MTP) */

/* Tensor dtypes in `uma_part0.strata` & `uma_part1.strata` */
#define STRATA_QWEN_FLASH_DTYPE_NONE             0U
#define STRATA_QWEN_FLASH_DTYPE_F32              1U /* IEEE-754 float32 (`float`) */
#define STRATA_QWEN_FLASH_DTYPE_F16              2U /* IEEE-754 float16 (`uint16_t` / `half`) */
#define STRATA_QWEN_FLASH_DTYPE_U32              3U /* Packed 8x 4-bit unsigned nibbles per `uint32_t` */
#define STRATA_QWEN_FLASH_DTYPE_Q8_0             4U /* Planar symmetric Int8 gs=64: `int8_t[rows, cols]` + 256B-aligned `half[rows, cols/64]` */
#define STRATA_QWEN_FLASH_DTYPE_Q4_A             5U /* Planar affine Q4 gs=64: `uint32_t[rows_pad, cols/8]` + `half` scales + `half` biases */

/* Canonical Qwen3.8 / Qwen4-Exp special token IDs (`vocab_size = 248320`) */
#define STRATA_QWEN_FLASH_TOKEN_ENDOFTEXT        248044U /* `<|endoftext|>` (PLE boundary EOS & secondary EOS) */
#define STRATA_QWEN_FLASH_TOKEN_IM_START         248045U /* `<|im_start|>` */
#define STRATA_QWEN_FLASH_TOKEN_IM_END           248046U /* `<|im_end|>` (primary chat EOS) */
#define STRATA_QWEN_FLASH_TOKEN_THINK_START      248068U /* `<think>` */
#define STRATA_QWEN_FLASH_TOKEN_THINK_END        248069U /* `</think>` */

/* Default bundle filenames inside `$HOME/.cache/strata-apex/qwen3.8-flash-next/` */
#define STRATA_QWEN_FLASH_PART0_FILENAME         "uma_part0.strata"
#define STRATA_QWEN_FLASH_PART1_FILENAME         "uma_part1.strata"
#define STRATA_QWEN_FLASH_PLE_FILENAME           "ple_ngram.strata"
#define STRATA_QWEN_FLASH_DEFAULT_HF_REPO        "Qwen/Qwen3.8-Flash-Next"
#ifndef __METAL_VERSION__
#include "strata_paths.h"
#define STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR     strata_apex_bundle_path("qwen3.8-flash-next")
#endif

/* ============================================================================
 * 2. Page-Aligned On-Disk `.strata` Descriptors (POD, Exact 8-Byte Aligned)
 * ============================================================================ */

/*
 * 4-Bit Affine Matrix Descriptor (`group_size = 64`, `sizeof = 64` bytes):
 *   - `weight_off`: 256B-aligned offset (within `uma_part0` or `uma_part1`) to `uint32_t [rows_padded, cols / 8]`
 *   - `scales_off`: 256B-aligned offset to IEEE-754 FP16 `uint16_t [rows_padded, cols / 64]`
 *   - `biases_off`: 256B-aligned offset to IEEE-754 FP16 `uint16_t [rows_padded, cols / 64]`
 *   - Dequantization: `w[r, c] = float((u32 >> (4 * (c & 7))) & 0xF) * float(scale[r, c >> 6]) + float(bias[r, c >> 6])`
 *   - For 3D stacked experts (`[E, R_e, C]`): `rows_logical = rows_padded = E * R_e`, `cols = C`.
 *     Because `R_e % 64 == 0` and `C % 64 == 0`, expert `e` (`0 .. E-1`) begins at exact 256B-aligned offsets:
 *       `weight_off + e * R_e * (C / 2)`, `scales_off + e * R_e * (C / 64) * 2`, `biases_off + e * R_e * (C / 64) * 2`.
 */
typedef struct StrataQwenFlashQuantMatrix {
    sqf_u64 weight_off;
    sqf_u64 weight_bytes;
    sqf_u64 scales_off;
    sqf_u64 scales_bytes;
    sqf_u64 biases_off;
    sqf_u64 biases_bytes;
    sqf_u32 rows_logical;
    sqf_u32 rows_padded;
    sqf_u32 cols;
    sqf_u32 group_size;
} StrataQwenFlashQuantMatrix;

/*
 * Dense 1D/2D Tensor Slice Descriptor (`sizeof = 32` bytes, 256B-aligned offset):
 *   - Sensitive small projections (`HC down/up/inject`, `mlp.gate` router, `shared_expert` `gate/up/down`,
 *     `linear_attn.in_proj_a/b`, `indexer.index_qk_proj`, `ple.key_proj/value_proj`, `mtp.fc_embedding/fc_hidden`)
 *     are stored in `STRATA_QWEN_FLASH_DTYPE_F16` (`uint16_t` / `half`).
 *   - All RMSNorm weights (with `+1.0f` pre-folded at conversion EXCEPT `linear_attn.norm`),
 *     `shared_expert_gate [1, D]`, `conv1d.weight [10240, 4]`, `A_log [n_v]`, and `dt_bias [n_v]`
 *     are stored in lossless `STRATA_QWEN_FLASH_DTYPE_F32` (`float`).
 */
typedef struct StrataQwenFlashTensorSlice {
    sqf_u64 offset;
    sqf_u64 bytes;
    sqf_u32 elem_count;
    sqf_u32 dtype;
    sqf_u32 shape0;
    sqf_u32 shape1;
} StrataQwenFlashTensorSlice;

/* Pre-Attn / Pre-MLP 4-Stream Hyper-Connection Descriptor (`use_combine = true`, `sizeof = 128` B) */
typedef struct StrataQwenFlashHCDesc {
    StrataQwenFlashTensorSlice hc_norm;               /* F32 [hc_count * D] (grouped RMSNorm, +1.0 folded) */
    StrataQwenFlashTensorSlice input_mix_weight_down; /* F16 [hc_lowrank, hc_count * D] */
    StrataQwenFlashTensorSlice input_mix_weight_up;   /* F16 [hc_count * D, hc_lowrank] */
    StrataQwenFlashTensorSlice block_inject_weight;   /* F16 [hc_count, hc_count * D] */
} StrataQwenFlashHCDesc;

/* Final 4-Stream -> 1-Stream Hyper-Connection Mixer (`use_combine = false`, `sizeof = 96` B) */
typedef struct StrataQwenFlashMixerDesc {
    StrataQwenFlashTensorSlice hc_norm;               /* F32 [hc_count * D] (grouped RMSNorm, +1.0 folded) */
    StrataQwenFlashTensorSlice input_mix_weight_down; /* F16 [hc_lowrank, hc_count * D] */
    StrataQwenFlashTensorSlice input_mix_weight_up;   /* F16 [hc_count * D, hc_lowrank] */
} StrataQwenFlashMixerDesc;

/* Per-Layer Embedding (PLE) GPU Projection/Norm/Conv Weights (`layer_idx == 1`, `sizeof = 192` B) */
typedef struct StrataQwenFlashPleProjDesc {
    StrataQwenFlashTensorSlice key_proj;              /* F16 [hc_count * D, ple_embed_dim] ([10240, 2560]) */
    StrataQwenFlashTensorSlice value_proj;            /* F16 [D, ple_embed_dim]            ([2560,  2560]) */
    StrataQwenFlashTensorSlice norm_key;              /* F32 [hc_count * D] (grouped RMSNorm, +1.0 folded) */
    StrataQwenFlashTensorSlice norm_query;            /* F32 [hc_count * D] (grouped RMSNorm, +1.0 folded) */
    StrataQwenFlashTensorSlice norm_conv;             /* F32 [hc_count * D] (grouped RMSNorm, +1.0 folded) */
    StrataQwenFlashTensorSlice conv1d_weight;         /* F32 [hc_count * D, 4] (dilated by `ngram_size=3`!) */
} StrataQwenFlashPleProjDesc;

/* Decoder Layer Descriptor (`layers[0..23]` in `uma_part0.strata`, `layers[24..47]` & `mtp` in `uma_part1.strata`) */
typedef struct StrataQwenFlashLayerDesc {
    sqf_u32 global_layer_idx;  /* 0..47 for main backbone, 48 for `mtp.layers.0` */
    sqf_u32 part_id;           /* 0 (`uma_part0.strata`) or 1 (`uma_part1.strata`) */
    sqf_u32 layer_type;        /* STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN (0) or FULL_ATTN (1) */
    sqf_u32 has_ple;           /* 1 iff `global_layer_idx == 1` (`ple_layer_ids = [2]`), else 0 */
    sqf_u64 layer_page_off;    /* 16 KiB-aligned start offset inside `uma_part{part_id}.strata` */
    sqf_u64 layer_page_bytes;  /* 16 KiB-aligned byte length of this layer */

    /* Optional PLE projections (`populated iff has_ple == 1`) */
    StrataQwenFlashPleProjDesc ple;

    /* 4-Stream Hyper-Connections before Attention & before MoE */
    StrataQwenFlashHCDesc      attn_hc;
    StrataQwenFlashHCDesc      mlp_hc;

    /* Populated iff `layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN` (36 layers) */
    StrataQwenFlashQuantMatrix lin_in_proj_qkv;       /* Q4  [10240, 2560] -> q[2048], k[2048], v[6144] */
    StrataQwenFlashQuantMatrix lin_in_proj_z;         /* Q4  [6144,  2560] -> z[48, 128] sigmoid gate   */
    StrataQwenFlashQuantMatrix lin_out_proj;          /* Q4  [2560,  6144]                              */
    StrataQwenFlashTensorSlice lin_in_proj_a;         /* F16 [48,    2560]                              */
    StrataQwenFlashTensorSlice lin_in_proj_b;         /* F16 [48,    2560]                              */
    StrataQwenFlashTensorSlice lin_conv1d_weight;     /* F32 [10240, 4] causal taps [t-3, t-2, t-1, t]  */
    StrataQwenFlashTensorSlice lin_A_log;             /* F32 [48]                                       */
    StrataQwenFlashTensorSlice lin_dt_bias;           /* F32 [48]                                       */
    StrataQwenFlashTensorSlice lin_norm;              /* F32 [128] RMSNormGated (NO +1.0, sigmoid gate) */

    /* Populated iff `layer_type == STRATA_QWEN_FLASH_LAYER_FULL_ATTN` (12 layers + `mtp.layers.0`) */
    StrataQwenFlashQuantMatrix attn_q_proj;           /* Q4  [12288, 2560] -> [24, 512]: q[256], gate[256] */
    StrataQwenFlashQuantMatrix attn_k_proj;           /* Q4  [512,   2560] -> [2, 256]                  */
    StrataQwenFlashQuantMatrix attn_v_proj;           /* Q4  [512,   2560] -> [2, 256]                  */
    StrataQwenFlashQuantMatrix attn_o_proj;           /* Q4  [2560,  6144]                              */
    StrataQwenFlashTensorSlice attn_q_norm;           /* F32 [256] (+1.0 folded)                        */
    StrataQwenFlashTensorSlice attn_k_norm;           /* F32 [256] (+1.0 folded)                        */
    StrataQwenFlashTensorSlice indexer_qk_proj;       /* F16 [640,   2560] -> idx_q[4,128] + raw_k[128] */
    StrataQwenFlashTensorSlice indexer_q_norm;        /* F32 [128] (+1.0 folded)                        */
    StrataQwenFlashTensorSlice indexer_k_norm;        /* F32 [128] (+1.0 folded)                        */

    /* 512-Expert Top-10 SwiGLU MoE + Gated Shared Expert (populated for ALL 48 + 1 MTP layers) */
    StrataQwenFlashTensorSlice moe_router_gate;       /* F16 [512, 2560] (`mlp.gate.weight`)            */
    StrataQwenFlashQuantMatrix moe_experts_gate_up;   /* Q4  [512 * 1280, 2560] (`[E, 2*d_ff, D]`)      */
    StrataQwenFlashQuantMatrix moe_experts_down;      /* Q4  [512 * 2560, 640]  (`[E, D, d_ff]`)        */
    StrataQwenFlashTensorSlice shared_gate_proj;      /* F16 [640,  2560]                               */
    StrataQwenFlashTensorSlice shared_up_proj;        /* F16 [640,  2560]                               */
    StrataQwenFlashTensorSlice shared_down_proj;      /* F16 [2560, 640]                                */
    StrataQwenFlashTensorSlice shared_expert_gate;    /* F32 [2560] (`[1, 2560]` sigmoid scalar gate)   */
} StrataQwenFlashLayerDesc;

/* Native 1-Layer Multi-Token Prediction (`mtp.*`) Descriptor inside `uma_part1.strata` */
typedef struct StrataQwenFlashMtpDesc {
    sqf_u32                    enabled;               /* 1 when MTP weights present */
    sqf_u32                    reserved;
    sqf_u64                    mtp_page_off;          /* 16 KiB-aligned start offset in `uma_part1.strata` */
    sqf_u64                    mtp_page_bytes;        /* 16 KiB-aligned total bytes of MTP module */
    StrataQwenFlashTensorSlice pre_fc_norm_embedding; /* F32 [2560]  (+1.0 folded, ungrouped 2560-dim) */
    StrataQwenFlashTensorSlice pre_fc_norm_hidden;    /* F32 [10240] (+1.0 folded, UNGROUPED 10240-dim!) */
    StrataQwenFlashTensorSlice fc_embedding;          /* F16 [2560, 2560] */
    StrataQwenFlashTensorSlice fc_hidden;             /* F16 [2560, 2560] */
    StrataQwenFlashLayerDesc   layer0;                /* `mtp.layers.0.*` (`FULL_ATTN` + 512-expert MoE) */
    StrataQwenFlashMixerDesc   mixer;                 /* `mtp.hyper_connection_mixer.*` -> `lm_head` */
} StrataQwenFlashMtpDesc;

/* Complete Model Hyperparameters stored identically in `uma_part0`, `uma_part1`, and `ple_ngram` */
typedef struct StrataQwenFlashConfig {
    sqf_u32 num_hidden_layers;          /* 48 */
    sqf_u32 split_layer_idx;            /* 24 (`0..23` in `uma_part0`, `24..47` in `uma_part1`) */
    sqf_u32 hidden_size;                /* 2560 */
    sqf_u32 vocab_size;                 /* 248320 */
    sqf_u32 num_linear_layers;          /* 36 */
    sqf_u32 num_full_attn_layers;       /* 12 */
    sqf_u32 full_attention_interval;    /* 4 (`(l + 1) % 4 == 0` => full_attention) */

    /* Hyper-Connections (`Qwen4ExpTextGatedResidual`) */
    sqf_u32 hc_count;                   /* 4 (`hc_dim = 4 * 2560 = 10240`) */
    sqf_u32 hc_lowrank;                 /* 320 */

    /* Full Attention (`QSA`) + Sparse Block Indexer */
    sqf_u32 num_attention_heads;        /* 24 */
    sqf_u32 num_key_value_heads;        /* 2 (`12` Q-heads per KV-head) */
    sqf_u32 head_dim;                   /* 256 */
    sqf_u32 rotary_dim;                 /* 64 (`256 * 0.25`) */
    float   rope_theta;                 /* 10000000.0f (`1e7`) */
    sqf_u32 indexer_n_heads;            /* 4 */
    sqf_u32 indexer_kv_heads;           /* 1 */
    sqf_u32 indexer_head_dim;           /* 128 */
    sqf_u32 indexer_budget;             /* 2048 */
    sqf_u32 indexer_compress_ratio;     /* 4 (`block_topk = 512`) */

    /* Linear Attention (`GatedDeltaNet`) */
    sqf_u32 linear_num_key_heads;       /* 16 */
    sqf_u32 linear_key_head_dim;        /* 128 (`key_dim = 2048`) */
    sqf_u32 linear_num_value_heads;     /* 48 (`v_heads / k_heads = 3`) */
    sqf_u32 linear_value_head_dim;      /* 128 (`value_dim = 6144`) */
    sqf_u32 linear_conv_kernel_dim;     /* 4 */
    sqf_u32 linear_output_gate_sigmoid; /* 1 (`output_gate_type = "sigmoid"`, NOT silu!) */

    /* 512-Expert MoE + Shared Expert */
    sqf_u32 num_experts;                /* 512 */
    sqf_u32 num_experts_per_tok;        /* 10 */
    sqf_u32 moe_intermediate_size;      /* 640 */
    sqf_u32 shared_intermediate_size;   /* 640 */
    sqf_u32 norm_topk_prob;             /* 1 (renormalize top-10 probabilities) */

    /* Per-Layer Embedding (`PLE`) & Native `MTP` */
    sqf_u32 has_ple;                    /* 1 */
    sqf_u32 ple_layer_idx;              /* 1 (0-indexed layer for `ple_layer_ids = [2]`) */
    sqf_u32 ple_embed_dim;              /* 2560 */
    sqf_u32 ple_conv_kernel_size;       /* 4 (with `dilation = ngram_size = 3` -> state len = 9!) */
    sqf_u32 ngram_size;                 /* 3 */
    sqf_u32 heads_per_ngram;            /* 8 (`ngram_heads = 16`, `head_dim_per_ngram = 160`) */
    sqf_u32 ngram_vocab_size_base;      /* 20000000 */
    sqf_u32 ngram_vocab_pad_div;        /* 128 */
    sqf_u32 split_ngram_parts;          /* 128 */
    sqf_u32 ple_seed;                   /* 1234 */
    sqf_u32 ple_eos_token_id;           /* 248044 (`config.text_config.eos_token_id`) */
    sqf_u32 has_mtp;                    /* 1 */

    /* Normalization, Quantization & Generation Tokens */
    float   rms_norm_eps;               /* 1e-6f */
    sqf_u32 quant_group_size;           /* 64 */
    sqf_u32 quant_bits;                 /* 4 */
    sqf_u32 tie_word_embeddings;        /* 0 */
    sqf_u32 max_position_embeddings;    /* 262144 */
    sqf_u32 bos_token_id;               /* 248044 */
    sqf_u32 eos_token_ids[4];           /* {248046, 248044, 0, 0} */
    sqf_u32 num_eos_token_ids;          /* 2 */
    sqf_u32 reserved_pad;
} StrataQwenFlashConfig;

/*
 * Header at byte offset `0` of `<bundle_dir>/uma_part0.strata` (`~33.65 GiB << 80.64 GiB` MTLBuffer limit).
 * Holds `embed_tokens` + Layers `0 .. split_layer_idx - 1` (`0 .. 23`, including `layers[1].ple` projections).
 */
typedef struct StrataQwenFlashPart0Header {
    sqf_u64                     magic;                    /* STRATA_QWEN_FLASH_PART0_MAGIC_U64 */
    sqf_u32                     version;                  /* STRATA_QWEN_FLASH_FORMAT_VERSION */
    sqf_u32                     header_padded_bytes;      /* 131072 (8 * 16 KiB) */
    sqf_u64                     total_file_bytes;         /* Exact multiple of 16384 */
    sqf_u64                     active_bytes_per_tok;     /* Total 48-layer + head M=1 physical bytes */
    sqf_u64                     mtp_active_bytes_per_tok; /* 1-layer MTP draft step physical bytes */

    StrataQwenFlashConfig       cfg;

    /* Global Token Embedding (`[248320, 2560]` Q4 affine) */
    sqf_u64                     embed_page_off;
    sqf_u64                     embed_page_bytes;
    StrataQwenFlashQuantMatrix  embed_tokens;

    /* Layers `0 .. num_part0_layers - 1` (`0 .. 23`) */
    sqf_u32                     num_part0_layers;
    sqf_u32                     reserved0;
    StrataQwenFlashLayerDesc    layers[STRATA_QWEN_FLASH_MAX_PART_LAYERS];
} StrataQwenFlashPart0Header;

/*
 * Header at byte offset `0` of `<bundle_dir>/uma_part1.strata` (`~35.07 GiB << 80.64 GiB` MTLBuffer limit).
 * Holds Layers `split_layer_idx .. num_hidden_layers - 1` (`24 .. 47`) + `final_mixer` + `lm_head` + `mtp`.
 */
typedef struct StrataQwenFlashPart1Header {
    sqf_u64                     magic;                    /* STRATA_QWEN_FLASH_PART1_MAGIC_U64 */
    sqf_u32                     version;                  /* STRATA_QWEN_FLASH_FORMAT_VERSION */
    sqf_u32                     header_padded_bytes;      /* 131072 (8 * 16 KiB) */
    sqf_u64                     total_file_bytes;         /* Exact multiple of 16384 */

    StrataQwenFlashConfig       cfg;

    /* Layers `split_layer_idx .. num_hidden_layers - 1` (indexed `0 .. num_part1_layers - 1` here!) */
    sqf_u32                     first_global_layer_idx;   /* 24 */
    sqf_u32                     num_part1_layers;         /* 24 */
    StrataQwenFlashLayerDesc    layers[STRATA_QWEN_FLASH_MAX_PART_LAYERS];

    /* Final 4-Stream Mixer + Untied LM Head (`[248320, 2560]` Q4 affine) */
    sqf_u64                     head_page_off;
    sqf_u64                     head_page_bytes;
    StrataQwenFlashMixerDesc    final_mixer;
    StrataQwenFlashQuantMatrix  lm_head;

    /* Native 1-Layer Multi-Token Prediction (`mtp.*`) */
    StrataQwenFlashMtpDesc      mtp;
} StrataQwenFlashPart1Header;

/*
 * Header at byte offset `0` of `<bundle_dir>/ple_ngram.strata` (`27.418 GiB`, accessed via `F_NOCACHE` `pread`).
 * Each of the `padded_vocab_rows = 320,001,536` rows (`128` shards x `2,500,012` rows/shard) has exact
 * compact affine Q4 layout of `row_bytes = 92` bytes (`head_dim_per_ngram = 160`, `group_size = 64` -> `3` groups):
 *   - `uint32_t quants[head_dim_per_ngram / 8]`  (`80` B: 160 unsigned 4-bit nibbles)
 *   - `uint16_t scales[num_groups_per_row]`      (`6` B:  IEEE FP16 scales for `[0..63]`, `[64..127]`, `[128..159]`)
 *   - `uint16_t biases[num_groups_per_row]`      (`6` B:  IEEE FP16 biases for `[0..63]`, `[64..127]`, `[128..159]`)
 * Shard `s` (`0 .. 127`) starts at exact deterministic offset `data_page_off + s * rows_per_shard * row_bytes`,
 * allowing `@flash-stream` to convert & `pwrite` shards `0 .. 127` in ANY parallel arrival order!
 */
typedef struct StrataQwenFlashPleHeader {
    sqf_u64 magic;                                              /* STRATA_QWEN_FLASH_PLE_MAGIC_U64 */
    sqf_u32 version;                                            /* STRATA_QWEN_FLASH_FORMAT_VERSION */
    sqf_u32 header_padded_bytes;                                /* 16384 (1 * 16 KiB) */
    sqf_u64 total_file_bytes;                                   /* Exact multiple of 16384 */
    sqf_u64 data_page_off;                                      /* 16384 */

    sqf_u32 ngram_size;                                         /* 3 */
    sqf_u32 heads_per_ngram;                                    /* 8 */
    sqf_u32 ngram_heads;                                        /* 16 */
    sqf_u32 head_dim_per_ngram;                                 /* 160 (`ple_embed_dim / ngram_heads`) */
    sqf_u32 quant_group_size;                                   /* 64 */
    sqf_u32 num_groups_per_row;                                 /* 3 (`ceil(160 / 64)`) */
    sqf_u32 row_bytes;                                          /* 92 (`(160 / 2) + 3 * 4`) */
    sqf_u32 num_shards;                                         /* 128 */
    sqf_u64 rows_per_shard;                                     /* 2500012 (`padded_vocab_rows / num_shards`) */
    sqf_u64 total_logical_rows;                                 /* 320001446 */
    sqf_u64 padded_vocab_rows;                                  /* 320001536 */
    sqf_u32 ple_eos_token_id;                                   /* 248044 */
    sqf_u32 ple_seed;                                           /* 1234 */

    sqf_u64 layer_multipliers[STRATA_QWEN_FLASH_MAX_NGRAM_SIZE];    /* {23703573157769, 20109073645365, 8052911324071, 0} */
    sqf_u64 head_vocab_sizes[STRATA_QWEN_FLASH_MAX_NGRAM_HEADS];    /* 16 consecutive primes >= 20,000,000 */
    sqf_u64 head_offsets[STRATA_QWEN_FLASH_MAX_NGRAM_HEADS];        /* Cumulative row offsets for heads 0..15 */
    sqf_u64 shard_byte_offsets[STRATA_QWEN_FLASH_MAX_PLE_SHARDS];   /* Deterministic pwrite start offset per shard 0..127 */
} StrataQwenFlashPleHeader;

/* ============================================================================
 * 3. Host Deterministic Layout Builder, Lock-Free `mlock` PLE LFU Cache & ABIs
 * ============================================================================ */

#ifndef __METAL_VERSION__

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __cplusplus
static_assert(sizeof(StrataQwenFlashPart0Header) <= STRATA_QWEN_FLASH_HEADER_PADDED_BYTES,
              "Part0Header exceeds 128 KiB padded header region");
static_assert(sizeof(StrataQwenFlashPart1Header) <= STRATA_QWEN_FLASH_HEADER_PADDED_BYTES,
              "Part1Header exceeds 128 KiB padded header region");
static_assert(sizeof(StrataQwenFlashPleHeader) <= STRATA_QWEN_FLASH_PLE_HEADER_BYTES,
              "PleHeader exceeds 16 KiB padded header region");
extern "C" {
#endif

/* ---- 3.1 Bit-Exact IEEE-754 FP16 <-> FP32 Helpers & Alignment Utilities ---- */

static inline uint64_t sqf_align_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1ULL) & ~(align - 1ULL);
}

static inline float sqf_bf16_to_f32(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline float sqf_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000U)) << 16;
    uint32_t exp  = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;
    uint32_t bits;
    if (exp == 0U) {
        if (mant == 0U) {
            bits = sign;
        } else {
            exp = 1U;
            while ((mant & 0x0400U) == 0U) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFU;
            bits = sign | ((exp + 112U) << 23) | (mant << 13);
        }
    } else if (exp == 31U) {
        bits = sign | 0x7F800000U | (mant << 13);
    } else {
        bits = sign | ((exp + 112U) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint16_t sqf_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000U;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFU;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000U;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_bit = 1U << (shift - 1U);
        uint32_t sticky = (mant & (half_bit - 1U)) != 0U ? 1U : 0U;
        uint32_t lsb = (mant >> shift) & 1U;
        mant = (mant + half_bit - 1U + (lsb | sticky)) >> shift;
        return (uint16_t)(sign | mant);
    } else if (exp >= 31) {
        if (((x >> 23) & 0xFFU) == 0xFFU && mant != 0U) {
            return (uint16_t)(sign | 0x7E00U); /* NaN */
        }
        return (uint16_t)(sign | 0x7C00U); /* Clamp / Inf */
    }
    uint32_t round_bit = (mant >> 12) & 1U;
    uint32_t sticky    = (mant & 0xFFFU) != 0U ? 1U : 0U;
    uint32_t lsb       = (mant >> 13) & 1U;
    mant = (mant >> 13) + (round_bit & (lsb | sticky));
    if (mant >= 0x0400U) {
        mant = 0U;
        ++exp;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00U);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

/* ---- 3.2 Exact SplitMix64 & Prime Generator matching `modeling_qwen4_exp.py` ---- */

static inline uint64_t sqf_splitmix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

static inline bool sqf_is_prime_u64(uint64_t v) {
    if (v < 2ULL) return false;
    if ((v & 1ULL) == 0ULL) return v == 2ULL;
    for (uint64_t d = 3ULL; d * d <= v; d += 2ULL) {
        if ((v % d) == 0ULL) return false;
    }
    return true;
}

/* ---- 3.3 Canonical `Qwen/Qwen3.8-Flash-Next` Default Config & Deterministic Offset Planner ---- */

static inline void strata_qwen_flash_default_config(StrataQwenFlashConfig* cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->num_hidden_layers          = 48U;
    cfg->split_layer_idx            = 24U;
    cfg->hidden_size                = 2560U;
    cfg->vocab_size                 = 248320U;
    cfg->num_linear_layers          = 36U;
    cfg->num_full_attn_layers       = 12U;
    cfg->full_attention_interval    = 4U;

    cfg->hc_count                   = 4U;
    cfg->hc_lowrank                 = 320U;

    cfg->num_attention_heads        = 24U;
    cfg->num_key_value_heads        = 2U;
    cfg->head_dim                   = 256U;
    cfg->rotary_dim                 = 64U;
    cfg->rope_theta                 = 10000000.0f;
    cfg->indexer_n_heads            = 4U;
    cfg->indexer_kv_heads           = 1U;
    cfg->indexer_head_dim           = 128U;
    cfg->indexer_budget             = 2048U;
    cfg->indexer_compress_ratio     = 4U;

    cfg->linear_num_key_heads       = 16U;
    cfg->linear_key_head_dim        = 128U;
    cfg->linear_num_value_heads     = 48U;
    cfg->linear_value_head_dim      = 128U;
    cfg->linear_conv_kernel_dim     = 4U;
    cfg->linear_output_gate_sigmoid = 1U;

    cfg->num_experts                = 512U;
    cfg->num_experts_per_tok        = 10U;
    cfg->moe_intermediate_size      = 640U;
    cfg->shared_intermediate_size   = 640U;
    cfg->norm_topk_prob             = 1U;

    cfg->has_ple                    = 1U;
    cfg->ple_layer_idx              = 1U; /* 0-indexed (`ple_layer_ids = [2]`) */
    cfg->ple_embed_dim              = 2560U;
    cfg->ple_conv_kernel_size       = 4U;
    cfg->ngram_size                 = 3U;
    cfg->heads_per_ngram            = 8U;
    cfg->ngram_vocab_size_base      = 20000000U;
    cfg->ngram_vocab_pad_div        = 128U;
    cfg->split_ngram_parts          = 128U;
    cfg->ple_seed                   = 1234U;
    cfg->ple_eos_token_id           = STRATA_QWEN_FLASH_TOKEN_ENDOFTEXT;
    cfg->has_mtp                    = 1U;

    cfg->rms_norm_eps               = 1e-6f;
    cfg->quant_group_size           = 64U;
    cfg->quant_bits                 = 4U;
    cfg->tie_word_embeddings        = 0U;
    cfg->max_position_embeddings    = 262144U;
    cfg->bos_token_id               = STRATA_QWEN_FLASH_TOKEN_ENDOFTEXT;
    cfg->eos_token_ids[0]           = STRATA_QWEN_FLASH_TOKEN_IM_END;
    cfg->eos_token_ids[1]           = STRATA_QWEN_FLASH_TOKEN_ENDOFTEXT;
    cfg->num_eos_token_ids          = 2U;
}

static inline void sqf_q8_slice_layout(
    uint32_t  rows,
    uint32_t  cols,
    uint64_t* w_bytes_out,
    uint64_t* s_bytes_out
) {
    const uint64_t wb = sqf_align_up_u64((uint64_t)rows * (uint64_t)cols, STRATA_QWEN_FLASH_TENSOR_ALIGN);
    const uint64_t sb = sqf_align_up_u64((uint64_t)rows * (uint64_t)(cols / STRATA_QWEN_FLASH_QUANT_GROUP_SIZE) * 2ULL,
                                         STRATA_QWEN_FLASH_TENSOR_ALIGN);
    if (w_bytes_out) *w_bytes_out = wb;
    if (s_bytes_out) *s_bytes_out = sb;
}

static inline StrataQwenFlashQuantMatrix sqf_q4_slice_as_qmat(const StrataQwenFlashTensorSlice* sl) {
    StrataQwenFlashQuantMatrix m;
    memset(&m, 0, sizeof(m));
    const uint32_t rows        = sl->shape0;
    const uint32_t cols        = sl->shape1;
    const uint32_t gs          = STRATA_QWEN_FLASH_QUANT_GROUP_SIZE;
    const uint32_t rows_padded = (rows + 63U) & ~63U;
    const uint64_t wb = sqf_align_up_u64(((uint64_t)rows_padded * (uint64_t)cols) / 2ULL, STRATA_QWEN_FLASH_TENSOR_ALIGN);
    const uint64_t sb = sqf_align_up_u64((uint64_t)rows_padded * (uint64_t)(cols / gs) * 2ULL, STRATA_QWEN_FLASH_TENSOR_ALIGN);
    m.weight_off   = sl->offset;
    m.weight_bytes = wb;
    m.scales_off   = sl->offset + wb;
    m.scales_bytes = sb;
    m.biases_off   = sl->offset + wb + sb;
    m.biases_bytes = sb;
    m.rows_logical = rows;
    m.rows_padded  = rows_padded;
    m.cols         = cols;
    m.group_size   = gs;
    return m;
}

static inline StrataQwenFlashTensorSlice sqf_alloc_slice(
    uint64_t* cursor,
    uint32_t  dtype,
    uint32_t  shape0,
    uint32_t  shape1
) {
    StrataQwenFlashTensorSlice s;
    memset(&s, 0, sizeof(s));
    const uint32_t c1    = (shape1 > 0U) ? shape1 : 1U;
    const uint64_t elems = (uint64_t)shape0 * (uint64_t)c1;
    const uint64_t off   = sqf_align_up_u64(*cursor, STRATA_QWEN_FLASH_TENSOR_ALIGN);
    uint64_t bytes = 0ULL;
    if (dtype == STRATA_QWEN_FLASH_DTYPE_Q8_0) {
        uint64_t wb = 0ULL, sb = 0ULL;
        sqf_q8_slice_layout(shape0, c1, &wb, &sb);
        bytes = wb + sb;
    } else if (dtype == STRATA_QWEN_FLASH_DTYPE_Q4_A) {
        const uint32_t rp = (shape0 + 63U) & ~63U;
        const uint64_t wb = sqf_align_up_u64(((uint64_t)rp * (uint64_t)c1) / 2ULL, STRATA_QWEN_FLASH_TENSOR_ALIGN);
        const uint64_t sb = sqf_align_up_u64((uint64_t)rp * (uint64_t)(c1 / STRATA_QWEN_FLASH_QUANT_GROUP_SIZE) * 2ULL,
                                             STRATA_QWEN_FLASH_TENSOR_ALIGN);
        bytes = wb + 2ULL * sb;
    } else {
        const uint64_t bpe = (dtype == STRATA_QWEN_FLASH_DTYPE_F32 || dtype == STRATA_QWEN_FLASH_DTYPE_U32) ? 4ULL : 2ULL;
        bytes = sqf_align_up_u64(elems * bpe, STRATA_QWEN_FLASH_TENSOR_ALIGN);
    }
    s.offset     = off;
    s.bytes      = bytes;
    s.elem_count = (uint32_t)elems;
    s.dtype      = dtype;
    s.shape0     = shape0;
    s.shape1     = shape1;
    *cursor      = off + bytes;
    return s;
}

static inline StrataQwenFlashQuantMatrix sqf_alloc_q4(
    uint64_t* cursor,
    uint32_t  rows_logical,
    uint32_t  cols,
    uint32_t  group_size
) {
    StrataQwenFlashQuantMatrix m;
    memset(&m, 0, sizeof(m));
    uint32_t rows_padded = (rows_logical + 63U) & ~63U;
    uint64_t w_bytes = sqf_align_up_u64(((uint64_t)rows_padded * (uint64_t)cols) / 2ULL, STRATA_QWEN_FLASH_TENSOR_ALIGN);
    uint64_t s_bytes = sqf_align_up_u64((uint64_t)rows_padded * (uint64_t)(cols / group_size) * 2ULL, STRATA_QWEN_FLASH_TENSOR_ALIGN);
    uint64_t off     = sqf_align_up_u64(*cursor, STRATA_QWEN_FLASH_TENSOR_ALIGN);

    m.weight_off   = off;
    m.weight_bytes = w_bytes;
    off += w_bytes;

    m.scales_off   = off;
    m.scales_bytes = s_bytes;
    off += s_bytes;

    m.biases_off   = off;
    m.biases_bytes = s_bytes;
    off += s_bytes;

    m.rows_logical = rows_logical;
    m.rows_padded  = rows_padded;
    m.cols         = cols;
    m.group_size   = group_size;
    *cursor        = off;
    return m;
}

static inline void sqf_plan_hc_ex(
    uint64_t* cursor,
    const StrataQwenFlashConfig* cfg,
    uint32_t mix_dtype,
    StrataQwenFlashHCDesc* hc
) {
    const uint32_t hc_dim = cfg->hc_count * cfg->hidden_size;
    hc->hc_norm               = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, hc_dim, 1U);
    hc->input_mix_weight_down = sqf_alloc_slice(cursor, mix_dtype, cfg->hc_lowrank, hc_dim);
    hc->input_mix_weight_up   = sqf_alloc_slice(cursor, mix_dtype, hc_dim, cfg->hc_lowrank);
    hc->block_inject_weight   = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F16, cfg->hc_count, hc_dim);
}

static inline void sqf_plan_hc(
    uint64_t* cursor,
    const StrataQwenFlashConfig* cfg,
    StrataQwenFlashHCDesc* hc
) {
    sqf_plan_hc_ex(cursor, cfg, STRATA_QWEN_FLASH_DTYPE_F16, hc);
}

static inline void sqf_plan_mixer_ex(
    uint64_t* cursor,
    const StrataQwenFlashConfig* cfg,
    uint32_t mix_dtype,
    StrataQwenFlashMixerDesc* mx
) {
    const uint32_t hc_dim = cfg->hc_count * cfg->hidden_size;
    mx->hc_norm               = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, hc_dim, 1U);
    mx->input_mix_weight_down = sqf_alloc_slice(cursor, mix_dtype, cfg->hc_lowrank, hc_dim);
    mx->input_mix_weight_up   = sqf_alloc_slice(cursor, mix_dtype, hc_dim, cfg->hc_lowrank);
}

static inline void sqf_plan_mixer(
    uint64_t* cursor,
    const StrataQwenFlashConfig* cfg,
    StrataQwenFlashMixerDesc* mx
) {
    sqf_plan_mixer_ex(cursor, cfg, STRATA_QWEN_FLASH_DTYPE_F16, mx);
}

static inline uint64_t sqf_plan_decoder_layer_ex(
    uint64_t* cursor,
    const StrataQwenFlashConfig* cfg,
    uint32_t global_layer_idx,
    uint32_t part_id,
    uint32_t layer_type,
    uint32_t has_ple,
    bool quantize_dense,
    StrataQwenFlashLayerDesc* ld
) {
    memset(ld, 0, sizeof(*ld));
    const uint64_t start_off = sqf_align_up_u64(*cursor, STRATA_QWEN_FLASH_PAGE_BYTES);
    *cursor = start_off;

    ld->global_layer_idx = global_layer_idx;
    ld->part_id          = part_id;
    ld->layer_type       = layer_type;
    ld->has_ple          = has_ple;
    ld->layer_page_off   = start_off;

    const uint32_t d         = cfg->hidden_size;
    const uint32_t hc_dim    = cfg->hc_count * d;
    const uint32_t gs        = (cfg->quant_group_size > 0U) ? cfg->quant_group_size : STRATA_QWEN_FLASH_QUANT_GROUP_SIZE;
    const uint32_t hc_dt     = (quantize_dense && (hc_dim % gs == 0U) && (cfg->hc_lowrank % gs == 0U))
                                   ? STRATA_QWEN_FLASH_DTYPE_Q8_0 : STRATA_QWEN_FLASH_DTYPE_F16;
    const uint32_t ple_dt    = (quantize_dense && (cfg->ple_embed_dim % gs == 0U))
                                   ? STRATA_QWEN_FLASH_DTYPE_Q8_0 : STRATA_QWEN_FLASH_DTYPE_F16;
    const uint32_t shared_dt = (quantize_dense && (d % gs == 0U) && (cfg->shared_intermediate_size % gs == 0U))
                                   ? STRATA_QWEN_FLASH_DTYPE_Q4_A : STRATA_QWEN_FLASH_DTYPE_F16;
    uint64_t active_bytes = 0ULL;

    if (has_ple) {
        ld->ple.key_proj      = sqf_alloc_slice(cursor, ple_dt, hc_dim, cfg->ple_embed_dim);
        ld->ple.value_proj    = sqf_alloc_slice(cursor, ple_dt, d, cfg->ple_embed_dim);
        ld->ple.norm_key      = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, hc_dim, 1U);
        ld->ple.norm_query    = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, hc_dim, 1U);
        ld->ple.norm_conv     = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, hc_dim, 1U);
        ld->ple.conv1d_weight = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, hc_dim, cfg->ple_conv_kernel_size);
        active_bytes += ld->ple.key_proj.bytes + ld->ple.value_proj.bytes +
                        ld->ple.norm_key.bytes + ld->ple.norm_query.bytes +
                        ld->ple.norm_conv.bytes + ld->ple.conv1d_weight.bytes;
    }

    const uint64_t hc_before = *cursor;
    sqf_plan_hc_ex(cursor, cfg, hc_dt, &ld->attn_hc);
    sqf_plan_hc_ex(cursor, cfg, hc_dt, &ld->mlp_hc);
    active_bytes += (*cursor - hc_before);

    const uint64_t attn_before = *cursor;
    if (layer_type == STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN) {
        const uint32_t k_dim    = cfg->linear_num_key_heads * cfg->linear_key_head_dim;
        const uint32_t v_dim    = cfg->linear_num_value_heads * cfg->linear_value_head_dim;
        const uint32_t conv_dim = 2U * k_dim + v_dim;
        const uint32_t nv       = cfg->linear_num_value_heads;
        ld->lin_in_proj_qkv   = sqf_alloc_q4(cursor, conv_dim, d, gs);
        ld->lin_in_proj_z     = sqf_alloc_q4(cursor, v_dim, d, gs);
        ld->lin_out_proj      = sqf_alloc_q4(cursor, d, v_dim, gs);
        ld->lin_in_proj_a     = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F16, nv, d);
        ld->lin_in_proj_b     = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F16, nv, d);
        ld->lin_conv1d_weight = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, conv_dim, cfg->linear_conv_kernel_dim);
        ld->lin_A_log         = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, nv, 1U);
        ld->lin_dt_bias       = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, nv, 1U);
        ld->lin_norm          = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, cfg->linear_value_head_dim, 1U);
    } else {
        const uint32_t q_out   = cfg->num_attention_heads * cfg->head_dim * 2U;
        const uint32_t kv_out  = cfg->num_key_value_heads * cfg->head_dim;
        const uint32_t o_in    = cfg->num_attention_heads * cfg->head_dim;
        const uint32_t idx_out = (cfg->indexer_n_heads + cfg->indexer_kv_heads) * cfg->indexer_head_dim;
        ld->attn_q_proj       = sqf_alloc_q4(cursor, q_out, d, gs);
        ld->attn_k_proj       = sqf_alloc_q4(cursor, kv_out, d, gs);
        ld->attn_v_proj       = sqf_alloc_q4(cursor, kv_out, d, gs);
        ld->attn_o_proj       = sqf_alloc_q4(cursor, d, o_in, gs);
        ld->attn_q_norm       = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, cfg->head_dim, 1U);
        ld->attn_k_norm       = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, cfg->head_dim, 1U);
        ld->indexer_qk_proj   = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F16, idx_out, d);
        ld->indexer_q_norm    = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, cfg->indexer_head_dim, 1U);
        ld->indexer_k_norm    = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, cfg->indexer_head_dim, 1U);
    }
    active_bytes += (*cursor - attn_before);

    /* Router + Shared Expert (always active) + 512 Routed Experts (top_k active) */
    const uint64_t shared_before = *cursor;
    ld->moe_router_gate    = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F16, cfg->num_experts, d);
    ld->shared_gate_proj   = sqf_alloc_slice(cursor, shared_dt, cfg->shared_intermediate_size, d);
    ld->shared_up_proj     = sqf_alloc_slice(cursor, shared_dt, cfg->shared_intermediate_size, d);
    ld->shared_down_proj   = sqf_alloc_slice(cursor, shared_dt, d, cfg->shared_intermediate_size);
    ld->shared_expert_gate = sqf_alloc_slice(cursor, STRATA_QWEN_FLASH_DTYPE_F32, d, 1U);
    active_bytes += (*cursor - shared_before);

    ld->moe_experts_gate_up = sqf_alloc_q4(cursor, cfg->num_experts * (2U * cfg->moe_intermediate_size), d, gs);
    ld->moe_experts_down    = sqf_alloc_q4(cursor, cfg->num_experts * d, cfg->moe_intermediate_size, gs);

    const uint64_t one_gu = ((uint64_t)(2U * cfg->moe_intermediate_size) * (uint64_t)d) / 2ULL +
                            (uint64_t)(2U * cfg->moe_intermediate_size) * (uint64_t)(d / gs) * 4ULL;
    const uint64_t one_dn = ((uint64_t)d * (uint64_t)cfg->moe_intermediate_size) / 2ULL +
                            (uint64_t)d * (uint64_t)(cfg->moe_intermediate_size / gs) * 4ULL;
    active_bytes += (uint64_t)cfg->num_experts_per_tok * (one_gu + one_dn);

    const uint64_t end_off = sqf_align_up_u64(*cursor, STRATA_QWEN_FLASH_PAGE_BYTES);
    *cursor = end_off;
    ld->layer_page_bytes = end_off - start_off;
    return active_bytes;
}

static inline uint64_t sqf_plan_decoder_layer(
    uint64_t* cursor,
    const StrataQwenFlashConfig* cfg,
    uint32_t global_layer_idx,
    uint32_t part_id,
    uint32_t layer_type,
    uint32_t has_ple,
    StrataQwenFlashLayerDesc* ld
) {
    return sqf_plan_decoder_layer_ex(cursor, cfg, global_layer_idx, part_id, layer_type, has_ple, false, ld);
}

/*
 * Deterministically computes EVERY file size and tensor byte offset across `uma_part0.strata`,
 * `uma_part1.strata`, and `ple_ngram.strata` purely from `cfg` (zero shard reads required!).
 * When `quantize_dense == true`, HC mixers + PLE projs + MTP FC use `Q8_0` and Shared Experts use `Q4_A`.
 */
static inline void strata_qwen_flash_init_layout_quant(
    const StrataQwenFlashConfig* cfg,
    bool                         quantize_dense,
    StrataQwenFlashPart0Header*  part0_out,
    StrataQwenFlashPart1Header*  part1_out,
    StrataQwenFlashPleHeader*    ple_out
) {
    memset(part0_out, 0, sizeof(*part0_out));
    memset(part1_out, 0, sizeof(*part1_out));
    if (ple_out) memset(ple_out, 0, sizeof(*ple_out));

    const uint32_t gs       = (cfg->quant_group_size > 0U) ? cfg->quant_group_size : STRATA_QWEN_FLASH_QUANT_GROUP_SIZE;
    const uint32_t full_int = (cfg->full_attention_interval > 0U) ? cfg->full_attention_interval : 1U;
    const uint32_t n_layers = (cfg->num_hidden_layers <= STRATA_QWEN_FLASH_MAX_LAYERS)
                                  ? cfg->num_hidden_layers : STRATA_QWEN_FLASH_MAX_LAYERS;
    const uint32_t split_i  = (cfg->split_layer_idx <= n_layers) ? cfg->split_layer_idx : n_layers;
    const uint32_t p0_cnt   = (split_i <= STRATA_QWEN_FLASH_MAX_PART_LAYERS)
                                  ? split_i : STRATA_QWEN_FLASH_MAX_PART_LAYERS;
    const uint32_t p1_raw   = n_layers - split_i;
    const uint32_t p1_cnt   = (p1_raw <= STRATA_QWEN_FLASH_MAX_PART_LAYERS)
                                  ? p1_raw : STRATA_QWEN_FLASH_MAX_PART_LAYERS;

    /* ---- Part 0 (`uma_part0.strata`: Embed + Layers `0 .. split_layer_idx - 1`) ---- */
    part0_out->magic               = STRATA_QWEN_FLASH_PART0_MAGIC_U64;
    part0_out->version             = STRATA_QWEN_FLASH_FORMAT_VERSION;
    part0_out->header_padded_bytes = STRATA_QWEN_FLASH_HEADER_PADDED_BYTES;
    part0_out->cfg                 = *cfg;

    uint64_t cur0 = STRATA_QWEN_FLASH_HEADER_PADDED_BYTES;
    part0_out->embed_page_off   = cur0;
    part0_out->embed_tokens     = sqf_alloc_q4(&cur0, cfg->vocab_size, cfg->hidden_size, gs);
    cur0                        = sqf_align_up_u64(cur0, STRATA_QWEN_FLASH_PAGE_BYTES);
    part0_out->embed_page_bytes = cur0 - part0_out->embed_page_off;

    uint64_t total_active = ((uint64_t)cfg->hidden_size / 2ULL) + (uint64_t)(cfg->hidden_size / gs) * 4ULL;
    part0_out->num_part0_layers = p0_cnt;
    for (uint32_t l = 0U; l < p0_cnt; ++l) {
        const uint32_t ltype = ((l + 1U) % full_int == 0U)
                                   ? STRATA_QWEN_FLASH_LAYER_FULL_ATTN
                                   : STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN;
        const uint32_t has_ple = (cfg->has_ple && l == cfg->ple_layer_idx) ? 1U : 0U;
        total_active += sqf_plan_decoder_layer_ex(&cur0, cfg, l, 0U, ltype, has_ple, quantize_dense, &part0_out->layers[l]);
    }
    part0_out->total_file_bytes = sqf_align_up_u64(cur0, STRATA_QWEN_FLASH_PAGE_BYTES);

    /* ---- Part 1 (`uma_part1.strata`: Layers `split..L-1` + Final Mixer + LM Head + MTP) ---- */
    part1_out->magic                  = STRATA_QWEN_FLASH_PART1_MAGIC_U64;
    part1_out->version                = STRATA_QWEN_FLASH_FORMAT_VERSION;
    part1_out->header_padded_bytes    = STRATA_QWEN_FLASH_HEADER_PADDED_BYTES;
    part1_out->cfg                    = *cfg;
    part1_out->first_global_layer_idx = split_i;
    part1_out->num_part1_layers       = p1_cnt;

    uint64_t cur1 = STRATA_QWEN_FLASH_HEADER_PADDED_BYTES;
    for (uint32_t i = 0U; i < p1_cnt; ++i) {
        const uint32_t l     = split_i + i;
        const uint32_t ltype = ((l + 1U) % full_int == 0U)
                                   ? STRATA_QWEN_FLASH_LAYER_FULL_ATTN
                                   : STRATA_QWEN_FLASH_LAYER_LINEAR_ATTN;
        total_active += sqf_plan_decoder_layer_ex(
            &cur1, cfg, l, 1U, ltype, 0U, quantize_dense, &part1_out->layers[i]
        );
    }

    const uint32_t hc_dim = cfg->hc_count * cfg->hidden_size;
    const uint32_t hc_dt  = (quantize_dense && (hc_dim % gs == 0U) && (cfg->hc_lowrank % gs == 0U))
                                ? STRATA_QWEN_FLASH_DTYPE_Q8_0 : STRATA_QWEN_FLASH_DTYPE_F16;
    const uint32_t fc_dt  = (quantize_dense && (cfg->hidden_size % gs == 0U))
                                ? STRATA_QWEN_FLASH_DTYPE_Q8_0 : STRATA_QWEN_FLASH_DTYPE_F16;

    part1_out->head_page_off = sqf_align_up_u64(cur1, STRATA_QWEN_FLASH_PAGE_BYTES);
    cur1 = part1_out->head_page_off;
    sqf_plan_mixer_ex(&cur1, cfg, hc_dt, &part1_out->final_mixer);
    part1_out->lm_head = sqf_alloc_q4(&cur1, cfg->vocab_size, cfg->hidden_size, gs);
    const uint64_t head_raw_end = cur1;
    cur1 = sqf_align_up_u64(cur1, STRATA_QWEN_FLASH_PAGE_BYTES);
    part1_out->head_page_bytes = cur1 - part1_out->head_page_off;
    total_active += (head_raw_end - part1_out->head_page_off);

    uint64_t mtp_active = 0ULL;
    if (cfg->has_mtp) {
        const uint32_t d = cfg->hidden_size;
        part1_out->mtp.enabled      = 1U;
        part1_out->mtp.mtp_page_off = cur1;

        const uint64_t fc_start = cur1;
        part1_out->mtp.pre_fc_norm_embedding = sqf_alloc_slice(&cur1, STRATA_QWEN_FLASH_DTYPE_F32, d, 1U);
        part1_out->mtp.pre_fc_norm_hidden    = sqf_alloc_slice(&cur1, STRATA_QWEN_FLASH_DTYPE_F32, hc_dim, 1U);
        part1_out->mtp.fc_embedding          = sqf_alloc_slice(&cur1, fc_dt, d, d);
        part1_out->mtp.fc_hidden             = sqf_alloc_slice(&cur1, fc_dt, d, d);
        mtp_active += (cur1 - fc_start);

        mtp_active += sqf_plan_decoder_layer_ex(
            &cur1, cfg, cfg->num_hidden_layers, 1U, STRATA_QWEN_FLASH_LAYER_FULL_ATTN, 0U, quantize_dense, &part1_out->mtp.layer0
        );

        const uint64_t mx_start = cur1;
        sqf_plan_mixer_ex(&cur1, cfg, hc_dt, &part1_out->mtp.mixer);
        mtp_active += (cur1 - mx_start) +
                      part1_out->lm_head.weight_bytes + part1_out->lm_head.scales_bytes + part1_out->lm_head.biases_bytes;

        cur1 = sqf_align_up_u64(cur1, STRATA_QWEN_FLASH_PAGE_BYTES);
        part1_out->mtp.mtp_page_bytes = cur1 - part1_out->mtp.mtp_page_off;
    }
    part1_out->total_file_bytes         = sqf_align_up_u64(cur1, STRATA_QWEN_FLASH_PAGE_BYTES);
    part0_out->active_bytes_per_tok     = total_active;
    part0_out->mtp_active_bytes_per_tok = mtp_active;

    /* ---- PLE N-Gram Store (`ple_ngram.strata`: 128 shards -> `27.418 GiB` on NVMe) ---- */
    if (ple_out && cfg->has_ple) {
        ple_out->magic               = STRATA_QWEN_FLASH_PLE_MAGIC_U64;
        ple_out->version             = STRATA_QWEN_FLASH_FORMAT_VERSION;
        ple_out->header_padded_bytes = STRATA_QWEN_FLASH_PLE_HEADER_BYTES;
        ple_out->data_page_off       = STRATA_QWEN_FLASH_PLE_HEADER_BYTES;
        ple_out->ngram_size          = cfg->ngram_size;
        ple_out->heads_per_ngram     = cfg->heads_per_ngram;
        const uint32_t ng_heads      = (cfg->ngram_size >= 2U && cfg->heads_per_ngram > 0U)
                                           ? ((cfg->ngram_size - 1U) * cfg->heads_per_ngram) : 1U;
        ple_out->ngram_heads         = ng_heads;
        ple_out->head_dim_per_ngram  = cfg->ple_embed_dim / ng_heads;
        ple_out->quant_group_size    = gs;
        ple_out->num_groups_per_row  = (ple_out->head_dim_per_ngram + gs - 1U) / gs;
        ple_out->row_bytes           = (ple_out->head_dim_per_ngram / 2U) + ple_out->num_groups_per_row * 4U;
        ple_out->num_shards          = cfg->split_ngram_parts;
        ple_out->ple_eos_token_id    = cfg->ple_eos_token_id;
        ple_out->ple_seed            = cfg->ple_seed;

        /* `_build_layer_multipliers` (`ple_layer_index = 0`) */
        const uint64_t max_long       = 0x7FFFFFFFFFFFFFFFULL;
        const uint64_t multiplier_max = max_long / (cfg->vocab_size > 0U ? (uint64_t)cfg->vocab_size : 1ULL);
        const uint64_t half_bound     = (multiplier_max / 2ULL) > 0ULL ? (multiplier_max / 2ULL) : 1ULL;
        const uint64_t base_seed      = (uint64_t)cfg->ple_seed; /* + 10007 * 0 */
        for (uint32_t i = 0U; i < cfg->ngram_size && i < STRATA_QWEN_FLASH_MAX_NGRAM_SIZE; ++i) {
            uint64_t val = base_seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1U);
            ple_out->layer_multipliers[i] = 2ULL * (sqf_splitmix64(val) % half_bound) + 1ULL;
        }

        /* Consecutive primes >= `ngram_vocab_size_base` */
        uint64_t p   = (cfg->ngram_vocab_size_base >= 2U) ? ((uint64_t)cfg->ngram_vocab_size_base - 1ULL) : 1ULL;
        uint64_t tot = 0ULL;
        for (uint32_t h = 0U; h < ple_out->ngram_heads && h < STRATA_QWEN_FLASH_MAX_NGRAM_HEADS; ++h) {
            ++p;
            while (!sqf_is_prime_u64(p)) ++p;
            ple_out->head_vocab_sizes[h] = p;
            ple_out->head_offsets[h]     = tot;
            tot += p;
        }
        const uint64_t div    = cfg->ngram_vocab_pad_div > 0U ? (uint64_t)cfg->ngram_vocab_pad_div : 128ULL;
        const uint64_t padded = ((tot + div - 1ULL) / div) * div;
        ple_out->total_logical_rows = tot;
        ple_out->padded_vocab_rows  = padded;
        ple_out->rows_per_shard     = padded / (ple_out->num_shards > 0U ? (uint64_t)ple_out->num_shards : 1ULL);

        const uint64_t shard_bytes = ple_out->rows_per_shard * (uint64_t)ple_out->row_bytes;
        for (uint32_t s = 0U; s < ple_out->num_shards && s < STRATA_QWEN_FLASH_MAX_PLE_SHARDS; ++s) {
            ple_out->shard_byte_offsets[s] = ple_out->data_page_off + (uint64_t)s * shard_bytes;
        }
        ple_out->total_file_bytes = sqf_align_up_u64(
            ple_out->data_page_off + padded * (uint64_t)ple_out->row_bytes,
            STRATA_QWEN_FLASH_PAGE_BYTES
        );
    }
}

static inline void strata_qwen_flash_init_layout(
    const StrataQwenFlashConfig* cfg,
    StrataQwenFlashPart0Header*  part0_out,
    StrataQwenFlashPart1Header*  part1_out,
    StrataQwenFlashPleHeader*    ple_out
) {
    strata_qwen_flash_init_layout_quant(cfg, false, part0_out, part1_out, ple_out);
}

/* ---- 3.4 Lock-Free 16-Bank 4-Way `mlock` PLE LFU Cache + `F_NOCACHE` `pread` Dequantizer ---- */

#define STRATA_QWEN_FLASH_PLE_LFU_WAYS 4U

typedef struct StrataQwenFlashPleCache {
    uint8_t*  arena_base;       /* Single mmap + mlock buffer (`0` page faults!) */
    size_t    arena_bytes;
    bool      arena_mlocked;
    uint32_t  ngram_heads;      /* 16 */
    uint32_t  head_dim;         /* 160 */
    uint32_t  sets_per_head;    /* Power of 2 (e.g. 4096 -> 4096 * 4 * 16 = 262,144 rows) */
    uint32_t  set_mask;         /* `sets_per_head - 1` */
    uint64_t* tags;             /* `[ngram_heads * sets_per_head * 4]`, `UINT64_MAX` = empty */
    uint32_t* freqs;            /* `[ngram_heads * sets_per_head * 4]` LFU counters */
    uint16_t* values_f16;       /* `[ngram_heads * sets_per_head * 4 * head_dim]` dequantized FP16 */

    /* Telemetry counters (`atomic` or engine-thread updated, zero overhead) */
    uint64_t  lfu_hits;
    uint64_t  lfu_misses;
    uint64_t  nvme_preads;
    uint64_t  nvme_bytes_read;
} StrataQwenFlashPleCache;

static inline int strata_qwen_flash_ple_cache_init(
    StrataQwenFlashPleCache* cache,
    uint32_t ple_embed_dim,
    uint32_t ngram_heads,
    size_t   target_bytes       /* 0 => default ~96 MiB (`sets_per_head = 4096`, `262,144` rows) */
) {
    memset(cache, 0, sizeof(*cache));
    if (ngram_heads == 0U || ple_embed_dim == 0U || (ple_embed_dim % ngram_heads) != 0U) return -1;
    const uint32_t head_dim = ple_embed_dim / ngram_heads;
    if (target_bytes == 0U) target_bytes = 96ULL * 1024ULL * 1024ULL;

    const size_t bytes_per_set_all_heads =
        (size_t)ngram_heads * (size_t)STRATA_QWEN_FLASH_PLE_LFU_WAYS *
        (sizeof(uint64_t) + sizeof(uint32_t) + (size_t)head_dim * sizeof(uint16_t));
    uint32_t sets = 64U;
    while ((size_t)(sets * 2U) * bytes_per_set_all_heads <= target_bytes && sets < (1U << 20)) {
        sets <<= 1U;
    }

    const size_t total_slots = (size_t)ngram_heads * (size_t)sets * (size_t)STRATA_QWEN_FLASH_PLE_LFU_WAYS;
    const size_t off_tags    = 0U;
    const size_t off_freqs   = (size_t)sqf_align_up_u64(off_tags + total_slots * sizeof(uint64_t), 64ULL);
    const size_t off_vals    = (size_t)sqf_align_up_u64(off_freqs + total_slots * sizeof(uint32_t), 64ULL);
    const size_t total_bytes = (size_t)sqf_align_up_u64(
        off_vals + total_slots * (size_t)head_dim * sizeof(uint16_t),
        STRATA_QWEN_FLASH_PAGE_BYTES
    );

    void* ptr = mmap(NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (ptr == MAP_FAILED) return -1;
    memset(ptr, 0, total_bytes);

    cache->arena_base    = (uint8_t*)ptr;
    cache->arena_bytes   = total_bytes;
    cache->arena_mlocked = (mlock(ptr, total_bytes) == 0);
    cache->ngram_heads   = ngram_heads;
    cache->head_dim      = head_dim;
    cache->sets_per_head = sets;
    cache->set_mask      = sets - 1U;
    cache->tags          = (uint64_t*)(cache->arena_base + off_tags);
    cache->freqs         = (uint32_t*)(cache->arena_base + off_freqs);
    cache->values_f16    = (uint16_t*)(cache->arena_base + off_vals);

    for (size_t i = 0U; i < total_slots; ++i) {
        cache->tags[i] = UINT64_MAX;
    }
    return 0;
}

static inline void strata_qwen_flash_ple_cache_free(StrataQwenFlashPleCache* cache) {
    if (!cache) return;
    if (cache->arena_base && cache->arena_bytes > 0U) {
        if (cache->arena_mlocked) munlock(cache->arena_base, cache->arena_bytes);
        munmap(cache->arena_base, cache->arena_bytes);
    }
    memset(cache, 0, sizeof(*cache));
}

static inline void strata_qwen_flash_ple_cache_reset_stats(StrataQwenFlashPleCache* cache) {
    if (!cache) return;
    cache->lfu_hits        = 0ULL;
    cache->lfu_misses      = 0ULL;
    cache->nvme_preads     = 0ULL;
    cache->nvme_bytes_read = 0ULL;
}

/*
 * Computes the exact 16 global row indices in `ple_ngram.strata` for token `tok0` with preceding
 * tokens `prev1` (`t-1`) and `prev2` (`t-2`), implementing `_shift_right_ignore_eos` identically to HF.
 */
static inline void strata_qwen_flash_ple_hash_token(
    const StrataQwenFlashPleHeader* ple_hdr,
    uint32_t  tok0,
    uint32_t  prev1,
    uint32_t  prev2,
    uint64_t* out_global_rows /* `[ple_hdr->ngram_heads]` (16) */
) {
    const uint32_t eos = ple_hdr->ple_eos_token_id;
    const uint64_t s0  = (uint64_t)tok0;
    const uint64_t s1  = (uint64_t)prev1;
    const uint64_t s2  = (prev1 == eos) ? (uint64_t)eos : (uint64_t)prev2;

    uint64_t mixed = s0 * ple_hdr->layer_multipliers[0];
    const uint32_t h_per = ple_hdr->heads_per_ngram;
    for (uint32_t order = 1U; order < ple_hdr->ngram_size; ++order) {
        const uint64_t st = (order == 1U) ? s1 : s2;
        mixed ^= (st * ple_hdr->layer_multipliers[order]);
        const uint32_t base_h = (order - 1U) * h_per;
        for (uint32_t k = 0U; k < h_per; ++k) {
            const uint32_t h = base_h + k;
            out_global_rows[h] = ple_hdr->head_offsets[h] + (mixed % ple_hdr->head_vocab_sizes[h]);
        }
    }
}

/* Dequantizes one packed PLE Q4 affine row (`92` B for `head_dim=160`) into `out_f16[head_dim]`. */
static inline void strata_qwen_flash_ple_dequant_row_f16(
    const uint8_t* raw_row,
    uint32_t       head_dim,
    uint32_t       group_size,
    uint16_t*      out_f16
) {
    const uint32_t num_u32    = head_dim >> 3U;
    const uint32_t num_groups = (head_dim + group_size - 1U) / group_size;
    const uint8_t* q_ptr      = raw_row;
    const uint8_t* s_ptr      = raw_row + (size_t)num_u32 * 4U;
    const uint8_t* b_ptr      = s_ptr + (size_t)num_groups * 2U;

    for (uint32_t u = 0U; u < num_u32; ++u) {
        uint32_t packed;
        memcpy(&packed, q_ptr + (size_t)u * 4U, sizeof(packed));
        const uint32_t dim_base = u << 3U;
        const uint32_t grp      = dim_base / group_size;
        uint16_t sh, bh;
        memcpy(&sh, s_ptr + (size_t)grp * 2U, sizeof(sh));
        memcpy(&bh, b_ptr + (size_t)grp * 2U, sizeof(bh));
        const float sc = sqf_f16_to_f32(sh);
        const float bi = sqf_f16_to_f32(bh);
        for (uint32_t j = 0U; j < 8U; ++j) {
            const float nib = (float)((packed >> (4U * j)) & 0xFU);
            out_f16[dim_base + j] = sqf_f32_to_f16(nib * sc + bi);
        }
    }
}

/*
 * Looks up head `h` (`0..15`) at `global_row` via head-partitioned 4-way LFU cache + `pread` fallback.
 * Thread-safe across distinct `h` (`0..15`) with ZERO locks because each head owns a disjoint bank!
 */
static inline int strata_qwen_flash_ple_lookup_head_f32(
    int                             ple_fd,
    const StrataQwenFlashPleHeader* ple_hdr,
    StrataQwenFlashPleCache*        cache,
    uint32_t                        head_idx,
    uint64_t                        global_row,
    float*                          out_f32,      /* `[head_dim]` (160) */
    uint64_t*                       hit_delta_out,
    uint64_t*                       miss_delta_out
) {
    const uint32_t head_dim = ple_hdr->head_dim_per_ngram;
    if (cache && cache->arena_base && head_idx < cache->ngram_heads && head_dim == cache->head_dim) {
        const uint32_t set_idx   = (uint32_t)((global_row * 0x9E3779B97F4A7C15ULL) >> 32) & cache->set_mask;
        const size_t   base_slot = ((size_t)head_idx * (size_t)cache->sets_per_head + (size_t)set_idx) *
                                   (size_t)STRATA_QWEN_FLASH_PLE_LFU_WAYS;
        for (uint32_t w = 0U; w < STRATA_QWEN_FLASH_PLE_LFU_WAYS; ++w) {
            if (cache->tags[base_slot + w] == global_row) {
                if (cache->freqs[base_slot + w] < 0xFFFFFFFEU) ++cache->freqs[base_slot + w];
                const uint16_t* src = cache->values_f16 + (base_slot + w) * (size_t)head_dim;
                for (uint32_t d = 0U; d < head_dim; ++d) out_f32[d] = sqf_f16_to_f32(src[d]);
                if (hit_delta_out)  *hit_delta_out  = 1ULL;
                if (miss_delta_out) *miss_delta_out = 0ULL;
                return 0;
            }
        }

        /* Cache miss: `pread` single row from `F_NOCACHE` `ple_ngram.strata` */
        uint8_t raw_buf[512];
        const uint32_t rb = ple_hdr->row_bytes;
        if (rb > sizeof(raw_buf)) return -1;
        const off_t byte_off = (off_t)(ple_hdr->data_page_off + global_row * (uint64_t)rb);
        const ssize_t got = pread(ple_fd, raw_buf, rb, byte_off);
        if (got != (ssize_t)rb) return -1;

        /* Victim selection: empty slot first, else lowest LFU frequency (with halving decay) */
        uint32_t victim_w    = 0U;
        uint32_t victim_freq = cache->freqs[base_slot];
        for (uint32_t w = 0U; w < STRATA_QWEN_FLASH_PLE_LFU_WAYS; ++w) {
            if (cache->tags[base_slot + w] == UINT64_MAX) {
                victim_w    = w;
                victim_freq = 0U;
                break;
            }
            if (cache->freqs[base_slot + w] < victim_freq) {
                victim_freq = cache->freqs[base_slot + w];
                victim_w    = w;
            }
        }
        if (victim_freq > 16U) {
            for (uint32_t w = 0U; w < STRATA_QWEN_FLASH_PLE_LFU_WAYS; ++w) {
                cache->freqs[base_slot + w] >>= 1U;
            }
        }

        uint16_t* dst = cache->values_f16 + (base_slot + victim_w) * (size_t)head_dim;
        strata_qwen_flash_ple_dequant_row_f16(raw_buf, head_dim, ple_hdr->quant_group_size, dst);
        cache->tags[base_slot + victim_w]  = global_row;
        cache->freqs[base_slot + victim_w] = 1U;
        for (uint32_t d = 0U; d < head_dim; ++d) out_f32[d] = sqf_f16_to_f32(dst[d]);
        if (hit_delta_out)  *hit_delta_out  = 0ULL;
        if (miss_delta_out) *miss_delta_out = 1ULL;
        return 0;
    }

    /* Uncached fallback */
    uint8_t  raw_buf[512];
    uint16_t tmp_f16[512];
    const uint32_t rb = ple_hdr->row_bytes;
    if (rb > sizeof(raw_buf) || head_dim > 512U) return -1;
    const off_t byte_off = (off_t)(ple_hdr->data_page_off + global_row * (uint64_t)rb);
    if (pread(ple_fd, raw_buf, rb, byte_off) != (ssize_t)rb) return -1;
    strata_qwen_flash_ple_dequant_row_f16(raw_buf, head_dim, ple_hdr->quant_group_size, tmp_f16);
    for (uint32_t d = 0U; d < head_dim; ++d) out_f32[d] = sqf_f16_to_f32(tmp_f16[d]);
    if (hit_delta_out)  *hit_delta_out  = 0ULL;
    if (miss_delta_out) *miss_delta_out = 1ULL;
    return 0;
}

/* ============================================================================
 * 4. Host Bundle Handle, Pipelined Streaming HF Converter & BPE Tokenizer ABI
 *    (Implemented by `@flash-stream` in `src/model/strata_qwen_flash_convert.cpp`)
 * ============================================================================ */

typedef struct StrataQwenFlashTokenizer StrataQwenFlashTokenizer;

typedef struct StrataQwenFlashBundle {
    char                              bundle_dir[512];
    char                              part0_path[512];
    char                              part1_path[512];
    char                              ple_path[512];

    /* Two zero-copy `16 KiB` page-aligned read-only `mmap`s (`< 36 GiB` each << `80.64 GiB` Metal limit) */
    int                               part0_fd;
    const uint8_t*                    part0_base;
    size_t                            part0_bytes;
    const StrataQwenFlashPart0Header* header0;

    int                               part1_fd;
    const uint8_t*                    part1_base;
    size_t                            part1_bytes;
    const StrataQwenFlashPart1Header* header1;

    /* `F_NOCACHE` NVMe PLE N-Gram Store (`~27.4 GiB`, `0` GPU wired bytes!) + `mlock` LFU Cache */
    int                               ple_fd;
    uint64_t                          ple_file_bytes;
    StrataQwenFlashPleHeader          ple_header;
    StrataQwenFlashPleCache           ple_cache;

    StrataQwenFlashConfig             cfg;
    StrataQwenFlashTokenizer*         tokenizer;
} StrataQwenFlashBundle;

/* Inline convenience helpers to resolve offsets in `part0` / `part1` and lookup PLE sequences */
static inline const void* strata_qwen_flash_part0_ptr(const StrataQwenFlashBundle* b, uint64_t off) {
    return (const void*)(b->part0_base + off);
}

static inline const void* strata_qwen_flash_part1_ptr(const StrataQwenFlashBundle* b, uint64_t off) {
    return (const void*)(b->part1_base + off);
}

static inline const uint8_t* strata_qwen_flash_part_base(const StrataQwenFlashBundle* b, uint32_t part_id) {
    return (part_id == 0U) ? b->part0_base : b->part1_base;
}

static inline const StrataQwenFlashLayerDesc* strata_qwen_flash_get_layer_desc(
    const StrataQwenFlashBundle* b,
    uint32_t global_layer_idx
) {
    if (global_layer_idx < b->cfg.split_layer_idx) {
        return &b->header0->layers[global_layer_idx];
    }
    return &b->header1->layers[global_layer_idx - b->cfg.split_layer_idx];
}

static inline bool strata_qwen_flash_is_eos_token(const StrataQwenFlashConfig* cfg, uint32_t token_id) {
    for (uint32_t i = 0U; i < cfg->num_eos_token_ids; ++i) {
        if (token_id == cfg->eos_token_ids[i]) return true;
    }
    return false;
}

/*
 * Gathers `out_emb_f32 [num_tokens, ple_embed_dim]` across `token_ids[0 .. num_tokens - 1]`
 * starting from `prev_tokens_inout[2] = {prev2, prev1}` (initialized at `reset()` to `{248044, 248044}`),
 * updating `prev_tokens_inout[2]` in-place and updating `bundle->ple_cache` telemetry!
 */
static inline int strata_qwen_flash_ple_lookup_sequence_f32(
    StrataQwenFlashBundle* bundle,
    uint32_t               prev_tokens_inout[2], /* `{prev2, prev1}` */
    const uint32_t*        token_ids,
    uint32_t               num_tokens,
    float*                 out_emb_f32           /* `[num_tokens * cfg.ple_embed_dim]` */
) {
    if (!bundle || !bundle->cfg.has_ple || bundle->ple_fd < 0) return -1;
    const StrataQwenFlashPleHeader* ph       = &bundle->ple_header;
    const uint32_t                  n_heads  = ph->ngram_heads;
    const uint32_t                  head_dim = ph->head_dim_per_ngram;
    const uint32_t                  emb_dim  = bundle->cfg.ple_embed_dim;

    uint32_t prev2 = prev_tokens_inout[0];
    uint32_t prev1 = prev_tokens_inout[1];
    uint64_t total_hits   = 0ULL;
    uint64_t total_misses = 0ULL;

    for (uint32_t t = 0U; t < num_tokens; ++t) {
        const uint32_t tok0 = token_ids[t];
        uint64_t rows[STRATA_QWEN_FLASH_MAX_NGRAM_HEADS];
        strata_qwen_flash_ple_hash_token(ph, tok0, prev1, prev2, rows);
        float* dst_tok = out_emb_f32 + (size_t)t * (size_t)emb_dim;
        for (uint32_t h = 0U; h < n_heads; ++h) {
            uint64_t hd = 0ULL, md = 0ULL;
            if (strata_qwen_flash_ple_lookup_head_f32(
                    bundle->ple_fd, ph, &bundle->ple_cache, h, rows[h],
                    dst_tok + (size_t)h * (size_t)head_dim, &hd, &md) != 0) {
                return -1;
            }
            total_hits   += hd;
            total_misses += md;
        }
        prev2 = prev1;
        prev1 = tok0;
    }

    prev_tokens_inout[0] = prev2;
    prev_tokens_inout[1] = prev1;
    bundle->ple_cache.lfu_hits        += total_hits;
    bundle->ple_cache.lfu_misses      += total_misses;
    bundle->ple_cache.nvme_preads     += total_misses;
    bundle->ple_cache.nvme_bytes_read += total_misses * (uint64_t)ph->row_bytes;
    return 0;
}

typedef struct StrataQwenFlashConvertStats {
    uint32_t num_shards_total;
    uint32_t num_shards_converted;
    uint32_t num_shards_resumed;
    uint32_t num_text_tensors_converted;
    uint32_t num_ple_shards_converted;
    uint32_t num_mtp_tensors_converted;
    uint32_t num_vision_tensors_skipped;
    uint64_t hf_safetensors_bytes;
    uint64_t uma_part0_bytes;
    uint64_t uma_part1_bytes;
    uint64_t ple_ngram_bytes;
    uint64_t total_bundle_bytes;
    uint64_t active_bytes_per_tok;
    double   elapsed_sec;
    double   network_mbps;              /* MB/s across downloaded shards */
    double   peak_rss_gib;
    bool     page_aligned_verified;
    bool     staging_clean_verified;
} StrataQwenFlashConvertStats;

/*
 * Pipelined Multi-Shard Streaming Downloader + ARM NEON Converter + Immediate Shard Deleter:
 *   - `hf_repo_or_dir`: Hugging Face repo ID (`"Qwen/Qwen3.8-Flash-Next"`), base URL, OR local directory.
 *     If NULL or empty, defaults to `"Qwen/Qwen3.8-Flash-Next"`.
 *   - `out_bundle_dir`: Destination directory (defaults to `STRATA_QWEN_FLASH_DEFAULT_BUNDLE_DIR` if NULL).
 *   - `download_concurrency`: Number of concurrent `curl` workers (`0` => default `3`, max `4` staged shards on disk).
 *   - `convert_threads`: Number of parallel NEON quantization threads (`0` => default hardware core count).
 *   - Crash-safe resumable via `<out_bundle_dir>/.convert_done_shards` and deletes `<out_bundle_dir>/.staging` on completion.
 */
int strata_qwen_flash_convert_hf(
    const char* hf_repo_or_dir,
    const char* out_bundle_dir,
    uint32_t    download_concurrency,
    uint32_t    convert_threads,
    StrataQwenFlashConvertStats* stats_out
);

/* Convenience wrapper calling `strata_qwen_flash_convert_hf(hf_repo_or_dir, out_bundle_dir, 0, 0, stats_out)`. */
int strata_qwen_flash_convert_hf_directory(
    const char* hf_repo_or_dir,
    const char* out_bundle_dir,
    StrataQwenFlashConvertStats* stats_out
);

/*
 * Local Offline Phase-C Dense-Weight Repacker (zero HuggingFace download, 100% V1-compatible!):
 *   Reads `src_bundle_dir/uma_part{0,1}.strata`, quantizes HC mixers (`down/up`) + PLE `key/value` +
 *   MTP `fc_embedding/hidden` (`FP16 -> Q8_0`, gs=64) and Shared Experts (`gate/up/down`, `FP16 -> Q4_A`, gs=64),
 *   copies existing Q4/F32/F16 tensors verbatim, hardlinks `ple_ngram.strata`, and writes 16 KiB-aligned
 *   `dst_bundle_dir/uma_part{0,1}.strata` (-0.987 GB/step active weight reads!).
 */
int strata_qwen_flash_repack_bundle(
    const char* src_bundle_dir,
    const char* dst_bundle_dir,
    uint32_t    convert_threads,
    StrataQwenFlashConvertStats* stats_out
);

/*
 * Maps `uma_part0.strata` + `uma_part1.strata` read-only (`16 KiB` aligned, zero copy), opens
 * `ple_ngram.strata` with `O_RDONLY` + `fcntl(F_NOCACHE, 1)`, initializes the `mlock` PLE LFU cache
 * (`ple_cache_bytes == 0` => default `96 MiB`), and loads `<bundle_dir>/tokenizer.json`.
 */
int strata_qwen_flash_load_bundle(
    const char* bundle_dir,
    size_t      ple_cache_bytes,
    StrataQwenFlashBundle* bundle_out
);

void strata_qwen_flash_free_bundle(StrataQwenFlashBundle* bundle);

/* ---- 4.1 Byte-Level BPE Tokenizer (`248,320` vocab) & Qwen4-Exp ChatML Formatter ---- */

StrataQwenFlashTokenizer* strata_qwen_flash_tokenizer_load(const char* tokenizer_json_path);
void strata_qwen_flash_tokenizer_free(StrataQwenFlashTokenizer* tok);
uint32_t strata_qwen_flash_tokenizer_vocab_size(const StrataQwenFlashTokenizer* tok);

int strata_qwen_flash_encode(
    const StrataQwenFlashTokenizer* tok,
    const char* text,
    size_t      text_len,
    bool        parse_special,
    uint32_t*   out_ids,
    uint32_t    max_ids,
    uint32_t*   out_num_ids
);

/*
 * Formats `user_prompt` (and optional `system_prompt`, NULL for none) per `Qwen3.8-Flash-Next`'s
 * official `chat_template.jinja`:
 *   - `enable_thinking == false`: `...<|im_start|>assistant\n<think>\n\n</think>\n\n`
 *   - `enable_thinking == true`:  `...<|im_start|>assistant\n<think>\n` (with optional `reasoning_effort`
 *     in `{"low", "medium", "xhigh"}`; NULL defaults to `"low"` or `"medium"` concise reasoning).
 */
int strata_qwen_flash_format_chat_prompt(
    const char* user_prompt,
    const char* system_prompt,
    bool        enable_thinking,
    const char* reasoning_effort,
    char*       out_buf,
    size_t      out_buf_cap,
    size_t*     out_len
);

int strata_qwen_flash_decode(
    const StrataQwenFlashTokenizer* tok,
    const uint32_t* ids,
    uint32_t        num_ids,
    bool            skip_special,
    char*           out_utf8,
    size_t          out_cap,
    size_t*         out_len
);

typedef struct StrataQwenFlashStreamDecoder {
    uint8_t pending_bytes[8];
    uint8_t pending_len;
} StrataQwenFlashStreamDecoder;

static inline void strata_qwen_flash_stream_decoder_init(StrataQwenFlashStreamDecoder* dec) {
    for (int i = 0; i < 8; ++i) dec->pending_bytes[i] = 0;
    dec->pending_len = 0;
}

int strata_qwen_flash_stream_decode_step(
    const StrataQwenFlashTokenizer* tok,
    StrataQwenFlashStreamDecoder*   dec,
    uint32_t                        token_id,
    bool                            skip_special,
    char*                           out_utf8,
    size_t                          out_cap,
    size_t*                         out_len
);

/* ============================================================================
 * 5. Metal 4 (`NAX` + Two-Buffer Zero-Copy UMA + NVMe PLE + Native `MTP`) Engine ABI
 *    (Implemented by `@flash-gpu` in `src/gpu/strata_qwen_flash_gpu.mm` & `strata_qwen_flash_kernels.metal`)
 * ============================================================================ */

typedef struct StrataQwenFlashGpuEngine StrataQwenFlashGpuEngine;

typedef struct StrataQwenFlashSamplingParams {
    bool     greedy;             /* true => deterministic argmax (`temperature` ignored) */
    float    temperature;        /* e.g. 0.6f when `!greedy` */
    uint32_t top_k;              /* e.g. 20 (0 => disabled / default) */
    float    top_p;              /* e.g. 0.95f (<= 0 or >= 1 => disabled) */
    uint64_t rng_seed;           /* PRNG seed/state for Leviathan stochastic verification */
    float    presence_penalty;   /* OpenAI presence penalty [-2..2]; 0 => disabled */
    float    frequency_penalty;  /* OpenAI frequency penalty [-2..2]; 0 => disabled */
    float    repetition_penalty; /* HF multiplicative penalty (> 0; 0 or 1.0 => disabled) */
    uint32_t penalty_window;     /* Sliding window in completion tokens: 0 => full completion; >0 => last max(N, 8) tokens */
} StrataQwenFlashSamplingParams;

typedef struct StrataQwenFlashGpuInfo {
    char     device_name[128];
    bool     zero_copy_mmap;
    double   load_time_ms;
    uint32_t max_seq_len;
    uint64_t uma_part0_mapped_bytes;      /* ~33.65 GiB (< 80.64 GiB MTLBuffer limit) */
    uint64_t uma_part1_mapped_bytes;      /* ~35.07 GiB (< 80.64 GiB MTLBuffer limit) */
    uint64_t total_uma_mapped_bytes;      /* ~68.72 GiB (< 107.52 GiB recommended working set) */
    uint64_t ple_nvme_file_bytes;         /* ~27.42 GiB on NVMe (`0` GPU wired bytes!) */
    uint64_t ple_lfu_cache_bytes;         /* `mlock`ed CPU LFU cache size (~96 MiB) */
    uint64_t active_bytes_per_token;      /* ~5.17 GB physical DRAM read per K=0 token */
    uint64_t mtp_active_bytes_per_draft;  /* ~0.51 GB physical DRAM read per MTP draft step */
    uint64_t deltanet_state_bytes;        /* 36-layer FP16 conv + FP32 [48,128,128] state (~115 MiB) */
    uint64_t ple_state_bytes;             /* Dilated k=4, d=3 (len=9) conv state (~360 KiB) */
    uint64_t kv_cache_bytes_allocated;    /* (12 + 1 MTP) 2KV x 256 FP16 + QSA indexer cache */
    uint64_t kv_cache_bytes_per_token;    /* Main 12L: `12 * 2 * 2 * 256 * 2B = 24,576 B` + indexer */
    uint64_t workspace_bytes;             /* Pre-allocated scratch + rollback snapshot arenas */
    uint32_t int8_prefill_enabled;        /* 1 iff opt-in INT8 NAX prefill GEMMs enabled */
    uint32_t adaptive_k_enabled;          /* 1 iff 3-state SpecCtl Adaptive-K enabled */
    uint32_t frspec_enabled;              /* 1 iff 32K SLC-pinned FR-Spec draft shortlist active */
    uint32_t frspec_shortlist_size;       /* Active FR-Spec shortlist vocab size (e.g. 32768) */
    uint32_t metal_cache_hit;             /* 1 iff MTLBinaryArchive warm cache hit at startup */
    double   pso_load_ms;                 /* Wall-clock ms spent loading/compiling Metal PSOs */
} StrataQwenFlashGpuInfo;

typedef struct StrataQwenFlashPerfCounters {
    uint64_t decode_steps;
    uint64_t spec_verify_steps;
    uint64_t mtp_drafted_tokens;
    uint64_t mtp_accepted_tokens;
    uint64_t ple_nvme_preads;
    uint64_t ple_nvme_bytes_read;
    uint64_t ple_lfu_hits;
    uint64_t ple_lfu_misses;
    uint64_t physical_dram_bytes_read;    /* Cumulative physical DRAM bytes read across steps */
} StrataQwenFlashPerfCounters;

/*
 * Maps `bundle->part0_base` and `bundle->part1_base` via TWO zero-copy `newBufferWithBytesNoCopy:`
 * `MTLBuffer`s in an `MTLResidencySet`, compiles `strata_qwen_flash_kernels.metal` (`MTLLanguageVersion4_0`),
 * and pre-allocates 36-layer DeltaNet state, 12-layer + 1-MTP KV & QSA Indexer caches up to `max_seq_len`,
 * PLE state, and speculative rollback arenas. Note: `bundle` is non-const inside engine only to update
 * `bundle->ple_cache` on PLE lookups (`part0_base` & `part1_base` remain strictly read-only).
 */
StrataQwenFlashGpuEngine* strata_qwen_flash_gpu_create(
    StrataQwenFlashBundle* bundle,
    uint32_t               max_seq_len
);

void strata_qwen_flash_gpu_destroy(StrataQwenFlashGpuEngine* eng);

/* Resets `seq_pos = 0`, DeltaNet, KV, QSA Indexer, PLE `{248044, 248044}`, and MTP state without reallocating. */
void strata_qwen_flash_gpu_reset(StrataQwenFlashGpuEngine* eng);

uint32_t strata_qwen_flash_gpu_seq_pos(const StrataQwenFlashGpuEngine* eng);

void strata_qwen_flash_gpu_get_info(
    const StrataQwenFlashGpuEngine* eng,
    StrataQwenFlashGpuInfo*         info_out
);

void strata_qwen_flash_gpu_get_counters(
    const StrataQwenFlashGpuEngine* eng,
    StrataQwenFlashPerfCounters*    counters_out
);

void strata_qwen_flash_gpu_reset_counters(StrataQwenFlashGpuEngine* eng);

/*
 * Reset or append completion-scoped token frequency counters used when
 * `presence_penalty != 0`, `frequency_penalty != 0`, or `repetition_penalty != 1.0`.
 * Automatically reset on `prefill`/`reset`/`checkpoint_restore` and updated on emitted tokens;
 * explicit calls allow multi-turn servers to reset/seed counters across cached turns.
 */
void strata_qwen_flash_gpu_penalties_reset(StrataQwenFlashGpuEngine* eng);
void strata_qwen_flash_gpu_penalties_record(
    StrataQwenFlashGpuEngine* eng,
    const uint32_t*           tokens,
    uint32_t                  num_tokens
);

typedef void (*StrataQwenFlashPrefillProgressFn)(
    uint32_t done_tokens,
    uint32_t total_tokens,
    uint32_t seq_pos,
    void*    user_ctx
);

void strata_qwen_flash_gpu_set_prefill_progress_cb(
    StrataQwenFlashGpuEngine*        eng,
    StrataQwenFlashPrefillProgressFn cb,
    void*                            user_ctx
);

/*
 * Batched Prefill over `token_ids[0 .. num_tokens - 1]` starting at current `seq_pos`,
 * advancing `seq_pos` by `num_tokens` (and synchronizing/priming MTP state) and writing the
 * sampled/argmaxed next token into `*next_token_out`. `last_logits_out` may be NULL.
 */
int strata_qwen_flash_gpu_prefill(
    StrataQwenFlashGpuEngine*            eng,
    const uint32_t*                      token_ids,
    uint32_t                             num_tokens,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t*                            next_token_out,
    float*                               last_logits_out
);

/*
 * Standard `K = 0` Single-Token Decode (`M = 1`): feeds `token_id` at `seq_pos`, advances `seq_pos` by 1,
 * and writes `*next_token_out`. `logits_out` may be NULL for fast GPU argmax/top-k readback.
 */
int strata_qwen_flash_gpu_decode_step(
    StrataQwenFlashGpuEngine*            eng,
    uint32_t                             token_id,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t*                            next_token_out,
    float*                               logits_out
);

/*
 * Native `MTP` Speculative Decode Step (`draft_k = 1 .. 3`, or `0` -> delegates to `decode_step`):
 *   1. Given `pending_token_id` (the uncommitted token emitted by `prefill` or the previous decode/spec step),
 *      runs `mtp` `draft_k` times to draft `draft_tokens[0 .. draft_k - 1]`.
 *   2. Verifies `[pending_token_id, draft_tokens[0 .. draft_k - 1]]` (`M = 1 + draft_k`) in ONE batched
 *      48-layer Main Model pass with exact Leviathan greedy/stochastic verification.
 *   3. Suppose `a` (`0 <= a <= draft_k`) draft tokens are accepted:
 *      - Rolls back any rejected tail states (`seq_pos` advances by `1 + a`).
 *      - Writes `*num_emitted_out = 1 + a` committed tokens into `emitted_tokens_out[0 .. a]`, where
 *        `emitted_tokens_out[0 .. a - 1]` are the `a` accepted draft tokens (already in Main Model cache)
 *        and `emitted_tokens_out[a]` is the bonus/resampled token (pass `emitted_tokens_out[a]` as the next
 *        step's `pending_token_id`!). Sets `*num_draft_accepted_out = a` if non-NULL.
 *   Guarantees 100% bit-identical greedy token IDs between `draft_k == 0` and `draft_k >= 1`!
 */
int strata_qwen_flash_gpu_spec_decode_step(
    StrataQwenFlashGpuEngine*            eng,
    uint32_t                             pending_token_id,
    uint32_t                             draft_k,
    const StrataQwenFlashSamplingParams* sampling,
    uint32_t*                            emitted_tokens_out,    /* capacity >= `1 + draft_k` */
    uint32_t*                            num_emitted_out,       /* in `[1 .. 1 + draft_k]`   */
    uint32_t*                            num_draft_accepted_out /* optional, in `[0 .. draft_k]` */
);

/* Toggle Width-2 x Depth-3 Speculative Tree Verification (`M = 7`, exact greedy & lossless SpecInfer/Leviathan). */
void strata_qwen_flash_gpu_set_spec_tree(StrataQwenFlashGpuEngine* eng, bool enable);
bool strata_qwen_flash_gpu_get_spec_tree(const StrataQwenFlashGpuEngine* eng);

/* Toggle async MTP True-State KV Sync replay in `cb_fix` (keeps MTP KV synced to true main-model hidden states). */
void strata_qwen_flash_gpu_set_mtp_kv_sync(StrataQwenFlashGpuEngine* eng, bool enable);
bool strata_qwen_flash_gpu_get_mtp_kv_sync(const StrataQwenFlashGpuEngine* eng);

/* Configure Zero-Copy Shortlisted-Vocab MTP Draft Head (`0` = full vocab; default `32768` + special tail). */
void     strata_qwen_flash_gpu_set_mtp_shortlist(StrataQwenFlashGpuEngine* eng, uint32_t v_short);
uint32_t strata_qwen_flash_gpu_get_mtp_shortlist(const StrataQwenFlashGpuEngine* eng);

/* Toggle Zero-Cost Rolling Prompt/History N-Gram Suffix Drafting (100% exact greedy & Leviathan stochastic). */
void strata_qwen_flash_gpu_set_ngram_draft(StrataQwenFlashGpuEngine* eng, bool enable);
bool strata_qwen_flash_gpu_get_ngram_draft(const StrataQwenFlashGpuEngine* eng);

/* Toggle opt-in INT8 NAX Prefill GEMMs (prefill M >= 32 only; decode M <= 8 remains 100% bit-exact FP16/FP32). */
void strata_qwen_flash_gpu_set_int8_prefill(StrataQwenFlashGpuEngine* eng, bool enable);

/* Toggle 3-State SpecCtl Adaptive-K (CooldownK0 + HighConfFused + ActiveProbe; 100% bit-exact K>0 == K=0). */
void strata_qwen_flash_gpu_set_adaptive_k(StrataQwenFlashGpuEngine* eng, bool enable);

/* Toggle 100% Lossless 32K SLC-Pinned Draft-Head Shortlist (`FR-Spec`, drafting only; verify always full 248K). */
void strata_qwen_flash_gpu_set_frspec(StrataQwenFlashGpuEngine* eng, bool enable);

/*
 * O(1) Turn-Boundary GPU Checkpoint / Restore (~110.5 MiB on-demand GPU->GPU blit, < 0.5 ms):
 *   - Allocated lazily ONLY when `strata_qwen_flash_checkpoint_create` is called (`0` bytes in `bench-flash`).
 *   - Snapshots `seq_pos`, `mtp_seq_pos`, `ple_prev_tokens[2]`, `sample_step`, `b_conv_state`,
 *     `b_rec_state`, `b_ple_conv_state`, and `b_mtp_hc_prev` in a single `MTLBlitCommandEncoder`.
 */
typedef struct StrataQwenFlashCheckpoint StrataQwenFlashCheckpoint;

StrataQwenFlashCheckpoint* strata_qwen_flash_checkpoint_create(const StrataQwenFlashGpuEngine* eng);
void strata_qwen_flash_checkpoint_destroy(StrataQwenFlashCheckpoint* ckpt);
int strata_qwen_flash_checkpoint_save(StrataQwenFlashGpuEngine* eng, StrataQwenFlashCheckpoint* ckpt);
int strata_qwen_flash_checkpoint_restore(StrataQwenFlashGpuEngine* eng, const StrataQwenFlashCheckpoint* ckpt);
uint64_t strata_qwen_flash_checkpoint_bytes(const StrataQwenFlashCheckpoint* ckpt);

/* ============================================================================
 * 6. Step 4 — Two-Tier Zero-Pause Realtime Compaction ABI
 *    Tier 2: Global-`pos`-Invariant Paged `QSA` KV (`b_block_loc`) + `F_NOCACHE` NVMe `kv_swap` + 16-Core ANE
 *            query-lookahead prefetch + exact selected-block recall (100% bit-exact lossless).
 *    Tier 1: Dual `StrataQwenFlashContextSlot` (`state_live` / `state_shadow` / rollback) sharing the zero-copy
 *            weight buffers (`0 B` duplication) with an O(1) pointer/binding swap + preemptible prefill slices.
 * ============================================================================ */

/* Bit 31 of `block_loc[layer][b]`: 1 => UMA-resident at physical pool block `loc & 0x7FFFFFFF`,
 * 0 => paged out to `kv_swap` (NVMe slot `loc`) or unallocated. Logical block `b = pos >> 2`. */
#define STRATA_QWEN_FLASH_KV_LOC_UMA_BIT          0x80000000U
#define STRATA_QWEN_FLASH_KV_LOC_IDX_MASK         0x7FFFFFFFU
#define STRATA_QWEN_FLASH_KV_BLOCK_TOKENS         4U       /* == indexer_compress_ratio */
#define STRATA_QWEN_FLASH_KV_BLOCK_BYTES          8192U    /* 4 tok x 2 KV x 256 x 2 B (K) + same (V), per layer */
#define STRATA_QWEN_FLASH_COMPACT_SINK_BLOCKS     32U      /* 128 pinned attention-sink tokens */
#define STRATA_QWEN_FLASH_COMPACT_LOCAL_BLOCKS    256U     /* 1,024 pinned recent tokens */
#define STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS 2560U    /* >= top-512 blocks + sinks + local + chunk */
/* Largest `cfg.indexer_budget` the engine accepts. Attention stays dense (every block read through `block_loc`
 * without a residency sync) while the context is within the budget, so a paged pool must hold the whole dense
 * range: `COMPACT_MIN_POOL_TOKENS >= MAX_INDEXER_BUDGET + 512` (enforced by a static_assert in the engine). */
#define STRATA_QWEN_FLASH_MAX_INDEXER_BUDGET      2048U

typedef enum StrataCompactionMode {
    STRATA_COMPACT_OFF    = 0, /* Uncompacted linear KV up to max_seq_len (zero overhead) */
    STRATA_COMPACT_SHADOW = 1, /* Tier 1: Predictive Dual-Buffer Shadow RCU (O(1) swap) */
    STRATA_COMPACT_PAGED  = 2, /* Tier 2: Global-pos Paged QSA + F_NOCACHE NVMe swap (100% lossless) */
    STRATA_COMPACT_HYBRID = 3  /* Tier 1 + Tier 2 (each slot owns a paged UMA pool + kv_swap) */
} StrataCompactionMode;

typedef struct StrataQwenFlashCompactionStats {
    /* ---- Tier 2 (filled by `strata_qwen_flash_slot_get_stats`) ---- */
    uint32_t mode;                     /* StrataCompactionMode of the slot (PAGED if paged, else OFF) */
    uint32_t logical_seq_pos;          /* Monotonic global RoPE position of the slot */
    uint32_t uma_kv_blocks_capacity;   /* Physical UMA pool blocks per layer (e.g. 1,024 = 4,096 tokens) */
    uint32_t uma_kv_blocks_used;       /* Max resident blocks across the 13 QSA layers */
    uint32_t logical_blocks;           /* ceil(seq_pos / 4) logical blocks per layer */
    uint32_t nvme_blocks_now_paged;    /* Blocks currently living only on NVMe (sum over layers) */
    uint64_t nvme_blocks_paged_out;    /* Cumulative 8 KiB F_NOCACHE pwrite block evictions (layer-blocks) */
    uint64_t nvme_blocks_recalled;     /* Cumulative F_NOCACHE pread recalls back into UMA (layer-blocks) */
    uint64_t nvme_bytes_written;
    uint64_t nvme_bytes_read;
    double   nvme_write_ms;
    double   nvme_read_ms;
    uint64_t paged_sync_points;        /* Exact selected-block checks (one per sparse QSA layer step) */
    uint64_t sync_recall_misses;       /* Blocks the exact check had to recall synchronously */
    uint64_t ane_probe_calls;          /* 16-core ANE query-lookahead predictions issued */
    uint64_t ane_probe_on_ane;         /* ... of which executed on MLComputePlan-verified ANE */
    uint32_t ran_on_ane;               /* 1 iff >= 1 probe ran on the verified 16-core ANE */
    uint32_t reserved0;
    uint64_t ane_prefetch_issued;      /* Paged blocks pread ahead of the GPU by the ANE lookahead */
    uint64_t ane_prefetch_hits;        /* ... that the exact top-512 selection then actually needed */
    double   ane_probe_us_avg;
    char     nvme_swap_path[256];
    /* ---- Tier 1 (filled by the REPL compaction controller) ---- */
    uint32_t episodes_distilled;
    uint32_t shadow_seq_pos;
    bool     shadow_dual_synced;
    uint32_t rcu_swaps_completed;
    double   last_switchover_ms;
    double   max_user_stall_ms;
} StrataQwenFlashCompactionStats;

typedef struct StrataQwenFlashContextSlot StrataQwenFlashContextSlot;

/*
 * Creates a mutable context slot (DeltaNet conv/rec + PLE conv + MTP hc + QSA KV + indexer raw_k/block_k +
 * seq_pos/mtp_seq_pos/ple ring/sample_step) sharing the engine's zero-copy weights and pipelines.
 *   - `uma_kv_pool_tokens == 0`: linear unpaged KV for the full engine `max_seq_len` (Tier 1 shadow slot).
 *   - `uma_kv_pool_tokens >= STRATA_QWEN_FLASH_COMPACT_MIN_POOL_TOKENS`: Tier 2 paged QSA KV pool of
 *     `uma_kv_pool_tokens / 4` blocks per layer, `b_block_loc[13][max_blocks]`, and an `F_NOCACHE`
 *     `kv_swap` file (`nvme_swap_path`, or an auto path in `NSTemporaryDirectory()` when NULL) which is
 *     unlinked immediately after open (0 leftover bytes even on crash) and closed on destroy.
 *     Also lazily builds the engine's CoreML MLComputePlan-verified 16-core ANE query-lookahead probe.
 * The new slot starts reset (`seq_pos = 0`). Returns NULL on failure.
 */
StrataQwenFlashContextSlot* strata_qwen_flash_slot_create(
    StrataQwenFlashGpuEngine* eng,
    uint32_t                  uma_kv_pool_tokens,
    const char*               nvme_swap_path
);

/* Destroys `slot` (if bound, the engine first rebinds its built-in default slot). */
void strata_qwen_flash_slot_destroy(StrataQwenFlashContextSlot* slot);

/* O(1) (< 0.05 ms) RCU context switch: binds `slot` (NULL => engine built-in default slot). */
void strata_qwen_flash_bind_slot(StrataQwenFlashGpuEngine* eng, StrataQwenFlashContextSlot* slot);

/* Returns the currently bound slot (NULL => built-in default slot). */
StrataQwenFlashContextSlot* strata_qwen_flash_bound_slot(const StrataQwenFlashGpuEngine* eng);

uint32_t strata_qwen_flash_slot_seq_pos(const StrataQwenFlashContextSlot* slot);
bool     strata_qwen_flash_slot_is_paged(const StrataQwenFlashContextSlot* slot);
uint64_t strata_qwen_flash_slot_bytes(const StrataQwenFlashContextSlot* slot); /* GPU bytes owned by the slot */

/* Fills the Tier 2 fields of `stats_out` (Tier 1 fields zeroed). `slot == NULL` => zeros. */
void strata_qwen_flash_slot_get_stats(const StrataQwenFlashContextSlot* slot, StrataQwenFlashCompactionStats* stats_out);
void strata_qwen_flash_slot_reset_stats(StrataQwenFlashContextSlot* slot);

/* 1 iff the engine's 16-core ANE lookahead probe passed MLComputePlan verification + warmup on ANE. */
int strata_qwen_flash_ane_probe_ready(const StrataQwenFlashGpuEngine* eng, char* note_out, size_t note_cap);

/*
 * Preemptible single-slice prefill on the BOUND slot (`slice_tokens` in 1..256; idle-gap quanta use 32..64):
 * identical semantics to `strata_qwen_flash_gpu_prefill` (advances seq_pos, keeps MTP in sync); when
 * `compute_final_logits == false` the final mixer + lm_head + sampler are skipped (`*next_token_out` untouched).
 */
int strata_qwen_flash_gpu_prefill_slice(
    StrataQwenFlashGpuEngine* eng,
    const uint32_t*           token_ids,
    uint32_t                  slice_tokens,
    bool                      compute_final_logits,
    uint32_t*                 next_token_out
);

/*
 * Canonical mode-independent GPU/NVMe sequence state serialization on the BOUND slot:
 * saves/loads `seq_pos`, `mtp_seq_pos`, `ple_prev_tokens[2]`, `sample_step`, 36L GatedDeltaNet + PLE + MTP state,
 * and logical `0..seq_pos` 13L QSA KV + indexer caches (transparently spanning UMA + `kv_swap` when paged!).
 */
int strata_qwen_flash_state_save_fd(StrataQwenFlashGpuEngine* eng, int fd, uint64_t* bytes_written_out);
int strata_qwen_flash_state_load_fd(StrataQwenFlashGpuEngine* eng, int fd, uint64_t* bytes_read_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !__METAL_VERSION__ */

#endif /* STRATA_QWEN_FLASH_H */

#ifndef STRATA_QWEN38_H
#define STRATA_QWEN38_H

/*
 * STRATA-M5 v2.0 — Qwen3.8-27B (Qwen3_5ForConditionalGeneration / qwen3_5_text)
 * Zero-Copy 16 KiB Page-Aligned Bundle ABI, Streaming HF Converter, BPE Tokenizer & Metal 4 GPU Engine
 *
 * Target Hardware: Mac17,10 (Apple M5 Max: 40c Metal 4 NAX GPU, ~508 GB/s LPDDR5X, 128 GiB UMA).
 *
 * Verified Architecture & MLX 4-Bit Checkpoint Equations (`mlx_lm/models/qwen3_5.py`, `gated_delta.py`):
 *   - 64 layers (`hidden_size = 5120`, `intermediate_size = 17408`, `vocab_size = 248320`, `eps = 1e-6`):
 *       * 48 `linear_attention` layers (`l % 4 != 3`): Gated DeltaNet (`16` K-heads, `48` V-heads, `head_dim = 128`)
 *       * 16 `full_attention` layers (`l % 4 == 3`, i.e. `3, 7, ..., 63`): Gated GQA (`24` Q-heads, `4` KV-heads, `head_dim = 256`)
 *   - MLX 4-bit affine quantization (`group_size = 64`, `bits = 4`):
 *       * `weight`: row-major `uint32_t [rows_padded, cols / 8]` (`nib_j = (u32 >> (4 * j)) & 0xF` for `j = 0..7`)
 *       * `scales`: row-major IEEE-754 FP16 `uint16_t [rows_padded, cols / 64]`
 *       * `biases`: row-major IEEE-754 FP16 `uint16_t [rows_padded, cols / 64]`
 *       * Dequantization: `w[r, c] = nib * scale[r, c >> 6] + bias[r, c >> 6]`
 *       * Every matrix in `weights.strata` has `rows_padded % 64 == 0` and `cols % 1024 == 0`
 *         (`in_proj_a` and `in_proj_b` logical `48` rows are zero-padded to `64` rows).
 *   - RMSNorm (`input_layernorm`, `post_attention_layernorm`, `model.norm`, `linear_attn.norm`, `q_norm`, `k_norm`):
 *       * MLX `sanitize()` already added `+1.0` to HF `(1 + w)` norms in `.safetensors` (`mean ~ 1.0`).
 *       * ALL 6 norms use standard `y = (x * rsqrt(mean(x^2) + eps)) * weight` directly (NO `+1.0` at runtime).
 *   - Gated DeltaNet (`linear_attn`, `l % 4 != 3`):
 *       1. `qkv = in_proj_qkv(h)` -> `[10240]`; `z = in_proj_z(h)` -> `[6144]` (`[48, 128]`);
 *          `a = in_proj_a(h)[0..47]` -> `[48]`; `b = in_proj_b(h)[0..47]` -> `[48]`.
 *       2. Causal depthwise `conv1d` (`k = 4`, zero left-pad at `t < 3`) + `SiLU`:
 *          `conv_raw[c] = w[c,0]*x_{t-3}[c] + w[c,1]*x_{t-2}[c] + w[c,2]*x_{t-1}[c] + w[c,3]*x_t[c]`,
 *          `conv_out[c] = silu(conv_raw[c])`.
 *       3. Contiguous split of `conv_out [10240]` -> `q [2048] = [16, 128]`, `k [2048] = [16, 128]`, `v [6144] = [48, 128]`.
 *       4. Per-head L2 norm (`normalize_qk`, `eps = 1e-6`) + `q` scale (`1 / sqrt(128)`):
 *          `q_hat[hk] = (q[hk] * rsqrt(sum(q[hk]^2) + 1e-6f)) * (1.0f / sqrt(128.0f))`,
 *          `k_hat[hk] =  k[hk] * rsqrt(sum(k[hk]^2) + 1e-6f)`.
 *          GQA key/query head broadcast (`48 / 16 = 3`): value head `hv` (`0..47`) uses `hk = hv / 3` (`0..15`).
 *       5. Gates (`FP32`):
 *          `g[hv] = -exp(A_log[hv]) * log(1.0f + exp(a[hv] + dt_bias[hv]))` (use stable `softplus`),
 *          `decay[hv] = exp(g[hv])`, `beta[hv] = 1.0f / (1.0f + exp(-b[hv]))`.
 *       6. Recurrent state update per value head `hv` (`S[hv]` in `FP32 [128_k, 128_v]` or `[128_v, 128_k]`):
 *          `S <- decay * S`
 *          `kv_mem = S^T @ k_hat` (`[128_v]`: `kv_mem[dv] = sum_{dk} S[dk, dv] * k_hat[dk]`)
 *          `delta  = beta * (v - kv_mem)` (`[128_v]`)
 *          `S <- S + k_hat (outer) delta` (`S[dk, dv] += k_hat[dk] * delta[dv]`)
 *          `out[hv] = S^T @ q_hat` (`[128_v]`: `out[dv] = sum_{dk} S[dk, dv] * q_hat[dk]`).
 *       7. `Qwen3NextRMSNormGated` (`output_gate_type = "swish"`):
 *          Per head `hv` (`128`-dim): `y[hv] = rms_norm_128(out[hv], linear_attn.norm.weight, 1e-6f) * silu(z[hv])`.
 *       8. Flatten `y [48, 128] -> [6144]`, return `out_proj(y) -> [5120]`.
 *   - Gated Full Attention (`self_attn`, `l % 4 == 3`):
 *       1. `q_gate = q_proj(h)` -> `[12288]`, reshaped **per-head** as `[24, 512]`:
 *          for head `hq` (`0..23`): `q[hq] = q_gate[hq*512 + 0 .. hq*512 + 255]` (`[256]`),
 *                                   `gate[hq] = q_gate[hq*512 + 256 .. hq*512 + 511]` (`[256]`).
 *       2. `k = k_proj(h)` -> `[4, 256]`, `v = v_proj(h)` -> `[4, 256]`.
 *       3. Per-head `RMSNorm(256, eps=1e-6)`: `q[hq] = rms_norm(q[hq], q_norm.weight)`, `k[hkv] = rms_norm(k[hkv], k_norm.weight)`.
 *       4. Partial NeoX/half-split RoPE (`traditional = false`) on first `rotary_dim = 64` of `256` dims (`theta = 10,000,000.0`):
 *          for `i = 0..31`: `freq = pos * pow(10000000.0, -2.0 * i / 64.0)`,
 *          `(x[i], x[i+32]) <- (x[i]*cos - x[i+32]*sin, x[i+32]*cos + x[i]*sin)`; dims `64..255` unchanged.
 *       5. Causal GQA (`24` Q-heads / `4` KV-heads = `6` Q-heads per KV-head, `scale = 1 / sqrt(256) = 0.0625f`):
 *          `attn_out [24, 256] = Softmax(Q K^T * 0.0625f) V`.
 *       6. **Sigmoid** attention output gate (`attn_output_gate = true`):
 *          `gated_attn = flatten(attn_out) * sigmoid(flatten(gate))` (`[6144]`) — NOTE: `sigmoid`, NOT `silu`!
 *       7. Return `o_proj(gated_attn) -> [5120]`.
 *   - SwiGLU MLP (all 64 layers):
 *       `mlp_out = down_proj(silu(gate_proj(h)) * up_proj(h))`.
 */

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef uint8_t  sq38_u8;
typedef uint16_t sq38_u16;
typedef uint32_t sq38_u32;
typedef uint64_t sq38_u64;
typedef int32_t  sq38_i32;
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef uint8_t  sq38_u8;
typedef uint16_t sq38_u16;
typedef uint32_t sq38_u32;
typedef uint64_t sq38_u64;
typedef int32_t  sq38_i32;
#endif

/* ============================================================================
 * 1. Alignment, Magic & Architecture Constants (Shared Host C++ & Metal 4 MSL)
 * ============================================================================ */

#define STRATA_QWEN38_MAGIC_U64          0x3156354D38335153ULL /* "SQ38M5V1" */
#define STRATA_QWEN38_FORMAT_VERSION     1U
#define STRATA_QWEN38_PAGE_BYTES         16384ULL              /* 16 KiB Apple Silicon VM/UAT page */
#define STRATA_QWEN38_HEADER_PADDED_BYTES 81920U               /* 5 * 16 KiB (80 KiB >= 67,416 B)  */
#define STRATA_QWEN38_TENSOR_ALIGN       256ULL                /* 256 B Metal buffer offset align  */

#define STRATA_QWEN38_MAX_LAYERS         64U
#define STRATA_QWEN38_QUANT_GROUP_SIZE   64U
#define STRATA_QWEN38_QUANT_BITS         4U

/* Layer types in hybrid 3:1 schedule */
#define STRATA_QWEN38_LAYER_LINEAR_ATTN  0U /* Gated DeltaNet (`l % 4 != 3`, 48 layers) */
#define STRATA_QWEN38_LAYER_FULL_ATTN    1U /* Gated Full GQA (`l % 4 == 3`, 16 layers) */

/* Unquantized tensor dtypes in `weights.strata` */
#define STRATA_QWEN38_DTYPE_NONE         0U
#define STRATA_QWEN38_DTYPE_F32          1U /* IEEE-754 float32 (`float`) */
#define STRATA_QWEN38_DTYPE_F16          2U /* IEEE-754 float16 (`uint16_t` / `half`) */
#define STRATA_QWEN38_DTYPE_U32          3U /* Packed 8x 4-bit unsigned nibbles per `uint32_t` */

/* Canonical Qwen3.8 special token IDs */
#define STRATA_QWEN38_TOKEN_ENDOFTEXT    248044U /* `<|endoftext|>` (pad / secondary EOS) */
#define STRATA_QWEN38_TOKEN_IM_START     248045U /* `<|im_start|>` */
#define STRATA_QWEN38_TOKEN_IM_END       248046U /* `<|im_end|>` (primary chat EOS) */
#define STRATA_QWEN38_TOKEN_THINK_START  248068U /* `<think>` */
#define STRATA_QWEN38_TOKEN_THINK_END    248069U /* `</think>` */

/* ============================================================================
 * 2. Page-Aligned On-Disk `weights.strata` Descriptors (POD, 8-byte aligned)
 * ============================================================================ */

/*
 * MLX 4-Bit Affine Matrix Descriptor (`group_size = 64`):
 *   - `weight_off`: 256B-aligned offset to `uint32_t [rows_padded, cols / 8]`
 *   - `scales_off`: 256B-aligned offset to `uint16_t [rows_padded, cols / 64]` (IEEE FP16)
 *   - `biases_off`: 256B-aligned offset to `uint16_t [rows_padded, cols / 64]` (IEEE FP16)
 *   - Guarantee: `rows_padded % 64 == 0` and `cols % 1024 == 0` for EVERY matrix in Qwen3.8-27B.
 *     When `rows_logical < rows_padded` (`in_proj_a`/`in_proj_b`: `48 -> 64`), padded rows `48..63`
 *     are zero-filled in `weight`, `scales`, and `biases`.
 */
typedef struct StrataQwen38QuantMatrix {
    sq38_u64 weight_off;
    sq38_u64 weight_bytes;
    sq38_u64 scales_off;
    sq38_u64 scales_bytes;
    sq38_u64 biases_off;
    sq38_u64 biases_bytes;
    sq38_u32 rows_logical;
    sq38_u32 rows_padded;
    sq38_u32 cols;
    sq38_u32 group_size;
} StrataQwen38QuantMatrix;

/*
 * Dense 1D/2D Tensor Slice Descriptor (256B-aligned offset in `weights.strata`):
 *   - All RMSNorm weights (`input_layernorm`, `post_attention_layernorm`, `final_norm`,
 *     `linear_attn.norm`, `q_norm`, `k_norm`), `conv1d_weight [10240, 4]`, `A_log [64]`, and
 *     `dt_bias [64]` are stored as exact `STRATA_QWEN38_DTYPE_F32` (`float`) converted losslessly
 *     from BF16 (`u16 << 16`), with `A_log`/`dt_bias` indices `48..63` zero-padded.
 */
typedef struct StrataQwen38TensorSlice {
    sq38_u64 offset;
    sq38_u64 bytes;
    sq38_u32 elem_count;
    sq38_u32 dtype;
    sq38_u32 shape0;
    sq38_u32 shape1;
} StrataQwen38TensorSlice;

/* Per-layer descriptor inside `StrataQwen38WeightsHeader` */
typedef struct StrataQwen38LayerDesc {
    sq38_u32 layer_idx;
    sq38_u32 layer_type;       /* STRATA_QWEN38_LAYER_LINEAR_ATTN (0) or FULL_ATTN (1) */
    sq38_u64 layer_page_off;   /* 16 KiB-aligned start offset of this layer's payload */
    sq38_u64 layer_page_bytes; /* 16 KiB-aligned byte size of this layer's payload */

    /* Pre-attn & post-attn RMSNorms (`F32 [5120]`, standard `rms_norm(x, w, eps)`, NO +1.0) */
    StrataQwen38TensorSlice input_layernorm;
    StrataQwen38TensorSlice post_attention_layernorm;

    /* Populated iff `layer_type == STRATA_QWEN38_LAYER_LINEAR_ATTN` (`l % 4 != 3`) */
    StrataQwen38QuantMatrix lin_in_proj_qkv;   /* [10240, 5120] -> q[2048], k[2048], v[6144] */
    StrataQwen38QuantMatrix lin_in_proj_z;     /* [6144,  5120] -> z[48, 128] swish gate     */
    StrataQwen38QuantMatrix lin_in_proj_a;     /* logical [48, 5120], padded [64, 5120]      */
    StrataQwen38QuantMatrix lin_in_proj_b;     /* logical [48, 5120], padded [64, 5120]      */
    StrataQwen38QuantMatrix lin_out_proj;      /* [5120,  6144]                              */
    StrataQwen38TensorSlice lin_conv1d_weight; /* F32 [10240, 4]: taps [t-3, t-2, t-1, t]    */
    StrataQwen38TensorSlice lin_A_log;         /* F32 [64] (logical 48, padded 64)           */
    StrataQwen38TensorSlice lin_dt_bias;       /* F32 [64] (logical 48, padded 64)           */
    StrataQwen38TensorSlice lin_norm;          /* F32 [128] gated RMSNorm weight             */

    /* Populated iff `layer_type == STRATA_QWEN38_LAYER_FULL_ATTN` (`l % 4 == 3`) */
    StrataQwen38QuantMatrix attn_q_proj;       /* [12288, 5120] -> [24, 512] -> q[256], gate[256] */
    StrataQwen38QuantMatrix attn_k_proj;       /* [1024,  5120] -> [4, 256]                  */
    StrataQwen38QuantMatrix attn_v_proj;       /* [1024,  5120] -> [4, 256]                  */
    StrataQwen38QuantMatrix attn_o_proj;       /* [5120,  6144]                              */
    StrataQwen38TensorSlice attn_q_norm;       /* F32 [256] per-head RMSNorm                 */
    StrataQwen38TensorSlice attn_k_norm;       /* F32 [256] per-head RMSNorm                 */

    /* Dense SwiGLU MLP (populated for ALL 64 layers) */
    StrataQwen38QuantMatrix mlp_gate_proj;     /* [17408, 5120] */
    StrataQwen38QuantMatrix mlp_up_proj;       /* [17408, 5120] */
    StrataQwen38QuantMatrix mlp_down_proj;     /* [5120, 17408] */
} StrataQwen38LayerDesc;

/* Model hyperparameters stored in `weights.strata` header */
typedef struct StrataQwen38Config {
    sq38_u32 num_hidden_layers;         /* 64 */
    sq38_u32 hidden_size;               /* 5120 */
    sq38_u32 intermediate_size;         /* 17408 */
    sq38_u32 vocab_size;                /* 248320 */
    sq38_u32 num_linear_layers;         /* 48 */
    sq38_u32 num_full_attn_layers;      /* 16 */
    sq38_u32 full_attention_interval;   /* 4 (`(l + 1) % 4 == 0` => full_attention) */

    /* Full Attention (`self_attn`) parameters */
    sq38_u32 num_attention_heads;       /* 24 */
    sq38_u32 num_key_value_heads;       /* 4 */
    sq38_u32 head_dim;                  /* 256 */
    sq38_u32 rotary_dim;                /* 64 (`head_dim * partial_rotary_factor = 256 * 0.25`) */
    float    rope_theta;                /* 10000000.0f */
    sq38_u32 attn_output_gate;          /* 1 (`attn_out * sigmoid(gate)`) */

    /* Linear Attention (`GatedDeltaNet`) parameters */
    sq38_u32 linear_num_key_heads;      /* 16 */
    sq38_u32 linear_key_head_dim;       /* 128 (`key_dim = 2048`) */
    sq38_u32 linear_num_value_heads;    /* 48 (`v_heads / k_heads = 3`) */
    sq38_u32 linear_value_head_dim;     /* 128 (`value_dim = 6144`) */
    sq38_u32 linear_conv_kernel_dim;    /* 4 */

    /* Normalization, Quantization & Token IDs */
    float    rms_norm_eps;              /* 1e-6f */
    sq38_u32 quant_group_size;          /* 64 */
    sq38_u32 quant_bits;                /* 4 */
    sq38_u32 tie_word_embeddings;       /* 0 */
    sq38_u32 max_position_embeddings;   /* 262144 */
    sq38_u32 bos_token_id;              /* 248044 */
    sq38_u32 eos_token_ids[4];          /* {248046, 248044, 0, 0} */
    sq38_u32 num_eos_token_ids;         /* 2 */
} StrataQwen38Config;

/*
 * Master Header at byte offset `0` of `<bundle_dir>/weights.strata`.
 * Padded to `header_padded_bytes` (`STRATA_QWEN38_HEADER_PADDED_BYTES = 81920` B = `5 * 16 KiB`)
 * so payload tensors begin on an exact `16 KiB` VM page boundary, and `total_file_bytes` is also
 * an exact multiple of `16384` bytes for single-call `newBufferWithBytesNoCopy:` mapping.
 */
typedef struct StrataQwen38WeightsHeader {
    sq38_u64              magic;                /* STRATA_QWEN38_MAGIC_U64 */
    sq38_u32              version;              /* STRATA_QWEN38_FORMAT_VERSION */
    sq38_u32              header_padded_bytes;  /* 81920 (5 * 16 KiB) */
    sq38_u64              total_file_bytes;     /* Exact multiple of 16384 */
    sq38_u64              active_bytes_per_tok; /* Physical weight bytes read per M=1 forward step */

    StrataQwen38Config    cfg;

    /* Global Embedding (`[248320, 5120]` 4-bit affine) — starts on 16 KiB page boundary */
    sq38_u64              embed_page_off;
    sq38_u64              embed_page_bytes;
    StrataQwen38QuantMatrix embed_tokens;

    /* 64 Hybrid Decoder Layers — each starts on a 16 KiB page boundary */
    StrataQwen38LayerDesc layers[STRATA_QWEN38_MAX_LAYERS];

    /* Final RMSNorm (`F32 [5120]`) + Untied LM Head (`[248320, 5120]` 4-bit affine) — 16 KiB aligned */
    sq38_u64              head_page_off;
    sq38_u64              head_page_bytes;
    StrataQwen38TensorSlice final_norm;
    StrataQwen38QuantMatrix lm_head;
} StrataQwen38WeightsHeader;

#ifndef __METAL_VERSION__

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 3. Host Bundle Handle, Multi-Shard Streaming Converter & Zero-Copy `mmap` Loader
 *    (Implemented in `src/model/strata_qwen38_convert.cpp`)
 * ============================================================================ */

typedef struct StrataQwen38Tokenizer StrataQwen38Tokenizer;

typedef struct StrataQwen38Bundle {
    char                             bundle_dir[512];
    char                             weights_path[512];
    int                              mmap_fd;
    const uint8_t*                   mmap_base;   /* 16 KiB page-aligned read-only mmap */
    size_t                           mmap_bytes;  /* Exact multiple of 16384 bytes      */
    const StrataQwen38WeightsHeader* header;      /* Points directly to `mmap_base`     */
    StrataQwen38Config               cfg;         /* Value copy of `header->cfg`        */
    StrataQwen38Tokenizer*           tokenizer;   /* Loaded from `<bundle_dir>/tokenizer.json` */
} StrataQwen38Bundle;

typedef struct StrataQwen38ConvertStats {
    uint32_t num_shards;
    uint32_t num_text_tensors_converted;
    uint32_t num_vision_tensors_skipped;
    uint64_t hf_safetensors_bytes;
    uint64_t out_weights_bytes;
    uint64_t active_bytes_per_tok;
    double   elapsed_sec;
    double   peak_rss_gib;
    bool     page_aligned_verified;
} StrataQwen38ConvertStats;

/*
 * Streams multi-shard Hugging Face `model.safetensors.index.json` (`Qwen3.8-27B` MLX 4-bit)
 * layer-by-layer with `< 2.5 GiB` Peak RSS into `<out_bundle_dir>/`:
 *   - `weights.strata` (`16 KiB` page-aligned zero-copy binary weight container)
 *   - `tokenizer.json`, `tokenizer_config.json`, `config.json`, `manifest.json`
 * Ignores `vision_tower.*` / `visual.*` tensors and strips `language_model.` prefix.
 * Returns `0` on success, non-zero on error.
 */
int strata_qwen38_convert_hf_directory(
    const char* hf_dir,
    const char* out_bundle_dir,
    StrataQwen38ConvertStats* stats_out
);

/*
 * Opens `<bundle_dir>/weights.strata` via read-only `mmap` (`16 KiB` aligned, zero RAM copy)
 * and loads the BPE tokenizer from `<bundle_dir>/tokenizer.json`.
 * Returns `0` on success, non-zero on error.
 */
int strata_qwen38_load_bundle(
    const char* bundle_dir,
    StrataQwen38Bundle* bundle_out
);

/* Unmaps `weights.strata`, closes fd, frees BPE tokenizer, and zeroes `bundle`. */
void strata_qwen38_free_bundle(StrataQwen38Bundle* bundle);

/* Inline convenience helper to resolve any byte offset inside `bundle->mmap_base`. */
static inline const void* strata_qwen38_ptr(const StrataQwen38Bundle* b, uint64_t byte_offset) {
    return (const void*)(b->mmap_base + byte_offset);
}

/* Returns true iff `token_id` matches any configured EOS token (`248046` `<|im_end|>` or `248044` `<|endoftext|>`). */
static inline bool strata_qwen38_is_eos_token(const StrataQwen38Config* cfg, uint32_t token_id) {
    for (uint32_t i = 0; i < cfg->num_eos_token_ids; ++i) {
        if (token_id == cfg->eos_token_ids[i]) return true;
    }
    return false;
}

/* ============================================================================
 * 4. Hugging Face `tokenizer.json` Byte-Level BPE Tokenizer & Streaming Decoder
 *    (Implemented in `src/model/strata_qwen38_convert.cpp`)
 * ============================================================================ */

/* Standalone tokenizer load/free (also automatically loaded inside `strata_qwen38_load_bundle`). */
StrataQwen38Tokenizer* strata_qwen38_tokenizer_load(const char* tokenizer_json_path);
void strata_qwen38_tokenizer_free(StrataQwen38Tokenizer* tok);
uint32_t strata_qwen38_tokenizer_vocab_size(const StrataQwen38Tokenizer* tok);

/*
 * Encodes UTF-8 `text` (`text_len` bytes, or `strlen(text)` if `text_len == (size_t)-1`)
 * into `out_ids` (`[0 .. max_ids - 1]`) using exact Qwen2/GPT-2 pre-tokenization +
 * byte-level BPE merges (`247,587` rules) + longest-match added/special tokens
 * (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<think>`, `</think>`, etc.) when `parse_special` is true.
 * Returns `0` on success and writes `*out_num_ids`; returns non-zero on error or buffer overflow.
 */
int strata_qwen38_encode(
    const StrataQwen38Tokenizer* tok,
    const char* text,
    size_t text_len,
    bool parse_special,
    uint32_t* out_ids,
    uint32_t max_ids,
    uint32_t* out_num_ids
);

/*
 * Formats a user prompt (and optional `system_prompt`, NULL for none) using the official
 * Qwen3.8 ChatML template (`chat_template.jinja`):
 *   - `enable_thinking == true`:  `...<|im_start|>assistant\n<think>\n`
 *   - `enable_thinking == false`: `...<|im_start|>assistant\n<think>\n\n</think>\n\n`
 * Writes null-terminated UTF-8 into `out_buf` (`out_buf_cap` bytes) and sets `*out_len` (excluding NUL).
 * Returns `0` on success, non-zero if `out_buf_cap` is too small.
 */
int strata_qwen38_format_chat_prompt(
    const char* user_prompt,
    const char* system_prompt,
    bool enable_thinking,
    char* out_buf,
    size_t out_buf_cap,
    size_t* out_len
);

/*
 * Batch-decodes `ids[0 .. num_ids - 1]` into null-terminated UTF-8 in `out_utf8` (`out_cap` bytes).
 * When `skip_special` is true, special control tokens (`<|im_end|>`, `<|endoftext|>`, `<|im_start|>`, etc.)
 * are omitted while keeping visible reasoning tags (`<think>`, `</think>`).
 */
int strata_qwen38_decode(
    const StrataQwen38Tokenizer* tok,
    const uint32_t* ids,
    uint32_t num_ids,
    bool skip_special,
    char* out_utf8,
    size_t out_cap,
    size_t* out_len
);

/*
 * Stateful token-by-token streaming UTF-8 decoder: buffers incomplete multi-byte UTF-8
 * byte-level BPE pieces across consecutive tokens so stdout streaming never prints broken UTF-8.
 */
typedef struct StrataQwen38StreamDecoder {
    uint8_t pending_bytes[8];
    uint8_t pending_len;
} StrataQwen38StreamDecoder;

static inline void strata_qwen38_stream_decoder_init(StrataQwen38StreamDecoder* dec) {
    for (int i = 0; i < 8; ++i) dec->pending_bytes[i] = 0;
    dec->pending_len = 0;
}

/*
 * Appends `token_id`'s raw bytes into `dec` and writes any complete valid UTF-8 codepoints
 * into `out_utf8` (null-terminated, up to `out_cap` bytes), setting `*out_len`.
 */
int strata_qwen38_stream_decode_step(
    const StrataQwen38Tokenizer* tok,
    StrataQwen38StreamDecoder* dec,
    uint32_t token_id,
    bool skip_special,
    char* out_utf8,
    size_t out_cap,
    size_t* out_len
);

/* ============================================================================
 * 5. Dedicated Metal 4 (`NAX` + Coalesced Decode) `Qwen3.8-27B` GPU Engine ABI
 *    (Implemented by `@qwen-gpu` in `src/gpu/strata_qwen38_gpu.mm`)
 * ============================================================================ */

typedef struct StrataQwen38GpuEngine StrataQwen38GpuEngine;

typedef struct StrataQwen38SamplingParams {
    bool     greedy;       /* true => fast GPU argmax (`temperature` ignored) */
    float    temperature;  /* e.g. 0.6f when `!greedy` */
    uint32_t top_k;        /* e.g. 20 (0 => disabled / default) */
    float    top_p;        /* e.g. 0.95f (<= 0 or >= 1 => disabled) */
    uint64_t rng_seed;     /* Per-step or stream PRNG seed/state */
} StrataQwen38SamplingParams;

typedef struct StrataQwen38GpuInfo {
    char     device_name[128];
    bool     zero_copy_mmap;
    double   load_time_ms;
    uint32_t max_seq_len;
    uint64_t weights_mapped_bytes;      /* Total zero-copy `MTLBuffer` bytes mapped */
    uint64_t active_bytes_per_token;    /* Physical weight bytes read per single-token step */
    uint64_t deltanet_state_bytes;      /* 48-layer `conv_state + recurrent_state` (~153.8 MiB) */
    uint64_t kv_cache_bytes_allocated;  /* 16-layer FP16 KV cache bytes for `max_seq_len` */
    uint64_t kv_cache_bytes_per_token;  /* `16 * 2 * 4 * 256 * 2 = 32,768` B/token */
    uint64_t workspace_bytes;           /* Pre-allocated scratch buffers */
} StrataQwen38GpuInfo;

/*
 * Maps `bundle->mmap_base` zero-copy via `newBufferWithBytesNoCopy:` + `MTLResidencySet`,
 * compiles `strata_qwen38_kernels.metal` (`MTLLanguageVersion4_0`), and pre-allocates
 * persistent 48-layer DeltaNet state + 16-layer FP16 KV cache up to `max_seq_len` tokens.
 */
StrataQwen38GpuEngine* strata_qwen38_gpu_create(
    const StrataQwen38Bundle* bundle,
    uint32_t max_seq_len
);

void strata_qwen38_gpu_destroy(StrataQwen38GpuEngine* eng);

/* Resets `seq_pos = 0`, `conv_state = 0`, and `recurrent_state = 0` without reallocating buffers. */
void strata_qwen38_gpu_reset(StrataQwen38GpuEngine* eng);

/* Current cached sequence length (`seq_pos`). */
uint32_t strata_qwen38_gpu_seq_pos(const StrataQwen38GpuEngine* eng);

/* Populates hardware residency, state footprint, and bandwidth accounting metadata. */
void strata_qwen38_gpu_get_info(
    const StrataQwen38GpuEngine* eng,
    StrataQwen38GpuInfo* info_out
);

/*
 * Batched Prefill over `token_ids[0 .. num_tokens - 1]` starting at current `seq_pos`
 * (internally chunked across Metal 4 `NAX` matmul2d + batched DeltaNet + tiled FlashAttention),
 * advancing `seq_pos` by `num_tokens` and sampling/argmaxing the next token from the final position
 * into `*next_token_out`. If `last_logits_out != NULL`, copies final-token FP32 `[248320]` logits.
 */
int strata_qwen38_gpu_prefill(
    StrataQwen38GpuEngine* eng,
    const uint32_t* token_ids,
    uint32_t num_tokens,
    const StrataQwen38SamplingParams* sampling,
    uint32_t* next_token_out,
    float* last_logits_out
);

/*
 * Single-Token Decode (`M = 1`): encodes ALL 64 layers + final RMSNorm + `248,320`-vocab LM head +
 * GPU argmax/top-k into a SINGLE `MTLCommandBuffer` (zero mid-network CPU-GPU sync!), advancing
 * `seq_pos` by 1 and writing `*next_token_out`. `logits_out` may be NULL for 4-byte readback fast path.
 */
int strata_qwen38_gpu_decode_step(
    StrataQwen38GpuEngine* eng,
    uint32_t token_id,
    const StrataQwen38SamplingParams* sampling,
    uint32_t* next_token_out,
    float* logits_out
);

/*
 * O(1) Turn-Boundary GPU Checkpoint / Restore (~146.8 MiB on-demand GPU->GPU blit, < 0.5 ms):
 *   - Allocated lazily ONLY when `strata_qwen38_checkpoint_create` is called (`0` bytes in `bench-hf`).
 *   - Snapshots `seq_pos`, `sample_step`, `b_conv_state`, and `b_rec_state` in a single `MTLBlitCommandEncoder`.
 */
typedef struct StrataQwen38Checkpoint StrataQwen38Checkpoint;

StrataQwen38Checkpoint* strata_qwen38_checkpoint_create(const StrataQwen38GpuEngine* eng);
void strata_qwen38_checkpoint_destroy(StrataQwen38Checkpoint* ckpt);
int strata_qwen38_checkpoint_save(StrataQwen38GpuEngine* eng, StrataQwen38Checkpoint* ckpt);
int strata_qwen38_checkpoint_restore(StrataQwen38GpuEngine* eng, const StrataQwen38Checkpoint* ckpt);
uint64_t strata_qwen38_checkpoint_bytes(const StrataQwen38Checkpoint* ckpt);

/* Streams active GPU sequence state (`seq_pos`, `sample_step`, 48L DeltaNet + `0..seq_pos` 16L KV) to/from `fd`. */
int strata_qwen38_state_save_fd(StrataQwen38GpuEngine* eng, int fd, uint64_t* bytes_written_out);
int strata_qwen38_state_load_fd(StrataQwen38GpuEngine* eng, int fd, uint64_t* bytes_read_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !__METAL_VERSION__ */

#endif /* STRATA_QWEN38_H */

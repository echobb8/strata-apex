#ifndef STRATA_CHECKPOINT_H
#define STRATA_CHECKPOINT_H

/*
 * STRATA-M5 v2.0 — 4-Axis Checkpoint Bundle & Real Tri-Engine Inference ABI
 * Target Hardware: Mac17,10 (Apple M5 Max: 40c Metal 4 NAX GPU, 3x SME2 CPU clusters,
 *                  16c CoreML ANE, ANS Gen5 NVMe @ 14.5 GB/s).
 *
 * Defines:
 *   1. Loaded 4-Axis `.strata` bundle representation (`StrataLayerWeights`, `StrataCheckpointBundle`):
 *      - Axis A (`engram.ngram`): 4 KiB-aligned INT4 nibbles + FP16 ZT0 codebook for SME2 Phase-1.
 *      - Axis B (`spine.bin`): 32x32 NAX `wCT`-tiled E4M3 + 32-way Tile-rANS Q/K/V/O & LM head,
 *        plus FP16 RMSNorms, Embeddings, Router weights, and FR-Spec MTP Draft Head.
 *      - Axis C (`uma_hot.bin`): 256B-aligned NAX `wCT` planar MXFP4 (17 B/32w) or QTIP (34 B/128w)
 *        hot experts, Barycenter/QTIP base pool, and W6A16 MoE-SVD `U_e` + FP16 `V_e` factors.
 *      - Axis D (`cold_experts.strata`): 256 KiB LBA-aligned (`STRATA_LBA_ALIGN_BYTES`) `F_NOCACHE`
 *        cold expert extents in direct GPU planar layout for zero-copy `MTLIOCommandQueue` ring DMA.
 *   2. Offline `.safetensors` synthesizer, 4-axis disaggregator/converter, and bundle loader APIs.
 *   3. Real Metal 4 `NAX` GPU weight residency & forward-pass execution APIs (`strata_gpu_exec_*`).
 */

#include "strata_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to resident Metal 4 GPU buffers + reusable zero-fault workspace arenas */
typedef struct StrataGpuWeights StrataGpuWeights;

/* ============================================================================
 * 1. Per-Layer & Bundle-Level Disaggregated Model Structures
 * ============================================================================ */

typedef struct StrataLayerWeights {
    uint32_t layer_idx;

    /* RoPE / GQA Attention Dimensions (mirrored from bundle for self-contained layer access) */
    uint32_t n_heads;               /* Query head count (e.g. 8; q_dim = n_heads * head_dim) */
    uint32_t n_kv_heads;            /* Key/Value head count (e.g. 4; kv_dim = n_kv_heads * head_dim) */
    uint32_t head_dim;              /* Per-head dimension (e.g. 32) */

    /* Axis B: Pre-Attn & Post-Attn RMSNorm Weights (IEEE FP16 bits, [d_model]) */
    uint16_t* attn_norm_f16;        /* [d_model] */
    uint16_t* ffn_norm_f16;         /* [d_model] */

    /* Axis B: 32x32 Metal 4 NAX `wCT`-tiled E4M3 + 32-way Tile-rANS Attention Blobs
     *   - q_rans_blob: shape [q_dim, d_model]   (d_in = d_model, d_out = n_heads * head_dim)
     *   - k_rans_blob: shape [kv_dim, d_model]  (d_in = d_model, d_out = n_kv_heads * head_dim)
     *   - v_rans_blob: shape [kv_dim, d_model]  (d_in = d_model, d_out = n_kv_heads * head_dim)
     *   - o_rans_blob: shape [d_model, q_dim]   (d_in = n_heads * head_dim, d_out = d_model)
     * Stable pointers for the lifetime of StrataCheckpointBundle (safe as keys for GPU buffer lookup). */
    uint8_t* q_rans_blob;
    size_t   q_rans_bytes;
    uint8_t* k_rans_blob;
    size_t   k_rans_bytes;
    uint8_t* v_rans_blob;
    size_t   v_rans_bytes;
    uint8_t* o_rans_blob;
    size_t   o_rans_bytes;

    /* Axis B: MoE Router Gate Matrix (IEEE FP16 bits, row-major [num_experts, d_model]) */
    uint16_t* router_w_f16;         /* [num_experts * d_model] */

    /* SLC Indirection & Expert Residency Tier Map ([num_experts])
     *   - is_uma_hot[e]:      true iff expert `e` is permanently wired in UMA (`STRATA_TIER_HOT_FULL_4B`).
     *   - tier_kind[e]:       StrataTierKind (`HOT_FULL_4B`, `WARM_QTIP_RING`, or `COLD_SVD_RING`).
     *   - hot_or_base_idx[e]:
     *       * When `is_uma_hot[e]` is true: slot index in `hot_pool_buf` (`0 .. hot_pool_slots - 1`).
     *         (In `STRATA_REGIME_I_BITNEST`, `base_pool_buf` is ordered so `hot_or_base_idx[e]` ALSO
     *         indexes expert `e`'s QTIP 2b base in `base_pool_buf`!)
     *       * When `is_uma_hot[e]` is false:
     *           - Regime II (`BARY_SVD`): expert `e`'s slot in `svd_u_w6_buf` (`0 .. base_pool_slots - 1`,
     *             where `base_pool_slots == num_experts` and `hot_or_base_idx[e] == e`), while
     *             `base_pool_buf` holds the single layer Barycenter expert at offset 0.
     *           - Regime I (`BITNEST`): expert `e`'s slot in `base_pool_buf` (`0 .. base_pool_slots - 1`).
     *   - cold_extent_idx[e]: index into `bundle->cold_extents[]` for Axis D NVMe DMA, or `-1` if UMA-hot.
     *   - default_slot_map[e]: pre-initialized `StrataSlotEntry` (`ring_slot_idx = STRATA_SLOT_INVALID`,
     *     `expected_epoch = 0`); callers copy per step and stamp `ring_slot_idx` + `expected_epoch`. */
    bool*            is_uma_hot;        /* [num_experts] */
    uint16_t*        tier_kind;         /* [num_experts] */
    uint16_t*        hot_or_base_idx;   /* [num_experts] */
    int32_t*         cold_extent_idx;   /* [num_experts], -1 when UMA-hot */
    StrataSlotEntry* default_slot_map;  /* [num_experts] */

    /* Axis C: UMA Hot Expert Pool & UMA Base/Barycenter Pool (exact `strata_kernels.metal` layout)
     *   - MXFP4 (`100PCT_UMA` / `BARY_SVD`): 32x32 NAX `wCT` planar nibbles (`total_sb32 * 16B`) +
     *     interleaved E4M3 scales (`total_sb32 * 1B`), padded to 256B (`hot_expert_bytes`).
     *   - QTIP (`BITNEST`): planar 2-bit codes (`total_blks128 * 32B`) + FP16 scales (`total_blks128 * 2B`),
     *     padded to 256B (`base_expert_bytes` for QTIP base, `hot_expert_bytes` for 4-lvl Laplacian refine).
     *   Note: `base_pool_slots` matches `StrataMoEKernelCfg.base_pool_slots` (`num_experts` for `BITNEST`
     *   and `BARY_SVD` `u_slot` indexing; `1` for `100PCT_UMA`), while `base_pool_bytes` is the actual
     *   byte size of `base_pool_buf` (`1 * base_expert_bytes` for `BARY_SVD` Barycenter,
     *   `num_experts * base_expert_bytes` for `BITNEST`). */
    uint32_t hot_pool_slots;        /* Number of wired hot expert slots in `hot_pool_buf` */
    uint64_t hot_expert_bytes;      /* 256B-aligned per-expert byte stride in `hot_pool_buf` & ring */
    uint8_t* hot_pool_buf;          /* [hot_pool_bytes] (non-NULL; >= hot_expert_bytes dummy if 0 hot) */
    size_t   hot_pool_bytes;

    uint32_t base_pool_slots;       /* Logical `cfg.base_pool_slots` passed to `StrataMoEKernelCfg` */
    uint64_t base_expert_bytes;     /* 256B-aligned per-expert byte stride in `base_pool_buf` */
    uint8_t* base_pool_buf;         /* [base_pool_bytes] */
    size_t   base_pool_bytes;

    /* Axis C: Regime II W6A16 MoE-SVD Factors (`svd_rank > 0`; NULL / 0 when `svd_rank == 0`)
     *   - `svd_rank_aligned = (svd_rank + 63) & ~63` (matches `strata_moe_svd_w6_fallback`).
     *   - `svd_u_w6_buf`: `[num_experts, d_ff, (svd_rank_aligned / 64) * 50]` bytes packed via
     *     `strata_codec_w6a16_pack` (indexed on GPU by `u_slot = slot.hot_or_base_idx % base_pool_slots`).
     *   - `svd_v_f16`: `[num_experts, svd_rank, d_model]` FP16 right singular vectors `V_e`.
     *   - `svd_v_shared_f16`: `[svd_rank_aligned, d_model]` FP16 shared layer subspace basis `V`
     *     (padded with zeros for rows `svd_rank .. svd_rank_aligned - 1`) so host/CPU or GPU can
     *     directly compute `v_proj_x_fp16 [batch_B, svd_rank_aligned] = x_fp16 @ V^T` for
     *     `strata_gpu_exec_moe_layer(..., v_proj_x_fp16, ...)`. */
    uint32_t  svd_rank_aligned;     /* (cfg.svd_rank + 63) & ~63, or 0 */
    uint32_t  svd_u_slots;          /* num_experts when svd_rank > 0, else 0 */
    uint8_t*  svd_u_w6_buf;         /* [svd_u_w6_bytes] */
    size_t    svd_u_w6_bytes;
    uint16_t* svd_v_f16;            /* [num_experts * svd_rank * d_model] FP16 */
    uint16_t* svd_v_shared_f16;     /* [svd_rank_aligned * d_model] FP16 */
} StrataLayerWeights;

typedef struct StrataCheckpointBundle {
    StrataModelConfig cfg;          /* `cfg.name` points to `arch_name` below */
    char arch_name[64];             /* e.g. "mini-deepseek", "mini-qwen", "mini-glm" */
    char bundle_dir[512];           /* Directory loaded from */

    /* Transformer Attention & Sequence Hyperparameters */
    uint32_t n_heads;               /* e.g. 8 */
    uint32_t n_kv_heads;            /* e.g. 4 */
    uint32_t head_dim;              /* e.g. 32 */
    uint32_t max_seq_len;           /* e.g. 512 */
    float    rms_norm_eps;          /* e.g. 1e-5f */
    float    rope_theta;            /* e.g. 10000.0f */
    uint32_t ane_probe_rank;        /* Bottleneck rank `r` for `strata_ane_probe_init` (e.g. 64) */

    /* Axis B Global Weights: Token Embeddings, Final RMSNorm, LM Head */
    uint16_t* tok_embed_f16;        /* [vocab_size * d_model] FP16 */
    uint16_t* final_norm_f16;       /* [d_model] FP16 */
    uint8_t*  lm_head_rans_blob;    /* [vocab_size, d_model] 32x32 NAX `wCT` E4M3 Tile-rANS */
    size_t    lm_head_rans_bytes;
    uint16_t* lm_head_f16;          /* [vocab_size * d_model] FP16 row-major (exact E4M3-decoded/FP16) */

    /* Axis B Speculative Decoding: Lightweight MTP Draft Module + FR-Spec Shortlist
     *   - `mtp_norm_f16`: `[d_model]` FP16 RMSNorm
     *   - `mtp_proj_f16`: `[d_model * (2 * d_model)]` FP16 state fusion `[h_prev; tok_emb] -> h_draft`
     *   - `mtp_head_f16`: `[shortlist_size * d_model]` FP16 draft logits over shortlist
     *   - `mtp_head_rans_blob`: `[shortlist_size, d_model]` 32x32 NAX `wCT` E4M3 Tile-rANS blob
     *   - `shortlist_vocab_ids`: `[shortlist_size]` mapping shortlist index -> `[0 .. vocab_size - 1]` */
    uint16_t* mtp_norm_f16;         /* [d_model] FP16 */
    uint16_t* mtp_proj_f16;         /* [d_model * (2 * d_model)] FP16 */
    uint16_t* mtp_head_f16;         /* [shortlist_size * d_model] FP16 */
    uint8_t*  mtp_head_rans_blob;   /* [shortlist_size, d_model] E4M3 Tile-rANS (NULL if % 32 != 0) */
    size_t    mtp_head_rans_bytes;
    uint32_t* shortlist_vocab_ids;  /* [shortlist_size] */

    /* Per-Layer Disaggregated Weights (`[cfg.num_layers]`) */
    StrataLayerWeights* layers;

    /* Axis A: 4 KiB-Aligned Conditional Memory Shard (`engram.ngram` for SME2 `LUTI4`/`FMOPA`) */
    char     ngram_path[512];       /* Full path to `<bundle_dir>/engram.ngram`, or "" if disabled */
    uint64_t ngram_file_bytes;      /* Multiple of 4096 B */
    uint32_t ngram_layer_idx;       /* Target layer index for Phase-1 N-gram injection (e.g. 1 or 2) */

    /* Axis D: 256 KiB LBA-Aligned Cold Experts Container (`cold_experts.strata`) */
    char                  cold_strata_path[512];
    StrataContainerHeader cold_header;
    StrataExtentRecord*   cold_extents;     /* [num_cold_extents] */
    uint32_t              num_cold_extents;

    /* Embedded Subword Tokenizer / Vocabulary (`[cfg.vocab_size]`) */
    char**   vocab_tokens;          /* [vocab_size] null-terminated UTF-8 token pieces */
    float*   vocab_scores;          /* [vocab_size] token merge/priority scores */
    uint32_t unk_token_id;          /* 0 */
    uint32_t bos_token_id;          /* 1 */
    uint32_t eos_token_id;          /* 2 */
} StrataCheckpointBundle;

/* ============================================================================
 * 2. `.safetensors` Synthesizer, 4-Axis Disaggregator/Converter & Bundle Loader
 *    (Implemented in `src/model/strata_convert.cpp`)
 * ============================================================================ */

typedef struct StrataConvertStats {
    uint32_t num_safetensors_tensors;
    uint64_t safetensors_payload_bytes;
    uint64_t axis_a_ngram_bytes;
    uint64_t axis_b_spine_bytes;
    uint64_t axis_c_uma_hot_bytes;
    uint64_t axis_d_cold_strata_bytes;
    uint32_t num_hot_experts_total;
    uint32_t num_cold_experts_total;
    bool     rans_roundtrip_verified;
    bool     lba_alignment_verified;
} StrataConvertStats;

/*
 * Synthesizes a valid HuggingFace RFC-8259 `.safetensors` checkpoint + `tokenizer.json` +
 * `vocab.txt` + `config.json` with deterministic, well-conditioned orthogonal/planted weights
 * and readable English subword tokens (< 80 MiB total footprint).
 *   - `out_path`: Either a `.safetensors` file path OR an output directory path (writes `model.safetensors`).
 *   - `preset_name`: `"mini-deepseek"` (default if NULL/empty: Regime II BARY_SVD + Engram),
 *                    `"mini-qwen"` (Regime I 100% UMA + PLE), or
 *                    `"mini-glm"` (Regime I BITNEST QTIP + 4-lvl Laplacian Refine).
 *   - `seed`: Deterministic PRNG seed (`0` uses default `42`).
 */
int strata_synth_safetensors_checkpoint(
    const char* out_path,
    const char* preset_name,
    uint64_t seed
);

/*
 * Disaggregates a `.safetensors` checkpoint (F32, F16, BF16, F8_E4M3) + tokenizer into
 * `<bundle_dir>/` across all 4 STRATA hardware axes (`engram.ngram`, `spine.bin`,
 * `uma_hot.bin`, `cold_experts.strata`, `tokenizer.json`, `manifest.json`).
 *   - `safetensors_path`: Path to `.safetensors` file OR directory containing `model.safetensors`.
 *   - `bundle_dir`: Destination directory (created if missing).
 *   - `stats_out`: Optional conversion & verification summary (may be NULL).
 */
int strata_convert_safetensors_to_bundle(
    const char* safetensors_path,
    const char* bundle_dir,
    StrataConvertStats* stats_out
);

/* Loads a converted `.strata` bundle directory into memory (`cold_experts.strata` stays on NVMe). */
int strata_bundle_load(
    const char* bundle_dir,
    StrataCheckpointBundle* bundle_out
);

/* Frees all host allocations inside `bundle` and zeroes the struct. */
void strata_bundle_free(StrataCheckpointBundle* bundle);

/* Greedy longest-match subword tokenizer & detokenizer using `bundle->vocab_tokens`. */
int strata_tokenize_prompt(
    const StrataCheckpointBundle* bundle,
    const char* prompt_utf8,
    bool add_bos,
    uint32_t* token_ids_out,
    uint32_t max_tokens,
    uint32_t* num_tokens_out
);

const char* strata_detokenize_token(
    const StrataCheckpointBundle* bundle,
    uint32_t token_id
);

/* ============================================================================
 * 3. Real Metal 4 `NAX` GPU Weight Residency & Forward-Pass Execution API
 *    (Implemented in `src/gpu/strata_gpu.mm`)
 * ============================================================================ */

/* Uploads Axis B Tile-rANS blobs & Axis C hot/base/W6A16 pools + pre-allocates zero-fault workspaces. */
StrataGpuWeights* strata_gpu_weights_upload(
    StrataGpuContext* ctx,
    const StrataCheckpointBundle* bundle
);

void strata_gpu_weights_free(StrataGpuWeights* gw);

/* Executes 32-way Tile-rANS E4M3 in-register decode -> Metal 4 NAX `cooperative_tensor` GEMM:
 *   y_f32_out [batch_B, d_out] = x_fp16 [batch_B, d_in] @ W [d_out, d_in]^T
 *   Supports any `d_in % 32 == 0` and `d_out % 32 == 0` (`1 <= batch_B <= STRATA_MAX_SPEC_BATCH`). */
int strata_gpu_exec_spine_rans(
    StrataGpuContext* ctx,
    const void* rans_blob,
    size_t blob_bytes,
    const uint16_t* x_fp16,
    uint32_t batch_B,
    uint32_t d_in,
    uint32_t d_out,
    float* y_f32_out
);

/* Executes a full Slot-Indexed Tiered MoE Layer on Metal 4 NAX (`gather_x16` -> `gate_up_swiglu` ->
 * `swiglu_pack16` -> optional `svd_w6_fallback` -> optional `fwht128 + pad_mid8_to_16` -> `down_accum`):
 *   - `layer_slot_map`: `[num_experts]` indirection table for `layer_idx`.
 *   - `routed_ids`:     `[batch_B * top_k]` selected expert IDs per token.
 *   - `routed_gates`:   `[batch_B * top_k]` normalized routing weights per token.
 *   - `v_proj_x_fp16`:  `[batch_B * svd_rank_aligned]` FP16 `x @ V^T` (may be NULL when `svd_rank == 0`).
 *   - `external_ring_mtl_buf` / `external_slot_ready`: shared `MTLIOCommandQueue` NVMe ring & atomic epoch table. */
int strata_gpu_exec_moe_layer(
    StrataGpuContext* ctx,
    StrataGpuWeights* gw,
    uint32_t layer_idx,
    StrataExecMode mode,
    uint32_t batch_B,
    const uint16_t* x_fp16,
    const StrataSlotEntry* layer_slot_map,
    const uint16_t* routed_ids,
    const float* routed_gates,
    const uint16_t* v_proj_x_fp16,
    void* external_ring_mtl_buf,
    const uint32_t* external_slot_ready,
    float* y_f32_out,
    uint32_t* used_ring_out,
    uint32_t* fallback_out
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STRATA_CHECKPOINT_H */

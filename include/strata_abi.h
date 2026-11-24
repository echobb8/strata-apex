#ifndef STRATA_ABI_H
#define STRATA_ABI_H

/*
 * STRATA-M5 v2.0 — Unified Cross-Engine Application Binary Interface
 * Target Hardware: 2026 MacBook Pro PROLED XDR (Mac17,10 — Apple M5 Max 40c GPU,
 *                  18c CPU w/ 3x SME2 clusters, 16c ANE w/ 32 MiB SRAM, 128 GiB UMA,
 *                  ANS Gen5 NVMe @ 14.5 GB/s).
 *
 * Shared cleanly across C++20 (.cpp), Objective-C++/Metal (.mm), and MSL (.metal).
 */

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef uint8_t  strata_u8;
typedef uint16_t strata_u16;
typedef uint32_t strata_u32;
typedef uint64_t strata_u64;
typedef int32_t  strata_i32;
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef uint8_t  strata_u8;
typedef uint16_t strata_u16;
typedef uint32_t strata_u32;
typedef uint64_t strata_u64;
typedef int32_t  strata_i32;
#endif

/* ============================================================================
 * 1. Global Physical & Alignment Constants (Verified v2.0 Spec #12 / #13)
 * ============================================================================ */

#define STRATA_MAGIC_U64            0x354D415441525453ULL /* "STRATAM5" */
#define STRATA_ABI_VERSION          200U                  /* v2.0 */
#define STRATA_LBA_ALIGN_BYTES      262144U               /* 256 KiB NAND coalesce rule */
#define STRATA_VM_PAGE_BYTES        16384U                /* 16 KiB XNU/UAT page size */
#define STRATA_CACHELINE_BYTES      128U                  /* Apple Silicon fabric line */
#define STRATA_RANS_LANES           32U                   /* 32-way SIMD-group Tile-rANS */
#define STRATA_ANE_SPATIAL_ALIGN    16U                   /* Orion [1, d, 1, S=16] rule */
#define STRATA_ANE_WINDOW_STRIDE    4U                    /* Predict L0+4 .. L0+7 */
#define STRATA_SHORTLIST_VOCAB      32768U                /* FR-Spec 32K draft shortlist */
#define STRATA_MAX_SPEC_BATCH       8U                    /* B=6 padded to 8 for NAX */
#define STRATA_MAX_TOP_K            16U                   /* Covers k=6, k=8, k=10 */
#define STRATA_SLOT_INVALID         0xFFFFU

/* Execution Modes (#12 §1.3) */
typedef enum StrataExecMode {
    /* Strictly prefix-stationary target (per-token g_{t,e} >= tau_tok, or 0-drop
     * when tau_tok == 0). Guarantees 100% unbiased Leviathan rejection sampling. */
    STRATA_MODE_A_STATIONARY = 0,

    /* Zero-stall hard real-time mode: runs 100% of UMA-resident hot hits at full
     * 4-bit for free, caps cold NVMe fetches via cross-batch S_e = sum_t g_{t,e},
     * and falls back in 0.00 ms to QTIP 2b Base or W6A16 MoE-SVD if slot unready. */
    STRATA_MODE_B_ZERO_STALL = 1
} StrataExecMode;

/* Tiered Capacity Regimes (#12 §1.1–1.2) */
typedef enum StrataRegime {
    STRATA_REGIME_I_100PCT_UMA  = 0, /* e.g., Qwen3.8-Flash-Next (PLE offloaded) */
    STRATA_REGIME_I_BITNEST     = 1, /* e.g., GLM-5.3-Flash / Sage S2.1 (2.125b QTIP + 1.85b rANS ref) */
    STRATA_REGIME_II_BARY_SVD   = 2  /* e.g., DeepSeek-V4.1-Flash / GLM-5.3 (Full 4b Hot + W6A16 MoE-SVD) */
} StrataRegime;

/* Per-Expert Residency Tier Flag inside SLC Slot Map */
typedef enum StrataTierKind {
    STRATA_TIER_HOT_FULL_4B     = 0, /* Full 4b/MXFP4/W6 wired permanently in UMA hot pool */
    STRATA_TIER_WARM_QTIP_RING  = 1, /* 2.125b QTIP base in UMA + 1.85b rANS refine in NVMe ring */
    STRATA_TIER_COLD_SVD_RING   = 2  /* W6A16 MoE-SVD + Barycenter in UMA + full 4b in NVMe ring */
} StrataTierKind;

/* ============================================================================
 * 2. Shared Host <-> Metal Shader POD Structures (8/16-byte aligned)
 * ============================================================================ */

/* SLC-resident indirection entry: slot_map[layer][expert] (8 bytes/entry) */
typedef struct StrataSlotEntry {
    strata_u16 hot_or_base_idx; /* Index in UMA Hot Pool or UMA Base/SVD table */
    strata_u16 ring_slot_idx;   /* Slot index in pre-wired NVMe Ring (or STRATA_SLOT_INVALID) */
    strata_u16 tier_kind;       /* StrataTierKind */
    strata_u16 reserved;
    strata_u32 expected_epoch;  /* Matches atomic slot_ready[ring_slot_idx] when DMA completes */
} StrataSlotEntry;

/* 32-way SIMD-interleaved Tile-rANS block header (32 bytes) */
typedef struct StrataTileRansHeader {
    strata_u32 magic;               /* 0x534E4152U ("RANS") */
    strata_u32 uncompressed_count;  /* Number of symbols encoded across 32 lanes */
    strata_u32 compressed_bytes;    /* Payload byte size followingstream_offsets */
    strata_u16 freq_bits;           /* Probability scale bits (e.g. 12 -> sum=4096) */
    strata_u16 alphabet_size;       /* e.g. 4 (Laplacian refine), 16 (MXFP4), 256 (E4M3/exp) */
    strata_u32 reserved[4];
} StrataTileRansHeader;

/* Per-dispatch MoE kernel control block passed to Metal `strata_moe_mm` */
typedef struct StrataMoEDispatchParams {
    strata_u32 d_model;
    strata_u32 d_ff;
    strata_u32 num_experts;
    strata_u32 top_k;
    strata_u32 batch_tokens;        /* Active B (e.g. 6, internally tiled to 8) */
    strata_u32 svd_rank;            /* e.g. 108 for DeepSeek-V4.1, 0 for Regime I */
    strata_u32 exec_mode;           /* StrataExecMode */
    float      tau_tok;             /* Mode A causal per-token threshold g_{t,e} >= tau_tok */
} StrataMoEDispatchParams;

#ifndef __METAL_VERSION__

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 3. Model Architecture Presets, Container Packer & Unbiased Speculative Sampler
 *    (Owned by `models-mlx` in `src/model/`)
 * ============================================================================ */

typedef struct StrataModelConfig {
    const char* name;
    StrataRegime regime;
    uint32_t num_layers;            /* e.g. 48 (Qwen3.8), 40 (DeepSeek CED), 45 (GLM-5.3-Flash) */
    uint32_t encoder_layers;        /* 20 for DeepSeek CED; 0 for standard decoder-only */
    uint32_t d_model;               /* 2560 (Qwen3.8), 5120 (DeepSeek), 4096 (GLM-5.3-Flash) */
    uint32_t d_ff;                  /* 640 (Qwen3.8), 2304 (DeepSeek), 2048 (GLM-5.3-Flash) */
    uint32_t num_experts;           /* 512 (Qwen3.8), 384 (DeepSeek), 288 (GLM-5.3-Flash) */
    uint32_t top_k;                 /* 10 (Qwen3.8), 6 (DeepSeek), 8 (GLM-5.3-Flash) */
    uint32_t vocab_size;            /* 248320 (Qwen3.8), 154880 (DeepSeek/GLM) */
    uint32_t shortlist_size;        /* 32768 (FR-Spec draft head) */
    uint32_t svd_rank;              /* 108 (DeepSeek), 110 (GLM-5.3 Flagship), 0 (Regime I) */
    uint32_t pinned_shallow_layers; /* e.g. 4..8 layers 100% wired in UMA */
    bool     has_ngram_table;       /* true for PLE (Qwen3.8 L2) and Engram (DeepSeek L1,14) */
    uint32_t ngram_heads_per_tok;   /* e.g. 2..4 (PLE) or 16..32 (Engram) */
    float    mode_a_tau_tok;        /* 0.0f (Qwen3.8), 0.095f (DeepSeek), 0.080f (GLM) */
    uint32_t mode_b_cold_cap;       /* Max cold NVMe expert fetches per dynamic layer */
    size_t   cold_extent_bytes;     /* 256 KiB-aligned cold payload per expert on NVMe */
} StrataModelConfig;

/* Verified Late-2026 Model Config Constructors (#7, #12) */
StrataModelConfig strata_model_config_qwen38_flash_next(void);
StrataModelConfig strata_model_config_deepseek_v41_flash(void);
StrataModelConfig strata_model_config_glm53_flash(void);

/* .strata 256 KiB-Aligned Container Format */
typedef struct StrataExtentRecord {
    uint32_t layer_idx;
    uint32_t expert_idx;
    uint64_t offset_bytes;          /* Guaranteed multiple of STRATA_LBA_ALIGN_BYTES (256 KiB) */
    uint64_t payload_bytes;         /* Actual Tile-rANS / packed byte length */
    uint64_t aligned_bytes;         /* Padded multiple of 256 KiB for F_NOCACHE DMA */
} StrataExtentRecord;

typedef struct StrataContainerHeader {
    uint64_t magic;                 /* STRATA_MAGIC_U64 */
    uint32_t abi_version;           /* STRATA_ABI_VERSION */
    uint32_t num_extents;
    uint64_t data_start_offset;     /* >= 256 KiB aligned */
    uint64_t total_file_bytes;
} StrataContainerHeader;

int strata_pack_write_container(
    const char* path,
    const StrataModelConfig* cfg,
    uint32_t num_extents,
    StrataExtentRecord* extents_out /* caller allocates num_extents if non-NULL */
);

int strata_pack_read_manifest(
    const char* path,
    StrataContainerHeader* hdr_out,
    StrataExtentRecord* extents_out,
    uint32_t max_extents
);

int strata_pack_inspect_safetensors_header(
    const char* path,
    uint32_t* num_tensors_out,
    uint64_t* total_payload_bytes_out
);

/* Routing & Speculative Sampling Primitives */
typedef struct StrataExpertPlan {
    /* Per-token mask: bit j of refine_mask_per_tok[t] is 1 iff token t's j-th top-k
     * expert uses full 4-bit precision (strictly causal in Mode A!). */
    uint16_t refine_mask_per_tok[STRATA_MAX_SPEC_BATCH];
    /* Cold experts requiring NVMe ring DMA for this layer (capped in Mode B) */
    uint16_t cold_fetch_experts[STRATA_MAX_SPEC_BATCH * STRATA_MAX_TOP_K];
    uint32_t num_cold_fetches;
    uint32_t num_uma_hot_hits;
} StrataExpertPlan;

int strata_plan_layer_experts(
    StrataExecMode mode,
    uint32_t batch_B,
    uint32_t top_k,
    const uint16_t* topk_expert_ids, /* [batch_B * top_k] */
    const float*    topk_gates,      /* [batch_B * top_k] */
    const bool*     is_uma_hot,      /* [num_experts] */
    float           mode_a_tau_tok,
    uint32_t        mode_b_cold_cap,
    StrataExpertPlan* plan_out
);

/* FR-Spec (32K Shortlist Draft -> Full Vocab Target) + Leviathan Rejection Sampler */
void strata_frspec_shortlist_probs(
    const float* draft_shortlist_logits, /* [shortlist_size] */
    const uint32_t* shortlist_vocab_ids, /* [shortlist_size] -> [0..vocab_size-1] */
    uint32_t shortlist_size,
    uint32_t full_vocab_size,
    float* full_draft_probs_out          /* [full_vocab_size], 0 outside shortlist */
);

typedef struct StrataVerifyResult {
    uint32_t accepted_count;             /* 1 .. K+1 (always emits >= 1 token) */
    uint32_t emitted_tokens[STRATA_MAX_SPEC_BATCH];
} StrataVerifyResult;

int strata_leviathan_verify_step(
    uint32_t draft_K,                    /* e.g. 5 -> B = K+1 = 6 target positions */
    uint32_t full_vocab_size,
    const uint32_t* draft_tokens,        /* [draft_K] */
    const float* draft_probs_KxV,        /* [draft_K * full_vocab_size] */
    const float* target_probs_BxV,       /* [(draft_K + 1) * full_vocab_size] */
    const float* uniform_rands_K,        /* [draft_K] in [0,1) + 1 bonus rand -> [draft_K+1] */
    StrataVerifyResult* result_out
);

/* ============================================================================
 * 4. GPU Codec & Metal 4 Execution Engine
 *    (Owned by `paradigm-hunter` in `src/gpu/`)
 * ============================================================================ */

typedef struct StrataGpuContext StrataGpuContext;

typedef struct StrataGpuMetrics {
    double kernel_us;
    double effective_gbps;      /* Uncompressed logical weight bytes / kernel_s */
    double physical_dram_gbps;  /* Actual UMA bus bytes read / kernel_s */
    double effective_tflops;
    double sqnr_db;
} StrataGpuMetrics;

/* Host Codec Functions (matching in-register GPU kernels) */
size_t strata_codec_rans_encode_32way(
    const uint8_t* symbols,
    size_t count,
    uint16_t alphabet_size,
    uint8_t* out_buf,
    size_t out_capacity
);

int strata_codec_rans_decode_32way_cpu_ref(
    const uint8_t* compressed_buf,
    size_t compressed_len,
    uint8_t* symbols_out,
    size_t expected_count
);

uint8_t  strata_codec_f32_to_e4m3(float val);
uint16_t strata_codec_e4m3_to_f16_bits(uint8_t e4m3);

void strata_codec_qtip_quantize_expert(
    const float* weights_f32,
    size_t count,
    uint8_t* qtip_base_2b_out,      /* Fixed-stride 2.125 bpw (2b + FP16 scale/128) */
    uint8_t* refine_4lvl_sym_out,   /* 0..3 symbols for Tile-rANS encoding */
    float*   refine_scales_out,
    double*  base_sqnr_db_out,
    double*  refined_sqnr_db_out
);

void strata_codec_w6a16_pack(
    const float* weights_f32,
    size_t count,                   /* Multiple of 32 */
    uint8_t* packed_w6_out          /* 0.78125 B/weight (24B 6-bit + 2B FP16 scale per 32) */
);

/* GPU Lifecycle & Kernel Dispatch */
StrataGpuContext* strata_gpu_init(void);
void strata_gpu_destroy(StrataGpuContext* ctx);

/* Returns Objective-C `id<MTLDevice>` cast to `void*` for zero-copy sharing with `strata_hw` */
void* strata_gpu_get_mtl_device(StrataGpuContext* ctx);

/* Benchmarks / executes Spine E4M3 + 32-way Tile-rANS in-register unpack -> FP16 GEMM */
int strata_gpu_bench_spine_e4m3_rans(
    StrataGpuContext* ctx,
    uint32_t batch_B,
    uint32_t d_in,
    uint32_t d_out,
    uint32_t warmup_iters,
    uint32_t timed_iters,
    bool* bit_exact_passed_out,
    StrataGpuMetrics* metrics_out
);

/* Benchmarks / executes Slot-Indexed MoE Layer (Mode A stationary & Mode B zero-stall fallback) */
int strata_gpu_bench_moe_layer(
    StrataGpuContext* ctx,
    const StrataModelConfig* cfg,
    StrataExecMode mode,
    uint32_t batch_B,
    void* external_ring_mtl_buffer,     /* Optional shared MTLBuffer from strata_hw_ring (NULL = internal) */
    const uint32_t* external_slot_ready,/* Optional coherent readiness table from strata_hw_ring */
    uint32_t simulated_unready_cold,    /* Tests Mode B 0.00 ms fallback vs Mode A wait */
    uint32_t warmup_iters,
    uint32_t timed_iters,
    StrataGpuMetrics* metrics_out
);

/* ============================================================================
 * 5. Hardware Co-Engines: ARMv9.2 SME2, Pre-Wired NVMe Ring & 16-Core ANE Probe
 *    (Owned by `silicon-arch` in `src/hw/`)
 * ============================================================================ */

typedef struct StrataHwContext StrataHwContext;
typedef struct StrataAneProbe  StrataAneProbe;

typedef struct StrataSme2IoMetrics {
    uint32_t total_lookups;
    uint32_t lfu_cache_hits;
    uint32_t nvme_pread_ios;
    double   phase1_io_us;          /* Isolated QD64 4 KiB preadv latency before expert DMA */
    double   sme2_compute_us;       /* LUTI4 + ZT0 + FMOPA/BFMOPA latency in CPU L2 */
    double   sme2_effective_gflops;
    bool     hardware_sme2_active;
} StrataSme2IoMetrics;

typedef struct StrataRingIoMetrics {
    uint32_t extents_streamed;
    uint64_t bytes_streamed;
    double   dma_wall_us;
    double   sustained_nvme_gbps;
    long     major_page_faults;     /* Must equal 0 (ru_majflt delta) */
    long     minor_page_faults;
} StrataRingIoMetrics;

StrataHwContext* strata_hw_init(
    void* shared_mtl_device,        /* Pass strata_gpu_get_mtl_device(gpu) or NULL */
    size_t ring_slot_bytes,         /* Must be multiple of STRATA_LBA_ALIGN_BYTES */
    uint32_t num_ring_slots,
    size_t cpu_ngram_cache_bytes    /* Non-IOGPU mlock pool (keep <= 256 MiB in PoC tests) */
);
void strata_hw_destroy(StrataHwContext* ctx);

void*     strata_hw_ring_get_mtl_buffer(StrataHwContext* ctx);
uint32_t* strata_hw_ring_get_ready_table(StrataHwContext* ctx);

/* Phase 1 Isolated N-Gram Gather (PLE / Engram) + ARMv9.2 SME2 LUTI4/FMOPA Gating */
int strata_hw_exec_ngram_phase1(
    StrataHwContext* ctx,
    int nvme_fd,                    /* Opened with F_NOCACHE on 4 KiB-aligned table shard */
    uint64_t table_file_bytes,
    const uint64_t* ngram_hashes,   /* [batch_B * heads_per_tok] */
    uint32_t batch_B,
    uint32_t heads_per_tok,
    uint32_t d_model,
    uint16_t* gated_fp16_out,       /* [batch_B * d_model] FP16 output in coherent SLC */
    StrataSme2IoMetrics* metrics_out
);

/* Phase 2 Pre-Wired MTLIOCommandQueue (F_NOCACHE) Cold-Expert Extent Streaming */
int strata_hw_ring_stream_extents(
    StrataHwContext* ctx,
    const char* container_path,
    const uint64_t* aligned_offsets,/* [num_transfers], multiples of 256 KiB */
    const size_t*   aligned_sizes,  /* [num_transfers], multiples of 256 KiB */
    const uint32_t* ring_slots,     /* [num_transfers] */
    const uint32_t* target_epochs,  /* [num_transfers] */
    uint32_t num_transfers,
    bool wait_completion,
    StrataRingIoMetrics* metrics_out
);

int strata_hw_ring_wait_slot(
    StrataHwContext* ctx,
    uint32_t ring_slot,
    uint32_t target_epoch,
    double* wait_us_out
);

/* 16-Core ANE Static [1, d, 1, S=16] Multi-Horizon Router Window Probe */
StrataAneProbe* strata_ane_probe_init(
    uint32_t d_model,
    uint32_t bottleneck_r,          /* e.g. 256 */
    uint32_t num_experts,           /* e.g. 384 or 512 */
    uint32_t num_windows            /* e.g. 2..10 in PoC */
);
void strata_ane_probe_destroy(StrataAneProbe* probe);

int strata_ane_probe_predict_window(
    StrataAneProbe* probe,
    uint32_t window_idx,
    const uint16_t* hidden_fp16,    /* [batch_B * d_model], internally padded to S=16 */
    uint32_t batch_B,
    float* horizon_scores_out,      /* [4 * num_experts] for L0+4 .. L0+7 */
    double* latency_us_out,
    bool* ran_on_ane_out
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* !__METAL_VERSION__ */
#endif /* STRATA_ABI_H */

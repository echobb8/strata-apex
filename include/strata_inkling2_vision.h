#ifndef STRATA_INKLING2_VISION_H
#define STRATA_INKLING2_VISION_H

/*
 * STRATA-M5 v2.0 — Native Apple M5 Max Metal 4.1 Vision Encoder ABI for `thinkingmachines/Inkling-2-Mini`
 *
 * Official Verified Architecture (`model.visual.*`, cross-verified against HF `transformers`
 * `src/transformers/models/inkling2/{modeling,image_processing,processing}_inkling2.py` and `llama.cpp`
 * `tools/mtmd/models/inkling2v.cpp`):
 *   - Image Preprocessing (`Inkling2ImageProcessor`):
 *       * Decode PNG/JPEG/WebP/GIF via native macOS `ImageIO`/`CoreGraphics` (zero external deps).
 *       * If image has alpha channel: composite unpremultiplied RGBA onto 8x8 checkerboard
 *         (`(row/8 + col/8) % 2 == 0 ? 204 : 153`) with rounded integer blend `((c*a + bg*(255-a) + 127) / 255)`.
 *       * No resizing for normal images (aspect-ratio downscale ONLY if padded tokens > `max_image_tokens = 4096`).
 *       * Bottom-right pad to multiple of `40` (`native_patch_size=20 * adapter_merge_size=2`) with `-1.0f`
 *         on the `[0..255]` scale (`grid_h = ceil(H/40)`, `grid_w = ceil(W/40)`).
 *       * Rescale & normalize: `val = (pixel * (1.0/255.0) - 0.5) / 0.5` (padded `-1.0` -> `-257/255`).
 *       * Patch layout: spatial-merge-block order `(block_row, block_col, in_row[0..1], in_col[0..1])`,
 *         each native 20x20 patch flattened in HWC `[py=0..19, px=0..19, c=0..2]` (`1200` floats).
 *   - Embeddings + Pre-LN:
 *       * `patch_emb = patch_pixels @ patch_embedding.weight^T` (`[num_patches, 1024]`, bias=False).
 *       * Learned 32x32 `position_embedding [1024, 1024]` resampled to `(2*grid_h, 2*grid_w)` via
 *         separable bilinear interpolation (`align_corners=False`, border clamp) and added to patches.
 *       * Prepend CLS token `embeddings[0] = class_embedding + cls_position_embedding` (`S = 1 + 4*grid_h*grid_w`).
 *       * Apply `ln_pre` (`LayerNorm(1024, eps=1e-5)`) to all `S` tokens (CLS + patches).
 *   - 23 ViT Encoder Layers (`hidden_size=1024`, `num_heads=16`, `head_dim=64`, `mlp_dim=4096`):
 *       * 2D Axial Pairwise RoPE (`rope_theta=10000.0`, `spatial_dim=32`, 16 inv_freqs):
 *         - CLS at `(pos_h=0, pos_w=0)` (identity); patch at `(row, col)` gets 1-indexed `(row+1, col+1)`.
 *         - Width (`pos_w`) rotates dims `[0..31]` (`16` adjacent pairs `(2m, 2m+1)`);
 *           Height (`pos_h`) rotates dims `[32..63]` (`16` adjacent pairs `(2m, 2m+1)`).
 *       * Pre-LN `ln_1` -> QKV linear w/ bias -> 2D RoPE on Q,K -> full bidirectional SDPA (`scale = 1/8`)
 *         -> `out_proj` w/ bias -> LayerScale `ls_1.gamma` -> residual.
 *       * Pre-LN `ln_2` -> `mlp.c_fc` w/ bias -> exact `GELU` (`0.5*x*(1 + erf(x/sqrt(2)))`)
 *         -> `mlp.c_proj` w/ bias -> LayerScale `ls_2.gamma` -> residual.
 *   - 2x2 Spatial Merger & Projector:
 *       * Drop token 0 (CLS); patches are already contiguous in `[TL, TR, BL, BR]` block order -> `[N_vis, 4096]`.
 *       * `proj_in [4096, 4096]` (no bias) -> exact `GELU` -> `proj_out [2304, 4096]` (no bias, no final norm).
 *   - LLM Token Placeholder Mapping (`chat_template.jinja` + `Inkling2Processor`):
 *       * `<|message_user|><|content_image|>` + `N_vis` copies of `<|unused_200054|>` (`token_id = 200054`)
 *         + `<|end_message|>`, where `N_vis = grid_h * grid_w`.
 *       * Projected `[N_vis, 2304]` FP32 embeddings replace the POST-`embed_norm` hidden states at those `N_vis`
 *         token positions (`strata_inkling2_gpu_prefill_with_images`).
 *   - Standalone Sidecar Format (`inkling2_vision.strata`, `~639 MB`, 16 KiB page-aligned zero-copy UMA):
 *       * Matrices: lossless IEEE FP16 (`half`, max abs weight `< 1.0`, 11-bit mantissa > BF16 8-bit).
 *       * Norms, biases, LayerScale gammas, class/position embeddings: lossless IEEE FP32 (`float`).
 *
 * Shared between C/C++/ObjC++ host code and Metal 4.1 MSL kernels via `// @@STRATA_INKLING2_VISION_HEADER@@`.
 */

#ifndef __METAL_VERSION__
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef uint8_t  sikv_u8;
typedef uint16_t sikv_u16;
typedef uint32_t sikv_u32;
typedef uint64_t sikv_u64;
#else
typedef uchar    sikv_u8;
typedef ushort   sikv_u16;
typedef uint     sikv_u32;
typedef ulong    sikv_u64;
#endif

/* ============================================================================
 * 1. Format Constants & Official Inkling-2-Mini Vision Architecture Constants
 * ============================================================================ */

#define STRATA_INKLING2_VISION_MAGIC_U64              0x30564B49354D5453ULL /* "STM5IKV0" */
#define STRATA_INKLING2_VISION_FORMAT_VERSION         1U
#define STRATA_INKLING2_VISION_PAGE_BYTES             16384ULL              /* 16 KiB Apple Silicon VM page */
#define STRATA_INKLING2_VISION_HEADER_PADDED_BYTES    16384U                /* 16 KiB padded header */
#define STRATA_INKLING2_VISION_TENSOR_ALIGN           256ULL                /* 256 B Metal buffer align */
#define STRATA_INKLING2_VISION_MAX_LAYERS             32U                   /* 23 layers in Inkling-2 */

#define STRATA_INKLING2_VISION_DTYPE_NONE             0U
#define STRATA_INKLING2_VISION_DTYPE_F32              1U                    /* IEEE-754 float32 */
#define STRATA_INKLING2_VISION_DTYPE_F16              2U                    /* IEEE-754 float16 */

#define STRATA_INKLING2_VISION_HIDDEN_SIZE            1024U
#define STRATA_INKLING2_VISION_NUM_LAYERS             23U
#define STRATA_INKLING2_VISION_NUM_HEADS              16U
#define STRATA_INKLING2_VISION_HEAD_DIM               64U
#define STRATA_INKLING2_VISION_MLP_DIM                4096U
#define STRATA_INKLING2_VISION_PATCH_SIZE             20U
#define STRATA_INKLING2_VISION_MERGE_SIZE             2U
#define STRATA_INKLING2_VISION_CELL_PX                40U                   /* patch_size * merge_size */
#define STRATA_INKLING2_VISION_PATCH_DIM              1200U                 /* 20 * 20 * 3 (HWC) */
#define STRATA_INKLING2_VISION_POSEMB_GRID            32U
#define STRATA_INKLING2_VISION_PROJ_HIDDEN            4096U
#define STRATA_INKLING2_VISION_TEXT_HIDDEN            2304U
#define STRATA_INKLING2_VISION_MAX_TOKENS             4096U                 /* matches llama.cpp limit */
#define STRATA_INKLING2_VISION_ROPE_THETA             10000.0f
#define STRATA_INKLING2_VISION_LN_EPS                 1.0e-5f

/* Canonical image placeholder token ID (`<|unused_200054|>`) */
#define STRATA_INKLING2_TOKEN_IMAGE_PLACEHOLDER       200054U
#define STRATA_INKLING2_VISION_FILENAME               "inkling2_vision.strata"
#ifndef __METAL_VERSION__
#include "strata_paths.h"
#define STRATA_INKLING2_VISION_DEFAULT_FAST_PATH      strata_apex_bundle_path("inkling-2-mini-fast/inkling2_vision.strata")
#define STRATA_INKLING2_VISION_DEFAULT_V1_PATH        strata_apex_bundle_path("inkling-2-mini/inkling2_vision.strata")
#endif

/* ============================================================================
 * 2. On-Disk Sidecar Binary Format Descriptors (POD, 8-Byte Aligned)
 * ============================================================================ */

typedef struct StrataInkling2VisionTensorSlice {
    sikv_u64 offset;
    sikv_u64 bytes;
    sikv_u32 elem_count;
    sikv_u32 dtype;   /* STRATA_INKLING2_VISION_DTYPE_F32 or F16 */
    sikv_u32 shape0;  /* Rows (outer dim) */
    sikv_u32 shape1;  /* Cols (inner dim, 1 for 1D) */
} StrataInkling2VisionTensorSlice;

typedef struct StrataInkling2VisionConfig {
    sikv_u32 hidden_size;           /* 1024 */
    sikv_u32 num_hidden_layers;     /* 23 */
    sikv_u32 num_attention_heads;   /* 16 */
    sikv_u32 head_dim;              /* 64 */
    sikv_u32 intermediate_size;     /* 4096 */
    sikv_u32 native_patch_size;     /* 20 */
    sikv_u32 adapter_merge_size;    /* 2 */
    sikv_u32 posemb_grid_size;      /* 32 */
    sikv_u32 projection_hidden_dim; /* 4096 */
    sikv_u32 text_hidden_size;      /* 2304 */
    sikv_u32 num_channels;          /* 3 */
    sikv_u32 projector_bias;        /* 0 */
    float    layer_norm_eps;        /* 1e-5f */
    float    rope_theta;            /* 10000.0f */
} StrataInkling2VisionConfig;

typedef struct StrataInkling2VisionLayerDesc {
    StrataInkling2VisionTensorSlice ln1_w;      /* F32 [1024] */
    StrataInkling2VisionTensorSlice ln1_b;      /* F32 [1024] */
    StrataInkling2VisionTensorSlice qkv_w;      /* F16 [3072, 1024] (fused [Q; K; V]) */
    StrataInkling2VisionTensorSlice qkv_b;      /* F32 [3072]       (fused [Q; K; V]) */
    StrataInkling2VisionTensorSlice out_w;      /* F16 [1024, 1024] */
    StrataInkling2VisionTensorSlice out_b;      /* F32 [1024] */
    StrataInkling2VisionTensorSlice ls1_gamma;  /* F32 [1024] */
    StrataInkling2VisionTensorSlice ln2_w;      /* F32 [1024] */
    StrataInkling2VisionTensorSlice ln2_b;      /* F32 [1024] */
    StrataInkling2VisionTensorSlice fc1_w;      /* F16 [4096, 1024] */
    StrataInkling2VisionTensorSlice fc1_b;      /* F32 [4096] */
    StrataInkling2VisionTensorSlice fc2_w;      /* F16 [1024, 4096] */
    StrataInkling2VisionTensorSlice fc2_b;      /* F32 [1024] */
    StrataInkling2VisionTensorSlice ls2_gamma;  /* F32 [1024] */
} StrataInkling2VisionLayerDesc;

typedef struct StrataInkling2VisionHeader {
    sikv_u64 magic;
    sikv_u32 format_version;
    sikv_u32 header_bytes;
    sikv_u64 total_file_bytes;
    sikv_u32 num_tensors;
    sikv_u32 reserved0;

    StrataInkling2VisionConfig cfg;

    /* Embeddings & Pre-LN */
    StrataInkling2VisionTensorSlice patch_emb_w;  /* F16 [1024, 1200] */
    StrataInkling2VisionTensorSlice class_emb;    /* F32 [1024] */
    StrataInkling2VisionTensorSlice cls_pos_emb;  /* F32 [1024] */
    StrataInkling2VisionTensorSlice pos_emb;      /* F32 [1024, 1024] (32x32 grid) */
    StrataInkling2VisionTensorSlice ln_pre_w;     /* F32 [1024] */
    StrataInkling2VisionTensorSlice ln_pre_b;     /* F32 [1024] */

    /* 2x2 Spatial Merger / Projector */
    StrataInkling2VisionTensorSlice proj_in_w;    /* F16 [4096, 4096] */
    StrataInkling2VisionTensorSlice proj_out_w;   /* F16 [2304, 4096] */

    /* 23 ViT Encoder Layers */
    StrataInkling2VisionLayerDesc layers[STRATA_INKLING2_VISION_MAX_LAYERS];
} StrataInkling2VisionHeader;

/* ============================================================================
 * 3. Shared Metal 4.1 Vision Kernel Parameter Structs
 * ============================================================================ */

typedef struct SikvGemmParams {
    sikv_u32 M_valid;    /* Logical token rows */
    sikv_u32 M_padded;   /* Padded token rows (multiple of 32) */
    sikv_u32 N;          /* Output features */
    sikv_u32 K;          /* Input features */
    sikv_u32 has_bias;   /* 1 if bias_f32 is non-null / enabled */
    sikv_u32 apply_gelu; /* 1 to apply exact erf GELU before storing */
    sikv_u32 out_f16;    /* 1 to write F16 output, 0 to write F32 output */
    sikv_u32 reserved;
} SikvGemmParams;

typedef struct SikvEmbedAssembleParams {
    sikv_u32 num_patches; /* 4 * grid_h * grid_w */
    sikv_u32 S;           /* 1 + num_patches */
    sikv_u32 S_padded;    /* multiple of 32 */
    sikv_u32 D;           /* 1024 */
    sikv_u32 grid_h;      /* merged token rows (H_p / 2) */
    sikv_u32 grid_w;      /* merged token cols (W_p / 2) */
    sikv_u32 posemb_grid; /* 32 */
    float    ln_eps;      /* 1e-5f */
} SikvEmbedAssembleParams;

typedef struct SikvLayerNormParams {
    sikv_u32 S;           /* 1 + num_patches */
    sikv_u32 S_padded;    /* multiple of 32 */
    sikv_u32 D;           /* 1024 */
    float    eps;         /* 1e-5f */
} SikvLayerNormParams;

typedef struct SikvQkvRopeParams {
    sikv_u32 S;           /* 1 + num_patches */
    sikv_u32 S_padded;    /* multiple of 32 */
    sikv_u32 D;           /* 1024 */
    sikv_u32 num_heads;   /* 16 */
    sikv_u32 head_dim;    /* 64 */
    sikv_u32 grid_w;      /* merged token cols */
    float    rope_theta;  /* 10000.0f */
    sikv_u32 reserved;
} SikvQkvRopeParams;

typedef struct SikvAttnParams {
    sikv_u32 S;           /* 1 + num_patches */
    sikv_u32 S_padded;    /* multiple of 32 */
    sikv_u32 num_heads;   /* 16 */
    sikv_u32 head_dim;    /* 64 */
    float    scale;       /* 1.0f / sqrt(64.0f) = 0.125f */
    sikv_u32 reserved0;
    sikv_u32 reserved1;
    sikv_u32 reserved2;
} SikvAttnParams;

typedef struct SikvResScaleLnParams {
    sikv_u32 S;           /* 1 + num_patches */
    sikv_u32 S_padded;    /* multiple of 32 */
    sikv_u32 D;           /* 1024 */
    sikv_u32 do_ln;       /* 1 to also emit post-residual LayerNorm F16, 0 for residual only */
    float    ln_eps;      /* 1e-5f */
    sikv_u32 reserved;
} SikvResScaleLnParams;

/* ============================================================================
 * 4. Host C / C++ / ObjC++ API
 * ============================================================================ */

#ifndef __METAL_VERSION__

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
#include <vector>
#endif

typedef struct StrataInkling2VisionEngine StrataInkling2VisionEngine;

typedef struct StrataInkling2VisionInfo {
    uint32_t ready;             /* 1 if loaded and Metal pipelines ready */
    uint32_t metal_cache_hit;   /* 1 if PSOs loaded from MTLBinaryArchive cache */
    double   pso_load_ms;       /* Wall-clock ms to load/compile Metal PSOs */
    uint64_t file_bytes;        /* Size of mapped `inkling2_vision.strata` in bytes */
    uint32_t hidden_size;       /* 1024 */
    uint32_t text_hidden_size;  /* 2304 */
    uint32_t num_layers;        /* 23 */
    uint32_t num_heads;         /* 16 */
    uint32_t patch_size;        /* 20 */
    uint32_t merge_size;        /* 2 */
    uint32_t posemb_grid_size;  /* 32 */
    uint32_t max_image_tokens;  /* 4096 */
    double   last_encode_ms;    /* Most recent image encode latency (ms) */
    uint32_t last_num_tokens;   /* Most recent image visual token count (grid_h * grid_w) */
    uint32_t last_grid_h;       /* Most recent image merged grid height */
    uint32_t last_grid_w;       /* Most recent image merged grid width */
    char     device_name[128];
    char     vision_path[512];
} StrataInkling2VisionInfo;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Convert all 422 `model.visual.*` BF16 tensors from a HuggingFace Inkling-2-Mini directory
 * (`config.json` + `model-00001-of-00001.safetensors` or sharded `*.safetensors`) into a
 * standalone 16 KiB page-aligned zero-copy sidecar file `inkling2_vision.strata` (~639 MB).
 * Validates 0 NaN/Inf across all converted weights.
 * Returns 0 on success, negative error code on failure.
 */
int strata_inkling2_convert_vision_from_hf(
    const char* hf_dir,
    const char* out_vision_strata_path
);

/*
 * Locate `inkling2_vision.strata` for a given bundle directory (or `.strata` bundle path),
 * with automatic fallback to `<cache_root>/inkling-2-mini-fast/inkling2_vision.strata`
 * and `<cache_root>/inkling-2-mini/inkling2_vision.strata` so `--no-refusal` and
 * v1 bundles find the sidecar automatically.
 * Returns 1 if a valid readable file was found (writing NUL-terminated path into `out_path`),
 * or 0 if not found.
 */
int strata_inkling2_vision_resolve_path(
    const char* bundle_dir,
    char*       out_path,
    size_t      out_cap
);

/*
 * Create a Metal 4.1 Vision Engine on the system default Metal device (pure C/C++ callable).
 * Returns NULL on failure.
 */
StrataInkling2VisionEngine* strata_inkling2_vision_create_default(
    const char* vision_strata_path
);

/* Query engine metadata, Metal binary archive cache hit, PSO load latency, and last encode stats. */
int strata_inkling2_vision_get_info(
    const StrataInkling2VisionEngine* eng,
    StrataInkling2VisionInfo*         out_info
);

/* Release all GPU buffers, Metal pipelines, and mmap mappings (safe to call with NULL). */
void strata_inkling2_vision_destroy(
    StrataInkling2VisionEngine* eng
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef __OBJC__
/*
 * Create a Metal 4.1 Vision Engine using an existing `id<MTLDevice>` (or system default if nil),
 * mapping `vision_strata_path` zero-copy and loading/caching PSOs via `StrataMetalPsoCache`
 * (`engine_tag = "inkling2_vision"`).
 * Returns NULL on failure.
 */
StrataInkling2VisionEngine* strata_inkling2_vision_create(
    id<MTLDevice> dev_or_null,
    const char*   vision_strata_path
);
#endif

#ifdef __cplusplus
/*
 * Decode base64 data (accepts either raw base64 or RFC 2397 `data:image/...;base64,<payload>` URI,
 * ignoring ASCII whitespace). Returns 0 on success, negative on invalid input.
 */
int strata_inkling2_vision_decode_base64(
    const char*           b64_or_data_uri,
    size_t                len,
    std::vector<uint8_t>* out_bytes
);

/*
 * Encode an in-memory encoded image (PNG, JPEG, WebP, GIF, TIFF, BMP, HEIC via native macOS ImageIO)
 * into projected Inkling-2-Mini visual embeddings `[N_vis * 2304]` FP32 (post-`embed_norm` space,
 * ready for `strata_inkling2_gpu_prefill_with_images`).
 * Returns 0 on success, negative error code on failure.
 */
int strata_inkling2_vision_encode_image_bytes(
    StrataInkling2VisionEngine* eng,
    const uint8_t*              img_bytes,
    size_t                      img_len,
    std::vector<float>*         out_emb_f32,
    uint32_t*                   out_num_tokens,
    uint32_t*                   out_grid_h,
    uint32_t*                   out_grid_w
);

/*
 * Convenience wrapper loading an image from a local file path (also strips optional `file://` prefix)
 * and calling `strata_inkling2_vision_encode_image_bytes`.
 */
int strata_inkling2_vision_encode_image_file(
    StrataInkling2VisionEngine* eng,
    const char*                 image_path,
    std::vector<float>*         out_emb_f32,
    uint32_t*                   out_num_tokens,
    uint32_t*                   out_grid_h,
    uint32_t*                   out_grid_w
);

/*
 * Encode raw interleaved RGB uint8 pixels (`[height, width, 3]`, row-major, 0..255) directly
 * without ImageIO decoding. Useful for synthetic tests and reference parity verification.
 */
int strata_inkling2_vision_encode_rgb_u8(
    StrataInkling2VisionEngine* eng,
    const uint8_t*              rgb_u8,
    uint32_t                    width,
    uint32_t                    height,
    std::vector<float>*         out_emb_f32,
    uint32_t*                   out_num_tokens,
    uint32_t*                   out_grid_h,
    uint32_t*                   out_grid_w
);
#endif /* __cplusplus */

#endif /* !__METAL_VERSION__ */

#endif /* STRATA_INKLING2_VISION_H */

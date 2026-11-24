/*
 * STRATA-M5 v2.0 — Native Apple M5 Max Metal 4.1 Vision Encoder, Sidecar Converter,
 * Zero-Copy UMA Loader & Native ImageIO/CoreGraphics Preprocessor for `thinkingmachines/Inkling-2-Mini`
 *
 * File: src/model/strata_inkling2_vision.mm
 *
 * Implements `include/strata_inkling2_vision.h`:
 *   1. Zero-dep HF safetensors -> 16 KiB-aligned `inkling2_vision.strata` converter (`~639 MB`):
 *        - Extracts all 422 `model.visual.*` BF16 tensors from `model-*.safetensors`.
 *        - Fuses Q/K/V weights `[3072, 1024]` FP16 + biases `[3072]` FP32 per layer.
 *        - Preserves all 1D norms/biases/LayerScale/embeddings in lossless FP32, and all 2D weight
 *          matrices in lossless IEEE FP16 (max abs weight < 1.0, 11-bit mantissa > BF16 8-bit).
 *        - Validates 0 NaN / 0 Inf across all converted elements; atomic `.tmp.<pid>` -> `rename()`.
 *   2. Native macOS `ImageIO` / `CoreGraphics` decoder + exact `Inkling2ImageProcessor` preprocessor:
 *        - Supports PNG, JPEG, WebP, GIF, TIFF, BMP, HEIC + RFC 2397 base64 data URIs.
 *        - Alpha-channel compositing onto 8x8 checkerboard (204 / 153).
 *        - Aspect-preserving downscale ONLY when padded token count > 4096.
 *        - Bottom-right constant `-1.0f` padding to multiple of 40, `(p/255 - 0.5)/0.5` normalization,
 *          and spatial-merge-block `(br, bc, in_r, in_c)` HWC `[20, 20, 3]` patch packing.
 *   3. Metal 4.1 NAX Vision Encoder:
 *        - Zero-copy `mmap` + `newBufferWithBytesNoCopy` weight mapping.
 *        - All 9 PSOs cached via `StrataMetalPsoCache` (`engine_tag = "inkling2_vision"`),
 *          `MTLMathModeSafe` + `MTLMathFloatingPointFunctionsFast`, 0 spill-risk PSOs (`<1024`).
 *        - Single `MTLCommandBuffer` encoding patch_emb -> CLS + bilinear 32x32 pos_emb + ln_pre ->
 *          23 ViT layers (2D Axial Pairwise RoPE + O(S) FlashAttention + LayerScale + erf-GELU) ->
 *          drop CLS + 2x2 merge -> proj_in -> erf-GELU -> proj_out `[N_vis, 2304]` FP32.
 */

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include "../../include/strata_inkling2_vision.h"
#include "../../include/strata_embedded_shaders.h"
#include "../../include/strata_metal_cache.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

static_assert(sizeof(StrataInkling2VisionTensorSlice) == 32,
              "StrataInkling2VisionTensorSlice ABI size must be 32 bytes");
static_assert(sizeof(StrataInkling2VisionHeader) <= STRATA_INKLING2_VISION_HEADER_PADDED_BYTES,
              "StrataInkling2VisionHeader must fit within 16 KiB padded header");

namespace {

/* ============================================================================
 * 1. Low-Level Numeric, Alignment & File Utilities
 * ============================================================================ */

static inline uint64_t sikv_align_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1ULL) & ~(align - 1ULL);
}

static inline uint32_t sikv_align_up_u32(uint32_t v, uint32_t align) {
    return (v + align - 1U) & ~(align - 1U);
}

static inline float sikv_bf16_to_f32(uint16_t bf) {
    uint32_t u = static_cast<uint32_t>(bf) << 16U;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

static inline float sikv_f16_to_f32(uint16_t h) {
    __fp16 fp;
    std::memcpy(&fp, &h, sizeof(uint16_t));
    return static_cast<float>(fp);
}

static inline uint16_t sikv_f32_to_f16(float f) {
    const float clamped = std::max(-65504.0f, std::min(65504.0f, f));
    __fp16 fp = static_cast<__fp16>(clamped);
    uint16_t h = 0U;
    std::memcpy(&h, &fp, sizeof(uint16_t));
    return h;
}

static bool sikv_file_exists_nonempty(const char* path) {
    if (!path || path[0] == '\0') return false;
    struct stat st;
    if (::stat(path, &st) != 0) return false;
    return S_ISREG(st.st_mode) && st.st_size >= static_cast<off_t>(STRATA_INKLING2_VISION_HEADER_PADDED_BYTES);
}

static bool sikv_validate_sidecar_magic(const char* path) {
    if (!sikv_file_exists_nonempty(path)) return false;
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    uint64_t magic = 0ULL;
    uint32_t ver   = 0U;
    const ssize_t r1 = ::pread(fd, &magic, sizeof(magic), 0);
    const ssize_t r2 = ::pread(fd, &ver,   sizeof(ver),   8);
    ::close(fd);
    return (r1 == static_cast<ssize_t>(sizeof(magic))) &&
           (r2 == static_cast<ssize_t>(sizeof(ver))) &&
           (magic == STRATA_INKLING2_VISION_MAGIC_U64) &&
           (ver == STRATA_INKLING2_VISION_FORMAT_VERSION);
}

static std::string sikv_read_text_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.good()) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static std::string sikv_resolve_repo_path(const char* rel_from_src_model, const char* fallback_from_cwd) {
    std::string self_file = __FILE__;
    const size_t slash = self_file.find_last_of('/');
    if (slash != std::string::npos) {
        std::string candidate = self_file.substr(0, slash + 1) + rel_from_src_model;
        std::ifstream test(candidate);
        if (test.good()) return candidate;
    }
    return std::string(fallback_from_cwd);
}

static bool sikv_pwrite_all(int fd, const void* buf, size_t len, uint64_t offset) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t done = 0U;
    while (done < len) {
        const ssize_t n = ::pwrite(fd, p + done, len - done, static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

static bool sikv_pread_all(int fd, void* buf, size_t len, uint64_t offset) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t done = 0U;
    while (done < len) {
        const ssize_t n = ::pread(fd, p + done, len - done, static_cast<off_t>(offset + done));
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

/* ============================================================================
 * 2. Minimal Safetensors Indexer & HF `config.json` Parser (Zero External Deps)
 * ============================================================================ */

enum class SikvSrcDtype : uint32_t {
    Unknown = 0,
    BF16    = 1,
    F16     = 2,
    F32     = 3,
};

struct SikvSrcTensor {
    std::string           file_path;
    uint64_t              file_offset = 0ULL; // byte offset inside file (8 + hdr_len + data_off0)
    uint64_t              byte_len    = 0ULL;
    SikvSrcDtype          dtype       = SikvSrcDtype::Unknown;
    std::vector<uint64_t> shape;
};

static bool sikv_index_safetensors_file(
    const std::string& path,
    std::unordered_map<std::string, SikvSrcTensor>& out_map
) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    uint64_t hdr_len = 0ULL;
    if (!sikv_pread_all(fd, &hdr_len, sizeof(hdr_len), 0ULL) || hdr_len == 0ULL || hdr_len > (256ULL << 20U)) {
        ::close(fd);
        return false;
    }

    std::string hdr_json(static_cast<size_t>(hdr_len), '\0');
    if (!sikv_pread_all(fd, hdr_json.data(), static_cast<size_t>(hdr_len), 8ULL)) {
        ::close(fd);
        return false;
    }
    ::close(fd);

    @autoreleasepool {
        NSData* data = [NSData dataWithBytesNoCopy:const_cast<char*>(hdr_json.data())
                                            length:hdr_json.size()
                                      freeWhenDone:NO];
        NSError* err = nil;
        id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
        if (!obj || ![obj isKindOfClass:[NSDictionary class]]) {
            return false;
        }
        NSDictionary* dict = (NSDictionary*)obj;
        const uint64_t data_base = 8ULL + hdr_len;
        for (id key_obj in dict) {
            if (![key_obj isKindOfClass:[NSString class]]) continue;
            NSString* key_ns = (NSString*)key_obj;
            if (![key_ns hasPrefix:@"model.visual."]) continue;
            id val_obj = dict[key_ns];
            if (![val_obj isKindOfClass:[NSDictionary class]]) continue;
            NSDictionary* info = (NSDictionary*)val_obj;
            NSString* dt_ns = info[@"dtype"];
            NSArray*  sh_ns = info[@"shape"];
            NSArray*  of_ns = info[@"data_offsets"];
            if (![dt_ns isKindOfClass:[NSString class]] ||
                ![sh_ns isKindOfClass:[NSArray class]] ||
                ![of_ns isKindOfClass:[NSArray class]] ||
                [of_ns count] != 2) {
                continue;
            }
            SikvSrcTensor t;
            t.file_path = path;
            const uint64_t off0 = [of_ns[0] unsignedLongLongValue];
            const uint64_t off1 = [of_ns[1] unsignedLongLongValue];
            if (off1 < off0) continue;
            t.file_offset = data_base + off0;
            t.byte_len    = off1 - off0;
            if ([dt_ns isEqualToString:@"BF16"])      t.dtype = SikvSrcDtype::BF16;
            else if ([dt_ns isEqualToString:@"F16"])  t.dtype = SikvSrcDtype::F16;
            else if ([dt_ns isEqualToString:@"F32"])  t.dtype = SikvSrcDtype::F32;
            else                                      t.dtype = SikvSrcDtype::Unknown;

            t.shape.reserve([sh_ns count]);
            for (id dim_num in sh_ns) {
                t.shape.push_back([dim_num unsignedLongLongValue]);
            }
            out_map[std::string([key_ns UTF8String])] = std::move(t);
        }
    }
    return true;
}

static bool sikv_parse_vision_config(const std::string& hf_dir, StrataInkling2VisionConfig& cfg) {
    cfg = StrataInkling2VisionConfig{};
    cfg.hidden_size           = STRATA_INKLING2_VISION_HIDDEN_SIZE;
    cfg.num_hidden_layers     = STRATA_INKLING2_VISION_NUM_LAYERS;
    cfg.num_attention_heads   = STRATA_INKLING2_VISION_NUM_HEADS;
    cfg.head_dim              = STRATA_INKLING2_VISION_HEAD_DIM;
    cfg.intermediate_size     = STRATA_INKLING2_VISION_MLP_DIM;
    cfg.native_patch_size     = STRATA_INKLING2_VISION_PATCH_SIZE;
    cfg.adapter_merge_size    = STRATA_INKLING2_VISION_MERGE_SIZE;
    cfg.posemb_grid_size      = STRATA_INKLING2_VISION_POSEMB_GRID;
    cfg.projection_hidden_dim = STRATA_INKLING2_VISION_PROJ_HIDDEN;
    cfg.text_hidden_size      = STRATA_INKLING2_VISION_TEXT_HIDDEN;
    cfg.num_channels          = 3U;
    cfg.projector_bias        = 0U;
    cfg.layer_norm_eps        = STRATA_INKLING2_VISION_LN_EPS;
    cfg.rope_theta            = STRATA_INKLING2_VISION_ROPE_THETA;

    const std::string cfg_path = hf_dir + "/config.json";
    const std::string txt = sikv_read_text_file(cfg_path);
    if (txt.empty()) return false;

    @autoreleasepool {
        NSData* data = [NSData dataWithBytesNoCopy:const_cast<char*>(txt.data())
                                            length:txt.size()
                                      freeWhenDone:NO];
        NSError* err = nil;
        id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
        if (!obj || ![obj isKindOfClass:[NSDictionary class]]) return false;
        NSDictionary* root = (NSDictionary*)obj;
        NSDictionary* vc = root[@"vision_config"];
        if (![vc isKindOfClass:[NSDictionary class]]) return false;

        auto read_u32 = [&](NSString* k, uint32_t& dst) {
            id v = vc[k];
            if ([v isKindOfClass:[NSNumber class]]) dst = [v unsignedIntValue];
        };
        read_u32(@"hidden_size",           cfg.hidden_size);
        read_u32(@"num_hidden_layers",     cfg.num_hidden_layers);
        read_u32(@"num_attention_heads",   cfg.num_attention_heads);
        read_u32(@"native_patch_size",     cfg.native_patch_size);
        read_u32(@"patch_size",            cfg.native_patch_size);
        read_u32(@"adapter_merge_size",    cfg.adapter_merge_size);
        read_u32(@"spatial_merge_size",    cfg.adapter_merge_size);
        read_u32(@"posemb_grid_size",      cfg.posemb_grid_size);
        read_u32(@"projection_hidden_dim", cfg.projection_hidden_dim);
        read_u32(@"text_hidden_size",      cfg.text_hidden_size);

        if (cfg.num_attention_heads > 0U) {
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }
        id mlp_ratio_v = vc[@"mlp_ratio"];
        if ([mlp_ratio_v isKindOfClass:[NSNumber class]]) {
            cfg.intermediate_size = static_cast<uint32_t>(std::lround([mlp_ratio_v doubleValue] * cfg.hidden_size));
        }
        id ln_eps_v = vc[@"layer_norm_eps"];
        if ([ln_eps_v isKindOfClass:[NSNumber class]]) {
            cfg.layer_norm_eps = [ln_eps_v floatValue];
        }
        id pb_v = vc[@"projector_bias"];
        if ([pb_v isKindOfClass:[NSNumber class]]) {
            cfg.projector_bias = [pb_v boolValue] ? 1U : 0U;
        }
    }
    return true;
}

static bool sikv_load_tensor_f32(
    const std::unordered_map<std::string, SikvSrcTensor>& index,
    const std::string& name,
    uint64_t expected_elems,
    std::vector<float>& out,
    uint64_t& nan_inf_count
) {
    auto it = index.find(name);
    if (it == index.end()) {
        std::fprintf(stderr, "[strata_inkling2_vision] Missing HF tensor: %s\n", name.c_str());
        return false;
    }
    const SikvSrcTensor& t = it->second;
    uint64_t elems = 1ULL;
    for (uint64_t d : t.shape) elems *= d;
    if (elems != expected_elems) {
        std::fprintf(stderr, "[strata_inkling2_vision] Shape mismatch for %s: got %llu elems, expected %llu\n",
                     name.c_str(), static_cast<unsigned long long>(elems),
                     static_cast<unsigned long long>(expected_elems));
        return false;
    }

    int fd = ::open(t.file_path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    std::vector<uint8_t> raw(static_cast<size_t>(t.byte_len));
    const bool ok = sikv_pread_all(fd, raw.data(), raw.size(), t.file_offset);
    ::close(fd);
    if (!ok) return false;

    out.resize(static_cast<size_t>(elems));
    if (t.dtype == SikvSrcDtype::BF16) {
        if (raw.size() != elems * 2ULL) return false;
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < static_cast<size_t>(elems); ++i) {
            const float f = sikv_bf16_to_f32(src[i]);
            if (!std::isfinite(f)) ++nan_inf_count;
            out[i] = f;
        }
    } else if (t.dtype == SikvSrcDtype::F16) {
        if (raw.size() != elems * 2ULL) return false;
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < static_cast<size_t>(elems); ++i) {
            const float f = sikv_f16_to_f32(src[i]);
            if (!std::isfinite(f)) ++nan_inf_count;
            out[i] = f;
        }
    } else if (t.dtype == SikvSrcDtype::F32) {
        if (raw.size() != elems * 4ULL) return false;
        const float* src = reinterpret_cast<const float*>(raw.data());
        for (size_t i = 0; i < static_cast<size_t>(elems); ++i) {
            const float f = src[i];
            if (!std::isfinite(f)) ++nan_inf_count;
            out[i] = f;
        }
    } else {
        std::fprintf(stderr, "[strata_inkling2_vision] Unsupported dtype for %s\n", name.c_str());
        return false;
    }
    return true;
}

/* ============================================================================
 * 3. Vision Engine Internal State & Scratch Buffer Manager
 * ============================================================================ */

struct SikvPsoSpec {
    __strong id<MTLComputePipelineState>* dst;
    const char*                           name;
};

} // namespace

struct StrataInkling2VisionEngine {
    id<MTLDevice>               device        = nil;
    id<MTLCommandQueue>         queue         = nil;
    id<MTLLibrary>              library       = nil;
    id<MTLBuffer>               b_uma         = nil;

    void*                       mmap_base     = nullptr;
    size_t                      mmap_size     = 0U;
    int                         mmap_fd       = -1;

    StrataInkling2VisionHeader  hdr           = {};
    bool                        has_nax       = false;
    uint32_t                    ready         = 0U;
    uint32_t                    metal_cache_hit = 0U;
    double                      pso_load_ms   = 0.0;
    double                      last_encode_ms = 0.0;
    uint32_t                    last_num_tokens = 0U;
    uint32_t                    last_grid_h   = 0U;
    uint32_t                    last_grid_w   = 0U;
    char                        device_name[128] = {};
    char                        vision_path[512] = {};

    /* Metal 4.1 Vision Compute Pipelines (all 10 routed through StrataMetalPsoCache) */
    id<MTLComputePipelineState> pso_gemm_64x128       = nil;
    id<MTLComputePipelineState> pso_gemm_32x64        = nil;
    id<MTLComputePipelineState> pso_gemm_fallback     = nil;
    id<MTLComputePipelineState> pso_embed_assemble    = nil;
    id<MTLComputePipelineState> pso_layernorm_f16     = nil;
    id<MTLComputePipelineState> pso_qkv_split_rope    = nil;
    id<MTLComputePipelineState> pso_flash_attn_nax    = nil;
    id<MTLComputePipelineState> pso_flash_attn_hd64   = nil;
    id<MTLComputePipelineState> pso_res_scale_ln      = nil;
    id<MTLComputePipelineState> pso_drop_cls_f16      = nil;

    /* Lazily grown scratch buffers (serialized by `encode_mu` for server concurrency) */
    std::mutex                  encode_mu;
    uint32_t                    cap_n_vis_padded   = 0U;
    uint32_t                    cap_patches_padded = 0U;
    uint32_t                    cap_s_padded       = 0U;

    id<MTLBuffer>               b_patches_f16      = nil; // Shared  [cap_patches_padded, 1200] F16
    id<MTLBuffer>               b_patch_emb_f32    = nil; // Private [cap_patches_padded, 1024] F32
    id<MTLBuffer>               b_x_f32            = nil; // Private [cap_s_padded, 1024] F32
    id<MTLBuffer>               b_x_ln_f16         = nil; // Private [cap_s_padded, 1024] F16
    id<MTLBuffer>               b_qkv_f32          = nil; // Private [cap_s_padded, 3072] F32
    id<MTLBuffer>               b_q_f16            = nil; // Private [16, cap_s_padded, 64] F16
    id<MTLBuffer>               b_k_f16            = nil; // Private [16, cap_s_padded, 64] F16
    id<MTLBuffer>               b_v_f16            = nil; // Private [16, cap_s_padded, 64] F16
    id<MTLBuffer>               b_attn_out_f16     = nil; // Private [cap_s_padded, 1024] F16
    id<MTLBuffer>               b_tmp_f32          = nil; // Private [cap_s_padded, 1024] F32
    id<MTLBuffer>               b_mlp_f16          = nil; // Private [cap_s_padded, 4096] F16
    id<MTLBuffer>               b_merged_f16       = nil; // Private [cap_n_vis_padded, 4096] F16
    id<MTLBuffer>               b_proj_mid_f16     = nil; // Private [cap_n_vis_padded, 4096] F16
    id<MTLBuffer>               b_out_f32          = nil; // Shared  [cap_n_vis_padded, 2304] F32
};

namespace {

static bool sikv_ensure_scratch(
    StrataInkling2VisionEngine* eng,
    uint32_t n_vis_padded,
    uint32_t patches_padded,
    uint32_t s_padded
) {
    if (n_vis_padded   <= eng->cap_n_vis_padded &&
        patches_padded <= eng->cap_patches_padded &&
        s_padded       <= eng->cap_s_padded) {
        return true;
    }

    const uint32_t new_nvis    = std::max(eng->cap_n_vis_padded,   n_vis_padded);
    const uint32_t new_patches = std::max(eng->cap_patches_padded, patches_padded);
    const uint32_t new_s       = std::max(eng->cap_s_padded,       s_padded);

    const uint32_t D         = eng->hdr.cfg.hidden_size;
    const uint32_t MLP       = eng->hdr.cfg.intermediate_size;
    const uint32_t PROJ_H    = eng->hdr.cfg.projection_hidden_dim;
    const uint32_t TEXT_D    = eng->hdr.cfg.text_hidden_size;
    const uint32_t PATCH_DIM = eng->hdr.cfg.native_patch_size * eng->hdr.cfg.native_patch_size * eng->hdr.cfg.num_channels;
    const uint32_t MERGED_D  = eng->hdr.cfg.adapter_merge_size * eng->hdr.cfg.adapter_merge_size * D;

    auto alloc_buf = [&](size_t bytes, MTLResourceOptions opts) -> id<MTLBuffer> {
        const size_t aligned = std::max<size_t>(4096U, static_cast<size_t>(sikv_align_up_u64(bytes, 4096ULL)));
        id<MTLBuffer> buf = [eng->device newBufferWithLength:aligned options:opts];
        if (buf && (opts & MTLResourceStorageModePrivate) == 0U) {
            std::memset([buf contents], 0, aligned);
        }
        return buf;
    };

    eng->b_patches_f16   = alloc_buf(size_t(new_patches) * PATCH_DIM * sizeof(uint16_t), MTLResourceStorageModeShared);
    eng->b_patch_emb_f32 = alloc_buf(size_t(new_patches) * D         * sizeof(float),    MTLResourceStorageModePrivate);
    eng->b_x_f32         = alloc_buf(size_t(new_s)       * D         * sizeof(float),    MTLResourceStorageModePrivate);
    eng->b_x_ln_f16      = alloc_buf(size_t(new_s)       * D         * sizeof(uint16_t), MTLResourceStorageModePrivate);
    eng->b_qkv_f32       = alloc_buf(size_t(new_s)       * (3U * D)  * sizeof(float),    MTLResourceStorageModePrivate);
    eng->b_q_f16         = alloc_buf(size_t(new_s)       * D         * sizeof(uint16_t), MTLResourceStorageModePrivate);
    eng->b_k_f16         = alloc_buf(size_t(new_s)       * D         * sizeof(uint16_t), MTLResourceStorageModePrivate);
    eng->b_v_f16         = alloc_buf(size_t(new_s)       * D         * sizeof(uint16_t), MTLResourceStorageModePrivate);
    eng->b_attn_out_f16  = alloc_buf(size_t(new_s)       * D         * sizeof(uint16_t), MTLResourceStorageModePrivate);
    eng->b_tmp_f32       = alloc_buf(size_t(new_s)       * D         * sizeof(float),    MTLResourceStorageModePrivate);
    eng->b_mlp_f16       = alloc_buf(size_t(new_s)       * MLP       * sizeof(uint16_t), MTLResourceStorageModePrivate);
    eng->b_merged_f16    = alloc_buf(size_t(new_nvis)    * MERGED_D  * sizeof(uint16_t), MTLResourceStorageModePrivate);
    eng->b_proj_mid_f16  = alloc_buf(size_t(new_nvis)    * PROJ_H    * sizeof(uint16_t), MTLResourceStorageModePrivate);
    eng->b_out_f32       = alloc_buf(size_t(new_nvis)    * TEXT_D    * sizeof(float),    MTLResourceStorageModeShared);

    if (!eng->b_patches_f16 || !eng->b_patch_emb_f32 || !eng->b_x_f32 || !eng->b_x_ln_f16 ||
        !eng->b_qkv_f32 || !eng->b_q_f16 || !eng->b_k_f16 || !eng->b_v_f16 ||
        !eng->b_attn_out_f16 || !eng->b_tmp_f32 || !eng->b_mlp_f16 ||
        !eng->b_merged_f16 || !eng->b_proj_mid_f16 || !eng->b_out_f32) {
        std::fprintf(stderr, "[strata_inkling2_vision] Failed to allocate Metal scratch buffers (S=%u)\n", new_s);
        return false;
    }

    eng->cap_n_vis_padded   = new_nvis;
    eng->cap_patches_padded = new_patches;
    eng->cap_s_padded       = new_s;
    return true;
}

static void sikv_encode_gemm(
    StrataInkling2VisionEngine*  eng,
    id<MTLComputeCommandEncoder> enc,
    uint64_t                     w_uma_off,
    id<MTLBuffer>                x_f16_buf,
    uint64_t                     bias_uma_off,
    bool                         has_bias,
    bool                         apply_gelu,
    bool                         out_f16,
    id<MTLBuffer>                y_f16_buf,
    id<MTLBuffer>                y_f32_buf,
    uint32_t                     M_valid,
    uint32_t                     M_padded,
    uint32_t                     N,
    uint32_t                     K
) {
    SikvGemmParams p{};
    p.M_valid    = M_valid;
    p.M_padded   = M_padded;
    p.N          = N;
    p.K          = K;
    p.has_bias   = has_bias   ? 1U : 0U;
    p.apply_gelu = apply_gelu ? 1U : 0U;
    p.out_f16    = out_f16    ? 1U : 0U;
    p.reserved   = 0U;

    const bool use_64x128 = eng->has_nax && eng->pso_gemm_64x128 != nil &&
                            (M_padded >= 64U) && ((M_padded % 64U) == 0U) && ((N % 128U) == 0U);
    const bool use_32x64  = !use_64x128 && eng->has_nax && eng->pso_gemm_32x64 != nil &&
                            ((M_padded % 32U) == 0U) && ((N % 64U) == 0U);

    id<MTLComputePipelineState> pso = use_64x128 ? eng->pso_gemm_64x128
                                    : (use_32x64 ? eng->pso_gemm_32x64 : eng->pso_gemm_fallback);

    [enc setComputePipelineState:pso];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(w_uma_off) atIndex:0];
    [enc setBuffer:x_f16_buf  offset:0 atIndex:1];
    [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(has_bias ? bias_uma_off : 0ULL) atIndex:2];
    [enc setBuffer:(y_f16_buf ? y_f16_buf : eng->b_attn_out_f16) offset:0 atIndex:3];
    [enc setBuffer:(y_f32_buf ? y_f32_buf : eng->b_tmp_f32)      offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];

    if (use_64x128) {
        [enc dispatchThreadgroups:MTLSizeMake(N / 128U, M_padded / 64U, 1U)
            threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
    } else if (use_32x64) {
        [enc dispatchThreadgroups:MTLSizeMake(N / 64U, M_padded / 32U, 1U)
            threadsPerThreadgroup:MTLSizeMake(64U, 1U, 1U)];
    } else {
        [enc dispatchThreads:MTLSizeMake(N, M_padded, 1U)
       threadsPerThreadgroup:MTLSizeMake(16U, 16U, 1U)];
    }
}

/* ============================================================================
 * 4. Native macOS ImageIO + CoreGraphics Decoder (Zero External Dependencies)
 * ============================================================================ */

static int sikv_decode_image_bytes_to_rgb(
    const uint8_t*        img_bytes,
    size_t                img_len,
    uint32_t              cell_px,
    uint32_t              max_image_tokens,
    std::vector<uint8_t>& out_rgb,
    uint32_t&             out_w,
    uint32_t&             out_h
) {
    if (!img_bytes || img_len == 0U) return -1;

    @autoreleasepool {
        CFDataRef cf_data = CFDataCreateWithBytesNoCopy(
            kCFAllocatorDefault, img_bytes, static_cast<CFIndex>(img_len), kCFAllocatorNull);
        if (!cf_data) return -2;

        CGImageSourceRef src = CGImageSourceCreateWithData(cf_data, nullptr);
        CFRelease(cf_data);
        if (!src) return -3;

        CGImageRef cg_img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
        CFRelease(src);
        if (!cg_img) return -4;

        const size_t orig_w = CGImageGetWidth(cg_img);
        const size_t orig_h = CGImageGetHeight(cg_img);
        if (orig_w == 0U || orig_h == 0U) {
            CGImageRelease(cg_img);
            return -5;
        }

        // Exact official Inkling2ImageProcessor behavior: do NOT resize normal images;
        // downscale preserving aspect ratio ONLY if padded tokens exceed `max_image_tokens` (4096).
        uint32_t dst_w = static_cast<uint32_t>(orig_w);
        uint32_t dst_h = static_cast<uint32_t>(orig_h);
        const uint64_t gw0 = (uint64_t(dst_w) + cell_px - 1ULL) / cell_px;
        const uint64_t gh0 = (uint64_t(dst_h) + cell_px - 1ULL) / cell_px;
        if (gw0 * gh0 > max_image_tokens) {
            const double max_px = double(max_image_tokens) * double(cell_px) * double(cell_px);
            const double scale  = std::sqrt(max_px / (double(dst_w) * double(dst_h)));
            dst_w = std::max<uint32_t>(cell_px, static_cast<uint32_t>(std::floor(dst_w * scale / cell_px)) * cell_px);
            dst_h = std::max<uint32_t>(cell_px, static_cast<uint32_t>(std::floor(dst_h * scale / cell_px)) * cell_px);
            while ((uint64_t(dst_w) / cell_px) * (uint64_t(dst_h) / cell_px) > max_image_tokens) {
                if (dst_w >= dst_h && dst_w > cell_px) dst_w -= cell_px;
                else if (dst_h > cell_px)              dst_h -= cell_px;
                else break;
            }
        }

        const CGImageAlphaInfo alpha_info = CGImageGetAlphaInfo(cg_img);
        const bool has_alpha = (alpha_info == kCGImageAlphaPremultipliedLast ||
                                alpha_info == kCGImageAlphaPremultipliedFirst ||
                                alpha_info == kCGImageAlphaLast ||
                                alpha_info == kCGImageAlphaFirst ||
                                alpha_info == kCGImageAlphaOnly);

        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        if (!cs) cs = CGColorSpaceCreateDeviceRGB();
        if (!cs) {
            CGImageRelease(cg_img);
            return -6;
        }

        std::vector<uint8_t> rgba(size_t(dst_w) * size_t(dst_h) * 4U, 0U);
        const CGBitmapInfo bmi = has_alpha
            ? (static_cast<CGBitmapInfo>(kCGImageAlphaPremultipliedLast) | kCGBitmapByteOrder32Big)
            : (static_cast<CGBitmapInfo>(kCGImageAlphaNoneSkipLast)      | kCGBitmapByteOrder32Big);

        CGContextRef ctx = CGBitmapContextCreate(
            rgba.data(), dst_w, dst_h, 8U, size_t(dst_w) * 4U, cs, bmi);
        CGColorSpaceRelease(cs);
        if (!ctx) {
            CGImageRelease(cg_img);
            return -7;
        }

        CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
        CGContextSetBlendMode(ctx, kCGBlendModeCopy);
        CGContextDrawImage(ctx, CGRectMake(0.0, 0.0, double(dst_w), double(dst_h)), cg_img);
        CGContextRelease(ctx);
        CGImageRelease(cg_img);

        out_rgb.resize(size_t(dst_w) * size_t(dst_h) * 3U);
        if (!has_alpha) {
            for (size_t i = 0; i < size_t(dst_w) * size_t(dst_h); ++i) {
                out_rgb[i * 3U + 0U] = rgba[i * 4U + 0U];
                out_rgb[i * 3U + 1U] = rgba[i * 4U + 1U];
                out_rgb[i * 3U + 2U] = rgba[i * 4U + 2U];
            }
        } else {
            // Composite onto official 8x8 checkerboard (204 / 153) matching Inkling2ImageProcessor.convert_to_rgb
            for (uint32_t y = 0U; y < dst_h; ++y) {
                const uint32_t by = y >> 3U;
                for (uint32_t x = 0U; x < dst_w; ++x) {
                    const size_t idx = size_t(y) * dst_w + x;
                    const uint32_t a = rgba[idx * 4U + 3U];
                    if (a == 255U) {
                        out_rgb[idx * 3U + 0U] = rgba[idx * 4U + 0U];
                        out_rgb[idx * 3U + 1U] = rgba[idx * 4U + 1U];
                        out_rgb[idx * 3U + 2U] = rgba[idx * 4U + 2U];
                    } else {
                        const uint32_t bg = (((by + (x >> 3U)) & 1U) == 0U) ? 204U : 153U;
                        const uint32_t bg_term = (bg * (255U - a) + 127U) / 255U;
                        out_rgb[idx * 3U + 0U] = static_cast<uint8_t>(std::min<uint32_t>(255U, rgba[idx * 4U + 0U] + bg_term));
                        out_rgb[idx * 3U + 1U] = static_cast<uint8_t>(std::min<uint32_t>(255U, rgba[idx * 4U + 1U] + bg_term));
                        out_rgb[idx * 3U + 2U] = static_cast<uint8_t>(std::min<uint32_t>(255U, rgba[idx * 4U + 2U] + bg_term));
                    }
                }
            }
        }

        out_w = dst_w;
        out_h = dst_h;
    }
    return 0;
}

} // namespace

/* ============================================================================
 * 5. Public API Implementation
 * ============================================================================ */

extern "C" int strata_inkling2_convert_vision_from_hf(
    const char* hf_dir,
    const char* out_vision_strata_path
) {
    if (!hf_dir || !out_vision_strata_path || hf_dir[0] == '\0' || out_vision_strata_path[0] == '\0') {
        return -1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const std::string hf_root(hf_dir);

    StrataInkling2VisionConfig cfg{};
    if (!sikv_parse_vision_config(hf_root, cfg)) {
        std::fprintf(stderr, "[strata_inkling2_vision] Failed to parse vision_config in %s/config.json\n", hf_dir);
        return -2;
    }
    if (cfg.num_hidden_layers == 0U || cfg.num_hidden_layers > STRATA_INKLING2_VISION_MAX_LAYERS) {
        std::fprintf(stderr, "[strata_inkling2_vision] Unsupported num_hidden_layers=%u\n", cfg.num_hidden_layers);
        return -3;
    }

    // Index all `model.visual.*` tensors across all `*.safetensors` in `hf_dir`
    std::vector<std::string> st_files;
    DIR* dir = ::opendir(hf_dir);
    if (!dir) {
        std::fprintf(stderr, "[strata_inkling2_vision] Cannot open HF dir: %s\n", hf_dir);
        return -4;
    }
    while (struct dirent* ent = ::readdir(dir)) {
        const std::string nm(ent->d_name);
        if (nm.size() > 12U && nm.compare(nm.size() - 12U, 12U, ".safetensors") == 0) {
            if (nm == "mtp.safetensors") continue;
            st_files.push_back(hf_root + "/" + nm);
        }
    }
    ::closedir(dir);
    std::sort(st_files.begin(), st_files.end());

    std::unordered_map<std::string, SikvSrcTensor> index;
    for (const auto& sf : st_files) {
        sikv_index_safetensors_file(sf, index);
    }
    if (index.empty()) {
        std::fprintf(stderr, "[strata_inkling2_vision] No `model.visual.*` tensors found in %s\n", hf_dir);
        return -5;
    }

    // Open temporary output file for atomic write + rename
    const std::string out_path(out_vision_strata_path);
    const std::string tmp_path = out_path + ".tmp." + std::to_string(::getpid());
    int out_fd = ::open(tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) {
        std::fprintf(stderr, "[strata_inkling2_vision] Cannot create output file %s (errno=%d)\n",
                     tmp_path.c_str(), errno);
        return -6;
    }

    StrataInkling2VisionHeader hdr{};
    hdr.magic          = STRATA_INKLING2_VISION_MAGIC_U64;
    hdr.format_version = STRATA_INKLING2_VISION_FORMAT_VERSION;
    hdr.header_bytes   = static_cast<uint32_t>(sizeof(StrataInkling2VisionHeader));
    hdr.cfg            = cfg;

    uint64_t cursor        = STRATA_INKLING2_VISION_HEADER_PADDED_BYTES;
    uint64_t nan_inf_count = 0ULL;
    uint32_t written_cnt   = 0U;

    auto write_f32_slice = [&](const std::vector<float>& vals, uint32_t s0, uint32_t s1,
                               StrataInkling2VisionTensorSlice& dst) -> bool {
        cursor = sikv_align_up_u64(cursor, STRATA_INKLING2_VISION_TENSOR_ALIGN);
        const uint64_t bytes = uint64_t(vals.size()) * sizeof(float);
        if (!sikv_pwrite_all(out_fd, vals.data(), static_cast<size_t>(bytes), cursor)) return false;
        dst.offset     = cursor;
        dst.bytes      = bytes;
        dst.elem_count = static_cast<uint32_t>(vals.size());
        dst.dtype      = STRATA_INKLING2_VISION_DTYPE_F32;
        dst.shape0     = s0;
        dst.shape1     = s1;
        cursor += bytes;
        ++written_cnt;
        return true;
    };

    auto write_f16_slice = [&](const std::vector<float>& vals, uint32_t s0, uint32_t s1,
                               StrataInkling2VisionTensorSlice& dst) -> bool {
        cursor = sikv_align_up_u64(cursor, STRATA_INKLING2_VISION_TENSOR_ALIGN);
        std::vector<uint16_t> f16(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            f16[i] = sikv_f32_to_f16(vals[i]);
        }
        const uint64_t bytes = uint64_t(f16.size()) * sizeof(uint16_t);
        if (!sikv_pwrite_all(out_fd, f16.data(), static_cast<size_t>(bytes), cursor)) return false;
        dst.offset     = cursor;
        dst.bytes      = bytes;
        dst.elem_count = static_cast<uint32_t>(f16.size());
        dst.dtype      = STRATA_INKLING2_VISION_DTYPE_F16;
        dst.shape0     = s0;
        dst.shape1     = s1;
        cursor += bytes;
        ++written_cnt;
        return true;
    };

    const uint32_t D         = cfg.hidden_size;
    const uint32_t MLP       = cfg.intermediate_size;
    const uint32_t PATCH_DIM = cfg.native_patch_size * cfg.native_patch_size * cfg.num_channels;
    const uint32_t POSEMB_N  = cfg.posemb_grid_size * cfg.posemb_grid_size;
    const uint32_t MERGED_D  = cfg.adapter_merge_size * cfg.adapter_merge_size * D;
    const uint32_t PROJ_H    = cfg.projection_hidden_dim;
    const uint32_t TEXT_D    = cfg.text_hidden_size;

    std::vector<float> buf, buf_q, buf_k, buf_v, fused;
    bool ok = true;

    // 1. Embeddings & Pre-LN
    ok = ok && sikv_load_tensor_f32(index, "model.visual.backbone.embeddings.patch_embedding.weight", uint64_t(D) * PATCH_DIM, buf, nan_inf_count)
            && write_f16_slice(buf, D, PATCH_DIM, hdr.patch_emb_w);
    ok = ok && sikv_load_tensor_f32(index, "model.visual.backbone.embeddings.class_embedding", D, buf, nan_inf_count)
            && write_f32_slice(buf, D, 1U, hdr.class_emb);
    ok = ok && sikv_load_tensor_f32(index, "model.visual.backbone.embeddings.cls_position_embedding", D, buf, nan_inf_count)
            && write_f32_slice(buf, D, 1U, hdr.cls_pos_emb);
    ok = ok && sikv_load_tensor_f32(index, "model.visual.backbone.embeddings.position_embedding", uint64_t(POSEMB_N) * D, buf, nan_inf_count)
            && write_f32_slice(buf, POSEMB_N, D, hdr.pos_emb);
    ok = ok && sikv_load_tensor_f32(index, "model.visual.backbone.ln_pre.weight", D, buf, nan_inf_count)
            && write_f32_slice(buf, D, 1U, hdr.ln_pre_w);
    ok = ok && sikv_load_tensor_f32(index, "model.visual.backbone.ln_pre.bias", D, buf, nan_inf_count)
            && write_f32_slice(buf, D, 1U, hdr.ln_pre_b);

    // 2. 23 ViT Layers (fusing Q/K/V weights + biases per layer)
    for (uint32_t l = 0U; ok && l < cfg.num_hidden_layers; ++l) {
        const std::string pfx = "model.visual.backbone.layers." + std::to_string(l) + ".";
        StrataInkling2VisionLayerDesc& ld = hdr.layers[l];

        ok = ok && sikv_load_tensor_f32(index, pfx + "ln_1.weight", D, buf, nan_inf_count)
                && write_f32_slice(buf, D, 1U, ld.ln1_w);
        ok = ok && sikv_load_tensor_f32(index, pfx + "ln_1.bias", D, buf, nan_inf_count)
                && write_f32_slice(buf, D, 1U, ld.ln1_b);

        ok = ok && sikv_load_tensor_f32(index, pfx + "self_attn.q_proj.weight", uint64_t(D) * D, buf_q, nan_inf_count)
                && sikv_load_tensor_f32(index, pfx + "self_attn.k_proj.weight", uint64_t(D) * D, buf_k, nan_inf_count)
                && sikv_load_tensor_f32(index, pfx + "self_attn.v_proj.weight", uint64_t(D) * D, buf_v, nan_inf_count);
        if (ok) {
            fused.clear();
            fused.reserve(buf_q.size() + buf_k.size() + buf_v.size());
            fused.insert(fused.end(), buf_q.begin(), buf_q.end());
            fused.insert(fused.end(), buf_k.begin(), buf_k.end());
            fused.insert(fused.end(), buf_v.begin(), buf_v.end());
            ok = write_f16_slice(fused, 3U * D, D, ld.qkv_w);
        }

        ok = ok && sikv_load_tensor_f32(index, pfx + "self_attn.q_proj.bias", D, buf_q, nan_inf_count)
                && sikv_load_tensor_f32(index, pfx + "self_attn.k_proj.bias", D, buf_k, nan_inf_count)
                && sikv_load_tensor_f32(index, pfx + "self_attn.v_proj.bias", D, buf_v, nan_inf_count);
        if (ok) {
            fused.clear();
            fused.reserve(3U * D);
            fused.insert(fused.end(), buf_q.begin(), buf_q.end());
            fused.insert(fused.end(), buf_k.begin(), buf_k.end());
            fused.insert(fused.end(), buf_v.begin(), buf_v.end());
            ok = write_f32_slice(fused, 3U * D, 1U, ld.qkv_b);
        }

        ok = ok && sikv_load_tensor_f32(index, pfx + "self_attn.out_proj.weight", uint64_t(D) * D, buf, nan_inf_count)
                && write_f16_slice(buf, D, D, ld.out_w);
        ok = ok && sikv_load_tensor_f32(index, pfx + "self_attn.out_proj.bias", D, buf, nan_inf_count)
                && write_f32_slice(buf, D, 1U, ld.out_b);
        ok = ok && sikv_load_tensor_f32(index, pfx + "ls_1.gamma", D, buf, nan_inf_count)
                && write_f32_slice(buf, D, 1U, ld.ls1_gamma);

        ok = ok && sikv_load_tensor_f32(index, pfx + "ln_2.weight", D, buf, nan_inf_count)
                && write_f32_slice(buf, D, 1U, ld.ln2_w);
        ok = ok && sikv_load_tensor_f32(index, pfx + "ln_2.bias", D, buf, nan_inf_count)
                && write_f32_slice(buf, D, 1U, ld.ln2_b);
        ok = ok && sikv_load_tensor_f32(index, pfx + "mlp.c_fc.weight", uint64_t(MLP) * D, buf, nan_inf_count)
                && write_f16_slice(buf, MLP, D, ld.fc1_w);
        ok = ok && sikv_load_tensor_f32(index, pfx + "mlp.c_fc.bias", MLP, buf, nan_inf_count)
                && write_f32_slice(buf, MLP, 1U, ld.fc1_b);
        ok = ok && sikv_load_tensor_f32(index, pfx + "mlp.c_proj.weight", uint64_t(D) * MLP, buf, nan_inf_count)
                && write_f16_slice(buf, D, MLP, ld.fc2_w);
        ok = ok && sikv_load_tensor_f32(index, pfx + "mlp.c_proj.bias", D, buf, nan_inf_count)
                && write_f32_slice(buf, D, 1U, ld.fc2_b);
        ok = ok && sikv_load_tensor_f32(index, pfx + "ls_2.gamma", D, buf, nan_inf_count)
                && write_f32_slice(buf, D, 1U, ld.ls2_gamma);
    }

    // 3. 2x2 Spatial Merger / Projector
    ok = ok && sikv_load_tensor_f32(index, "model.visual.proj_in.weight", uint64_t(PROJ_H) * MERGED_D, buf, nan_inf_count)
            && write_f16_slice(buf, PROJ_H, MERGED_D, hdr.proj_in_w);
    ok = ok && sikv_load_tensor_f32(index, "model.visual.proj_out.weight", uint64_t(TEXT_D) * PROJ_H, buf, nan_inf_count)
            && write_f16_slice(buf, TEXT_D, PROJ_H, hdr.proj_out_w);

    if (!ok || nan_inf_count > 0ULL) {
        std::fprintf(stderr, "[strata_inkling2_vision] Conversion failed (ok=%d, nan_inf=%llu)\n",
                     ok ? 1 : 0, static_cast<unsigned long long>(nan_inf_count));
        ::close(out_fd);
        ::unlink(tmp_path.c_str());
        return -7;
    }

    // Pad file length to 16 KiB VM page boundary for Apple Silicon zero-copy mmap
    const uint64_t total_file_bytes = sikv_align_up_u64(cursor, STRATA_INKLING2_VISION_PAGE_BYTES);
    if (::ftruncate(out_fd, static_cast<off_t>(total_file_bytes)) != 0) {
        ::close(out_fd);
        ::unlink(tmp_path.c_str());
        return -8;
    }

    hdr.total_file_bytes = total_file_bytes;
    hdr.num_tensors      = written_cnt;

    std::vector<uint8_t> hdr_page(STRATA_INKLING2_VISION_HEADER_PADDED_BYTES, 0U);
    std::memcpy(hdr_page.data(), &hdr, sizeof(hdr));
    if (!sikv_pwrite_all(out_fd, hdr_page.data(), hdr_page.size(), 0ULL)) {
        ::close(out_fd);
        ::unlink(tmp_path.c_str());
        return -9;
    }
    ::close(out_fd);

    if (::rename(tmp_path.c_str(), out_path.c_str()) != 0) {
        ::unlink(tmp_path.c_str());
        return -10;
    }

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
                 "[strata_inkling2_vision] Converted %zu HF tensors -> %u packed slices in %s (%.2f MiB, 0 NaN/Inf, %.1f ms)\n",
                 index.size(), written_cnt, out_path.c_str(),
                 double(total_file_bytes) / (1024.0 * 1024.0), ms);
    return 0;
}

extern "C" int strata_inkling2_vision_resolve_path(
    const char* bundle_dir,
    char*       out_path,
    size_t      out_cap
) {
    if (!out_path || out_cap == 0U) return 0;
    out_path[0] = '\0';

    std::vector<std::string> candidates;
    if (bundle_dir && bundle_dir[0] != '\0') {
        std::string b(bundle_dir);
        while (b.size() > 1U && b.back() == '/') b.pop_back();
        // Direct sidecar file path
        candidates.push_back(b);
        // Inside bundle directory
        candidates.push_back(b + "/" + STRATA_INKLING2_VISION_FILENAME);
        // If caller passed a `.strata` file path, check sibling directory
        const size_t slash = b.find_last_of('/');
        if (slash != std::string::npos) {
            candidates.push_back(b.substr(0, slash) + "/" + STRATA_INKLING2_VISION_FILENAME);
        }
    }
    if (const char* env_p = std::getenv("STRATA_INKLING2_VISION_PATH"); env_p && env_p[0] != '\0') {
        candidates.push_back(std::string(env_p));
    }
    candidates.push_back(STRATA_INKLING2_VISION_DEFAULT_FAST_PATH);
    candidates.push_back(STRATA_INKLING2_VISION_DEFAULT_V1_PATH);

    for (const auto& cand : candidates) {
        if (sikv_validate_sidecar_magic(cand.c_str())) {
            if (cand.size() + 1U > out_cap) return 0;
            std::memcpy(out_path, cand.c_str(), cand.size() + 1U);
            return 1;
        }
    }
    return 0;
}

StrataInkling2VisionEngine* strata_inkling2_vision_create(
    id<MTLDevice> dev_or_null,
    const char*   vision_strata_path
) {
    if (!vision_strata_path || vision_strata_path[0] == '\0') return nullptr;

    @autoreleasepool {
        int fd = ::open(vision_strata_path, O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "[strata_inkling2_vision] Cannot open %s (errno=%d)\n",
                         vision_strata_path, errno);
            return nullptr;
        }
        struct stat st;
        if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(STRATA_INKLING2_VISION_HEADER_PADDED_BYTES)) {
            ::close(fd);
            return nullptr;
        }
        const size_t file_bytes = static_cast<size_t>(st.st_size);
        void* base = ::mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) {
            ::close(fd);
            std::fprintf(stderr, "[strata_inkling2_vision] mmap failed for %s\n", vision_strata_path);
            return nullptr;
        }
        ::madvise(base, file_bytes, MADV_WILLNEED);

        StrataInkling2VisionHeader hdr{};
        std::memcpy(&hdr, base, sizeof(hdr));
        auto vision_hdr_valid = [&]() -> bool {
            const auto& c = hdr.cfg;
            if (hdr.magic != STRATA_INKLING2_VISION_MAGIC_U64 ||
                hdr.format_version != STRATA_INKLING2_VISION_FORMAT_VERSION ||
                hdr.total_file_bytes != static_cast<uint64_t>(file_bytes) ||
                c.num_hidden_layers == 0U || c.num_hidden_layers > STRATA_INKLING2_VISION_MAX_LAYERS ||
                c.num_channels != 3U ||
                c.native_patch_size == 0U || c.native_patch_size > 64U ||
                c.adapter_merge_size == 0U || c.adapter_merge_size > 8U ||
                c.posemb_grid_size == 0U || c.posemb_grid_size > 256U ||
                c.num_attention_heads == 0U || c.head_dim != 64U ||
                c.hidden_size == 0U || c.hidden_size > 16384U ||
                static_cast<uint64_t>(c.num_attention_heads) * c.head_dim != c.hidden_size ||
                c.intermediate_size == 0U || c.intermediate_size > 65536U ||
                c.projection_hidden_dim == 0U || c.projection_hidden_dim > 65536U ||
                c.text_hidden_size == 0U || c.text_hidden_size > 16384U) {
                return false;
            }
            const uint64_t D         = c.hidden_size;
            const uint64_t MLP       = c.intermediate_size;
            const uint64_t PROJ_H    = c.projection_hidden_dim;
            const uint64_t TEXT_D    = c.text_hidden_size;
            const uint64_t MERGED_D  = static_cast<uint64_t>(c.adapter_merge_size) * c.adapter_merge_size * D;
            const uint64_t PATCH_DIM = static_cast<uint64_t>(c.native_patch_size) * c.native_patch_size * c.num_channels;
            const uint64_t GRID_POS  = static_cast<uint64_t>(c.posemb_grid_size) * c.posemb_grid_size;
            auto chk = [&](const StrataInkling2VisionTensorSlice& s, uint32_t exp_dtype, uint64_t exp_bytes) -> bool {
                if (s.dtype != exp_dtype) return false;
                if (s.offset < STRATA_INKLING2_VISION_HEADER_PADDED_BYTES) return false;
                if (s.bytes != exp_bytes) return false;
                if (s.offset > file_bytes || s.bytes > file_bytes - s.offset) return false;
                return true;
            };
            if (!chk(hdr.patch_emb_w, STRATA_INKLING2_VISION_DTYPE_F16, D * PATCH_DIM * 2ULL) ||
                !chk(hdr.class_emb,   STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                !chk(hdr.cls_pos_emb, STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                !chk(hdr.pos_emb,     STRATA_INKLING2_VISION_DTYPE_F32, GRID_POS * D * 4ULL) ||
                !chk(hdr.ln_pre_w,    STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                !chk(hdr.ln_pre_b,    STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                !chk(hdr.proj_in_w,   STRATA_INKLING2_VISION_DTYPE_F16, PROJ_H * MERGED_D * 2ULL) ||
                !chk(hdr.proj_out_w,  STRATA_INKLING2_VISION_DTYPE_F16, TEXT_D * PROJ_H * 2ULL)) {
                return false;
            }
            for (uint32_t l = 0U; l < c.num_hidden_layers; ++l) {
                const auto& ly = hdr.layers[l];
                if (!chk(ly.ln1_w,     STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                    !chk(ly.ln1_b,     STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                    !chk(ly.qkv_w,     STRATA_INKLING2_VISION_DTYPE_F16, 3ULL * D * D * 2ULL) ||
                    !chk(ly.qkv_b,     STRATA_INKLING2_VISION_DTYPE_F32, 3ULL * D * 4ULL) ||
                    !chk(ly.out_w,     STRATA_INKLING2_VISION_DTYPE_F16, D * D * 2ULL) ||
                    !chk(ly.out_b,     STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                    !chk(ly.ls1_gamma, STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                    !chk(ly.ln2_w,     STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                    !chk(ly.ln2_b,     STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                    !chk(ly.fc1_w,     STRATA_INKLING2_VISION_DTYPE_F16, MLP * D * 2ULL) ||
                    !chk(ly.fc1_b,     STRATA_INKLING2_VISION_DTYPE_F32, MLP * 4ULL) ||
                    !chk(ly.fc2_w,     STRATA_INKLING2_VISION_DTYPE_F16, D * MLP * 2ULL) ||
                    !chk(ly.fc2_b,     STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL) ||
                    !chk(ly.ls2_gamma, STRATA_INKLING2_VISION_DTYPE_F32, D * 4ULL)) {
                    return false;
                }
            }
            return true;
        };
        if (!vision_hdr_valid()) {
            ::munmap(base, file_bytes);
            ::close(fd);
            std::fprintf(stderr, "[strata_inkling2_vision] Invalid sidecar header in %s\n", vision_strata_path);
            return nullptr;
        }

        id<MTLDevice> dev = dev_or_null ? dev_or_null : MTLCreateSystemDefaultDevice();
        if (!dev) {
            ::munmap(base, file_bytes);
            ::close(fd);
            return nullptr;
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) {
            ::munmap(base, file_bytes);
            ::close(fd);
            return nullptr;
        }

        const std::string msl_path = sikv_resolve_repo_path(
            "../gpu/strata_inkling2_vision_kernels.metal",
            "src/gpu/strata_inkling2_vision_kernels.metal");
        const std::string hdr_path = sikv_resolve_repo_path(
            "../../include/strata_inkling2_vision.h",
            "include/strata_inkling2_vision.h");
        std::string msl_src = strata_load_shader_asset("strata_inkling2_vision_kernels.metal", "");
        if (msl_src.empty()) msl_src = sikv_read_text_file(msl_path);
        std::string hdr_src = strata_load_shader_asset("strata_inkling2_vision.h", "");
        if (hdr_src.empty()) hdr_src = sikv_read_text_file(hdr_path);
        if (msl_src.empty() || hdr_src.empty()) {
            std::fprintf(stderr, "[strata_inkling2_vision] Failed to read MSL (%s) or header (%s)\n",
                         msl_path.c_str(), hdr_path.c_str());
            ::munmap(base, file_bytes);
            ::close(fd);
            return nullptr;
        }
        const std::string marker = "// @@STRATA_INKLING2_VISION_HEADER@@";
        const size_t pos = msl_src.find(marker);
        if (pos != std::string::npos) {
            msl_src.replace(pos, marker.size(), hdr_src);
        }

        NSString* msl_ns = [NSString stringWithUTF8String:msl_src.c_str()];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        bool compiled_m4 = false;
        if (@available(macOS 27.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_1;
            compiled_m4 = true;
        } else if (@available(macOS 26.0, *)) {
            opts.languageVersion = MTLLanguageVersion4_0;
            compiled_m4 = true;
        } else {
            opts.languageVersion = MTLLanguageVersion3_1;
        }
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
            opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
        }

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
        if (!lib && compiled_m4) {
            if (@available(macOS 26.0, *)) {
                err = nil;
                opts.languageVersion = MTLLanguageVersion4_0;
                lib = [dev newLibraryWithSource:msl_ns options:opts error:&err];
            }
        }
        if (!lib) {
            std::fprintf(stderr, "[strata_inkling2_vision] Metal shader compile failed:\n%s\n",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            ::munmap(base, file_bytes);
            ::close(fd);
            return nullptr;
        }
        if (err) {
            std::fprintf(stderr, "[strata_inkling2_vision] Metal shader warnings:\n%s\n",
                         [[err localizedDescription] UTF8String]);
        }

        StrataInkling2VisionEngine* eng = new StrataInkling2VisionEngine();
        eng->device    = dev;
        eng->queue     = queue;
        eng->library   = lib;
        eng->has_nax   = compiled_m4;
        eng->mmap_base = base;
        eng->mmap_size = file_bytes;
        eng->mmap_fd   = fd;
        eng->hdr       = hdr;

        const char* dev_utf8 = [[dev name] UTF8String];
        std::snprintf(eng->device_name, sizeof(eng->device_name), "%s", dev_utf8 ? dev_utf8 : "Apple Silicon");
        std::snprintf(eng->vision_path, sizeof(eng->vision_path), "%s", vision_strata_path);

        // Build all 10 Vision Compute PSOs via persistent `StrataMetalPsoCache` (`engine_tag = "inkling2_vision"`)
        std::vector<SikvPsoSpec> specs;
        if (compiled_m4) {
            specs.push_back(SikvPsoSpec{&eng->pso_gemm_64x128,    "sikv_f16_gemm_nax_64x128_sg4"});
            specs.push_back(SikvPsoSpec{&eng->pso_gemm_32x64,     "sikv_f16_gemm_nax_32x64_sg2"});
            specs.push_back(SikvPsoSpec{&eng->pso_flash_attn_nax, "sikv_flash_attn_hd64_nax"});
        }
        specs.push_back(SikvPsoSpec{&eng->pso_gemm_fallback,   "sikv_f16_gemm_fallback"});
        specs.push_back(SikvPsoSpec{&eng->pso_embed_assemble,  "sikv_embed_assemble_ln_pre"});
        specs.push_back(SikvPsoSpec{&eng->pso_layernorm_f16,   "sikv_layernorm_f32_to_f16"});
        specs.push_back(SikvPsoSpec{&eng->pso_qkv_split_rope,  "sikv_qkv_split_rope_f16"});
        specs.push_back(SikvPsoSpec{&eng->pso_flash_attn_hd64, "sikv_flash_attn_hd64"});
        specs.push_back(SikvPsoSpec{&eng->pso_res_scale_ln,    "sikv_res_scale_ln"});
        specs.push_back(SikvPsoSpec{&eng->pso_drop_cls_f16,    "sikv_drop_cls_f32_to_f16"});

        StrataMetalPsoCache* pso_cache = strata_metal_cache_open_with_opts(
            dev, lib, "inkling2_vision", msl_src.data(), msl_src.size(), opts);
        if (!pso_cache) {
            strata_inkling2_vision_destroy(eng);
            return nullptr;
        }

        std::atomic<bool> pso_ok{true};
        std::atomic<bool>* pso_ok_ptr = &pso_ok;
        const SikvPsoSpec* specs_ptr  = specs.data();
        const size_t       num_specs  = specs.size();
        dispatch_apply(num_specs, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t idx) {
            @autoreleasepool {
                NSError* p_err = nil;
                id<MTLComputePipelineState> pso = strata_metal_cache_make_pso(
                    pso_cache, specs_ptr[idx].name, nil, &p_err);
                if (!pso) {
                    std::fprintf(stderr, "[strata_inkling2_vision] PSO failed for `%s`: %s\n",
                                 specs_ptr[idx].name,
                                 p_err ? [[p_err localizedDescription] UTF8String] : "unknown");
                    pso_ok_ptr->store(false, std::memory_order_relaxed);
                } else {
                    *(specs_ptr[idx].dst) = pso;
                }
            }
        });
        strata_metal_cache_finish(pso_cache);
        eng->metal_cache_hit = pso_cache->cache_hit ? 1U : 0U;
        eng->pso_load_ms     = pso_cache->elapsed_ms;
        strata_metal_cache_destroy(pso_cache);

        if (!pso_ok.load(std::memory_order_relaxed)) {
            strata_inkling2_vision_destroy(eng);
            return nullptr;
        }

        // Zero-Copy 16 KiB-Aligned UMA Weight Buffer
        const size_t aligned_len = static_cast<size_t>(
            sikv_align_up_u64(file_bytes, STRATA_INKLING2_VISION_PAGE_BYTES));
        eng->b_uma = [dev newBufferWithBytesNoCopy:base
                                            length:aligned_len
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
        if (!eng->b_uma) {
            std::fprintf(stderr, "[strata_inkling2_vision] Failed to create zero-copy UMA buffer (%zu B)\n", aligned_len);
            strata_inkling2_vision_destroy(eng);
            return nullptr;
        }

        eng->ready = 1U;
        return eng;
    }
}

extern "C" StrataInkling2VisionEngine* strata_inkling2_vision_create_default(
    const char* vision_strata_path
) {
    return strata_inkling2_vision_create(nil, vision_strata_path);
}

extern "C" int strata_inkling2_vision_get_info(
    const StrataInkling2VisionEngine* eng,
    StrataInkling2VisionInfo*         out_info
) {
    if (!out_info) return -1;
    std::memset(out_info, 0, sizeof(*out_info));
    if (!eng) return -2;

    out_info->ready            = eng->ready;
    out_info->metal_cache_hit  = eng->metal_cache_hit;
    out_info->pso_load_ms      = eng->pso_load_ms;
    out_info->file_bytes       = eng->hdr.total_file_bytes;
    out_info->hidden_size      = eng->hdr.cfg.hidden_size;
    out_info->text_hidden_size = eng->hdr.cfg.text_hidden_size;
    out_info->num_layers       = eng->hdr.cfg.num_hidden_layers;
    out_info->num_heads        = eng->hdr.cfg.num_attention_heads;
    out_info->patch_size       = eng->hdr.cfg.native_patch_size;
    out_info->merge_size       = eng->hdr.cfg.adapter_merge_size;
    out_info->posemb_grid_size = eng->hdr.cfg.posemb_grid_size;
    out_info->max_image_tokens = STRATA_INKLING2_VISION_MAX_TOKENS;
    out_info->last_encode_ms   = eng->last_encode_ms;
    out_info->last_num_tokens  = eng->last_num_tokens;
    out_info->last_grid_h      = eng->last_grid_h;
    out_info->last_grid_w      = eng->last_grid_w;
    std::snprintf(out_info->device_name, sizeof(out_info->device_name), "%s", eng->device_name);
    std::snprintf(out_info->vision_path, sizeof(out_info->vision_path), "%s", eng->vision_path);
    return 0;
}

extern "C" void strata_inkling2_vision_destroy(
    StrataInkling2VisionEngine* eng
) {
    if (!eng) return;
    @autoreleasepool {
        eng->b_uma = nil;
        if (eng->mmap_base && eng->mmap_base != MAP_FAILED && eng->mmap_size > 0U) {
            ::munmap(eng->mmap_base, eng->mmap_size);
        }
        if (eng->mmap_fd >= 0) {
            ::close(eng->mmap_fd);
        }
        delete eng;
    }
}

int strata_inkling2_vision_decode_base64(
    const char*           b64_or_data_uri,
    size_t                len,
    std::vector<uint8_t>* out_bytes
) {
    if (!b64_or_data_uri || !out_bytes) return -1;
    out_bytes->clear();

    const char* p = b64_or_data_uri;
    const char* end = p + len;

    // Strip RFC 2397 `data:...;base64,` prefix if present
    if (len >= 5U && std::memcmp(p, "data:", 5U) == 0) {
        const char* comma = static_cast<const char*>(std::memchr(p, ',', len));
        if (!comma) return -2;
        p = comma + 1;
    }

    static const int8_t kDec[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,62,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,63,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    };

    out_bytes->reserve((static_cast<size_t>(end - p) * 3U) / 4U + 4U);
    uint32_t acc = 0U;
    int bits = 0;
    for (; p < end; ++p) {
        const unsigned char ch = static_cast<unsigned char>(*p);
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') continue;
        if (ch == '=') break;
        const int8_t v = kDec[ch];
        if (v < 0) {
            out_bytes->clear();
            return -3;
        }
        acc = (acc << 6U) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out_bytes->push_back(static_cast<uint8_t>((acc >> bits) & 0xFFU));
        }
    }
    return out_bytes->empty() ? -4 : 0;
}

int strata_inkling2_vision_encode_rgb_u8(
    StrataInkling2VisionEngine* eng,
    const uint8_t*              rgb_u8,
    uint32_t                    width,
    uint32_t                    height,
    std::vector<float>*         out_emb_f32,
    uint32_t*                   out_num_tokens,
    uint32_t*                   out_grid_h,
    uint32_t*                   out_grid_w
) {
    if (!eng || !eng->ready || !rgb_u8 || width == 0U || height == 0U || !out_emb_f32) {
        return -1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(eng->encode_mu);

    const auto& cfg        = eng->hdr.cfg;
    const uint32_t patch   = cfg.native_patch_size;
    const uint32_t merge   = cfg.adapter_merge_size;
    const uint32_t cell_px = patch * merge; // 40
    const uint32_t D       = cfg.hidden_size; // 1024
    const uint32_t MLP     = cfg.intermediate_size; // 4096
    const uint32_t PROJ_H  = cfg.projection_hidden_dim; // 4096
    const uint32_t TEXT_D  = cfg.text_hidden_size; // 2304
    const uint32_t MERGED_D = merge * merge * D; // 4096
    const uint32_t PATCH_DIM = patch * patch * cfg.num_channels; // 1200

    const uint32_t grid_w = (width  + cell_px - 1U) / cell_px;
    const uint32_t grid_h = (height + cell_px - 1U) / cell_px;
    const uint32_t n_vis  = grid_h * grid_w;
    if (n_vis == 0U || n_vis > STRATA_INKLING2_VISION_MAX_TOKENS) {
        std::fprintf(stderr, "[strata_inkling2_vision] Image grid %ux%u (%u tokens) exceeds max %u\n",
                     grid_h, grid_w, n_vis, STRATA_INKLING2_VISION_MAX_TOKENS);
        return -2;
    }

    const uint32_t num_patches = n_vis * merge * merge;
    const uint32_t S           = 1U + num_patches;

    // Align padded M rows to 64 when >= 64 (for 64x128 NAX), else 32 (for 32x64 NAX)
    const uint32_t patches_padded = (num_patches >= 64U) ? sikv_align_up_u32(num_patches, 64U) : sikv_align_up_u32(num_patches, 32U);
    const uint32_t s_padded       = (S >= 64U)           ? sikv_align_up_u32(S, 64U)           : sikv_align_up_u32(S, 32U);
    const uint32_t n_vis_padded   = (n_vis >= 64U)       ? sikv_align_up_u32(n_vis, 64U)       : sikv_align_up_u32(n_vis, 32U);

    @autoreleasepool {
        if (!sikv_ensure_scratch(eng, n_vis_padded, patches_padded, s_padded)) {
            return -3;
        }

        // 1. Pack normalized HWC [20, 20, 3] patches in spatial-merge-block order (br, bc, in_r, in_c)
        //    Exact Inkling2ImageProcessor formula:
        //      pad bottom-right with -1.0f on [0..255] scale, then (val / 255.0 - 0.5) / 0.5
        uint16_t* dst_patches = static_cast<uint16_t*>([eng->b_patches_f16 contents]);
        const uint16_t pad_h16 = sikv_f32_to_f16((-1.0f / 255.0f - 0.5f) / 0.5f);
        uint16_t lut_u8_to_h16[256];
        for (int v = 0; v < 256; ++v) {
            lut_u8_to_h16[v] = sikv_f32_to_f16((float(v) / 255.0f - 0.5f) / 0.5f);
        }

        for (uint32_t br = 0U; br < grid_h; ++br) {
            for (uint32_t bc = 0U; bc < grid_w; ++bc) {
                for (uint32_t ir = 0U; ir < merge; ++ir) {
                    for (uint32_t ic = 0U; ic < merge; ++ic) {
                        const uint32_t p_idx  = (((br * grid_w + bc) * merge + ir) * merge) + ic;
                        const uint32_t base_y = (br * merge + ir) * patch;
                        const uint32_t base_x = (bc * merge + ic) * patch;
                        uint16_t* row_out = dst_patches + size_t(p_idx) * PATCH_DIM;

                        for (uint32_t py = 0U; py < patch; ++py) {
                            const uint32_t y = base_y + py;
                            uint16_t* line_out = row_out + size_t(py) * patch * 3U;
                            if (y < height) {
                                const uint8_t* src_row = rgb_u8 + size_t(y) * width * 3U;
                                for (uint32_t px = 0U; px < patch; ++px) {
                                    const uint32_t x = base_x + px;
                                    if (x < width) {
                                        const uint8_t* pix = src_row + size_t(x) * 3U;
                                        line_out[px * 3U + 0U] = lut_u8_to_h16[pix[0]];
                                        line_out[px * 3U + 1U] = lut_u8_to_h16[pix[1]];
                                        line_out[px * 3U + 2U] = lut_u8_to_h16[pix[2]];
                                    } else {
                                        line_out[px * 3U + 0U] = pad_h16;
                                        line_out[px * 3U + 1U] = pad_h16;
                                        line_out[px * 3U + 2U] = pad_h16;
                                    }
                                }
                            } else {
                                for (uint32_t k = 0U; k < patch * 3U; ++k) {
                                    line_out[k] = pad_h16;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (patches_padded > num_patches) {
            std::memset(dst_patches + size_t(num_patches) * PATCH_DIM, 0,
                        size_t(patches_padded - num_patches) * PATCH_DIM * sizeof(uint16_t));
        }

        // 2. Encode entire Vision Transformer + 2x2 Projector in ONE single Metal command buffer
        id<MTLCommandBuffer> cb = [eng->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // 2a. Patch embedding projection: [num_patches, 1200] @ [1024, 1200]^T -> [num_patches, 1024] F32
        sikv_encode_gemm(eng, enc,
                         eng->hdr.patch_emb_w.offset, eng->b_patches_f16,
                         0ULL, false, false, false,
                         nil, eng->b_patch_emb_f32,
                         num_patches, patches_padded, D, PATCH_DIM);

        // 2b. Assemble CLS + bilinear 32x32 pos_emb + ln_pre -> b_x_f32 [S, 1024]
        {
            SikvEmbedAssembleParams ep{};
            ep.num_patches = num_patches;
            ep.S           = S;
            ep.S_padded    = s_padded;
            ep.D           = D;
            ep.grid_h      = grid_h;
            ep.grid_w      = grid_w;
            ep.posemb_grid = cfg.posemb_grid_size;
            ep.ln_eps      = cfg.layer_norm_eps;

            [enc setComputePipelineState:eng->pso_embed_assemble];
            [enc setBuffer:eng->b_patch_emb_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.class_emb.offset)   atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.cls_pos_emb.offset) atIndex:2];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.pos_emb.offset)     atIndex:3];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.ln_pre_w.offset)    atIndex:4];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.ln_pre_b.offset)    atIndex:5];
            [enc setBuffer:eng->b_x_f32 offset:0 atIndex:6];
            [enc setBytes:&ep length:sizeof(ep) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(s_padded, 1U, 1U)
                threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
        }

        // 2c. Layer 0 pre-attention LayerNorm (`ln_1`) -> b_x_ln_f16 [S_padded, 1024]
        {
            SikvLayerNormParams lp{};
            lp.S        = S;
            lp.S_padded = s_padded;
            lp.D        = D;
            lp.eps      = cfg.layer_norm_eps;

            [enc setComputePipelineState:eng->pso_layernorm_f16];
            [enc setBuffer:eng->b_x_f32 offset:0 atIndex:0];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.layers[0].ln1_w.offset) atIndex:1];
            [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(eng->hdr.layers[0].ln1_b.offset) atIndex:2];
            [enc setBuffer:eng->b_x_ln_f16 offset:0 atIndex:3];
            [enc setBytes:&lp length:sizeof(lp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(s_padded, 1U, 1U)
                threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
        }

        // 2d. 23 ViT Encoder Layers
        for (uint32_t l = 0U; l < cfg.num_hidden_layers; ++l) {
            const StrataInkling2VisionLayerDesc& ld = eng->hdr.layers[l];

            // QKV fused projection: [S, 1024] @ [3072, 1024]^T + qkv_b -> b_qkv_f32 [S, 3072]
            sikv_encode_gemm(eng, enc,
                             ld.qkv_w.offset, eng->b_x_ln_f16,
                             ld.qkv_b.offset, true, false, false,
                             nil, eng->b_qkv_f32,
                             S, s_padded, 3U * D, D);

            // Split Q/K/V + 2D Axial Pairwise RoPE on Q,K -> [16, S, 64] F16
            {
                SikvQkvRopeParams rp{};
                rp.S          = S;
                rp.S_padded   = s_padded;
                rp.D          = D;
                rp.num_heads  = cfg.num_attention_heads;
                rp.head_dim   = cfg.head_dim;
                rp.grid_w     = grid_w;
                rp.rope_theta = cfg.rope_theta;
                rp.reserved   = 0U;

                const uint32_t total_pairs = cfg.num_attention_heads * (cfg.head_dim / 2U); // 512
                [enc setComputePipelineState:eng->pso_qkv_split_rope];
                [enc setBuffer:eng->b_qkv_f32 offset:0 atIndex:0];
                [enc setBuffer:eng->b_q_f16   offset:0 atIndex:1];
                [enc setBuffer:eng->b_k_f16   offset:0 atIndex:2];
                [enc setBuffer:eng->b_v_f16   offset:0 atIndex:3];
                [enc setBytes:&rp length:sizeof(rp) atIndex:4];
                [enc dispatchThreads:MTLSizeMake(total_pairs, s_padded, 1U)
               threadsPerThreadgroup:MTLSizeMake(64U, 2U, 1U)];
            }

            // Full bidirectional FlashAttention -> b_attn_out_f16 [S_padded, 1024]
            {
                SikvAttnParams ap{};
                ap.S         = S;
                ap.S_padded  = s_padded;
                ap.num_heads = cfg.num_attention_heads;
                ap.head_dim  = cfg.head_dim;
                ap.scale     = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

                if (eng->has_nax && eng->pso_flash_attn_nax != nil && ((s_padded % 32U) == 0U)) {
                    const uint32_t q_tgs = (s_padded + 63U) / 64U;
                    [enc setComputePipelineState:eng->pso_flash_attn_nax];
                    [enc setBuffer:eng->b_q_f16        offset:0 atIndex:0];
                    [enc setBuffer:eng->b_k_f16        offset:0 atIndex:1];
                    [enc setBuffer:eng->b_v_f16        offset:0 atIndex:2];
                    [enc setBuffer:eng->b_attn_out_f16 offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(q_tgs, cfg.num_attention_heads, 1U)
                        threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
                } else {
                    const uint32_t q_tgs = (s_padded + 3U) / 4U;
                    [enc setComputePipelineState:eng->pso_flash_attn_hd64];
                    [enc setBuffer:eng->b_q_f16        offset:0 atIndex:0];
                    [enc setBuffer:eng->b_k_f16        offset:0 atIndex:1];
                    [enc setBuffer:eng->b_v_f16        offset:0 atIndex:2];
                    [enc setBuffer:eng->b_attn_out_f16 offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(q_tgs, cfg.num_attention_heads, 1U)
                        threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
                }
            }

            // Attention output projection: [S, 1024] @ [1024, 1024]^T + out_b -> b_tmp_f32
            sikv_encode_gemm(eng, enc,
                             ld.out_w.offset, eng->b_attn_out_f16,
                             ld.out_b.offset, true, false, false,
                             nil, eng->b_tmp_f32,
                             S, s_padded, D, D);

            // Fused LayerScale ls_1 + Residual Add + Pre-MLP LayerNorm (`ln_2`) -> b_x_ln_f16
            {
                SikvResScaleLnParams rlp{};
                rlp.S        = S;
                rlp.S_padded = s_padded;
                rlp.D        = D;
                rlp.do_ln    = 1U;
                rlp.ln_eps   = cfg.layer_norm_eps;
                rlp.reserved = 0U;

                [enc setComputePipelineState:eng->pso_res_scale_ln];
                [enc setBuffer:eng->b_x_f32   offset:0 atIndex:0];
                [enc setBuffer:eng->b_tmp_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.ls1_gamma.offset) atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.ln2_w.offset)     atIndex:3];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.ln2_b.offset)     atIndex:4];
                [enc setBuffer:eng->b_x_ln_f16 offset:0 atIndex:5];
                [enc setBytes:&rlp length:sizeof(rlp) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(s_padded, 1U, 1U)
                    threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
            }

            // MLP fc1 + bias + exact erf-GELU -> b_mlp_f16 [S_padded, 4096]
            sikv_encode_gemm(eng, enc,
                             ld.fc1_w.offset, eng->b_x_ln_f16,
                             ld.fc1_b.offset, true, true, true,
                             eng->b_mlp_f16, nil,
                             S, s_padded, MLP, D);

            // MLP fc2 + bias -> b_tmp_f32 [S, 1024]
            sikv_encode_gemm(eng, enc,
                             ld.fc2_w.offset, eng->b_mlp_f16,
                             ld.fc2_b.offset, true, false, false,
                             nil, eng->b_tmp_f32,
                             S, s_padded, D, MLP);

            // Fused LayerScale ls_2 + Residual Add (+ next layer's `ln_1` if l + 1 < num_layers)
            {
                const bool has_next = (l + 1U < cfg.num_hidden_layers);
                SikvResScaleLnParams rlp{};
                rlp.S        = S;
                rlp.S_padded = s_padded;
                rlp.D        = D;
                rlp.do_ln    = has_next ? 1U : 0U;
                rlp.ln_eps   = cfg.layer_norm_eps;
                rlp.reserved = 0U;

                const uint64_t next_w = has_next ? eng->hdr.layers[l + 1U].ln1_w.offset : ld.ln1_w.offset;
                const uint64_t next_b = has_next ? eng->hdr.layers[l + 1U].ln1_b.offset : ld.ln1_b.offset;

                [enc setComputePipelineState:eng->pso_res_scale_ln];
                [enc setBuffer:eng->b_x_f32   offset:0 atIndex:0];
                [enc setBuffer:eng->b_tmp_f32 offset:0 atIndex:1];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(ld.ls2_gamma.offset) atIndex:2];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(next_w)              atIndex:3];
                [enc setBuffer:eng->b_uma offset:static_cast<NSUInteger>(next_b)              atIndex:4];
                [enc setBuffer:eng->b_x_ln_f16 offset:0 atIndex:5];
                [enc setBytes:&rlp length:sizeof(rlp) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(s_padded, 1U, 1U)
                    threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
            }
        }

        // 2e. Drop CLS token + cast [N_vis, 4096] F32 -> b_merged_f16 [N_vis_padded, 4096]
        {
            const uint32_t dims[4] = { n_vis, n_vis_padded, MERGED_D, D };
            [enc setComputePipelineState:eng->pso_drop_cls_f16];
            [enc setBuffer:eng->b_x_f32      offset:0 atIndex:0];
            [enc setBuffer:eng->b_merged_f16 offset:0 atIndex:1];
            [enc setBytes:dims length:sizeof(dims) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(MERGED_D, n_vis_padded, 1U)
           threadsPerThreadgroup:MTLSizeMake(128U, 1U, 1U)];
        }

        // 2f. Projector Layer 1 (`proj_in`, no bias) + exact erf-GELU -> b_proj_mid_f16 [N_vis_padded, 4096]
        sikv_encode_gemm(eng, enc,
                         eng->hdr.proj_in_w.offset, eng->b_merged_f16,
                         0ULL, false, true, true,
                         eng->b_proj_mid_f16, nil,
                         n_vis, n_vis_padded, PROJ_H, MERGED_D);

        // 2g. Projector Layer 2 (`proj_out`, no bias, no final norm) -> b_out_f32 [N_vis, 2304]
        sikv_encode_gemm(eng, enc,
                         eng->hdr.proj_out_w.offset, eng->b_proj_mid_f16,
                         0ULL, false, false, false,
                         nil, eng->b_out_f32,
                         n_vis, n_vis_padded, TEXT_D, PROJ_H);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if ([cb status] != MTLCommandBufferStatusCompleted) {
            NSError* cb_err = [cb error];
            std::fprintf(stderr, "[strata_inkling2_vision] CommandBuffer failed: %s\n",
                         cb_err ? [[cb_err localizedDescription] UTF8String] : "unknown");
            return -4;
        }

        const size_t total_out = size_t(n_vis) * size_t(TEXT_D);
        out_emb_f32->resize(total_out);
        std::memcpy(out_emb_f32->data(), [eng->b_out_f32 contents], total_out * sizeof(float));
    }

    const double encode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    eng->last_encode_ms  = encode_ms;
    eng->last_num_tokens = n_vis;
    eng->last_grid_h     = grid_h;
    eng->last_grid_w     = grid_w;

    if (out_num_tokens) *out_num_tokens = n_vis;
    if (out_grid_h)     *out_grid_h     = grid_h;
    if (out_grid_w)     *out_grid_w     = grid_w;
    return 0;
}

int strata_inkling2_vision_encode_image_bytes(
    StrataInkling2VisionEngine* eng,
    const uint8_t*              img_bytes,
    size_t                      img_len,
    std::vector<float>*         out_emb_f32,
    uint32_t*                   out_num_tokens,
    uint32_t*                   out_grid_h,
    uint32_t*                   out_grid_w
) {
    if (!eng || !eng->ready || !img_bytes || img_len == 0U || !out_emb_f32) {
        return -1;
    }

    const uint32_t cell_px = eng->hdr.cfg.native_patch_size * eng->hdr.cfg.adapter_merge_size;
    std::vector<uint8_t> rgb;
    uint32_t w = 0U, h = 0U;
    const int dec_rc = sikv_decode_image_bytes_to_rgb(
        img_bytes, img_len, cell_px, STRATA_INKLING2_VISION_MAX_TOKENS, rgb, w, h);
    if (dec_rc != 0) {
        std::fprintf(stderr, "[strata_inkling2_vision] Failed to decode image bytes (%zu B, rc=%d)\n",
                     img_len, dec_rc);
        return dec_rc - 10;
    }

    return strata_inkling2_vision_encode_rgb_u8(
        eng, rgb.data(), w, h, out_emb_f32, out_num_tokens, out_grid_h, out_grid_w);
}

int strata_inkling2_vision_encode_image_file(
    StrataInkling2VisionEngine* eng,
    const char*                 image_path,
    std::vector<float>*         out_emb_f32,
    uint32_t*                   out_num_tokens,
    uint32_t*                   out_grid_h,
    uint32_t*                   out_grid_w
) {
    if (!eng || !image_path || image_path[0] == '\0' || !out_emb_f32) {
        return -1;
    }

    std::string path(image_path);
    if (path.rfind("file://", 0) == 0) {
        path = path.substr(7U);
    }

    std::ifstream ifs(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!ifs.good()) {
        std::fprintf(stderr, "[strata_inkling2_vision] Cannot read image file: %s\n", path.c_str());
        return -2;
    }
    const std::streamsize sz = ifs.tellg();
    if (sz <= 0 || sz > (256LL << 20U)) {
        std::fprintf(stderr, "[strata_inkling2_vision] Invalid image file size (%lld B): %s\n",
                     static_cast<long long>(sz), path.c_str());
        return -3;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    if (!ifs.read(reinterpret_cast<char*>(bytes.data()), sz)) {
        return -4;
    }

    return strata_inkling2_vision_encode_image_bytes(
        eng, bytes.data(), bytes.size(), out_emb_f32, out_num_tokens, out_grid_h, out_grid_w);
}

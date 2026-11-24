/*
 * STRATA-M5 v2.0 — Model Architecture Presets, 256 KiB Container Packer & Safetensors Hook
 * Owned by: models-mlx (src/model/strata_pack.cpp)
 */

#include "../../include/strata_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kMaxSafeContainerBytes = 1500ULL * 1000ULL * 1000ULL; /* 1.5 GB guardrail (#15) */
constexpr uint64_t kMaxSafetensorsHeader  = 100ULL * 1024ULL * 1024ULL;  /* 100 MiB sanity cap */
constexpr uint32_t kRansMagic             = 0x534E4152U;                 /* "RANS" */

static inline uint64_t align_up_u64(uint64_t val, uint64_t align) {
    return (val + align - 1ULL) & ~(align - 1ULL);
}

/* Fast deterministic SplitMix64 PRNG so packed test extents contain high-entropy
 * non-zero pages (preventing APFS sparse-hole or zero-page shortcuts on ANS NAND). */
static inline uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Computes exact logical compressed payload bytes per cold expert per #12 v2.0 spec */
static uint64_t compute_logical_payload_bytes(const StrataModelConfig* cfg) {
    const uint64_t expert_params = 3ULL * static_cast<uint64_t>(cfg->d_model) *
                                   static_cast<uint64_t>(cfg->d_ff);
    double bpw = 3.52;
    if (cfg->regime == STRATA_REGIME_I_BITNEST) {
        bpw = 1.85; /* 4-level Laplacian delta-W_refine Tile-rANS (#12 §1.2: 5.82 MB) */
    } else if (cfg->regime == STRATA_REGIME_II_BARY_SVD) {
        bpw = 3.52; /* Standalone lossless Tile-rANS MXFP4 (#12 §1.1: 15.57 MB) */
    } else {
        bpw = 3.52; /* Lossless Tile-rANS MXFP4 (#12 §2.A) */
    }
    uint64_t raw = static_cast<uint64_t>(llround(static_cast<double>(expert_params) * bpw / 8.0));
    const uint64_t aligned = align_up_u64(cfg->cold_extent_bytes, STRATA_LBA_ALIGN_BYTES);
    if (raw < sizeof(StrataTileRansHeader)) {
        raw = sizeof(StrataTileRansHeader);
    }
    if (raw > aligned) {
        raw = aligned;
    }
    return raw;
}

/* Minimal, zero-dependency RFC-8259 JSON scanner tailored for Hugging Face .safetensors headers.
 * Structure of a .safetensors JSON header:
 *   {
 *     "__metadata__": { "format": "pt", ... },
 *     "model.layers.0.mlp.experts.0.w1.weight": {
 *       "dtype": "F8_E4M3",
 *       "shape": [2304, 5120],
 *       "data_offsets": [0, 11796480]
 *     }, ...
 *   }
 */
struct JsonCursor {
    const char* s;
    size_t len;
    size_t pos;

    void skip_ws() {
        while (pos < len) {
            char c = s[pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos;
            } else {
                break;
            }
        }
    }

    bool consume(char expected) {
        skip_ws();
        if (pos < len && s[pos] == expected) {
            ++pos;
            return true;
        }
        return false;
    }

    bool parse_string(std::string* out) {
        skip_ws();
        if (pos >= len || s[pos] != '"') return false;
        ++pos;
        out->clear();
        while (pos < len) {
            char c = s[pos++];
            if (c == '"') {
                return true;
            }
            if (c == '\\') {
                if (pos >= len) return false;
                char esc = s[pos++];
                if (esc == 'u') {
                    if (pos + 4 > len) return false;
                    pos += 4;
                    out->push_back('?');
                } else {
                    out->push_back(esc);
                }
            } else {
                out->push_back(c);
            }
        }
        return false;
    }

    bool parse_u64(uint64_t* out) {
        skip_ws();
        if (pos >= len || s[pos] < '0' || s[pos] > '9') return false;
        uint64_t v = 0;
        while (pos < len && s[pos] >= '0' && s[pos] <= '9') {
            uint64_t digit = static_cast<uint64_t>(s[pos] - '0');
            if (v > (UINT64_MAX - digit) / 10ULL) return false;
            v = v * 10ULL + digit;
            ++pos;
        }
        *out = v;
        return true;
    }

    bool skip_value(int depth = 0) {
        if (depth > 128) return false;
        skip_ws();
        if (pos >= len) return false;
        char c = s[pos];
        if (c == '"') {
            std::string dummy;
            return parse_string(&dummy);
        }
        if (c == '{') {
            ++pos;
            if (consume('}')) return true;
            while (true) {
                std::string k;
                if (!parse_string(&k)) return false;
                if (!consume(':')) return false;
                if (!skip_value(depth + 1)) return false;
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == '[') {
            ++pos;
            if (consume(']')) return true;
            while (true) {
                if (!skip_value(depth + 1)) return false;
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        /* primitive: number, true, false, null */
        size_t start = pos;
        while (pos < len) {
            char ch = s[pos];
            if (ch == ',' || ch == '}' || ch == ']' ||
                ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                break;
            }
            ++pos;
        }
        return pos > start;
    }
};

}  // namespace

extern "C" {

StrataModelConfig strata_model_config_qwen38_flash_next(void) {
    StrataModelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.name                  = "Qwen3.8-Flash-Next";
    cfg.regime                = STRATA_REGIME_I_100PCT_UMA;
    cfg.num_layers            = 48U;     /* 12 x (3 GDN + 1 QSA) */
    cfg.encoder_layers        = 0U;
    cfg.d_model               = 2560U;
    cfg.d_ff                  = 640U;
    cfg.num_experts           = 512U;
    cfg.top_k                 = 10U;
    cfg.vocab_size            = 248320U;
    cfg.shortlist_size        = STRATA_SHORTLIST_VOCAB; /* 32,768 */
    cfg.svd_rank              = 0U;
    cfg.pinned_shallow_layers = 48U;     /* 100% UMA resident once 51.2B PLE is offloaded */
    cfg.has_ngram_table       = true;    /* 51.2B PLE table (320M x 160) @ Layer 2 */
    cfg.ngram_heads_per_tok   = 4U;      /* Bigram + trigram split hash lookups */
    cfg.mode_a_tau_tok        = 0.0f;    /* 0 expert drop in Mode A -> exact U(B) */
    cfg.mode_b_cold_cap       = 0U;      /* 0 cold NVMe expert fetches needed */
    cfg.cold_extent_bytes     = 9ULL * STRATA_LBA_ALIGN_BYTES; /* 2.25 MiB (9 x 256 KiB) */
    return cfg;
}

StrataModelConfig strata_model_config_deepseek_v41_flash(void) {
    StrataModelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.name                  = "DeepSeek-V4.1-Flash-1017";
    cfg.regime                = STRATA_REGIME_II_BARY_SVD;
    cfg.num_layers            = 40U;     /* 20-layer Causal Encoder + 20-layer Decoder (CED) */
    cfg.encoder_layers        = 20U;
    cfg.d_model               = 5120U;
    cfg.d_ff                  = 2304U;
    cfg.num_experts           = 384U;
    cfg.top_k                 = 6U;
    cfg.vocab_size            = 154880U;
    cfg.shortlist_size        = STRATA_SHORTLIST_VOCAB; /* 32,768 */
    cfg.svd_rank              = 108U;    /* W6A16 MoE-SVD r=108 (0.5832 MB/expert, #12 §1.1) */
    cfg.pinned_shallow_layers = 8U;      /* L0..L3 (Encoder) + L20..L23 (Decoder) */
    cfg.has_ngram_table       = true;    /* 196B Engram conditional memory @ [1, 14] */
    cfg.ngram_heads_per_tok   = 24U;     /* Multi-head N-gram hashes across L1 & L14 */
    cfg.mode_a_tau_tok        = 0.095f;  /* Causal per-token threshold (k_crit ~ 2.25) */
    cfg.mode_b_cold_cap       = 1U;      /* <= 1 cold NVMe fetch/layer -> rho_NVMe = 0.923 < 1 */
    cfg.cold_extent_bytes     = 60ULL * STRATA_LBA_ALIGN_BYTES; /* 15.00 MiB (15.73 MB >= 15.57 MB) */
    return cfg;
}

StrataModelConfig strata_model_config_glm53_flash(void) {
    StrataModelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.name                  = "GLM-5.3-Flash";
    cfg.regime                = STRATA_REGIME_I_BITNEST;
    cfg.num_layers            = 45U;     /* 3 dense + 42 MoE (34 KDA linear + 11 NoPE Sparse MLA) */
    cfg.encoder_layers        = 0U;
    cfg.d_model               = 4096U;
    cfg.d_ff                  = 2048U;
    cfg.num_experts           = 288U;
    cfg.top_k                 = 8U;
    cfg.vocab_size            = 154880U;
    cfg.shortlist_size        = STRATA_SHORTLIST_VOCAB; /* 32,768 */
    cfg.svd_rank              = 0U;
    cfg.pinned_shallow_layers = 7U;      /* L0..L2 dense + L3..L6 pinned MoE -> 38 dynamic layers */
    cfg.has_ngram_table       = false;
    cfg.ngram_heads_per_tok   = 0U;
    cfg.mode_a_tau_tok        = 0.080f;  /* Causal per-token threshold (k_crit ~ 2.65) */
    cfg.mode_b_cold_cap       = 2U;      /* Cold cap -> rho_NVMe = 0.747 < 1 */
    cfg.cold_extent_bytes     = 23ULL * STRATA_LBA_ALIGN_BYTES; /* 5.75 MiB (6.03 MB >= 5.82 MB) */
    return cfg;
}

int strata_pack_write_container(
    const char* path,
    const StrataModelConfig* cfg,
    uint32_t num_extents,
    StrataExtentRecord* extents_out
) {
    if (!path || !cfg || cfg->cold_extent_bytes == 0 || cfg->num_layers == 0 || cfg->num_experts == 0) {
        return -EINVAL;
    }

    const uint64_t aligned_extent = align_up_u64(cfg->cold_extent_bytes, STRATA_LBA_ALIGN_BYTES);
    const uint64_t payload_bytes  = compute_logical_payload_bytes(cfg);
    const uint64_t manifest_bytes = sizeof(StrataContainerHeader) +
                                    static_cast<uint64_t>(num_extents) * sizeof(StrataExtentRecord);
    const uint64_t data_start     = std::max<uint64_t>(
        STRATA_LBA_ALIGN_BYTES,
        align_up_u64(manifest_bytes, STRATA_LBA_ALIGN_BYTES)
    );
    const uint64_t total_bytes    = data_start + static_cast<uint64_t>(num_extents) * aligned_extent;

    if (total_bytes > kMaxSafeContainerBytes) {
        /* Enforce strict <= 1.5 GB PoC disk guardrail (#15) */
        return -EFBIG;
    }

    std::vector<StrataExtentRecord> records(num_extents);
    const uint32_t start_layer = (cfg->pinned_shallow_layers < cfg->num_layers)
                                     ? cfg->pinned_shallow_layers
                                     : 0U;
    const uint32_t dyn_layers  = (cfg->num_layers > start_layer)
                                     ? (cfg->num_layers - start_layer)
                                     : cfg->num_layers;

    for (uint32_t i = 0; i < num_extents; ++i) {
        records[i].layer_idx     = start_layer + (i / cfg->num_experts) % dyn_layers;
        records[i].expert_idx    = i % cfg->num_experts;
        records[i].offset_bytes  = data_start + static_cast<uint64_t>(i) * aligned_extent;
        records[i].payload_bytes = payload_bytes;
        records[i].aligned_bytes = aligned_extent;
    }

    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return -errno;
    }
    (void)::fcntl(fd, F_NOCACHE, 1);

    /* Allocate a 256 KiB-aligned staging chunk for clean F_NOCACHE writes */
    void* raw_chunk = nullptr;
    if (::posix_memalign(&raw_chunk, STRATA_VM_PAGE_BYTES, STRATA_LBA_ALIGN_BYTES) != 0 || !raw_chunk) {
        ::close(fd);
        return -ENOMEM;
    }

    auto write_all = [&](const void* buf, size_t len) -> bool {
        const uint8_t* p = static_cast<const uint8_t*>(buf);
        size_t rem = len;
        while (rem > 0) {
            ssize_t nw = ::write(fd, p, rem);
            if (nw < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            p   += static_cast<size_t>(nw);
            rem -= static_cast<size_t>(nw);
        }
        return true;
    };

    /* 1. Build and pad header + manifest region up to data_start */
    std::vector<uint8_t> meta_region(static_cast<size_t>(data_start), 0U);
    StrataContainerHeader hdr;
    hdr.magic             = STRATA_MAGIC_U64;
    hdr.abi_version       = STRATA_ABI_VERSION;
    hdr.num_extents       = num_extents;
    hdr.data_start_offset = data_start;
    hdr.total_file_bytes  = total_bytes;

    memcpy(meta_region.data(), &hdr, sizeof(hdr));
    if (num_extents > 0) {
        memcpy(meta_region.data() + sizeof(hdr),
               records.data(),
               static_cast<size_t>(num_extents) * sizeof(StrataExtentRecord));
    }

    if (!write_all(meta_region.data(), meta_region.size())) {
        int err = -errno;
        free(raw_chunk);
        ::close(fd);
        return err ? err : -EIO;
    }

    /* 2. Write each 256 KiB-aligned cold expert extent with embedded Tile-rANS framing */
    const uint64_t chunks_per_extent = aligned_extent / STRATA_LBA_ALIGN_BYTES;
    uint64_t* chunk_u64 = static_cast<uint64_t*>(raw_chunk);
    const size_t words_per_chunk = STRATA_LBA_ALIGN_BYTES / sizeof(uint64_t);

    for (uint32_t i = 0; i < num_extents; ++i) {
        uint64_t rng = 0x5354524154414D35ULL ^
                       (static_cast<uint64_t>(records[i].layer_idx) << 32) ^
                       static_cast<uint64_t>(records[i].expert_idx);
        for (uint64_t c = 0; c < chunks_per_extent; ++c) {
            for (size_t w = 0; w < words_per_chunk; ++w) {
                chunk_u64[w] = splitmix64(&rng);
            }
            if (c == 0) {
                /* Stamp valid StrataTileRansHeader at start of extent */
                StrataTileRansHeader rhdr;
                memset(&rhdr, 0, sizeof(rhdr));
                rhdr.magic              = kRansMagic;
                rhdr.uncompressed_count = 3U * cfg->d_model * cfg->d_ff;
                rhdr.compressed_bytes   = static_cast<uint32_t>(
                    std::min<uint64_t>(payload_bytes - sizeof(StrataTileRansHeader), 0xFFFFFFFFULL));
                rhdr.freq_bits          = 12U;
                rhdr.alphabet_size      = (cfg->regime == STRATA_REGIME_I_BITNEST) ? 4U : 16U;
                rhdr.reserved[0]        = records[i].layer_idx;
                rhdr.reserved[1]        = records[i].expert_idx;
                memcpy(raw_chunk, &rhdr, sizeof(rhdr));
            }
            if (!write_all(raw_chunk, STRATA_LBA_ALIGN_BYTES)) {
                int err = -errno;
                free(raw_chunk);
                ::close(fd);
                return err ? err : -EIO;
            }
        }
    }

    (void)::fsync(fd);
    free(raw_chunk);
    ::close(fd);

    if (extents_out && num_extents > 0) {
        memcpy(extents_out, records.data(), static_cast<size_t>(num_extents) * sizeof(StrataExtentRecord));
    }
    return 0;
}

int strata_pack_read_manifest(
    const char* path,
    StrataContainerHeader* hdr_out,
    StrataExtentRecord* extents_out,
    uint32_t max_extents
) {
    if (!path || !hdr_out) {
        return -EINVAL;
    }

    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        return -errno;
    }

    struct stat st;
    if (::fstat(fd, &st) != 0) {
        int err = -errno;
        ::close(fd);
        return err;
    }

    StrataContainerHeader hdr;
    if (::pread(fd, &hdr, sizeof(hdr), 0) != static_cast<ssize_t>(sizeof(hdr))) {
        ::close(fd);
        return -EIO;
    }

    if (hdr.magic != STRATA_MAGIC_U64 ||
        hdr.abi_version != STRATA_ABI_VERSION ||
        (hdr.data_start_offset % STRATA_LBA_ALIGN_BYTES) != 0 ||
        hdr.total_file_bytes != static_cast<uint64_t>(st.st_size)) {
        ::close(fd);
        return -EBADMSG;
    }

    *hdr_out = hdr;

    const uint32_t to_read = (extents_out != nullptr)
                                 ? std::min<uint32_t>(hdr.num_extents, max_extents)
                                 : 0U;
    if (to_read > 0) {
        const size_t bytes = static_cast<size_t>(to_read) * sizeof(StrataExtentRecord);
        if (::pread(fd, extents_out, bytes, static_cast<off_t>(sizeof(hdr))) !=
            static_cast<ssize_t>(bytes)) {
            ::close(fd);
            return -EIO;
        }
        for (uint32_t i = 0; i < to_read; ++i) {
            if ((extents_out[i].offset_bytes % STRATA_LBA_ALIGN_BYTES) != 0 ||
                (extents_out[i].aligned_bytes % STRATA_LBA_ALIGN_BYTES) != 0 ||
                extents_out[i].offset_bytes + extents_out[i].aligned_bytes > hdr.total_file_bytes) {
                ::close(fd);
                return -EBADMSG;
            }
        }
    }

    ::close(fd);
    return 0;
}

int strata_pack_inspect_safetensors_header(
    const char* path,
    uint32_t* num_tensors_out,
    uint64_t* total_payload_bytes_out
) {
    if (!path || !num_tensors_out || !total_payload_bytes_out) {
        return -EINVAL;
    }

    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        return -errno;
    }

    uint8_t len_le[8];
    if (::pread(fd, len_le, 8, 0) != 8) {
        ::close(fd);
        return -EIO;
    }

    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) {
        header_len |= (static_cast<uint64_t>(len_le[i]) << (8 * i));
    }

    if (header_len == 0 || header_len > kMaxSafetensorsHeader) {
        ::close(fd);
        return -EBADMSG;
    }

    std::string json(static_cast<size_t>(header_len), '\0');
    size_t got = 0;
    while (got < header_len) {
        ssize_t nr = ::pread(fd,
                             &json[got],
                             static_cast<size_t>(header_len - got),
                             static_cast<off_t>(8ULL + got));
        if (nr <= 0) {
            ::close(fd);
            return -EIO;
        }
        got += static_cast<size_t>(nr);
    }
    ::close(fd);

    JsonCursor cur{json.data(), json.size(), 0};
    if (!cur.consume('{')) {
        return -EBADMSG;
    }

    uint32_t tensor_count = 0;
    uint64_t total_bytes  = 0;

    if (!cur.consume('}')) {
        while (true) {
            std::string key;
            if (!cur.parse_string(&key) || !cur.consume(':')) {
                return -EBADMSG;
            }
            if (key == "__metadata__") {
                if (!cur.skip_value()) return -EBADMSG;
            } else {
                /* Parse tensor descriptor object: must contain data_offsets: [begin, end] */
                if (!cur.consume('{')) return -EBADMSG;
                bool found_offsets = false;
                uint64_t begin_off = 0;
                uint64_t end_off   = 0;

                if (!cur.consume('}')) {
                    while (true) {
                        std::string subkey;
                        if (!cur.parse_string(&subkey) || !cur.consume(':')) {
                            return -EBADMSG;
                        }
                        if (subkey == "data_offsets") {
                            if (!cur.consume('[') ||
                                !cur.parse_u64(&begin_off) ||
                                !cur.consume(',') ||
                                !cur.parse_u64(&end_off) ||
                                !cur.consume(']')) {
                                return -EBADMSG;
                            }
                            if (end_off < begin_off) return -EBADMSG;
                            found_offsets = true;
                        } else {
                            if (!cur.skip_value()) return -EBADMSG;
                        }
                        if (cur.consume('}')) break;
                        if (!cur.consume(',')) return -EBADMSG;
                    }
                }
                if (!found_offsets) return -EBADMSG;
                ++tensor_count;
                total_bytes += (end_off - begin_off);
            }

            if (cur.consume('}')) break;
            if (!cur.consume(',')) return -EBADMSG;
        }
    }

    *num_tensors_out         = tensor_count;
    *total_payload_bytes_out = total_bytes;
    return 0;
}

/* Extra model-slice helper used by test_model to verify 4-Axis Disaggregation
 * routing on real Qwen3.8-Flash-Next (PLE) and DeepSeek-V4.1-Flash-1017 (Engram) tensor names. */
int strata_pack_classify_safetensors_axes(
    const char* path,
    uint32_t* axis_a_ngram_out,
    uint32_t* axis_b_spine_router_out,
    uint32_t* axis_cd_routed_experts_out
) {
    if (!path || !axis_a_ngram_out || !axis_b_spine_router_out || !axis_cd_routed_experts_out) {
        return -EINVAL;
    }

    int fd = ::open(path, O_RDONLY);
    if (fd < 0) return -errno;

    uint8_t len_le[8];
    if (::pread(fd, len_le, 8, 0) != 8) {
        ::close(fd);
        return -EIO;
    }
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) {
        header_len |= (static_cast<uint64_t>(len_le[i]) << (8 * i));
    }
    if (header_len == 0 || header_len > kMaxSafetensorsHeader) {
        ::close(fd);
        return -EBADMSG;
    }

    std::string json(static_cast<size_t>(header_len), '\0');
    if (::pread(fd, &json[0], static_cast<size_t>(header_len), 8) !=
        static_cast<ssize_t>(header_len)) {
        ::close(fd);
        return -EIO;
    }
    ::close(fd);

    JsonCursor cur{json.data(), json.size(), 0};
    if (!cur.consume('{')) return -EBADMSG;

    uint32_t a_cnt = 0, b_cnt = 0, cd_cnt = 0;
    if (!cur.consume('}')) {
        while (true) {
            std::string key;
            if (!cur.parse_string(&key) || !cur.consume(':')) return -EBADMSG;
            if (!cur.skip_value()) return -EBADMSG;

            if (key != "__metadata__") {
                if (key.find(".ple.") != std::string::npos ||
                    key.find("ngram") != std::string::npos ||
                    key.find("engram") != std::string::npos) {
                    ++a_cnt;
                } else if (key.find("mlp.experts.") != std::string::npos) {
                    ++cd_cnt;
                } else {
                    ++b_cnt;
                }
            }
            if (cur.consume('}')) break;
            if (!cur.consume(',')) return -EBADMSG;
        }
    }

    *axis_a_ngram_out           = a_cnt;
    *axis_b_spine_router_out    = b_cnt;
    *axis_cd_routed_experts_out = cd_cnt;
    return 0;
}

}  // extern "C"

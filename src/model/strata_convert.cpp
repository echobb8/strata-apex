/*
 * STRATA-M5 v2.0 — `.safetensors` Synthesizer, 4-Axis Disaggregator/Converter & Bundle Loader
 * File: src/model/strata_convert.cpp
 *
 * Implements:
 *   1. Full RFC-8259 `.safetensors` reader supporting F32, F16, BF16, and F8_E4M3 tensors,
 *      plus `tokenizer.json` / `vocab.txt` / `config.json` parsing.
 *   2. `strata_synth_safetensors_checkpoint`: generates valid HuggingFace `.safetensors` +
 *      `tokenizer.json` + `vocab.txt` + `config.json` (`mini-deepseek`, `mini-qwen`, `mini-glm`)
 *      with well-conditioned orthogonal/planted weights & readable English subwords (< 80 MiB).
 *   3. `strata_convert_safetensors_to_bundle`: disaggregates `.safetensors` into `<bundle_dir>/`
 *      across all 4 STRATA hardware axes:
 *        - Axis A (`engram.ngram`): 4 KiB-aligned INT4 nibbles + FP16 ZT0 codebook (`NGRM`).
 *        - Axis B (`spine.bin`): 32x32 Metal 4 NAX `wCT`-tiled E4M3 + 32-way Tile-rANS Q/K/V/O
 *          & LM head (verified round-trip via `strata_codec_rans_decode_32way_cpu_ref`), plus
 *          FP16 RMSNorms, Embeddings, Routers, and FR-Spec MTP Draft Head.
 *        - Axis C (`uma_hot.bin`): 256B-aligned NAX `wCT` planar MXFP4 (17 B/32w) or QTIP (34 B/128w)
 *          hot experts, Barycenter (`mean_e W_e`) / QTIP base pool, and truncated-SVD W6A16 `U_e` + FP16 `V_e`.
 *        - Axis D (`cold_experts.strata`): 256 KiB LBA-aligned (`STRATA_LBA_ALIGN_BYTES`) `F_NOCACHE`
 *          cold expert extents in exact Metal NAX planar layout for zero-copy `MTLIOCommandQueue` DMA.
 *   4. `strata_bundle_load`, `strata_bundle_free`, `strata_tokenize_prompt`, `strata_detokenize_token`.
 */

#include "../../include/strata_checkpoint.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint32_t kSpineFileMagic  = 0x424E5053U; /* "SPNB" */
constexpr uint32_t kUmaHotFileMagic = 0x43414D55U; /* "UMAC" */
constexpr uint32_t kBundleFileVer   = 200U;
constexpr uint32_t kNgramPageMagic  = 0x4D52474EU; /* "NGRM" (matches strata_sme2_io.mm) */
constexpr size_t   kNgramPageBytes  = 4096U;
constexpr uint64_t kMaxHeaderBytes  = 100ULL * 1024ULL * 1024ULL;

/* ============================================================================
 * 1. Bit-Exact Scalar Conversion & PRNG Helpers
 * ============================================================================ */

static inline uint64_t align_up_u64(uint64_t val, uint64_t align) {
    return (val + align - 1ULL) & ~(align - 1ULL);
}

static inline uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

static inline float rng_uniform_f32(uint64_t* state) {
    return static_cast<float>((splitmix64(state) >> 40U) * (1.0 / 16777216.0));
}

static inline float rng_normal_f32(uint64_t* state, float sigma) {
    /* Marsaglia polar or Irwin-Hall 12-uniform Gaussian (exact bounded support, fast, deterministic) */
    float acc = -6.0f;
    for (int i = 0; i < 12; ++i) {
        acc += rng_uniform_f32(state);
    }
    return acc * sigma;
}

static inline uint16_t f32_to_fp16_bits(float val) {
    uint32_t fbits = 0;
    std::memcpy(&fbits, &val, sizeof(uint32_t));
    uint32_t sign = (fbits >> 16U) & 0x8000U;
    int32_t  exp  = static_cast<int32_t>((fbits >> 23U) & 0xFFU) - 127 + 15;
    uint32_t man  = fbits & 0x007FFFFFU;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        man |= 0x00800000U;
        uint32_t shift    = static_cast<uint32_t>(14 - exp);
        uint32_t half_man = man >> shift;
        uint32_t rem      = man & ((1U << shift) - 1U);
        uint32_t halfway  = 1U << (shift - 1U);
        if (rem > halfway || (rem == halfway && (half_man & 1U))) {
            ++half_man;
        }
        return static_cast<uint16_t>(sign | half_man);
    } else if (exp >= 31) {
        if (((fbits >> 23U) & 0xFFU) == 0xFFU && man != 0U) {
            return static_cast<uint16_t>(sign | 0x7E00U);
        }
        return static_cast<uint16_t>(sign | 0x7BFFU);
    }
    uint32_t half_man = man >> 13U;
    uint32_t rem      = man & 0x1FFFU;
    if (rem > 0x1000U || (rem == 0x1000U && (half_man & 1U))) {
        ++half_man;
        if (half_man == 0x0400U) {
            half_man = 0U;
            ++exp;
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFU);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10U) | half_man);
}

static inline float fp16_bits_to_f32(uint16_t h) {
    uint32_t sign  = static_cast<uint32_t>(h & 0x8000U) << 16U;
    uint32_t exp   = (h >> 10U) & 0x1FU;
    uint32_t man   = h & 0x03FFU;
    uint32_t fbits = 0U;
    if (exp == 0U) {
        if (man == 0U) {
            fbits = sign;
        } else {
            int shift = 0;
            while ((man & 0x0400U) == 0U) {
                man <<= 1U;
                ++shift;
            }
            man &= 0x03FFU;
            uint32_t fexp = static_cast<uint32_t>(127 - 15 - shift + 1);
            fbits = sign | (fexp << 23U) | (man << 13U);
        }
    } else if (exp == 31U) {
        fbits = sign | 0x7F800000U | (man << 13U);
    } else {
        uint32_t fexp = exp + (127U - 15U);
        fbits = sign | (fexp << 23U) | (man << 13U);
    }
    float out = 0.0f;
    std::memcpy(&out, &fbits, sizeof(float));
    return out;
}

static inline uint16_t f32_to_bf16_bits(float val) {
    uint32_t fbits = 0U;
    std::memcpy(&fbits, &val, sizeof(uint32_t));
    if (((fbits >> 23U) & 0xFFU) == 0xFFU && (fbits & 0x7FFFFFU) != 0U) {
        return static_cast<uint16_t>((fbits >> 16U) | 0x0040U);
    }
    uint32_t rounding_bias = 0x00007FFFU + ((fbits >> 16U) & 1U);
    return static_cast<uint16_t>((fbits + rounding_bias) >> 16U);
}

static inline float bf16_bits_to_f32(uint16_t b) {
    uint32_t fbits = static_cast<uint32_t>(b) << 16U;
    float out = 0.0f;
    std::memcpy(&out, &fbits, sizeof(float));
    return out;
}

/* ============================================================================
 * 2. Filesystem & Binary I/O Helpers
 * ============================================================================ */

static bool is_directory_path(const std::string& p) {
    struct stat st;
    if (::stat(p.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return false;
}

static bool file_exists_path(const std::string& p) {
    struct stat st;
    return (::stat(p.c_str(), &st) == 0) && S_ISREG(st.st_mode);
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    return std::memcmp(s.data() + (s.size() - suffix.size()), suffix.data(), suffix.size()) == 0;
}

static std::string parent_dir_of(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

static int mkdir_p(const std::string& dir) {
    if (dir.empty() || dir == "." || dir == "/") return 0;
    std::string cur;
    for (size_t i = 0; i < dir.size(); ++i) {
        cur.push_back(dir[i]);
        if (dir[i] == '/' && cur.size() > 1U) {
            if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return -errno;
        }
    }
    if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return -errno;
    return 0;
}

static bool write_all_fd(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t rem = len;
    while (rem > 0U) {
        ssize_t nw = ::write(fd, p, rem);
        if (nw < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p   += static_cast<size_t>(nw);
        rem -= static_cast<size_t>(nw);
    }
    return true;
}

static bool read_all_fd(int fd, void* buf, size_t len, off_t offset) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t got = 0U;
    while (got < len) {
        ssize_t nr = ::pread(fd, p + got, len - got, offset + static_cast<off_t>(got));
        if (nr <= 0) {
            if (nr < 0 && errno == EINTR) continue;
            return false;
        }
        got += static_cast<size_t>(nr);
    }
    return true;
}

static bool write_binary_file(const std::string& path, const void* data, size_t bytes) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    bool ok = write_all_fd(fd, data, bytes);
    ::close(fd);
    return ok;
}

static bool read_entire_file(const std::string& path, std::vector<uint8_t>* out) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
        ::close(fd);
        return false;
    }
    out->resize(static_cast<size_t>(st.st_size));
    bool ok = out->empty() || read_all_fd(fd, out->data(), out->size(), 0);
    ::close(fd);
    return ok;
}

struct BinaryWriter {
    std::vector<uint8_t> buf;

    void write_bytes(const void* src, size_t n) {
        if (n == 0U) return;
        const uint8_t* p = static_cast<const uint8_t*>(src);
        buf.insert(buf.end(), p, p + n);
    }
    void write_u32(uint32_t v) { write_bytes(&v, sizeof(v)); }
    void write_u64(uint64_t v) { write_bytes(&v, sizeof(v)); }
    void write_f32(float v)    { write_bytes(&v, sizeof(v)); }
    void align_to(size_t a) {
        while ((buf.size() % a) != 0U) buf.push_back(0U);
    }
};

struct BinaryReader {
    const uint8_t* data;
    size_t size;
    size_t pos;

    size_t remaining() const { return (pos <= size) ? (size - pos) : 0U; }

    bool read_bytes(void* dst, size_t n) {
        if (pos > size || n > size - pos) return false;
        if (n > 0U) std::memcpy(dst, data + pos, n);
        pos += n;
        return true;
    }
    bool read_u32(uint32_t* v) { return read_bytes(v, sizeof(*v)); }
    bool read_u64(uint64_t* v) { return read_bytes(v, sizeof(*v)); }
    bool read_f32(float* v)    { return read_bytes(v, sizeof(*v)); }
    bool align_to(size_t a) {
        size_t rem = pos % a;
        if (rem != 0U) {
            size_t pad = a - rem;
            if (pos + pad > size) return false;
            pos += pad;
        }
        return true;
    }
};

/* ============================================================================
 * 3. Full RFC-8259 JSON Value Parser & String Escaper
 * ============================================================================ */

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4U);
    for (unsigned char ch : s) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (ch < 0x20U) {
                    char hex[8];
                    std::snprintf(hex, sizeof(hex), "\\u%04x", static_cast<unsigned>(ch));
                    out += hex;
                } else {
                    out.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return out;
}

struct JsonValue {
    enum Type { kNull, kBool, kNumber, kString, kArray, kObject } type = kNull;
    bool bval = false;
    double nval = 0.0;
    uint64_t uval = 0ULL;
    std::string sval;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* get(const char* key) const {
        if (type != kObject) return nullptr;
        for (const auto& kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

struct Rfc8259Parser {
    const char* s;
    size_t len;
    size_t pos;

    void skip_ws() {
        while (pos < len) {
            char c = s[pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos;
            else break;
        }
    }

    bool consume(char ch) {
        skip_ws();
        if (pos < len && s[pos] == ch) {
            ++pos;
            return true;
        }
        return false;
    }

    static void append_utf8(std::string* out, uint32_t cp) {
        if (cp <= 0x7FU) {
            out->push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FFU) {
            out->push_back(static_cast<char>(0xC0U | (cp >> 6U)));
            out->push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        } else if (cp <= 0xFFFFU) {
            out->push_back(static_cast<char>(0xE0U | (cp >> 12U)));
            out->push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
            out->push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        } else {
            out->push_back(static_cast<char>(0xF0U | (cp >> 18U)));
            out->push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
            out->push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
            out->push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        }
    }

    bool parse_hex4(uint32_t* out) {
        if (pos + 4U > len) return false;
        uint32_t v = 0U;
        for (int i = 0; i < 4; ++i) {
            char c = s[pos++];
            v <<= 4U;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else return false;
        }
        *out = v;
        return true;
    }

    bool parse_string(std::string* out) {
        skip_ws();
        if (pos >= len || s[pos] != '"') return false;
        ++pos;
        out->clear();
        while (pos < len) {
            unsigned char c = static_cast<unsigned char>(s[pos++]);
            if (c == '"') return true;
            if (c < 0x20U) return false;
            if (c == '\\') {
                if (pos >= len) return false;
                char esc = s[pos++];
                switch (esc) {
                    case '"':  out->push_back('"');  break;
                    case '\\': out->push_back('\\'); break;
                    case '/':  out->push_back('/');  break;
                    case 'b':  out->push_back('\b'); break;
                    case 'f':  out->push_back('\f'); break;
                    case 'n':  out->push_back('\n'); break;
                    case 'r':  out->push_back('\r'); break;
                    case 't':  out->push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = 0U;
                        if (!parse_hex4(&cp)) return false;
                        if (cp >= 0xD800U && cp <= 0xDBFFU && pos + 6U <= len &&
                            s[pos] == '\\' && s[pos + 1U] == 'u') {
                            pos += 2U;
                            uint32_t low = 0U;
                            if (!parse_hex4(&low) || low < 0xDC00U || low > 0xDFFFU) return false;
                            cp = 0x10000U + (((cp - 0xD800U) << 10U) | (low - 0xDC00U));
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: return false;
                }
            } else {
                out->push_back(static_cast<char>(c));
            }
        }
        return false;
    }

    bool parse_value(JsonValue* v, int depth = 0) {
        if (depth > 128) return false;
        skip_ws();
        if (pos >= len) return false;
        char c = s[pos];
        if (c == '"') {
            v->type = JsonValue::kString;
            return parse_string(&v->sval);
        }
        if (c == '{') {
            ++pos;
            v->type = JsonValue::kObject;
            v->obj.clear();
            if (consume('}')) return true;
            while (true) {
                std::string key;
                if (!parse_string(&key)) return false;
                if (!consume(':')) return false;
                JsonValue child;
                if (!parse_value(&child, depth + 1)) return false;
                v->obj.emplace_back(std::move(key), std::move(child));
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == '[') {
            ++pos;
            v->type = JsonValue::kArray;
            v->arr.clear();
            if (consume(']')) return true;
            while (true) {
                JsonValue elem;
                if (!parse_value(&elem, depth + 1)) return false;
                v->arr.push_back(std::move(elem));
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        if (pos + 4U <= len && std::memcmp(s + pos, "true", 4U) == 0) {
            pos += 4U;
            v->type = JsonValue::kBool;
            v->bval = true;
            return true;
        }
        if (pos + 5U <= len && std::memcmp(s + pos, "false", 5U) == 0) {
            pos += 5U;
            v->type = JsonValue::kBool;
            v->bval = false;
            return true;
        }
        if (pos + 4U <= len && std::memcmp(s + pos, "null", 4U) == 0) {
            pos += 4U;
            v->type = JsonValue::kNull;
            return true;
        }
        /* Number */
        size_t start = pos;
        bool is_neg = false;
        bool is_flt = false;
        if (pos < len && s[pos] == '-') { is_neg = true; ++pos; }
        uint64_t uacc = 0ULL;
        size_t digits = 0U;
        while (pos < len && s[pos] >= '0' && s[pos] <= '9') {
            uacc = uacc * 10ULL + static_cast<uint64_t>(s[pos] - '0');
            ++pos;
            ++digits;
        }
        if (digits == 0U) return false;
        if (pos < len && s[pos] == '.') {
            is_flt = true;
            ++pos;
            while (pos < len && s[pos] >= '0' && s[pos] <= '9') ++pos;
        }
        if (pos < len && (s[pos] == 'e' || s[pos] == 'E')) {
            is_flt = true;
            ++pos;
            if (pos < len && (s[pos] == '+' || s[pos] == '-')) ++pos;
            while (pos < len && s[pos] >= '0' && s[pos] <= '9') ++pos;
        }
        std::string num_str(s + start, pos - start);
        char* endp = nullptr;
        double dv = std::strtod(num_str.c_str(), &endp);
        if (!endp || endp == num_str.c_str()) return false;
        v->type = JsonValue::kNumber;
        v->nval = dv;
        v->uval = (!is_neg && !is_flt) ? uacc : static_cast<uint64_t>(std::max(0.0, dv));
        return true;
    }
};

/* ============================================================================
 * 4. `.safetensors` Reader & Writer (F32, F16, BF16, F8_E4M3)
 * ============================================================================ */

enum class StDtype { kF32, kF16, kBF16, kF8_E4M3, kU32 };

static size_t dtype_elem_bytes(StDtype dt) {
    switch (dt) {
        case StDtype::kF32:     return 4U;
        case StDtype::kF16:     return 2U;
        case StDtype::kBF16:    return 2U;
        case StDtype::kF8_E4M3: return 1U;
        case StDtype::kU32:     return 4U;
    }
    return 4U;
}

static const char* dtype_name_str(StDtype dt) {
    switch (dt) {
        case StDtype::kF32:     return "F32";
        case StDtype::kF16:     return "F16";
        case StDtype::kBF16:    return "BF16";
        case StDtype::kF8_E4M3: return "F8_E4M3";
        case StDtype::kU32:     return "U32";
    }
    return "F32";
}

static bool parse_st_dtype(const std::string& s, StDtype* out) {
    if (s == "F32")     { *out = StDtype::kF32;     return true; }
    if (s == "F16")     { *out = StDtype::kF16;     return true; }
    if (s == "BF16")    { *out = StDtype::kBF16;    return true; }
    if (s == "F8_E4M3" || s == "F8_E4M3FN") { *out = StDtype::kF8_E4M3; return true; }
    if (s == "U32" || s == "I32") { *out = StDtype::kU32; return true; }
    return false;
}

struct SafetensorEntry {
    std::string name;
    StDtype dtype = StDtype::kF32;
    std::vector<uint64_t> shape;
    uint64_t begin_off = 0ULL;
    uint64_t end_off   = 0ULL;

    uint64_t numel() const {
        if (shape.empty()) return 0ULL;
        uint64_t n = 1ULL;
        for (uint64_t d : shape) n *= d;
        return n;
    }
};

struct SafetensorsReader {
    int fd = -1;
    uint64_t header_len = 0ULL;
    uint64_t data_base  = 0ULL;
    uint64_t file_size  = 0ULL;
    std::unordered_map<std::string, std::string> metadata;
    std::unordered_map<std::string, SafetensorEntry> tensors;

    ~SafetensorsReader() {
        if (fd >= 0) ::close(fd);
    }

    int open_file(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return -errno;
        struct stat st;
        if (::fstat(fd, &st) != 0) return -errno;
        file_size = static_cast<uint64_t>(st.st_size);
        if (file_size < 8ULL) return -EBADMSG;

        uint8_t len_le[8];
        if (!read_all_fd(fd, len_le, 8U, 0)) return -EIO;
        header_len = 0ULL;
        for (int i = 0; i < 8; ++i) {
            header_len |= (static_cast<uint64_t>(len_le[i]) << (8 * i));
        }
        if (header_len == 0ULL || header_len > kMaxHeaderBytes || 8ULL + header_len > file_size) {
            return -EBADMSG;
        }
        data_base = 8ULL + header_len;

        std::string hdr_json(static_cast<size_t>(header_len), '\0');
        if (!read_all_fd(fd, hdr_json.data(), static_cast<size_t>(header_len), 8)) {
            return -EIO;
        }

        Rfc8259Parser parser{hdr_json.data(), hdr_json.size(), 0U};
        JsonValue root;
        if (!parser.parse_value(&root) || root.type != JsonValue::kObject) {
            return -EBADMSG;
        }

        for (const auto& kv : root.obj) {
            if (kv.first == "__metadata__") {
                if (kv.second.type == JsonValue::kObject) {
                    for (const auto& mkv : kv.second.obj) {
                        if (mkv.second.type == JsonValue::kString) {
                            metadata[mkv.first] = mkv.second.sval;
                        }
                    }
                }
                continue;
            }
            if (kv.second.type != JsonValue::kObject) return -EBADMSG;
            const JsonValue* j_dt  = kv.second.get("dtype");
            const JsonValue* j_shp = kv.second.get("shape");
            const JsonValue* j_off = kv.second.get("data_offsets");
            if (!j_dt || j_dt->type != JsonValue::kString ||
                !j_shp || j_shp->type != JsonValue::kArray ||
                !j_off || j_off->type != JsonValue::kArray || j_off->arr.size() != 2U) {
                return -EBADMSG;
            }

            SafetensorEntry ent;
            ent.name = kv.first;
            if (!parse_st_dtype(j_dt->sval, &ent.dtype)) return -EBADMSG;
            for (const auto& dim_v : j_shp->arr) {
                if (dim_v.type != JsonValue::kNumber) return -EBADMSG;
                ent.shape.push_back(dim_v.uval);
            }
            ent.begin_off = j_off->arr[0].uval;
            ent.end_off   = j_off->arr[1].uval;
            if (ent.end_off < ent.begin_off || data_base + ent.end_off > file_size) {
                return -EBADMSG;
            }
            uint64_t expected_bytes = ent.numel() * dtype_elem_bytes(ent.dtype);
            if ((ent.end_off - ent.begin_off) != expected_bytes) {
                return -EBADMSG;
            }
            tensors[ent.name] = std::move(ent);
        }
        return 0;
    }

    const SafetensorEntry* find(const std::string& name) const {
        auto it = tensors.find(name);
        return (it != tensors.end()) ? &it->second : nullptr;
    }

    bool read_tensor_f32(const SafetensorEntry& ent, std::vector<float>* out) const {
        const size_t n     = static_cast<size_t>(ent.numel());
        const size_t bytes = static_cast<size_t>(ent.end_off - ent.begin_off);
        out->resize(n);
        if (n == 0U) return true;

        if (ent.dtype == StDtype::kF32) {
            return read_all_fd(fd, out->data(), bytes, static_cast<off_t>(data_base + ent.begin_off));
        }

        std::vector<uint8_t> raw(bytes);
        if (!read_all_fd(fd, raw.data(), bytes, static_cast<off_t>(data_base + ent.begin_off))) {
            return false;
        }
        float* dst = out->data();
        if (ent.dtype == StDtype::kF16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
            for (size_t i = 0; i < n; ++i) dst[i] = fp16_bits_to_f32(src[i]);
        } else if (ent.dtype == StDtype::kBF16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
            for (size_t i = 0; i < n; ++i) dst[i] = bf16_bits_to_f32(src[i]);
        } else if (ent.dtype == StDtype::kF8_E4M3) {
            for (size_t i = 0; i < n; ++i) {
                dst[i] = fp16_bits_to_f32(strata_codec_e4m3_to_f16_bits(raw[i]));
            }
        } else if (ent.dtype == StDtype::kU32) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(raw.data());
            for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
        }
        return true;
    }
};

struct OutTensorSpec {
    std::string name;
    StDtype dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> raw_bytes;
};

static void append_safetensor_from_f32(
    std::vector<OutTensorSpec>* list,
    const std::string& name,
    StDtype dtype,
    const std::vector<uint64_t>& shape,
    const float* data_f32
) {
    uint64_t n = 1ULL;
    for (uint64_t d : shape) n *= d;

    OutTensorSpec spec;
    spec.name  = name;
    spec.dtype = dtype;
    spec.shape = shape;
    spec.raw_bytes.resize(static_cast<size_t>(n) * dtype_elem_bytes(dtype));

    if (dtype == StDtype::kF32) {
        std::memcpy(spec.raw_bytes.data(), data_f32, spec.raw_bytes.size());
    } else if (dtype == StDtype::kF16) {
        uint16_t* dst = reinterpret_cast<uint16_t*>(spec.raw_bytes.data());
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            dst[i] = f32_to_fp16_bits(data_f32[i]);
        }
    } else if (dtype == StDtype::kBF16) {
        uint16_t* dst = reinterpret_cast<uint16_t*>(spec.raw_bytes.data());
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            dst[i] = f32_to_bf16_bits(data_f32[i]);
        }
    } else if (dtype == StDtype::kF8_E4M3) {
        uint8_t* dst = spec.raw_bytes.data();
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            dst[i] = strata_codec_f32_to_e4m3(data_f32[i]);
        }
    }
    list->push_back(std::move(spec));
}

static int write_safetensors_file(
    const std::string& path,
    const std::vector<std::pair<std::string, std::string>>& meta,
    const std::vector<OutTensorSpec>& tensors
) {
    std::string hdr = "{\"__metadata__\":{";
    for (size_t i = 0; i < meta.size(); ++i) {
        if (i > 0U) hdr += ",";
        hdr += "\"" + json_escape(meta[i].first) + "\":\"" + json_escape(meta[i].second) + "\"";
    }
    hdr += "}";

    uint64_t cur_off = 0ULL;
    for (const auto& t : tensors) {
        uint64_t next_off = cur_off + static_cast<uint64_t>(t.raw_bytes.size());
        hdr += ",\"" + json_escape(t.name) + "\":{\"dtype\":\"";
        hdr += dtype_name_str(t.dtype);
        hdr += "\",\"shape\":[";
        for (size_t d = 0; d < t.shape.size(); ++d) {
            if (d > 0U) hdr += ",";
            hdr += std::to_string(t.shape[d]);
        }
        hdr += "],\"data_offsets\":[" + std::to_string(cur_off) + "," + std::to_string(next_off) + "]}";
        cur_off = next_off;
    }
    hdr += "}";
    /* Pad JSON header with trailing spaces to 8-byte alignment per HuggingFace safetensors convention */
    while ((hdr.size() % 8U) != 0U) {
        hdr.push_back(' ');
    }

    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -errno;

    uint64_t hlen = static_cast<uint64_t>(hdr.size());
    uint8_t len_le[8];
    for (int i = 0; i < 8; ++i) {
        len_le[i] = static_cast<uint8_t>((hlen >> (8 * i)) & 0xFFULL);
    }

    bool ok = write_all_fd(fd, len_le, 8U) && write_all_fd(fd, hdr.data(), hdr.size());
    for (const auto& t : tensors) {
        if (!ok) break;
        if (!t.raw_bytes.empty()) {
            ok = write_all_fd(fd, t.raw_bytes.data(), t.raw_bytes.size());
        }
    }
    ::close(fd);
    return ok ? 0 : -EIO;
}

/* ============================================================================
 * 5. Exact Metal 4 `NAX` `wCT` Bijection, Planar MXFP4 & Truncated-SVD Solvers
 * ============================================================================ */

/* Encodes row-major `[d_out, d_in]` into 32x32 NAX `wCT`-tiled E4M3 + 32-way Tile-rANS,
 * and verifies bit-exact round-trip via `strata_codec_rans_decode_32way_cpu_ref`! */
static bool encode_spine_wct_e4m3_rans(
    const float* w_rowmajor,
    uint32_t d_out,
    uint32_t d_in,
    std::vector<uint8_t>* rans_blob_out,
    std::vector<uint16_t>* rowmajor_f16_decoded_out
) {
    if ((d_out % 32U) != 0U || (d_in % 32U) != 0U) return false;
    const size_t n_weights = static_cast<size_t>(d_out) * static_cast<size_t>(d_in);
    std::vector<uint8_t> w_e4m3_rowmajor(n_weights);
    for (size_t i = 0; i < n_weights; ++i) {
        w_e4m3_rowmajor[i] = strata_codec_f32_to_e4m3(w_rowmajor[i]);
    }
    if (rowmajor_f16_decoded_out) {
        rowmajor_f16_decoded_out->resize(n_weights);
        for (size_t i = 0; i < n_weights; ++i) {
            (*rowmajor_f16_decoded_out)[i] = strata_codec_e4m3_to_f16_bits(w_e4m3_rowmajor[i]);
        }
    }

    const uint32_t num_row_blocks = d_out / 32U;
    const uint32_t num_k_tiles    = d_in / 32U;
    std::vector<uint8_t> w_tiled_e4m3(n_weights);

    for (uint32_t rb = 0; rb < num_row_blocks; ++rb) {
        for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
            const size_t tile_idx  = static_cast<size_t>(rb) * num_k_tiles + kt;
            const size_t tile_base = tile_idx * 1024U;
            for (uint32_t cl = 0; cl < 32U; ++cl) {
                for (uint32_t ln = 0; ln < 32U; ++ln) {
                    const uint32_t k_loc = (((cl >> 4U) & 1U) << 4U) |
                                           (((ln >> 3U) & 1U) << 3U) |
                                           (( ln        & 1U) << 2U) |
                                           (  cl        & 3U);
                    const uint32_t n_loc = (((cl >> 2U) & 3U) << 3U) |
                                           (((ln >> 4U) & 1U) << 2U) |
                                           (( ln >> 1U) & 3U);
                    const size_t src_idx = static_cast<size_t>(rb * 32U + n_loc) * d_in + (kt * 32U + k_loc);
                    const size_t dst_idx = tile_base + static_cast<size_t>(cl) * 32U + ln;
                    w_tiled_e4m3[dst_idx] = w_e4m3_rowmajor[src_idx];
                }
            }
        }
    }

    rans_blob_out->assign(n_weights + (n_weights / 2U) + 65536U, 0U);
    size_t comp_bytes = strata_codec_rans_encode_32way(
        w_tiled_e4m3.data(), n_weights, 256U, rans_blob_out->data(), rans_blob_out->size());
    if (comp_bytes == 0U) return false;
    rans_blob_out->resize(comp_bytes);

    std::vector<uint8_t> decoded(n_weights, 0U);
    int rc = strata_codec_rans_decode_32way_cpu_ref(
        rans_blob_out->data(), comp_bytes, decoded.data(), n_weights);
    if (rc != 0 || std::memcmp(decoded.data(), w_tiled_e4m3.data(), n_weights) != 0) {
        return false;
    }
    return true;
}

/* Exact planar MXFP4 row quantizer matching `strata_gpu.mm` & `strata_kernels.metal` bit-for-bit */
static void quantize_mxfp4_row_planar_host(
    const float* src,
    size_t n_weights,
    uint8_t* nib_out,   /* (n_weights / 32) * 16 bytes */
    uint8_t* scale_out  /* (n_weights / 32) * 1 byte */
) {
    constexpr float kGrid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const size_t groups = n_weights / 32U;
    for (size_t g = 0; g < groups; ++g) {
        const float* w = src + g * 32U;
        float max_abs = 0.0f;
        for (size_t i = 0; i < 32U; ++i) max_abs = std::max(max_abs, std::fabs(w[i]));

        uint8_t s_center = (max_abs > 0.0f) ? strata_codec_f32_to_e4m3((max_abs / 4.5f) * 16.0f) : 0U;
        uint8_t best_s_e4m3 = 0U;
        uint8_t best_packed[16] = {0};
        double best_mse = 1e30;

        for (int delta = -4; delta <= 4; ++delta) {
            int cand_s = static_cast<int>(s_center & 0x7FU) + delta;
            if (cand_s < 0) cand_s = 0;
            if (cand_s > 0x7E) cand_s = 0x7E;
            uint8_t s_e4m3  = static_cast<uint8_t>(cand_s);
            float eff_scale = fp16_bits_to_f32(strata_codec_e4m3_to_f16_bits(s_e4m3)) * 0.0625f;
            float inv_scale = (eff_scale > 0.0f) ? (1.0f / eff_scale) : 0.0f;

            uint8_t cand_packed[16];
            double mse = 0.0;
            for (size_t i = 0; i < 16U; ++i) {
                uint8_t nib[2];
                for (size_t k = 0; k < 2U; ++k) {
                    float orig = w[2U * i + k];
                    float v    = orig * inv_scale;
                    uint8_t sign = (v < 0.0f) ? 8U : 0U;
                    float av = std::fabs(v);
                    uint8_t best_idx = 0U;
                    float best_err = std::fabs(av - kGrid[0]);
                    for (uint8_t m = 1U; m < 8U; ++m) {
                        float err = std::fabs(av - kGrid[m]);
                        if (err < best_err) {
                            best_err = err;
                            best_idx = m;
                        }
                    }
                    nib[k] = static_cast<uint8_t>(sign | best_idx);
                    float rec = ((sign != 0U) ? -kGrid[best_idx] : kGrid[best_idx]) * eff_scale;
                    double diff = static_cast<double>(orig - rec);
                    mse += diff * diff;
                }
                cand_packed[i] = static_cast<uint8_t>(nib[0] | (nib[1] << 4U));
            }
            if (mse < best_mse) {
                best_mse    = mse;
                best_s_e4m3 = s_e4m3;
                std::memcpy(best_packed, cand_packed, 16U);
            }
        }
        scale_out[g] = best_s_e4m3;
        std::memcpy(nib_out + g * 16U, best_packed, 16U);
    }
}

/* Packs a single expert (`w_gate [d_ff, d_model]`, `w_up [d_ff, d_model]`, `w_down [d_model, d_ff]`)
 * into exact 32x32 Metal 4 NAX `wCT` planar MXFP4 (`hot_expert_bytes`, 256B-aligned). */
static void pack_expert_mxfp4_nax_planar(
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    uint32_t d_model,
    uint32_t d_ff,
    uint8_t* dst_expert_buf,
    uint64_t expert_bytes_aligned
) {
    std::memset(dst_expert_buf, 0, static_cast<size_t>(expert_bytes_aligned));

    const uint32_t sb_g       = d_model / 32U;
    const uint32_t sb_d       = d_ff / 32U;
    const uint32_t dff_tiles  = d_ff / 32U;
    const uint32_t dm_tiles   = d_model / 32U;
    const uint64_t total_sb32 = (3ULL * d_model * d_ff) / 32ULL;
    const uint64_t scales_off = total_sb32 * 16ULL;

    std::vector<uint8_t> gate_nib(static_cast<size_t>(d_ff) * sb_g * 16U);
    std::vector<uint8_t> gate_scl(static_cast<size_t>(d_ff) * sb_g);
    quantize_mxfp4_row_planar_host(w_gate, static_cast<size_t>(d_ff) * d_model, gate_nib.data(), gate_scl.data());

    std::vector<uint8_t> up_nib(static_cast<size_t>(d_ff) * sb_g * 16U);
    std::vector<uint8_t> up_scl(static_cast<size_t>(d_ff) * sb_g);
    quantize_mxfp4_row_planar_host(w_up, static_cast<size_t>(d_ff) * d_model, up_nib.data(), up_scl.data());

    std::vector<uint8_t> down_nib(static_cast<size_t>(d_model) * sb_d * 16U);
    std::vector<uint8_t> down_scl(static_cast<size_t>(d_model) * sb_d);
    quantize_mxfp4_row_planar_host(w_down, static_cast<size_t>(d_model) * d_ff, down_nib.data(), down_scl.data());

    uint8_t wct_k_lut[32][32];
    uint8_t wct_n_lut[32][32];
    for (uint32_t ln = 0; ln < 32U; ++ln) {
        for (uint32_t i = 0; i < 32U; ++i) {
            wct_k_lut[ln][i] = static_cast<uint8_t>(
                (((i  >> 4U) & 1U) << 4U) |
                (((ln >> 3U) & 1U) << 3U) |
                (( ln        & 1U) << 2U) |
                (  i         & 3U));
            wct_n_lut[ln][i] = static_cast<uint8_t>(
                (((i  >> 2U) & 3U) << 3U) |
                (((ln >> 4U) & 1U) << 2U) |
                (( ln >> 1U) & 3U));
        }
    }

    uint32_t* dst_nib_u32 = reinterpret_cast<uint32_t*>(dst_expert_buf);
    uint32_t* dst_scl_u32 = reinterpret_cast<uint32_t*>(dst_expert_buf + scales_off);

    auto pack_tile_rowblock = [&](uint64_t dst_tile_base,
                                  uint32_t k_tiles,
                                  uint32_t row_start,
                                  const uint8_t* src_nib,
                                  const uint8_t* src_scl) {
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            const uint64_t t_idx = dst_tile_base + kt;
            const uint8_t* row_b16[32];
            uint8_t        row_s[32];
            for (uint32_t n = 0; n < 32U; ++n) {
                const size_t sb_idx = static_cast<size_t>(row_start + n) * k_tiles + kt;
                row_b16[n] = src_nib + sb_idx * 16U;
                row_s[n]   = src_scl[sb_idx];
            }

            uint32_t* sp = dst_scl_u32 + t_idx * 8ULL;
            for (uint32_t n_sub = 0; n_sub < 8U; ++n_sub) {
                sp[n_sub] = static_cast<uint32_t>(row_s[n_sub +  0U]) |
                            (static_cast<uint32_t>(row_s[n_sub +  8U]) <<  8U) |
                            (static_cast<uint32_t>(row_s[n_sub + 16U]) << 16U) |
                            (static_cast<uint32_t>(row_s[n_sub + 24U]) << 24U);
            }

            uint32_t* np = dst_nib_u32 + t_idx * 128ULL;
            for (uint32_t ln = 0; ln < 32U; ++ln) {
                uint32_t q[4] = {0U, 0U, 0U, 0U};
                for (uint32_t i = 0; i < 32U; ++i) {
                    const uint32_t k_loc = wct_k_lut[ln][i];
                    const uint32_t n_loc = wct_n_lut[ln][i];
                    const uint8_t  b8    = row_b16[n_loc][k_loc >> 1U];
                    const uint32_t nib   = (k_loc & 1U) ? ((b8 >> 4U) & 0xFU) : (b8 & 0xFU);
                    q[i >> 3U] |= (nib << ((i & 7U) * 4U));
                }
                np[ln * 4U + 0U] = q[0];
                np[ln * 4U + 1U] = q[1];
                np[ln * 4U + 2U] = q[2];
                np[ln * 4U + 3U] = q[3];
            }
        }
    };

    for (uint32_t nt = 0; nt < dff_tiles; ++nt) {
        pack_tile_rowblock(static_cast<uint64_t>(nt) * sb_g, sb_g, nt * 32U, gate_nib.data(), gate_scl.data());
        pack_tile_rowblock(static_cast<uint64_t>(dff_tiles + nt) * sb_g, sb_g, nt * 32U, up_nib.data(), up_scl.data());
    }
    const uint64_t down_tile_off = 2ULL * dff_tiles * sb_g;
    for (uint32_t nt = 0; nt < dm_tiles; ++nt) {
        pack_tile_rowblock(down_tile_off + static_cast<uint64_t>(nt) * sb_d, sb_d, nt * 32U, down_nib.data(), down_scl.data());
    }
}

/* Packs a single expert (`w_gate`, `w_up`, `w_down`) into Regime I BitNest QTIP 2b planar base
 * (`dst_base_buf`) AND 4-level Laplacian planar refine (`dst_refine_buf`), matching `strata_kernels.metal`! */
static void pack_expert_qtip_bitnest_planar(
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    uint32_t d_model,
    uint32_t d_ff,
    uint8_t* dst_base_buf,
    uint8_t* dst_refine_buf,
    uint64_t expert_bytes_aligned
) {
    std::memset(dst_base_buf, 0, static_cast<size_t>(expert_bytes_aligned));
    std::memset(dst_refine_buf, 0, static_cast<size_t>(expert_bytes_aligned));

    const uint32_t blks_g     = d_model / 128U;
    const uint32_t blks_d     = d_ff / 128U;
    const uint64_t total_blks = (3ULL * d_model * d_ff) / 128ULL;
    const uint64_t scales_off = total_blks * 32ULL;

    const size_t gu_cnt   = static_cast<size_t>(d_ff) * d_model;
    const size_t down_cnt = static_cast<size_t>(d_model) * d_ff;

    std::vector<uint8_t> g_qbase(static_cast<size_t>(d_ff) * blks_g * 34U);
    std::vector<uint8_t> g_qsym(gu_cnt);
    std::vector<float>   g_rscl(static_cast<size_t>(d_ff) * blks_g);
    strata_codec_qtip_quantize_expert(w_gate, gu_cnt, g_qbase.data(), g_qsym.data(), g_rscl.data(), nullptr, nullptr);

    std::vector<uint8_t> u_qbase(static_cast<size_t>(d_ff) * blks_g * 34U);
    std::vector<uint8_t> u_qsym(gu_cnt);
    std::vector<float>   u_rscl(static_cast<size_t>(d_ff) * blks_g);
    strata_codec_qtip_quantize_expert(w_up, gu_cnt, u_qbase.data(), u_qsym.data(), u_rscl.data(), nullptr, nullptr);

    std::vector<uint8_t> d_qbase(static_cast<size_t>(d_model) * blks_d * 34U);
    std::vector<uint8_t> d_qsym(down_cnt);
    std::vector<float>   d_rscl(static_cast<size_t>(d_model) * blks_d);
    strata_codec_qtip_quantize_expert(w_down, down_cnt, d_qbase.data(), d_qsym.data(), d_rscl.data(), nullptr, nullptr);

    uint8_t*  base_codes = dst_base_buf;
    uint16_t* base_scls  = reinterpret_cast<uint16_t*>(dst_base_buf + scales_off);
    uint8_t*  ref_codes  = dst_refine_buf;
    uint16_t* ref_scls   = reinterpret_cast<uint16_t*>(dst_refine_buf + scales_off);

    auto write_rows = [&](uint64_t dst_blk_base,
                          uint32_t n_rows,
                          uint32_t blks_per_row,
                          const uint8_t* src_qbase,
                          const uint8_t* src_qsym,
                          const float* src_rscl) {
        const uint64_t total_r_blks = static_cast<uint64_t>(n_rows) * blks_per_row;
        for (uint64_t b = 0; b < total_r_blks; ++b) {
            const uint64_t bidx = dst_blk_base + b;
            const uint8_t* ib   = src_qbase + b * 34ULL;
            uint16_t bs_h = 0U;
            std::memcpy(&bs_h, ib, 2U);
            base_scls[bidx] = bs_h;
            std::memcpy(base_codes + bidx * 32ULL, ib + 2U, 32U);

            ref_scls[bidx] = f32_to_fp16_bits(src_rscl[b]);
            uint8_t* rc_dst = ref_codes + bidx * 32ULL;
            const uint8_t* syms = src_qsym + b * 128ULL;
            for (uint32_t bi = 0; bi < 32U; ++bi) {
                const uint8_t* s = syms + bi * 4U;
                rc_dst[bi] = static_cast<uint8_t>(s[0] | (s[1] << 2U) | (s[2] << 4U) | (s[3] << 6U));
            }
        }
    };

    write_rows(0ULL, d_ff, blks_g, g_qbase.data(), g_qsym.data(), g_rscl.data());
    write_rows(static_cast<uint64_t>(d_ff) * blks_g, d_ff, blks_g, u_qbase.data(), u_qsym.data(), u_rscl.data());
    write_rows(2ULL * d_ff * blks_g, d_model, blks_d, d_qbase.data(), d_qsym.data(), d_rscl.data());
}

/* Modified Gram-Schmidt orthonormalization of rows `[n_rows, dim]` */
static void orthonormalize_rows_f32(float* mat, uint32_t n_rows, uint32_t dim, uint64_t* fallback_rng) {
    for (uint32_t r = 0; r < n_rows; ++r) {
        float* row_r = mat + static_cast<size_t>(r) * dim;
        for (uint32_t p = 0; p < r; ++p) {
            const float* row_p = mat + static_cast<size_t>(p) * dim;
            double dot = 0.0;
            for (uint32_t c = 0; c < dim; ++c) dot += static_cast<double>(row_r[c]) * row_p[c];
            float fdot = static_cast<float>(dot);
            for (uint32_t c = 0; c < dim; ++c) row_r[c] -= fdot * row_p[c];
        }
        double norm_sq = 0.0;
        for (uint32_t c = 0; c < dim; ++c) norm_sq += static_cast<double>(row_r[c]) * row_r[c];
        if (norm_sq < 1e-12 && fallback_rng) {
            for (uint32_t c = 0; c < dim; ++c) row_r[c] = rng_normal_f32(fallback_rng, 1.0f);
            for (uint32_t p = 0; p < r; ++p) {
                const float* row_p = mat + static_cast<size_t>(p) * dim;
                double dot = 0.0;
                for (uint32_t c = 0; c < dim; ++c) dot += static_cast<double>(row_r[c]) * row_p[c];
                float fdot = static_cast<float>(dot);
                for (uint32_t c = 0; c < dim; ++c) row_r[c] -= fdot * row_p[c];
            }
            norm_sq = 0.0;
            for (uint32_t c = 0; c < dim; ++c) norm_sq += static_cast<double>(row_r[c]) * row_r[c];
        }
        float inv_norm = (norm_sq > 1e-20) ? static_cast<float>(1.0 / std::sqrt(norm_sq)) : 0.0f;
        for (uint32_t c = 0; c < dim; ++c) row_r[c] *= inv_norm;
    }
}

/* Computes real truncated-SVD shared right-singular subspace `V_shared [eff_rank, d_model]`
 * of stacked expert SwiGLU residuals `[num_experts * d_ff, d_model]` via block power QR iteration,
 * then projects `U_e [d_ff, rank_aligned] = Residual_e @ V_shared^T` and packs via `strata_codec_w6a16_pack`! */
static void compute_layer_moe_svd_w6a16(
    const std::vector<std::vector<float>>& exp_gate,
    const std::vector<std::vector<float>>& exp_up,
    const std::vector<float>& bary_gate,
    const std::vector<float>& bary_up,
    uint32_t num_experts,
    uint32_t d_model,
    uint32_t d_ff,
    uint32_t svd_rank,
    std::vector<uint8_t>* u_w6_out,
    std::vector<uint16_t>* v_experts_f16_out,
    std::vector<uint16_t>* v_shared_f16_out
) {
    const uint32_t rank_aligned = (svd_rank + 63U) & ~63U;
    const uint32_t eff_rank     = std::min(svd_rank, d_model);
    const size_t   row_w6_bytes = static_cast<size_t>(rank_aligned / 64U) * 50U;
    const size_t   total_rows   = static_cast<size_t>(num_experts) * d_ff;

    /* Residual matrix M [total_rows, d_model]: first-order SwiGLU residual wrt Barycenter */
    std::vector<float> M(total_rows * d_model, 0.0f);
    for (uint32_t e = 0; e < num_experts; ++e) {
        for (uint32_t r = 0; r < d_ff; ++r) {
            float* dst = &M[(static_cast<size_t>(e) * d_ff + r) * d_model];
            const size_t roff = static_cast<size_t>(r) * d_model;
            for (uint32_t c = 0; c < d_model; ++c) {
                float dg = exp_gate[e][roff + c] - bary_gate[roff + c];
                float du = exp_up[e][roff + c]   - bary_up[roff + c];
                dst[c] = 0.5f * (dg + du);
            }
        }
    }

    uint64_t svd_rng = 0x5356445F53545241ULL ^ (static_cast<uint64_t>(d_model) << 16U) ^ svd_rank;
    std::vector<float> V(static_cast<size_t>(eff_rank) * d_model);
    for (float& v : V) v = rng_normal_f32(&svd_rng, 1.0f);
    orthonormalize_rows_f32(V.data(), eff_rank, d_model, &svd_rng);

    /* 4-step Block Subspace Iteration: V <- orth((M @ V^T)^T @ M) = orth(V @ M^T @ M) */
    std::vector<float> P(total_rows * eff_rank, 0.0f);
    std::vector<float> V_next(static_cast<size_t>(eff_rank) * d_model, 0.0f);
    for (int iter = 0; iter < 4; ++iter) {
        /* P = M @ V^T [total_rows, eff_rank] */
        for (size_t row = 0; row < total_rows; ++row) {
            const float* m_row = &M[row * d_model];
            float* p_row       = &P[row * eff_rank];
            for (uint32_t k = 0; k < eff_rank; ++k) {
                const float* v_row = &V[static_cast<size_t>(k) * d_model];
                float acc = 0.0f;
                for (uint32_t c = 0; c < d_model; ++c) acc += m_row[c] * v_row[c];
                p_row[k] = acc;
            }
        }
        /* V_next = P^T @ M [eff_rank, d_model] */
        std::fill(V_next.begin(), V_next.end(), 0.0f);
        for (size_t row = 0; row < total_rows; ++row) {
            const float* m_row = &M[row * d_model];
            const float* p_row = &P[row * eff_rank];
            for (uint32_t k = 0; k < eff_rank; ++k) {
                float pk = p_row[k];
                float* vn = &V_next[static_cast<size_t>(k) * d_model];
                for (uint32_t c = 0; c < d_model; ++c) vn[c] += pk * m_row[c];
            }
        }
        V.swap(V_next);
        orthonormalize_rows_f32(V.data(), eff_rank, d_model, &svd_rng);
    }

    /* Quantize V to FP16 first so U_e is projected against the exact FP16 V seen at runtime! */
    v_shared_f16_out->assign(static_cast<size_t>(rank_aligned) * d_model, 0U);
    std::vector<float> V_f16_exact(static_cast<size_t>(eff_rank) * d_model, 0.0f);
    for (uint32_t k = 0; k < eff_rank; ++k) {
        for (uint32_t c = 0; c < d_model; ++c) {
            uint16_t h = f32_to_fp16_bits(V[static_cast<size_t>(k) * d_model + c]);
            (*v_shared_f16_out)[static_cast<size_t>(k) * d_model + c] = h;
            V_f16_exact[static_cast<size_t>(k) * d_model + c]        = fp16_bits_to_f32(h);
        }
    }

    v_experts_f16_out->assign(static_cast<size_t>(num_experts) * svd_rank * d_model, 0U);
    for (uint32_t e = 0; e < num_experts; ++e) {
        uint16_t* dst_ve = &(*v_experts_f16_out)[static_cast<size_t>(e) * svd_rank * d_model];
        for (uint32_t k = 0; k < eff_rank; ++k) {
            std::memcpy(dst_ve + static_cast<size_t>(k) * d_model,
                        &(*v_shared_f16_out)[static_cast<size_t>(k) * d_model],
                        d_model * sizeof(uint16_t));
        }
    }

    /* Compute U_e [d_ff, rank_aligned] = M_e @ V_f16_exact^T and pack via strata_codec_w6a16_pack */
    u_w6_out->assign(static_cast<size_t>(num_experts) * d_ff * row_w6_bytes, 0U);
    std::vector<float> u_row_f32(rank_aligned, 0.0f);
    for (uint32_t e = 0; e < num_experts; ++e) {
        for (uint32_t r = 0; r < d_ff; ++r) {
            std::fill(u_row_f32.begin(), u_row_f32.end(), 0.0f);
            const float* m_row = &M[(static_cast<size_t>(e) * d_ff + r) * d_model];
            for (uint32_t k = 0; k < eff_rank; ++k) {
                const float* v_row = &V_f16_exact[static_cast<size_t>(k) * d_model];
                float acc = 0.0f;
                for (uint32_t c = 0; c < d_model; ++c) acc += m_row[c] * v_row[c];
                u_row_f32[k] = acc;
            }
            uint8_t* dst_w6 = &(*u_w6_out)[(static_cast<size_t>(e) * d_ff + r) * row_w6_bytes];
            strata_codec_w6a16_pack(u_row_f32.data(), rank_aligned, dst_w6);
        }
    }
}

/* ============================================================================
 * 6. Readable English Subword Vocabulary Builder & Tokenizer I/O
 * ============================================================================ */

static std::vector<std::string> build_default_english_vocab(uint32_t vocab_size) {
    static const char* kCuratedSubwords[] = {
        "<unk>", "<s>", "</s>", "\n",
        /* High-frequency English + STRATA technical subwords (4 .. 127: inside 128-token FR-Spec shortlist!) */
        " STRATA", " executes", " four", "-", "axis", " speculative", " decoding", " on",
        " Apple", " M5", " Max", " silicon", " with", " zero", " page", " faults",
        " and", " unified", " memory", " bandwidth", ".", " The", " Metal", " NAX",
        " cooperative", " tensor", " engine", " streams", " cold", " MoE", " experts", " from",
        " ANS", " NVMe", " into", " pre", "wired", " ring", " buffers", " while",
        " the", " 16", "core", " ANE", " router", " probe", " predicts", " upcoming",
        " windows", " in", " parallel", ",", " achieving", " exact", " stationary", " target",
        " distributions", " via", " FR", "Spec", " shortlist", " rejection", " sampling", " across",
        " all", " three", " ARMv9", " SME2", " matrix", " clusters", "!", " In",
        " Mode", " A", " causal", " gating", " preserves", " bit", "identical", " greedy",
        " trajectories", ";", " B", " eliminates", " stalls", " using", " Barycenter", " plus",
        " W6A16", " SVD", " or", " BitNest", " QTIP", " trellis", " fallback", " at",
        " sub", "millisecond", " latency", ":", " of", " to", " is", " for",
        " by", " as", " an", " are", " tile", " rANS", " E4M3", " MXFP4",
        " FP16", " SLC", " cache", " UMA", " layer", " token", " draft", " model",
        " fast", " high", " throughput", " "
    };

    static const char* kExtendedWords[] = {
        "DeepSeek", "Qwen", "GLM", "Flash", "Engram", "PLE", "CoreML", "MLComputePlan",
        "LUTI4", "FMOPA", "ZT0", "SwiGLU", "RoPE", "GQA", "attention", "kv",
        "norm", "residual", "head", "embed", "vocab", "prompt", "generate", "convert",
        "synth", "inspect", "bundle", "checkpoint", "safetensors", "extent", "aligned", "LBA",
        "KiB", "MiB", "GiB", "GB/s", "TFLOPS", "tok/s", "us", "ms",
        "dB", "SQNR", "verified", "hardware", "neural", "accelerator", "compute", "pipeline",
        "dispatch", "threadgroup", "simdgroup", "register", "unpack", "lossless", "entropy", "stream",
        "coding", "system", "architecture", "runtime", "inference", "autoregressive", "sequence", "context",
        "window", "horizon", "prefetch", "async", "DMA", "coherent", "atomic", "epoch",
        "slot", "indirection", "table", "active", "routed", "gate", "score", "softmax",
        "top", "rank", "orthogonal", "subspace", "singular", "vector", "projection", "factor",
        "codebook", "laplacian", "refine", "residual", "quantizer", "viterbi", "hadamard", "transform",
        "ing", "tion", "ed", "er", "ly", "al", "es", "st",
        "ch", "th", "sh", "ph", "wh", "qu", "ck", "ng",
        "pro", "con", "dis", "un", "re", "de", "in", "ex",
        "able", "ible", "ment", "ness", "less", "ful", "ous", "ive",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "32", "64", "128", "256", "512", "2026", "v2"
    };

    std::vector<std::string> vocab;
    vocab.reserve(vocab_size);
    std::unordered_map<std::string, bool> seen;

    auto push_unique = [&](const std::string& tok) {
        if (vocab.size() >= vocab_size) return;
        if (!seen[tok]) {
            seen[tok] = true;
            vocab.push_back(tok);
        }
    };

    for (const char* t : kCuratedSubwords) push_unique(t);

    /* Ensure every printable ASCII character (32..126) + '\t' exists as a single-char fallback token */
    push_unique("\t");
    for (int ch = 32; ch <= 126; ++ch) {
        char buf[2] = {static_cast<char>(ch), '\0'};
        push_unique(buf);
    }

    for (const char* t : kExtendedWords) {
        push_unique(std::string(" ") + t);
        push_unique(t);
    }

    uint32_t filler_idx = 0U;
    while (vocab.size() < vocab_size) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), " tok_%u", filler_idx++);
        push_unique(buf);
    }
    return vocab;
}

static bool write_tokenizer_files(
    const std::string& dir,
    const std::vector<std::string>& vocab
) {
    /* 1. HuggingFace-compatible RFC-8259 `tokenizer.json` */
    std::string js = "{\n  \"version\": \"1.0\",\n  \"model\": {\n    \"type\": \"BPE\",\n    \"vocab\": {\n";
    for (size_t i = 0; i < vocab.size(); ++i) {
        if (i > 0U) js += ",\n";
        js += "      \"" + json_escape(vocab[i]) + "\": " + std::to_string(i);
    }
    js += "\n    }\n  }\n}\n";
    if (!write_binary_file(dir + "/tokenizer.json", js.data(), js.size())) return false;

    /* 2. Escaped line-per-token `vocab.txt` */
    std::string vt;
    for (const auto& t : vocab) {
        vt += json_escape(t) + "\n";
    }
    return write_binary_file(dir + "/vocab.txt", vt.data(), vt.size());
}

static bool load_vocab_from_dir(
    const std::string& dir,
    uint32_t expected_vocab_size,
    std::vector<std::string>* vocab_out
) {
    /* Prefer `tokenizer.json` */
    std::vector<uint8_t> raw;
    if (read_entire_file(dir + "/tokenizer.json", &raw) && !raw.empty()) {
        Rfc8259Parser p{reinterpret_cast<const char*>(raw.data()), raw.size(), 0U};
        JsonValue root;
        if (p.parse_value(&root) && root.type == JsonValue::kObject) {
            const JsonValue* model = root.get("model");
            const JsonValue* vobj  = model ? model->get("vocab") : root.get("vocab");
            if (vobj && vobj->type == JsonValue::kObject && !vobj->obj.empty()) {
                uint32_t max_id = 0U;
                for (const auto& kv : vobj->obj) {
                    if (kv.second.type == JsonValue::kNumber && kv.second.uval > max_id) {
                        max_id = static_cast<uint32_t>(kv.second.uval);
                    }
                }
                uint32_t sz = std::max(expected_vocab_size, max_id + 1U);
                vocab_out->assign(sz, "");
                for (const auto& kv : vobj->obj) {
                    if (kv.second.type == JsonValue::kNumber && kv.second.uval < sz) {
                        (*vocab_out)[static_cast<size_t>(kv.second.uval)] = kv.first;
                    }
                }
                if (vocab_out->size() > expected_vocab_size && expected_vocab_size > 0U) {
                    vocab_out->resize(expected_vocab_size);
                }
                return true;
            }
        }
    }

    /* Fallback to `vocab.txt` */
    if (read_entire_file(dir + "/vocab.txt", &raw) && !raw.empty()) {
        vocab_out->clear();
        std::string content(reinterpret_cast<const char*>(raw.data()), raw.size());
        size_t pos = 0U;
        while (pos < content.size()) {
            size_t nl = content.find('\n', pos);
            std::string line = (nl == std::string::npos) ? content.substr(pos) : content.substr(pos, nl - pos);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::string quoted = "\"" + line + "\"";
            Rfc8259Parser lp{quoted.data(), quoted.size(), 0U};
            std::string unesc;
            if (lp.parse_string(&unesc)) vocab_out->push_back(unesc);
            else vocab_out->push_back(line);
            if (nl == std::string::npos) break;
            pos = nl + 1U;
        }
        if (expected_vocab_size > 0U) {
            while (vocab_out->size() < expected_vocab_size) {
                vocab_out->push_back("tok_" + std::to_string(vocab_out->size()));
            }
            vocab_out->resize(expected_vocab_size);
        }
        return !vocab_out->empty();
    }

    if (expected_vocab_size > 0U) {
        *vocab_out = build_default_english_vocab(expected_vocab_size);
        return true;
    }
    return false;
}

/* ============================================================================
 * 7. Mini Preset Definitions
 * ============================================================================ */

struct SynthPresetSpec {
    std::string arch_name;
    StrataRegime regime;
    uint32_t num_layers;
    uint32_t encoder_layers;
    uint32_t d_model;
    uint32_t d_ff;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t svd_rank;
    uint32_t vocab_size;
    uint32_t shortlist_size;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t pinned_shallow_layers;
    bool     has_ngram_table;
    uint32_t ngram_heads_per_tok;
    uint32_t ngram_layer_idx;
    uint32_t ngram_pages;
    float    mode_a_tau_tok;
    uint32_t mode_b_cold_cap;
    uint32_t ane_probe_rank;
};

static bool resolve_synth_preset(const char* preset_name, SynthPresetSpec* out) {
    std::string p = (preset_name && preset_name[0] != '\0') ? preset_name : "mini-deepseek";
    *out = SynthPresetSpec{};
    out->d_model        = 256U;
    out->d_ff           = 128U;
    out->num_layers     = 6U;
    out->num_experts    = 16U;
    out->top_k          = 4U;
    out->vocab_size     = 512U;
    out->shortlist_size = 128U;
    out->n_heads        = 8U;
    out->n_kv_heads     = 4U;
    out->head_dim       = 32U;
    out->ane_probe_rank = 64U;

    if (p == "mini-deepseek" || p == "deepseek") {
        out->arch_name             = "mini-deepseek";
        out->regime                = STRATA_REGIME_II_BARY_SVD;
        out->encoder_layers        = 2U;
        out->svd_rank              = 64U;
        out->pinned_shallow_layers = 2U;
        out->has_ngram_table       = true;
        out->ngram_heads_per_tok   = 4U;
        out->ngram_layer_idx       = 1U;
        out->ngram_pages           = 64U;
        out->mode_a_tau_tok        = 0.095f;
        out->mode_b_cold_cap       = 2U;
        return true;
    }
    if (p == "mini-qwen" || p == "qwen") {
        out->arch_name             = "mini-qwen";
        out->regime                = STRATA_REGIME_I_100PCT_UMA;
        out->encoder_layers        = 0U;
        out->svd_rank              = 0U;
        out->pinned_shallow_layers = 6U; /* 100% UMA resident */
        out->has_ngram_table       = true;
        out->ngram_heads_per_tok   = 4U;
        out->ngram_layer_idx       = 2U;
        out->ngram_pages           = 64U;
        out->mode_a_tau_tok        = 0.0f;
        out->mode_b_cold_cap       = 0U;
        return true;
    }
    if (p == "mini-glm" || p == "glm") {
        out->arch_name             = "mini-glm";
        out->regime                = STRATA_REGIME_I_BITNEST;
        out->encoder_layers        = 0U;
        out->svd_rank              = 0U;
        out->pinned_shallow_layers = 2U;
        out->has_ngram_table       = false;
        out->ngram_heads_per_tok   = 0U;
        out->ngram_layer_idx       = 0U;
        out->ngram_pages           = 0U;
        out->mode_a_tau_tok        = 0.080f;
        out->mode_b_cold_cap       = 2U;
        return true;
    }
    return false;
}

}  // namespace

/* ============================================================================
 * 8. Public C API Implementation
 * ============================================================================ */

extern "C" {

int strata_synth_safetensors_checkpoint(
    const char* out_path,
    const char* preset_name,
    uint64_t seed
) {
    if (!out_path || out_path[0] == '\0') return -EINVAL;

    SynthPresetSpec spec;
    if (!resolve_synth_preset(preset_name, &spec)) return -EINVAL;

    std::string raw_path(out_path);
    std::string st_file_path;
    std::string out_dir;
    if (ends_with(raw_path, ".safetensors")) {
        st_file_path = raw_path;
        out_dir      = parent_dir_of(raw_path);
    } else {
        out_dir      = raw_path;
        st_file_path = out_dir + "/model.safetensors";
    }

    int mrc = mkdir_p(out_dir);
    if (mrc != 0) return mrc;

    uint64_t rng = (seed != 0ULL) ? seed : 42ULL;
    rng ^= 0x5354524154414D35ULL;

    const uint32_t V  = spec.vocab_size;
    const uint32_t S  = spec.shortlist_size;
    const uint32_t D  = spec.d_model;
    const uint32_t F  = spec.d_ff;
    const uint32_t L  = spec.num_layers;
    const uint32_t E  = spec.num_experts;
    const uint32_t Qd = spec.n_heads * spec.head_dim;
    const uint32_t Kd = spec.n_kv_heads * spec.head_dim;

    std::vector<std::string> vocab = build_default_english_vocab(V);
    if (!write_tokenizer_files(out_dir, vocab)) return -EIO;

    std::vector<OutTensorSpec> tensors;
    tensors.reserve(12U + static_cast<size_t>(L) * (7U + 3U * E));

    /* 1. Token Embeddings `[V, D]` — normalized unit-RMS vectors (`RMS = 1.0`) */
    std::vector<float> tok_embed(static_cast<size_t>(V) * D);
    for (uint32_t v = 0; v < V; ++v) {
        float* row = &tok_embed[static_cast<size_t>(v) * D];
        double sq = 0.0;
        for (uint32_t d = 0; d < D; ++d) {
            float val = rng_normal_f32(&rng, 1.0f);
            row[d] = val;
            sq += static_cast<double>(val) * val;
        }
        float scale = static_cast<float>(std::sqrt(static_cast<double>(D) / std::max(sq, 1e-12)));
        for (uint32_t d = 0; d < D; ++d) row[d] *= scale * 0.5f;
    }
    append_safetensor_from_f32(&tensors, "model.embed_tokens.weight", StDtype::kF16, {V, D}, tok_embed.data());

    /* 2. Final RMSNorm `[D]` (F32) */
    std::vector<float> final_norm(D, 1.0f);
    for (uint32_t d = 0; d < D; ++d) {
        final_norm[d] = 1.0f + 0.02f * std::cos(static_cast<float>(d) * 0.13f);
    }
    append_safetensor_from_f32(&tensors, "model.norm.weight", StDtype::kF32, {D}, final_norm.data());

    /* 3. Planted Coherent Markov Next-Token Chain on `lm_head [V, D]` & `mtp_head [S, D]`
     *    Tokens `4 .. 103` form a fluent English STRATA narrative cycle; any other token `v`
     *    transitions into `[4 .. 67]` via `(v * 7 + 4)` so prompts immediately enter readable prose! */
    constexpr uint32_t kChainBegin = 4U;
    constexpr uint32_t kChainEnd   = 102U; /* "." */
    auto planted_next_tok = [&](uint32_t v) -> uint32_t {
        if (v >= kChainBegin && v < kChainEnd) return v + 1U;
        if (v == kChainEnd) return 25U; /* " The" -> continues second sentence */
        return kChainBegin + ((v * 7U + 3U) % 60U);
    };

    /* Compute RMS-normalized token signatures `u_tok[v, d]` (`||u_tok[v]||_2 = 1`) */
    std::vector<float> u_tok(static_cast<size_t>(V) * D, 0.0f);
    for (uint32_t v = 0; v < V; ++v) {
        const float* src = &tok_embed[static_cast<size_t>(v) * D];
        double mean_sq = 0.0;
        for (uint32_t d = 0; d < D; ++d) mean_sq += static_cast<double>(src[d]) * src[d];
        mean_sq /= static_cast<double>(D);
        float inv_rms = 1.0f / std::sqrt(static_cast<float>(mean_sq) + 1e-5f);
        float* dst = &u_tok[static_cast<size_t>(v) * D];
        double nsq = 0.0;
        for (uint32_t d = 0; d < D; ++d) {
            dst[d] = src[d] * inv_rms * final_norm[d];
            nsq += static_cast<double>(dst[d]) * dst[d];
        }
        float inv_l2 = 1.0f / std::sqrt(static_cast<float>(std::max(nsq, 1e-12)));
        for (uint32_t d = 0; d < D; ++d) dst[d] *= inv_l2;
    }

    std::vector<float> lm_head(static_cast<size_t>(V) * D);
    const float lm_bg_sig = 0.25f / std::sqrt(static_cast<float>(D));
    for (float& w : lm_head) w = rng_normal_f32(&rng, lm_bg_sig);

    /* Plant forward transition `v -> next_tok(v)` strongly into row `next_tok(v)` of `lm_head` */
    for (uint32_t v = 0; v < V; ++v) {
        uint32_t nxt = planted_next_tok(v);
        float* dst_row = &lm_head[static_cast<size_t>(nxt) * D];
        const float* sig = &u_tok[static_cast<size_t>(v) * D];
        float gain = (v >= kChainBegin && v <= kChainEnd) ? 0.55f : 0.14f;
        for (uint32_t d = 0; d < D; ++d) dst_row[d] += gain * sig[d];
    }
    append_safetensor_from_f32(&tensors, "lm_head.weight", StDtype::kF16, {V, D}, lm_head.data());

    /* 4. FR-Spec MTP Draft Module (`mtp.norm` F32, `mtp.proj` BF16, `mtp.head` F16, `mtp.shortlist_ids` F32)
     *    - `shortlist_ids`: `[0 .. S-1]` covering special tokens + all curated STRATA English subwords!
     *    - `mtp.proj [D, 2*D]`: maps `[h_prev; tok_embed[t]] -> tok_embed[t]` (+ mild `0.15 * h_prev` context)
     *      so multi-step speculative drafting `k = 0 .. K-1` tracks the planted target chain accurately!
     *    - `mtp.head [S, D]`: copies rows `0 .. S-1` of `lm_head` for high speculative acceptance! */
    std::vector<float> mtp_norm = final_norm;
    append_safetensor_from_f32(&tensors, "mtp.norm.weight", StDtype::kF32, {D}, mtp_norm.data());

    std::vector<float> mtp_proj(static_cast<size_t>(D) * (2U * D), 0.0f);
    for (uint32_t r = 0; r < D; ++r) {
        mtp_proj[static_cast<size_t>(r) * (2U * D) + r]     = 0.12f; /* mild h_prev carry */
        mtp_proj[static_cast<size_t>(r) * (2U * D) + D + r] = 0.94f; /* strong current token carry */
    }
    append_safetensor_from_f32(&tensors, "mtp.proj.weight", StDtype::kBF16, {D, 2U * D}, mtp_proj.data());

    std::vector<float> mtp_head(static_cast<size_t>(S) * D, 0.0f);
    std::memcpy(mtp_head.data(), lm_head.data(), static_cast<size_t>(S) * D * sizeof(float));
    append_safetensor_from_f32(&tensors, "mtp.head.weight", StDtype::kF16, {S, D}, mtp_head.data());

    std::vector<float> shortlist_f32(S);
    for (uint32_t i = 0; i < S; ++i) shortlist_f32[i] = static_cast<float>(i);
    append_safetensor_from_f32(&tensors, "mtp.shortlist_ids", StDtype::kF32, {S}, shortlist_f32.data());

    /* 5. Axis A N-Gram Conditional Memory Tensor (`model.engram.weight` or `model.ple.weight`) */
    if (spec.has_ngram_table && spec.ngram_pages > 0U) {
        std::vector<float> ngram_w(static_cast<size_t>(spec.ngram_pages) * D);
        const float ng_sig = 0.03f / std::sqrt(static_cast<float>(D));
        for (float& w : ngram_w) w = rng_normal_f32(&rng, ng_sig);
        std::string ng_name = (spec.regime == STRATA_REGIME_I_100PCT_UMA)
                                  ? "model.ple.weight"
                                  : "model.engram.weight";
        append_safetensor_from_f32(&tensors, ng_name, StDtype::kF16, {spec.ngram_pages, D}, ngram_w.data());
    }

    /* 6. Per-Layer Transformer Attention + Router + Tiered MoE Experts
     *    Uses F32, BF16, F8_E4M3, and F16 tensors so all 4 `.safetensors` dtypes are exercised! */
    const float attn_in_sig  = 0.65f / std::sqrt(static_cast<float>(D));
    const float attn_out_sig = 0.08f / (std::sqrt(static_cast<float>(Qd)) * std::sqrt(static_cast<float>(L)));
    const float router_sig   = 1.15f / std::sqrt(static_cast<float>(D));
    const float moe_in_sig   = 0.70f / std::sqrt(static_cast<float>(D));
    const float moe_out_sig  = 0.10f / (std::sqrt(static_cast<float>(F)) * std::sqrt(static_cast<float>(L)));

    for (uint32_t l = 0; l < L; ++l) {
        std::string pfx = "model.layers." + std::to_string(l) + ".";

        std::vector<float> in_norm(D, 1.0f);
        std::vector<float> post_norm(D, 1.0f);
        for (uint32_t d = 0; d < D; ++d) {
            in_norm[d]   = 1.0f + 0.015f * std::sin(static_cast<float>(l * 7U + d) * 0.11f);
            post_norm[d] = 1.0f + 0.015f * std::cos(static_cast<float>(l * 5U + d) * 0.09f);
        }
        append_safetensor_from_f32(&tensors, pfx + "input_layernorm.weight", StDtype::kF32, {D}, in_norm.data());
        append_safetensor_from_f32(&tensors, pfx + "post_attention_layernorm.weight", StDtype::kBF16, {D}, post_norm.data());

        std::vector<float> wq(static_cast<size_t>(Qd) * D);
        std::vector<float> wk(static_cast<size_t>(Kd) * D);
        std::vector<float> wv(static_cast<size_t>(Kd) * D);
        std::vector<float> wo(static_cast<size_t>(D) * Qd);
        for (float& v : wq) v = rng_normal_f32(&rng, attn_in_sig);
        for (float& v : wk) v = rng_normal_f32(&rng, attn_in_sig);
        for (float& v : wv) v = rng_normal_f32(&rng, attn_in_sig);
        for (float& v : wo) v = rng_normal_f32(&rng, attn_out_sig);

        append_safetensor_from_f32(&tensors, pfx + "self_attn.q_proj.weight", StDtype::kF16, {Qd, D}, wq.data());
        append_safetensor_from_f32(&tensors, pfx + "self_attn.k_proj.weight", StDtype::kF8_E4M3, {Kd, D}, wk.data());
        append_safetensor_from_f32(&tensors, pfx + "self_attn.v_proj.weight", StDtype::kF16, {Kd, D}, wv.data());
        append_safetensor_from_f32(&tensors, pfx + "self_attn.o_proj.weight", StDtype::kF16, {D, Qd}, wo.data());

        /* Balanced Router `[E, D]` so tokens activate BOTH UMA-hot (`0..3`) and NVMe-cold (`4..15`) experts! */
        std::vector<float> router_w(static_cast<size_t>(E) * D);
        for (uint32_t e = 0; e < E; ++e) {
            /* Give hot experts `0..3` slightly higher norm (`1.12x`) so `strata_convert`'s router-norm
             * hot-tier ranker deterministically wires experts `0..3` as hot while `4..15` still win
             * frequent top-k slots (`g >= 0.12 > tau_tok`), exercising real NVMe cold streaming! */
            float tier_scale = (e < (E / 4U)) ? 1.10f : 1.00f;
            for (uint32_t d = 0; d < D; ++d) {
                router_w[static_cast<size_t>(e) * D + d] = rng_normal_f32(&rng, router_sig * tier_scale);
            }
        }
        append_safetensor_from_f32(&tensors, pfx + "mlp.gate.weight", StDtype::kF32, {E, D}, router_w.data());

        /* Planted Shared Barycenter + Low-Rank Subspace (`r = 32`) + Fine Residual MoE Experts */
        std::vector<float> bary_gate(static_cast<size_t>(F) * D);
        std::vector<float> bary_up(static_cast<size_t>(F) * D);
        std::vector<float> bary_down(static_cast<size_t>(D) * F);
        for (float& v : bary_gate) v = rng_normal_f32(&rng, moe_in_sig * 0.80f);
        for (float& v : bary_up)   v = rng_normal_f32(&rng, moe_in_sig * 0.80f);
        for (float& v : bary_down) v = rng_normal_f32(&rng, moe_out_sig * 0.85f);

        const uint32_t plant_r = 32U;
        std::vector<float> sub_v(static_cast<size_t>(plant_r) * D);
        for (float& v : sub_v) v = rng_normal_f32(&rng, 1.0f);
        orthonormalize_rows_f32(sub_v.data(), plant_r, D, &rng);

        for (uint32_t e = 0; e < E; ++e) {
            std::vector<float> eg = bary_gate;
            std::vector<float> eu = bary_up;
            std::vector<float> ed = bary_down;

            std::vector<float> u_coef(static_cast<size_t>(F) * plant_r);
            for (float& c : u_coef) c = rng_normal_f32(&rng, moe_in_sig * 0.45f * std::sqrt(static_cast<float>(D) / plant_r));

            for (uint32_t r = 0; r < F; ++r) {
                for (uint32_t k = 0; k < plant_r; ++k) {
                    float uc = u_coef[static_cast<size_t>(r) * plant_r + k];
                    const float* sv = &sub_v[static_cast<size_t>(k) * D];
                    for (uint32_t d = 0; d < D; ++d) {
                        eg[static_cast<size_t>(r) * D + d] += uc * sv[d];
                        eu[static_cast<size_t>(r) * D + d] += (0.85f * uc) * sv[d];
                    }
                }
                for (uint32_t d = 0; d < D; ++d) {
                    eg[static_cast<size_t>(r) * D + d] += rng_normal_f32(&rng, moe_in_sig * 0.12f);
                    eu[static_cast<size_t>(r) * D + d] += rng_normal_f32(&rng, moe_in_sig * 0.12f);
                }
            }
            for (float& v : ed) v += rng_normal_f32(&rng, moe_out_sig * 0.20f);

            std::string epfx = pfx + "mlp.experts." + std::to_string(e) + ".";
            append_safetensor_from_f32(&tensors, epfx + "gate_proj.weight", StDtype::kF16, {F, D}, eg.data());
            append_safetensor_from_f32(&tensors, epfx + "up_proj.weight",   StDtype::kF16, {F, D}, eu.data());
            append_safetensor_from_f32(&tensors, epfx + "down_proj.weight", StDtype::kF16, {D, F}, ed.data());
        }
    }

    char tau_buf[32];
    std::snprintf(tau_buf, sizeof(tau_buf), "%.6f", spec.mode_a_tau_tok);

    std::vector<std::pair<std::string, std::string>> meta = {
        {"format", "pt"},
        {"strata_arch", spec.arch_name},
        {"regime", std::to_string(static_cast<uint32_t>(spec.regime))},
        {"num_layers", std::to_string(spec.num_layers)},
        {"encoder_layers", std::to_string(spec.encoder_layers)},
        {"d_model", std::to_string(spec.d_model)},
        {"d_ff", std::to_string(spec.d_ff)},
        {"num_experts", std::to_string(spec.num_experts)},
        {"top_k", std::to_string(spec.top_k)},
        {"svd_rank", std::to_string(spec.svd_rank)},
        {"vocab_size", std::to_string(spec.vocab_size)},
        {"shortlist_size", std::to_string(spec.shortlist_size)},
        {"n_heads", std::to_string(spec.n_heads)},
        {"n_kv_heads", std::to_string(spec.n_kv_heads)},
        {"head_dim", std::to_string(spec.head_dim)},
        {"pinned_shallow_layers", std::to_string(spec.pinned_shallow_layers)},
        {"has_ngram_table", spec.has_ngram_table ? "1" : "0"},
        {"ngram_heads_per_tok", std::to_string(spec.ngram_heads_per_tok)},
        {"ngram_layer_idx", std::to_string(spec.ngram_layer_idx)},
        {"mode_a_tau_tok", tau_buf},
        {"mode_b_cold_cap", std::to_string(spec.mode_b_cold_cap)},
        {"ane_probe_rank", std::to_string(spec.ane_probe_rank)}
    };

    int wrc = write_safetensors_file(st_file_path, meta, tensors);
    if (wrc != 0) return wrc;

    /* Also emit HuggingFace-style `config.json` alongside `.safetensors` */
    std::string cfg_json =
        "{\n"
        "  \"architectures\": [\"" + json_escape(spec.arch_name) + "\"],\n"
        "  \"model_type\": \"" + json_escape(spec.arch_name) + "\",\n"
        "  \"strata_regime\": " + std::to_string(static_cast<uint32_t>(spec.regime)) + ",\n"
        "  \"hidden_size\": " + std::to_string(spec.d_model) + ",\n"
        "  \"moe_intermediate_size\": " + std::to_string(spec.d_ff) + ",\n"
        "  \"num_hidden_layers\": " + std::to_string(spec.num_layers) + ",\n"
        "  \"encoder_layers\": " + std::to_string(spec.encoder_layers) + ",\n"
        "  \"n_routed_experts\": " + std::to_string(spec.num_experts) + ",\n"
        "  \"num_experts_per_tok\": " + std::to_string(spec.top_k) + ",\n"
        "  \"svd_rank\": " + std::to_string(spec.svd_rank) + ",\n"
        "  \"vocab_size\": " + std::to_string(spec.vocab_size) + ",\n"
        "  \"shortlist_size\": " + std::to_string(spec.shortlist_size) + ",\n"
        "  \"num_attention_heads\": " + std::to_string(spec.n_heads) + ",\n"
        "  \"num_key_value_heads\": " + std::to_string(spec.n_kv_heads) + ",\n"
        "  \"head_dim\": " + std::to_string(spec.head_dim) + ",\n"
        "  \"pinned_shallow_layers\": " + std::to_string(spec.pinned_shallow_layers) + ",\n"
        "  \"has_ngram_table\": " + std::string(spec.has_ngram_table ? "true" : "false") + ",\n"
        "  \"ngram_heads_per_tok\": " + std::to_string(spec.ngram_heads_per_tok) + ",\n"
        "  \"ngram_layer_idx\": " + std::to_string(spec.ngram_layer_idx) + ",\n"
        "  \"mode_a_tau_tok\": " + std::string(tau_buf) + ",\n"
        "  \"mode_b_cold_cap\": " + std::to_string(spec.mode_b_cold_cap) + "\n"
        "}\n";
    if (!write_binary_file(out_dir + "/config.json", cfg_json.data(), cfg_json.size())) {
        return -EIO;
    }
    return 0;
}

int strata_convert_safetensors_to_bundle(
    const char* safetensors_path,
    const char* bundle_dir,
    StrataConvertStats* stats_out
) {
    if (!safetensors_path || !bundle_dir || safetensors_path[0] == '\0' || bundle_dir[0] == '\0') {
        return -EINVAL;
    }
    if (stats_out) std::memset(stats_out, 0, sizeof(*stats_out));

    std::string st_path(safetensors_path);
    std::string src_dir;
    if (is_directory_path(st_path)) {
        src_dir = st_path;
        st_path = src_dir + "/model.safetensors";
    } else {
        src_dir = parent_dir_of(st_path);
    }

    SafetensorsReader reader;
    int orc = reader.open_file(st_path);
    if (orc != 0) return orc;

    std::string dst_dir(bundle_dir);
    int mrc = mkdir_p(dst_dir);
    if (mrc != 0) return mrc;

    /* Merge optional HuggingFace `config.json` keys into metadata fallback map */
    if (file_exists_path(src_dir + "/config.json")) {
        std::vector<uint8_t> cfg_raw;
        if (read_entire_file(src_dir + "/config.json", &cfg_raw) && !cfg_raw.empty()) {
            Rfc8259Parser cp{reinterpret_cast<const char*>(cfg_raw.data()), cfg_raw.size(), 0U};
            JsonValue croot;
            if (cp.parse_value(&croot) && croot.type == JsonValue::kObject) {
                auto sync_num = [&](const char* json_k, const char* meta_k) {
                    if (reader.metadata.find(meta_k) == reader.metadata.end()) {
                        const JsonValue* v = croot.get(json_k);
                        if (v && v->type == JsonValue::kNumber) {
                            char nb[64];
                            std::snprintf(nb, sizeof(nb), "%.6g", v->nval);
                            reader.metadata[meta_k] = nb;
                        } else if (v && v->type == JsonValue::kBool) {
                            reader.metadata[meta_k] = v->bval ? "1" : "0";
                        } else if (v && v->type == JsonValue::kString) {
                            reader.metadata[meta_k] = v->sval;
                        }
                    }
                };
                sync_num("model_type", "strata_arch");
                sync_num("strata_regime", "regime");
                sync_num("num_hidden_layers", "num_layers");
                sync_num("encoder_layers", "encoder_layers");
                sync_num("n_routed_experts", "num_experts");
                sync_num("num_experts_per_tok", "top_k");
                sync_num("svd_rank", "svd_rank");
                sync_num("shortlist_size", "shortlist_size");
                sync_num("num_attention_heads", "n_heads");
                sync_num("num_key_value_heads", "n_kv_heads");
                sync_num("head_dim", "head_dim");
                sync_num("pinned_shallow_layers", "pinned_shallow_layers");
                sync_num("has_ngram_table", "has_ngram_table");
                sync_num("ngram_heads_per_tok", "ngram_heads_per_tok");
                sync_num("ngram_layer_idx", "ngram_layer_idx");
                sync_num("mode_a_tau_tok", "mode_a_tau_tok");
                sync_num("mode_b_cold_cap", "mode_b_cold_cap");
            }
        }
    }

    auto meta_u32 = [&](const char* k, uint32_t def_val) -> uint32_t {
        auto it = reader.metadata.find(k);
        if (it == reader.metadata.end() || it->second.empty()) return def_val;
        return static_cast<uint32_t>(std::strtoul(it->second.c_str(), nullptr, 10));
    };
    auto meta_f32 = [&](const char* k, float def_val) -> float {
        auto it = reader.metadata.find(k);
        if (it == reader.metadata.end() || it->second.empty()) return def_val;
        return std::strtof(it->second.c_str(), nullptr);
    };

    /* Infer or read model dimensions from `.safetensors` shapes + metadata + optional `config.json` */
    const SafetensorEntry* e_emb = reader.find("model.embed_tokens.weight");
    if (!e_emb || e_emb->shape.size() != 2U) return -EBADMSG;

    const uint32_t vocab_size = static_cast<uint32_t>(e_emb->shape[0]);
    const uint32_t d_model    = static_cast<uint32_t>(e_emb->shape[1]);
    if ((d_model % 32U) != 0U) return -EINVAL;

    uint32_t inferred_layers = 0U;
    while (reader.find("model.layers." + std::to_string(inferred_layers) + ".self_attn.q_proj.weight") != nullptr) {
        ++inferred_layers;
    }
    const uint32_t num_layers = meta_u32("num_layers", inferred_layers);
    if (num_layers == 0U) return -EBADMSG;

    const SafetensorEntry* e_q0 = reader.find("model.layers.0.self_attn.q_proj.weight");
    const SafetensorEntry* e_k0 = reader.find("model.layers.0.self_attn.k_proj.weight");
    const SafetensorEntry* e_g0 = reader.find("model.layers.0.mlp.experts.0.gate_proj.weight");
    if (!e_g0) e_g0 = reader.find("model.layers.0.mlp.experts.0.w1.weight");
    if (!e_q0 || !e_k0 || !e_g0 || e_g0->shape.size() != 2U) return -EBADMSG;

    const uint32_t q_dim  = static_cast<uint32_t>(e_q0->shape[0]);
    const uint32_t kv_dim = static_cast<uint32_t>(e_k0->shape[0]);
    const uint32_t d_ff   = static_cast<uint32_t>(e_g0->shape[0]);
    if ((q_dim % 32U) != 0U || (kv_dim % 32U) != 0U || (d_ff % 32U) != 0U) return -EINVAL;

    uint32_t inferred_experts = 0U;
    while (reader.find("model.layers.0.mlp.experts." + std::to_string(inferred_experts) + ".gate_proj.weight") != nullptr ||
           reader.find("model.layers.0.mlp.experts." + std::to_string(inferred_experts) + ".w1.weight") != nullptr) {
        ++inferred_experts;
    }
    const uint32_t num_experts = meta_u32("num_experts", inferred_experts);
    if (num_experts == 0U) return -EBADMSG;

    const uint32_t head_dim   = meta_u32("head_dim", 32U);
    const uint32_t n_heads    = meta_u32("n_heads", q_dim / std::max(head_dim, 1U));
    const uint32_t n_kv_heads = meta_u32("n_kv_heads", kv_dim / std::max(head_dim, 1U));

    std::string arch_name = "mini-deepseek";
    auto arch_it = reader.metadata.find("strata_arch");
    if (arch_it != reader.metadata.end() && !arch_it->second.empty()) {
        arch_name = arch_it->second;
    }

    const SafetensorEntry* e_ngram = reader.find("model.engram.weight");
    if (!e_ngram) e_ngram = reader.find("model.ple.weight");

    const auto regime = static_cast<StrataRegime>(meta_u32("regime", static_cast<uint32_t>(STRATA_REGIME_II_BARY_SVD)));
    const uint32_t encoder_layers = meta_u32("encoder_layers", 0U);
    const uint32_t top_k          = meta_u32("top_k", std::min(4U, num_experts));
    const uint32_t svd_rank       = (regime == STRATA_REGIME_II_BARY_SVD) ? meta_u32("svd_rank", 64U) : 0U;
    const uint32_t pinned_layers  = (regime == STRATA_REGIME_I_100PCT_UMA)
                                        ? num_layers
                                        : meta_u32("pinned_shallow_layers", std::min(2U, num_layers));
    const bool     has_ngram      = (e_ngram != nullptr) && (meta_u32("has_ngram_table", 1U) != 0U);
    const uint32_t ngram_heads    = has_ngram ? meta_u32("ngram_heads_per_tok", 4U) : 0U;
    const uint32_t ngram_layer    = has_ngram ? meta_u32("ngram_layer_idx", std::min(1U, num_layers - 1U)) : 0U;
    const float    tau_tok        = meta_f32("mode_a_tau_tok", (regime == STRATA_REGIME_I_100PCT_UMA) ? 0.0f : 0.095f);
    const uint32_t cold_cap       = meta_u32("mode_b_cold_cap", (regime == STRATA_REGIME_I_100PCT_UMA) ? 0U : 2U);
    const uint32_t ane_probe_rank = meta_u32("ane_probe_rank", 64U);

    if (regime == STRATA_REGIME_I_BITNEST && ((d_model % 128U) != 0U || (d_ff % 128U) != 0U)) {
        return -EINVAL;
    }

    const SafetensorEntry* e_mtp_head = reader.find("mtp.head.weight");
    const uint32_t shortlist_size = e_mtp_head
                                        ? static_cast<uint32_t>(e_mtp_head->shape[0])
                                        : meta_u32("shortlist_size", std::min(128U, vocab_size));

    /* Copy / write tokenizer into `<bundle_dir>/` */
    std::vector<std::string> vocab;
    if (!load_vocab_from_dir(src_dir, vocab_size, &vocab)) {
        vocab = build_default_english_vocab(vocab_size);
    }
    if (!write_tokenizer_files(dst_dir, vocab)) return -EIO;

    /* ========================================================================
     * AXIS A: `<bundle_dir>/engram.ngram` (4 KiB-aligned INT4 + FP16 ZT0 LUT)
     * ======================================================================== */
    uint64_t axis_a_bytes = 0ULL;
    if (has_ngram && e_ngram) {
        std::vector<float> ng_f32;
        if (!reader.read_tensor_f32(*e_ngram, &ng_f32)) return -EIO;
        const uint32_t row_dim   = (e_ngram->shape.size() >= 2U) ? static_cast<uint32_t>(e_ngram->shape[1]) : d_model;
        const uint32_t num_pages = std::max(1U, static_cast<uint32_t>(ng_f32.size() / std::max(row_dim, 1U)));

        std::vector<uint8_t> ngram_blob(static_cast<size_t>(num_pages) * kNgramPageBytes, 0U);
        const uint32_t unpack_elems = std::min<uint32_t>(((d_model + 127U) / 128U) * 128U, 8064U);

        for (uint32_t p = 0; p < num_pages; ++p) {
            uint8_t* page = &ngram_blob[static_cast<size_t>(p) * kNgramPageBytes];
            const float* src_row = &ng_f32[static_cast<size_t>(p) * row_dim];

            float max_abs = 0.0f;
            for (uint32_t j = 0; j < row_dim; ++j) max_abs = std::max(max_abs, std::fabs(src_row[j]));
            float step = (max_abs > 1e-12f) ? (max_abs / 7.5f) : 0.001f;

            uint16_t cb_f16[16];
            float    cb_f32[16];
            for (int k = 0; k < 16; ++k) {
                float lvl = (static_cast<float>(k) - 7.5f) * step;
                cb_f16[k] = f32_to_fp16_bits(lvl);
                cb_f32[k] = fp16_bits_to_f32(cb_f16[k]);
            }
            std::memcpy(page, cb_f16, 32U);
            uint32_t mg = kNgramPageMagic;
            std::memcpy(page + 32U, &mg, sizeof(uint32_t));
            std::memcpy(page + 36U, &p, sizeof(uint32_t));
            uint16_t step_h = f32_to_fp16_bits(step);
            std::memcpy(page + 40U, &step_h, sizeof(uint16_t));

            uint8_t* nib_dst = page + 64U;
            for (uint32_t i = 0; i < unpack_elems; i += 2U) {
                uint8_t nib[2] = {8U, 8U};
                for (uint32_t sub = 0; sub < 2U; ++sub) {
                    uint32_t idx = i + sub;
                    float val = (idx < row_dim) ? src_row[idx] : 0.0f;
                    uint8_t best_k = 0U;
                    float best_d = std::fabs(val - cb_f32[0]);
                    for (uint8_t k = 1U; k < 16U; ++k) {
                        float d = std::fabs(val - cb_f32[k]);
                        if (d < best_d) { best_d = d; best_k = k; }
                    }
                    nib[sub] = best_k;
                }
                nib_dst[i / 2U] = static_cast<uint8_t>(nib[0] | (nib[1] << 4U));
            }
        }

        if (!write_binary_file(dst_dir + "/engram.ngram", ngram_blob.data(), ngram_blob.size())) {
            return -EIO;
        }
        axis_a_bytes = ngram_blob.size();
    }

    /* ========================================================================
     * AXIS B: `<bundle_dir>/spine.bin` (32x32 NAX `wCT` E4M3 Tile-rANS + FP16)
     * ======================================================================== */
    BinaryWriter spine_w;
    spine_w.write_u32(kSpineFileMagic);
    spine_w.write_u32(kBundleFileVer);

    char arch_fixed[64] = {0};
    std::snprintf(arch_fixed, sizeof(arch_fixed), "%s", arch_name.c_str());
    spine_w.write_bytes(arch_fixed, sizeof(arch_fixed));

    spine_w.write_u32(static_cast<uint32_t>(regime));
    spine_w.write_u32(num_layers);
    spine_w.write_u32(encoder_layers);
    spine_w.write_u32(d_model);
    spine_w.write_u32(d_ff);
    spine_w.write_u32(num_experts);
    spine_w.write_u32(top_k);
    spine_w.write_u32(vocab_size);
    spine_w.write_u32(shortlist_size);
    spine_w.write_u32(svd_rank);
    spine_w.write_u32(pinned_layers);
    spine_w.write_u32(has_ngram ? 1U : 0U);
    spine_w.write_u32(ngram_heads);
    spine_w.write_u32(ngram_layer);
    spine_w.write_f32(tau_tok);
    spine_w.write_u32(cold_cap);
    spine_w.write_u32(n_heads);
    spine_w.write_u32(n_kv_heads);
    spine_w.write_u32(head_dim);
    spine_w.write_u32(ane_probe_rank);
    spine_w.write_f32(1e-5f);     /* rms_norm_eps */
    spine_w.write_f32(10000.0f);  /* rope_theta */
    spine_w.write_u32(2048U);     /* max_seq_len */

    auto write_f16_vec_from_f32 = [&](const std::vector<float>& src) {
        std::vector<uint16_t> h(src.size());
        for (size_t i = 0; i < src.size(); ++i) h[i] = f32_to_fp16_bits(src[i]);
        spine_w.align_to(16U);
        spine_w.write_bytes(h.data(), h.size() * sizeof(uint16_t));
    };

    std::vector<float> tmp_f32;
    if (!reader.read_tensor_f32(*e_emb, &tmp_f32)) return -EIO;
    write_f16_vec_from_f32(tmp_f32);

    const SafetensorEntry* e_fnorm = reader.find("model.norm.weight");
    if (e_fnorm) {
        if (!reader.read_tensor_f32(*e_fnorm, &tmp_f32)) return -EIO;
    } else {
        tmp_f32.assign(d_model, 1.0f);
    }
    write_f16_vec_from_f32(tmp_f32);

    /* LM Head: encode 32x32 NAX `wCT` E4M3 Tile-rANS (if vocab_size % 32 == 0) + exact FP16 decoded weights */
    const SafetensorEntry* e_lm = reader.find("lm_head.weight");
    if (!e_lm) e_lm = e_emb; /* support weight-tied checkpoints */
    if (!reader.read_tensor_f32(*e_lm, &tmp_f32)) return -EIO;
    {
        std::vector<uint8_t> lm_rans;
        std::vector<uint16_t> lm_f16_dec;
        if ((vocab_size % 32U) == 0U) {
            if (!encode_spine_wct_e4m3_rans(tmp_f32.data(), vocab_size, d_model, &lm_rans, &lm_f16_dec)) {
                return -EBADMSG;
            }
        } else {
            lm_f16_dec.resize(tmp_f32.size());
            for (size_t i = 0; i < tmp_f32.size(); ++i) lm_f16_dec[i] = f32_to_fp16_bits(tmp_f32[i]);
        }
        spine_w.write_u64(lm_rans.size());
        spine_w.align_to(32U);
        spine_w.write_bytes(lm_rans.data(), lm_rans.size());
        spine_w.align_to(16U);
        spine_w.write_bytes(lm_f16_dec.data(), lm_f16_dec.size() * sizeof(uint16_t));
    }

    /* MTP Draft Module (`mtp_norm`, `mtp_proj`, `mtp_head`, `mtp_head_rans`, `shortlist_vocab_ids`) */
    const SafetensorEntry* e_mtp_norm = reader.find("mtp.norm.weight");
    if (e_mtp_norm) {
        if (!reader.read_tensor_f32(*e_mtp_norm, &tmp_f32)) return -EIO;
    } else {
        tmp_f32.assign(d_model, 1.0f);
    }
    write_f16_vec_from_f32(tmp_f32);

    const SafetensorEntry* e_mtp_proj = reader.find("mtp.proj.weight");
    if (e_mtp_proj) {
        if (!reader.read_tensor_f32(*e_mtp_proj, &tmp_f32)) return -EIO;
    } else {
        tmp_f32.assign(static_cast<size_t>(d_model) * (2U * d_model), 0.0f);
        for (uint32_t r = 0; r < d_model; ++r) {
            tmp_f32[static_cast<size_t>(r) * (2U * d_model) + d_model + r] = 1.0f;
        }
    }
    write_f16_vec_from_f32(tmp_f32);

    std::vector<uint32_t> shortlist_ids(shortlist_size);
    const SafetensorEntry* e_sl = reader.find("mtp.shortlist_ids");
    if (e_sl) {
        std::vector<float> sl_f32;
        if (!reader.read_tensor_f32(*e_sl, &sl_f32)) return -EIO;
        for (uint32_t i = 0; i < shortlist_size; ++i) {
            uint32_t id = (i < sl_f32.size()) ? static_cast<uint32_t>(sl_f32[i]) : i;
            shortlist_ids[i] = std::min(id, vocab_size - 1U);
        }
    } else {
        for (uint32_t i = 0; i < shortlist_size; ++i) shortlist_ids[i] = i % vocab_size;
    }

    if (e_mtp_head) {
        if (!reader.read_tensor_f32(*e_mtp_head, &tmp_f32)) return -EIO;
    } else {
        /* Slice shortlist rows from LM head if checkpoint has no dedicated MTP head */
        std::vector<float> full_lm;
        if (!reader.read_tensor_f32(*e_lm, &full_lm)) return -EIO;
        tmp_f32.resize(static_cast<size_t>(shortlist_size) * d_model);
        for (uint32_t i = 0; i < shortlist_size; ++i) {
            std::memcpy(&tmp_f32[static_cast<size_t>(i) * d_model],
                        &full_lm[static_cast<size_t>(shortlist_ids[i]) * d_model],
                        d_model * sizeof(float));
        }
    }
    {
        std::vector<uint8_t> mtp_rans;
        std::vector<uint16_t> mtp_f16_dec;
        if ((shortlist_size % 32U) == 0U) {
            if (!encode_spine_wct_e4m3_rans(tmp_f32.data(), shortlist_size, d_model, &mtp_rans, &mtp_f16_dec)) {
                return -EBADMSG;
            }
        } else {
            mtp_f16_dec.resize(tmp_f32.size());
            for (size_t i = 0; i < tmp_f32.size(); ++i) mtp_f16_dec[i] = f32_to_fp16_bits(tmp_f32[i]);
        }
        spine_w.write_u64(mtp_rans.size());
        spine_w.align_to(32U);
        spine_w.write_bytes(mtp_rans.data(), mtp_rans.size());
        spine_w.align_to(16U);
        spine_w.write_bytes(mtp_f16_dec.data(), mtp_f16_dec.size() * sizeof(uint16_t));
        spine_w.align_to(16U);
        spine_w.write_bytes(shortlist_ids.data(), shortlist_ids.size() * sizeof(uint32_t));
    }

    /* Per-layer Spine weights (`attn_norm`, `ffn_norm`, `router_w`, `q/k/v/o_rans_blob`) */
    std::vector<std::vector<float>> layer_routers_f32(num_layers);
    for (uint32_t l = 0; l < num_layers; ++l) {
        std::string pfx = "model.layers." + std::to_string(l) + ".";
        const SafetensorEntry* e_in  = reader.find(pfx + "input_layernorm.weight");
        const SafetensorEntry* e_pan = reader.find(pfx + "post_attention_layernorm.weight");
        const SafetensorEntry* e_q   = reader.find(pfx + "self_attn.q_proj.weight");
        const SafetensorEntry* e_k   = reader.find(pfx + "self_attn.k_proj.weight");
        const SafetensorEntry* e_v   = reader.find(pfx + "self_attn.v_proj.weight");
        const SafetensorEntry* e_o   = reader.find(pfx + "self_attn.o_proj.weight");
        const SafetensorEntry* e_rt  = reader.find(pfx + "mlp.gate.weight");
        if (!e_q || !e_k || !e_v || !e_o || !e_rt) return -EBADMSG;

        if (e_in) { if (!reader.read_tensor_f32(*e_in, &tmp_f32)) return -EIO; }
        else      { tmp_f32.assign(d_model, 1.0f); }
        write_f16_vec_from_f32(tmp_f32);

        if (e_pan) { if (!reader.read_tensor_f32(*e_pan, &tmp_f32)) return -EIO; }
        else       { tmp_f32.assign(d_model, 1.0f); }
        write_f16_vec_from_f32(tmp_f32);

        if (!reader.read_tensor_f32(*e_rt, &layer_routers_f32[l])) return -EIO;
        write_f16_vec_from_f32(layer_routers_f32[l]);

        const std::pair<const SafetensorEntry*, std::pair<uint32_t, uint32_t>> qkvo[4] = {
            {e_q, {q_dim, d_model}},
            {e_k, {kv_dim, d_model}},
            {e_v, {kv_dim, d_model}},
            {e_o, {d_model, q_dim}}
        };
        for (int p = 0; p < 4; ++p) {
            if (!reader.read_tensor_f32(*qkvo[p].first, &tmp_f32)) return -EIO;
            std::vector<uint8_t> rblob;
            if (!encode_spine_wct_e4m3_rans(tmp_f32.data(), qkvo[p].second.first, qkvo[p].second.second, &rblob, nullptr)) {
                return -EBADMSG;
            }
            spine_w.write_u64(rblob.size());
            spine_w.align_to(32U);
            spine_w.write_bytes(rblob.data(), rblob.size());
        }
    }

    if (!write_binary_file(dst_dir + "/spine.bin", spine_w.buf.data(), spine_w.buf.size())) {
        return -EIO;
    }

    /* ========================================================================
     * AXIS C (`uma_hot.bin`) & AXIS D (`cold_experts.strata`, 256 KiB aligned)
     * ======================================================================== */
    const uint64_t total_exp_weights = 3ULL * d_model * d_ff;
    const uint64_t exp_planar_bytes  = (regime == STRATA_REGIME_I_BITNEST)
                                           ? align_up_u64((total_exp_weights / 128ULL) * 34ULL, 256ULL)
                                           : align_up_u64((total_exp_weights / 32ULL) * 17ULL, 256ULL);
    const uint64_t cold_extent_bytes = align_up_u64(std::max<uint64_t>(exp_planar_bytes, STRATA_LBA_ALIGN_BYTES),
                                                    STRATA_LBA_ALIGN_BYTES);

    BinaryWriter uma_w;
    uma_w.write_u32(kUmaHotFileMagic);
    uma_w.write_u32(kBundleFileVer);
    uma_w.write_u32(num_layers);
    uma_w.write_u32(num_experts);

    struct PendingColdExtent {
        uint32_t layer_idx;
        uint32_t expert_idx;
        std::vector<uint8_t> planar_payload;
    };
    std::vector<PendingColdExtent> cold_list;
    uint32_t total_hot_experts  = 0U;
    uint32_t total_cold_experts = 0U;

    for (uint32_t l = 0; l < num_layers; ++l) {
        /* 1. Determine Hot vs Cold tier per expert in layer `l` */
        std::vector<uint8_t> is_hot_u8(num_experts, 0U);
        if (l < pinned_layers || regime == STRATA_REGIME_I_100PCT_UMA) {
            std::fill(is_hot_u8.begin(), is_hot_u8.end(), 1U);
        } else {
            /* Rank experts by Router L2 norm to pin the highest-affinity `max(1, E/4)` experts in UMA */
            const uint32_t target_hot = std::max(1U, num_experts / 4U);
            std::vector<std::pair<double, uint32_t>> norms(num_experts);
            for (uint32_t e = 0; e < num_experts; ++e) {
                const float* rw = &layer_routers_f32[l][static_cast<size_t>(e) * d_model];
                double sq = 0.0;
                for (uint32_t c = 0; c < d_model; ++c) sq += static_cast<double>(rw[c]) * rw[c];
                norms[e] = {sq, e};
            }
            std::sort(norms.begin(), norms.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
            for (uint32_t k = 0; k < target_hot; ++k) {
                is_hot_u8[norms[k].second] = 1U;
            }
        }

        /* 2. Read all `num_experts` FP32 expert weights from `.safetensors` */
        std::vector<std::vector<float>> exp_gate(num_experts);
        std::vector<std::vector<float>> exp_up(num_experts);
        std::vector<std::vector<float>> exp_down(num_experts);
        for (uint32_t e = 0; e < num_experts; ++e) {
            std::string epfx = "model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + ".";
            const SafetensorEntry* eg = reader.find(epfx + "gate_proj.weight");
            const SafetensorEntry* eu = reader.find(epfx + "up_proj.weight");
            const SafetensorEntry* ed = reader.find(epfx + "down_proj.weight");
            if (!eg) eg = reader.find(epfx + "w1.weight");
            if (!eu) eu = reader.find(epfx + "w3.weight");
            if (!ed) ed = reader.find(epfx + "w2.weight");
            if (!eg || !eu || !ed) return -EBADMSG;

            if (!reader.read_tensor_f32(*eg, &exp_gate[e]) ||
                !reader.read_tensor_f32(*eu, &exp_up[e]) ||
                !reader.read_tensor_f32(*ed, &exp_down[e])) {
                return -EIO;
            }
        }

        uint32_t hot_count = 0U;
        for (uint8_t h : is_hot_u8) if (h) ++hot_count;
        total_hot_experts  += hot_count;
        total_cold_experts += (num_experts - hot_count);

        const uint32_t hot_pool_slots = std::max(1U, hot_count);
        std::vector<uint8_t> hot_pool_buf(static_cast<size_t>(hot_pool_slots) * exp_planar_bytes, 0U);
        std::vector<uint8_t> base_pool_buf;
        uint32_t base_pool_slots = 1U;

        std::vector<uint16_t> tier_kind(num_experts, 0U);
        std::vector<uint16_t> hot_or_base_idx(num_experts, 0U);
        std::vector<int32_t>  cold_extent_idx(num_experts, -1);

        std::vector<uint8_t>  svd_u_w6;
        std::vector<uint16_t> svd_v_f16;
        std::vector<uint16_t> svd_v_shared_f16;
        uint32_t svd_rank_aligned = 0U;
        uint32_t svd_u_slots      = 0U;

        if (regime == STRATA_REGIME_I_BITNEST) {
            /* BitNest: every expert gets QTIP 2b base in `base_pool_buf[slot]`;
             * hot experts get 4-lvl refine in `hot_pool_buf[0 .. hot_count-1]`,
             * warm experts get 4-lvl refine in `cold_experts.strata`! */
            base_pool_slots = num_experts;
            base_pool_buf.assign(static_cast<size_t>(num_experts) * exp_planar_bytes, 0U);

            uint32_t cur_hot  = 0U;
            uint32_t cur_warm = 0U;
            std::vector<uint8_t> tmp_ref(static_cast<size_t>(exp_planar_bytes), 0U);

            for (uint32_t e = 0; e < num_experts; ++e) {
                uint32_t assigned_slot = is_hot_u8[e] ? (cur_hot++) : (hot_count + cur_warm++);
                hot_or_base_idx[e]     = static_cast<uint16_t>(assigned_slot);

                uint8_t* dst_base = &base_pool_buf[static_cast<size_t>(assigned_slot) * exp_planar_bytes];
                pack_expert_qtip_bitnest_planar(
                    exp_gate[e].data(), exp_up[e].data(), exp_down[e].data(),
                    d_model, d_ff, dst_base, tmp_ref.data(), exp_planar_bytes);

                if (is_hot_u8[e]) {
                    tier_kind[e]       = static_cast<uint16_t>(STRATA_TIER_HOT_FULL_4B);
                    cold_extent_idx[e] = -1;
                    std::memcpy(&hot_pool_buf[static_cast<size_t>(assigned_slot) * exp_planar_bytes],
                                tmp_ref.data(), static_cast<size_t>(exp_planar_bytes));
                } else {
                    tier_kind[e]       = static_cast<uint16_t>(STRATA_TIER_WARM_QTIP_RING);
                    cold_extent_idx[e] = static_cast<int32_t>(cold_list.size());
                    cold_list.push_back(PendingColdExtent{l, e, tmp_ref});
                }
            }
        } else {
            /* Regime I (100% UMA) or Regime II (Barycenter + W6A16 MoE-SVD + MXFP4):
             * Compute exact layer Barycenter `mean_e W_e` and pack into `base_pool_buf`! */
            std::vector<float> bary_gate(static_cast<size_t>(d_ff) * d_model, 0.0f);
            std::vector<float> bary_up(static_cast<size_t>(d_ff) * d_model, 0.0f);
            std::vector<float> bary_down(static_cast<size_t>(d_model) * d_ff, 0.0f);
            const float inv_e = 1.0f / static_cast<float>(num_experts);

            for (uint32_t e = 0; e < num_experts; ++e) {
                for (size_t i = 0; i < bary_gate.size(); ++i) {
                    bary_gate[i] += exp_gate[e][i] * inv_e;
                    bary_up[i]   += exp_up[e][i] * inv_e;
                }
                for (size_t i = 0; i < bary_down.size(); ++i) {
                    bary_down[i] += exp_down[e][i] * inv_e;
                }
            }

            base_pool_buf.assign(static_cast<size_t>(exp_planar_bytes), 0U);
            pack_expert_mxfp4_nax_planar(
                bary_gate.data(), bary_up.data(), bary_down.data(),
                d_model, d_ff, base_pool_buf.data(), exp_planar_bytes);

            if (svd_rank > 0U) {
                svd_rank_aligned = (svd_rank + 63U) & ~63U;
                svd_u_slots      = num_experts;
                base_pool_slots  = num_experts; /* matches `u_slot = hot_or_base_idx % base_pool_slots` in shader */
                compute_layer_moe_svd_w6a16(
                    exp_gate, exp_up, bary_gate, bary_up,
                    num_experts, d_model, d_ff, svd_rank,
                    &svd_u_w6, &svd_v_f16, &svd_v_shared_f16);
            } else {
                base_pool_slots = 1U;
            }

            uint32_t cur_hot = 0U;
            std::vector<uint8_t> tmp_cold(static_cast<size_t>(exp_planar_bytes), 0U);
            for (uint32_t e = 0; e < num_experts; ++e) {
                if (is_hot_u8[e]) {
                    uint32_t h_slot    = cur_hot++;
                    tier_kind[e]       = static_cast<uint16_t>(STRATA_TIER_HOT_FULL_4B);
                    hot_or_base_idx[e] = static_cast<uint16_t>(h_slot);
                    cold_extent_idx[e] = -1;
                    pack_expert_mxfp4_nax_planar(
                        exp_gate[e].data(), exp_up[e].data(), exp_down[e].data(),
                        d_model, d_ff,
                        &hot_pool_buf[static_cast<size_t>(h_slot) * exp_planar_bytes],
                        exp_planar_bytes);
                } else {
                    tier_kind[e]       = static_cast<uint16_t>(STRATA_TIER_COLD_SVD_RING);
                    hot_or_base_idx[e] = static_cast<uint16_t>(e); /* indexes `svd_u_w6[e]` */
                    cold_extent_idx[e] = static_cast<int32_t>(cold_list.size());
                    pack_expert_mxfp4_nax_planar(
                        exp_gate[e].data(), exp_up[e].data(), exp_down[e].data(),
                        d_model, d_ff, tmp_cold.data(), exp_planar_bytes);
                    cold_list.push_back(PendingColdExtent{l, e, tmp_cold});
                }
            }
        }

        /* Serialize Layer `l` Axis C payload into `uma_hot.bin` */
        uma_w.align_to(16U);
        uma_w.write_bytes(is_hot_u8.data(), num_experts);
        uma_w.align_to(16U);
        uma_w.write_bytes(tier_kind.data(), num_experts * sizeof(uint16_t));
        uma_w.write_bytes(hot_or_base_idx.data(), num_experts * sizeof(uint16_t));
        uma_w.write_bytes(cold_extent_idx.data(), num_experts * sizeof(int32_t));

        uma_w.write_u32(hot_pool_slots);
        uma_w.write_u64(exp_planar_bytes);
        uma_w.write_u64(hot_pool_buf.size());
        uma_w.align_to(256U);
        uma_w.write_bytes(hot_pool_buf.data(), hot_pool_buf.size());

        uma_w.write_u32(base_pool_slots);
        uma_w.write_u64(exp_planar_bytes);
        uma_w.write_u64(base_pool_buf.size());
        uma_w.align_to(256U);
        uma_w.write_bytes(base_pool_buf.data(), base_pool_buf.size());

        uma_w.write_u32(svd_rank_aligned);
        uma_w.write_u32(svd_u_slots);
        uma_w.write_u64(svd_u_w6.size());
        uma_w.write_u64(svd_v_f16.size() * sizeof(uint16_t));
        uma_w.write_u64(svd_v_shared_f16.size() * sizeof(uint16_t));
        uma_w.align_to(64U);
        if (!svd_u_w6.empty())         uma_w.write_bytes(svd_u_w6.data(), svd_u_w6.size());
        uma_w.align_to(16U);
        if (!svd_v_f16.empty())        uma_w.write_bytes(svd_v_f16.data(), svd_v_f16.size() * sizeof(uint16_t));
        if (!svd_v_shared_f16.empty()) uma_w.write_bytes(svd_v_shared_f16.data(), svd_v_shared_f16.size() * sizeof(uint16_t));
    }

    if (!write_binary_file(dst_dir + "/uma_hot.bin", uma_w.buf.data(), uma_w.buf.size())) {
        return -EIO;
    }

    /* Write Axis D: `<bundle_dir>/cold_experts.strata` (256 KiB LBA-aligned `F_NOCACHE`) */
    const uint32_t num_extents    = static_cast<uint32_t>(cold_list.size());
    const uint64_t manifest_bytes = sizeof(StrataContainerHeader) +
                                    static_cast<uint64_t>(num_extents) * sizeof(StrataExtentRecord);
    const uint64_t data_start     = std::max<uint64_t>(
        STRATA_LBA_ALIGN_BYTES, align_up_u64(manifest_bytes, STRATA_LBA_ALIGN_BYTES));
    const uint64_t total_cold_file_bytes = data_start + static_cast<uint64_t>(num_extents) * cold_extent_bytes;

    std::vector<StrataExtentRecord> records(num_extents);
    for (uint32_t i = 0; i < num_extents; ++i) {
        records[i].layer_idx     = cold_list[i].layer_idx;
        records[i].expert_idx    = cold_list[i].expert_idx;
        records[i].offset_bytes  = data_start + static_cast<uint64_t>(i) * cold_extent_bytes;
        records[i].payload_bytes = cold_list[i].planar_payload.size();
        records[i].aligned_bytes = cold_extent_bytes;
    }

    std::string cold_path = dst_dir + "/cold_experts.strata";
    int cfd = ::open(cold_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (cfd < 0) return -errno;
    (void)::fcntl(cfd, F_NOCACHE, 1);

    std::vector<uint8_t> meta_region(static_cast<size_t>(data_start), 0U);
    StrataContainerHeader chdr{};
    chdr.magic             = STRATA_MAGIC_U64;
    chdr.abi_version       = STRATA_ABI_VERSION;
    chdr.num_extents       = num_extents;
    chdr.data_start_offset = data_start;
    chdr.total_file_bytes  = total_cold_file_bytes;
    std::memcpy(meta_region.data(), &chdr, sizeof(chdr));
    if (num_extents > 0U) {
        std::memcpy(meta_region.data() + sizeof(chdr),
                    records.data(),
                    static_cast<size_t>(num_extents) * sizeof(StrataExtentRecord));
    }
    if (!write_all_fd(cfd, meta_region.data(), meta_region.size())) {
        ::close(cfd);
        return -EIO;
    }

    void* aligned_buf = nullptr;
    if (::posix_memalign(&aligned_buf, STRATA_VM_PAGE_BYTES, static_cast<size_t>(cold_extent_bytes)) != 0 || !aligned_buf) {
        ::close(cfd);
        return -ENOMEM;
    }

    for (uint32_t i = 0; i < num_extents; ++i) {
        /* Fill trailing padding past `exp_planar_bytes` with deterministic high-entropy bytes (no APFS sparse holes) */
        uint64_t pad_rng = 0x5354524154414D35ULL ^
                           (static_cast<uint64_t>(records[i].layer_idx) << 32U) ^
                           static_cast<uint64_t>(records[i].expert_idx);
        uint64_t* u64p = static_cast<uint64_t*>(aligned_buf);
        for (size_t w = 0; w < cold_extent_bytes / sizeof(uint64_t); ++w) {
            u64p[w] = splitmix64(&pad_rng);
        }
        std::memcpy(aligned_buf, cold_list[i].planar_payload.data(), cold_list[i].planar_payload.size());
        if (!write_all_fd(cfd, aligned_buf, static_cast<size_t>(cold_extent_bytes))) {
            std::free(aligned_buf);
            ::close(cfd);
            return -EIO;
        }
    }
    (void)::fsync(cfd);
    std::free(aligned_buf);
    ::close(cfd);

    /* Verify LBA alignment of every extent */
    bool lba_ok = ((chdr.data_start_offset % STRATA_LBA_ALIGN_BYTES) == 0ULL);
    for (const auto& r : records) {
        if ((r.offset_bytes % STRATA_LBA_ALIGN_BYTES) != 0ULL ||
            (r.aligned_bytes % STRATA_LBA_ALIGN_BYTES) != 0ULL) {
            lba_ok = false;
        }
    }
    if (!lba_ok) return -EBADMSG;

    /* Write human-readable `manifest.json` */
    uint64_t st_payload = 0ULL;
    for (const auto& kv : reader.tensors) st_payload += (kv.second.end_off - kv.second.begin_off);

    std::string manifest_json =
        "{\n"
        "  \"strata_abi_version\": 200,\n"
        "  \"arch_name\": \"" + json_escape(arch_name) + "\",\n"
        "  \"regime\": " + std::to_string(static_cast<uint32_t>(regime)) + ",\n"
        "  \"d_model\": " + std::to_string(d_model) + ",\n"
        "  \"d_ff\": " + std::to_string(d_ff) + ",\n"
        "  \"num_layers\": " + std::to_string(num_layers) + ",\n"
        "  \"num_experts\": " + std::to_string(num_experts) + ",\n"
        "  \"top_k\": " + std::to_string(top_k) + ",\n"
        "  \"svd_rank\": " + std::to_string(svd_rank) + ",\n"
        "  \"vocab_size\": " + std::to_string(vocab_size) + ",\n"
        "  \"shortlist_size\": " + std::to_string(shortlist_size) + ",\n"
        "  \"n_heads\": " + std::to_string(n_heads) + ",\n"
        "  \"n_kv_heads\": " + std::to_string(n_kv_heads) + ",\n"
        "  \"head_dim\": " + std::to_string(head_dim) + ",\n"
        "  \"axis_a_ngram_bytes\": " + std::to_string(axis_a_bytes) + ",\n"
        "  \"axis_b_spine_bytes\": " + std::to_string(spine_w.buf.size()) + ",\n"
        "  \"axis_c_uma_hot_bytes\": " + std::to_string(uma_w.buf.size()) + ",\n"
        "  \"axis_d_cold_strata_bytes\": " + std::to_string(total_cold_file_bytes) + ",\n"
        "  \"num_hot_experts\": " + std::to_string(total_hot_experts) + ",\n"
        "  \"num_cold_extents\": " + std::to_string(num_extents) + "\n"
        "}\n";
    if (!write_binary_file(dst_dir + "/manifest.json", manifest_json.data(), manifest_json.size())) {
        return -EIO;
    }

    if (stats_out) {
        stats_out->num_safetensors_tensors   = static_cast<uint32_t>(reader.tensors.size());
        stats_out->safetensors_payload_bytes = st_payload;
        stats_out->axis_a_ngram_bytes        = axis_a_bytes;
        stats_out->axis_b_spine_bytes        = spine_w.buf.size();
        stats_out->axis_c_uma_hot_bytes      = uma_w.buf.size();
        stats_out->axis_d_cold_strata_bytes  = total_cold_file_bytes;
        stats_out->num_hot_experts_total     = total_hot_experts;
        stats_out->num_cold_experts_total    = total_cold_experts;
        stats_out->rans_roundtrip_verified   = true;
        stats_out->lba_alignment_verified    = lba_ok;
    }
    return 0;
}

int strata_bundle_load(
    const char* bundle_dir,
    StrataCheckpointBundle* bundle_out
) {
    if (!bundle_dir || !bundle_out || bundle_dir[0] == '\0') return -EINVAL;
    std::memset(bundle_out, 0, sizeof(*bundle_out));

    std::string bdir(bundle_dir);
    std::snprintf(bundle_out->bundle_dir, sizeof(bundle_out->bundle_dir), "%s", bdir.c_str());

    std::vector<uint8_t> spine_raw;
    std::vector<uint8_t> uma_raw;
    if (!read_entire_file(bdir + "/spine.bin", &spine_raw) ||
        !read_entire_file(bdir + "/uma_hot.bin", &uma_raw)) {
        return -ENOENT;
    }

    BinaryReader sr{spine_raw.data(), spine_raw.size(), 0U};
    uint32_t smagic = 0U, sver = 0U, regime_u32 = 0U, has_ng_u32 = 0U;
    if (!sr.read_u32(&smagic) || !sr.read_u32(&sver) ||
        smagic != kSpineFileMagic || sver != kBundleFileVer) {
        return -EBADMSG;
    }
    if (!sr.read_bytes(bundle_out->arch_name, sizeof(bundle_out->arch_name))) return -EBADMSG;
    bundle_out->arch_name[sizeof(bundle_out->arch_name) - 1U] = '\0';

    StrataModelConfig& cfg = bundle_out->cfg;
    cfg.name = bundle_out->arch_name;

    if (!sr.read_u32(&regime_u32) ||
        !sr.read_u32(&cfg.num_layers) ||
        !sr.read_u32(&cfg.encoder_layers) ||
        !sr.read_u32(&cfg.d_model) ||
        !sr.read_u32(&cfg.d_ff) ||
        !sr.read_u32(&cfg.num_experts) ||
        !sr.read_u32(&cfg.top_k) ||
        !sr.read_u32(&cfg.vocab_size) ||
        !sr.read_u32(&cfg.shortlist_size) ||
        !sr.read_u32(&cfg.svd_rank) ||
        !sr.read_u32(&cfg.pinned_shallow_layers) ||
        !sr.read_u32(&has_ng_u32) ||
        !sr.read_u32(&cfg.ngram_heads_per_tok) ||
        !sr.read_u32(&bundle_out->ngram_layer_idx) ||
        !sr.read_f32(&cfg.mode_a_tau_tok) ||
        !sr.read_u32(&cfg.mode_b_cold_cap) ||
        !sr.read_u32(&bundle_out->n_heads) ||
        !sr.read_u32(&bundle_out->n_kv_heads) ||
        !sr.read_u32(&bundle_out->head_dim) ||
        !sr.read_u32(&bundle_out->ane_probe_rank) ||
        !sr.read_f32(&bundle_out->rms_norm_eps) ||
        !sr.read_f32(&bundle_out->rope_theta) ||
        !sr.read_u32(&bundle_out->max_seq_len)) {
        return -EBADMSG;
    }
    cfg.regime          = static_cast<StrataRegime>(regime_u32);
    cfg.has_ngram_table = (has_ng_u32 != 0U);
    if (cfg.num_layers == 0U || cfg.num_layers > 256U ||
        cfg.d_model == 0U || cfg.d_model > 32768U || (cfg.d_model % 32U) != 0U ||
        cfg.d_ff == 0U || cfg.d_ff > 131072U || (cfg.d_ff % 32U) != 0U ||
        cfg.num_experts == 0U || cfg.num_experts > 4096U ||
        cfg.top_k == 0U || cfg.top_k > cfg.num_experts || cfg.top_k > STRATA_MAX_TOP_K ||
        cfg.svd_rank > 4096U ||
        cfg.vocab_size == 0U || cfg.vocab_size > (1U << 22) ||
        cfg.shortlist_size > cfg.vocab_size ||
        bundle_out->n_heads == 0U || bundle_out->n_heads > 256U ||
        bundle_out->n_kv_heads == 0U || bundle_out->n_kv_heads > bundle_out->n_heads ||
        (bundle_out->n_heads % bundle_out->n_kv_heads) != 0U ||
        bundle_out->head_dim == 0U || bundle_out->head_dim > 1024U ||
        cfg.ngram_heads_per_tok > 64U) {
        return -EBADMSG;
    }
    const uint64_t q_dim_u64  = static_cast<uint64_t>(bundle_out->n_heads) * bundle_out->head_dim;
    const uint64_t kv_dim_u64 = static_cast<uint64_t>(bundle_out->n_kv_heads) * bundle_out->head_dim;
    if ((q_dim_u64 % 32ULL) != 0ULL || (kv_dim_u64 % 32ULL) != 0ULL) {
        return -EBADMSG;
    }
    const uint64_t total_exp_weights = 3ULL * cfg.d_model * cfg.d_ff;
    if ((cfg.regime == STRATA_REGIME_I_BITNEST && (total_exp_weights % 128ULL) != 0ULL) ||
        (cfg.regime != STRATA_REGIME_I_BITNEST && (total_exp_weights % 32ULL) != 0ULL)) {
        return -EBADMSG;
    }
    const uint64_t exp_planar_bytes = (cfg.regime == STRATA_REGIME_I_BITNEST)
                                          ? align_up_u64((total_exp_weights / 128ULL) * 34ULL, 256ULL)
                                          : align_up_u64((total_exp_weights / 32ULL) * 17ULL, 256ULL);
    const uint32_t exp_svd_alg = (cfg.svd_rank > 0U) ? ((cfg.svd_rank + 63U) & ~63U) : 0U;
    std::vector<uint8_t> rans_verify_buf;
    auto verify_rans_blob = [&](const uint8_t* blob, size_t len, uint64_t exp_elems) -> bool {
        if (!blob || len < sizeof(StrataTileRansHeader) || exp_elems == 0ULL || exp_elems > (1ULL << 28)) {
            return false;
        }
        if ((exp_elems % 1024ULL) != 0ULL) {
            return false;
        }
        StrataTileRansHeader rh{};
        std::memcpy(&rh, blob, sizeof(rh));
        if (rh.reserved[0] != 1024U ||
            static_cast<uint64_t>(rh.reserved[1]) != (exp_elems / 1024ULL)) {
            return false;
        }
        if (rans_verify_buf.size() < static_cast<size_t>(exp_elems)) {
            rans_verify_buf.resize(static_cast<size_t>(exp_elems));
        }
        return strata_codec_rans_decode_32way_cpu_ref(blob, len, rans_verify_buf.data(), static_cast<size_t>(exp_elems)) == 0;
    };

    auto fail_cleanup = [&](int err_code) -> int {
        strata_bundle_free(bundle_out);
        return err_code;
    };

    auto alloc_read = [&](void** dst_ptr, size_t nbytes, size_t align_pad) -> bool {
        if (!sr.align_to(align_pad)) return false;
        if (nbytes > sr.remaining()) return false;
        size_t alloc_sz = std::max<size_t>(nbytes, 16U);
        void* p = std::calloc(1U, alloc_sz);
        if (!p) return false;
        if (nbytes > 0U && !sr.read_bytes(p, nbytes)) {
            std::free(p);
            return false;
        }
        *dst_ptr = p;
        return true;
    };

    /* Global Axis B Tensors */
    if (!alloc_read(reinterpret_cast<void**>(&bundle_out->tok_embed_f16),
                    static_cast<size_t>(cfg.vocab_size) * cfg.d_model * sizeof(uint16_t), 16U)) {
        return fail_cleanup(-EBADMSG);
    }
    if (!alloc_read(reinterpret_cast<void**>(&bundle_out->final_norm_f16),
                    static_cast<size_t>(cfg.d_model) * sizeof(uint16_t), 16U)) {
        return fail_cleanup(-EBADMSG);
    }

    uint64_t lm_rans_u64 = 0ULL;
    if (!sr.read_u64(&lm_rans_u64)) return fail_cleanup(-EBADMSG);
    bundle_out->lm_head_rans_bytes = static_cast<size_t>(lm_rans_u64);
    if (lm_rans_u64 > 0ULL) {
        if (!alloc_read(reinterpret_cast<void**>(&bundle_out->lm_head_rans_blob),
                        bundle_out->lm_head_rans_bytes, 32U) ||
            !verify_rans_blob(bundle_out->lm_head_rans_blob, bundle_out->lm_head_rans_bytes,
                              static_cast<uint64_t>(cfg.vocab_size) * cfg.d_model)) {
            return fail_cleanup(-EBADMSG);
        }
    } else {
        sr.align_to(32U);
    }
    if (!alloc_read(reinterpret_cast<void**>(&bundle_out->lm_head_f16),
                    static_cast<size_t>(cfg.vocab_size) * cfg.d_model * sizeof(uint16_t), 16U)) {
        return fail_cleanup(-EBADMSG);
    }

    if (!alloc_read(reinterpret_cast<void**>(&bundle_out->mtp_norm_f16),
                    static_cast<size_t>(cfg.d_model) * sizeof(uint16_t), 16U)) {
        return fail_cleanup(-EBADMSG);
    }
    if (!alloc_read(reinterpret_cast<void**>(&bundle_out->mtp_proj_f16),
                    static_cast<size_t>(cfg.d_model) * (2U * cfg.d_model) * sizeof(uint16_t), 16U)) {
        return fail_cleanup(-EBADMSG);
    }

    uint64_t mtp_rans_u64 = 0ULL;
    if (!sr.read_u64(&mtp_rans_u64)) return fail_cleanup(-EBADMSG);
    bundle_out->mtp_head_rans_bytes = static_cast<size_t>(mtp_rans_u64);
    if (mtp_rans_u64 > 0ULL) {
        if (!alloc_read(reinterpret_cast<void**>(&bundle_out->mtp_head_rans_blob),
                        bundle_out->mtp_head_rans_bytes, 32U) ||
            !verify_rans_blob(bundle_out->mtp_head_rans_blob, bundle_out->mtp_head_rans_bytes,
                              static_cast<uint64_t>(cfg.shortlist_size) * cfg.d_model)) {
            return fail_cleanup(-EBADMSG);
        }
    } else {
        sr.align_to(32U);
    }
    if (!alloc_read(reinterpret_cast<void**>(&bundle_out->mtp_head_f16),
                    static_cast<size_t>(cfg.shortlist_size) * cfg.d_model * sizeof(uint16_t), 16U)) {
        return fail_cleanup(-EBADMSG);
    }
    if (!alloc_read(reinterpret_cast<void**>(&bundle_out->shortlist_vocab_ids),
                    static_cast<size_t>(cfg.shortlist_size) * sizeof(uint32_t), 16U)) {
        return fail_cleanup(-EBADMSG);
    }

    bundle_out->layers = static_cast<StrataLayerWeights*>(
        std::calloc(cfg.num_layers, sizeof(StrataLayerWeights)));
    if (!bundle_out->layers) return fail_cleanup(-ENOMEM);

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        StrataLayerWeights& lw = bundle_out->layers[l];
        lw.layer_idx  = l;
        lw.n_heads    = bundle_out->n_heads;
        lw.n_kv_heads = bundle_out->n_kv_heads;
        lw.head_dim   = bundle_out->head_dim;

        if (!alloc_read(reinterpret_cast<void**>(&lw.attn_norm_f16),
                        static_cast<size_t>(cfg.d_model) * sizeof(uint16_t), 16U) ||
            !alloc_read(reinterpret_cast<void**>(&lw.ffn_norm_f16),
                        static_cast<size_t>(cfg.d_model) * sizeof(uint16_t), 16U) ||
            !alloc_read(reinterpret_cast<void**>(&lw.router_w_f16),
                        static_cast<size_t>(cfg.num_experts) * cfg.d_model * sizeof(uint16_t), 16U)) {
            return fail_cleanup(-EBADMSG);
        }

        uint8_t** blob_ptrs[4]  = {&lw.q_rans_blob, &lw.k_rans_blob, &lw.v_rans_blob, &lw.o_rans_blob};
        size_t*   blob_sizes[4] = {&lw.q_rans_bytes, &lw.k_rans_bytes, &lw.v_rans_bytes, &lw.o_rans_bytes};
        const uint64_t blob_elems[4] = {
            q_dim_u64 * cfg.d_model,
            kv_dim_u64 * cfg.d_model,
            kv_dim_u64 * cfg.d_model,
            static_cast<uint64_t>(cfg.d_model) * q_dim_u64
        };
        for (int p = 0; p < 4; ++p) {
            uint64_t bsz = 0ULL;
            if (!sr.read_u64(&bsz) || bsz == 0ULL) return fail_cleanup(-EBADMSG);
            *blob_sizes[p] = static_cast<size_t>(bsz);
            if (!alloc_read(reinterpret_cast<void**>(blob_ptrs[p]), static_cast<size_t>(bsz), 32U) ||
                !verify_rans_blob(*blob_ptrs[p], static_cast<size_t>(bsz), blob_elems[p])) {
                return fail_cleanup(-EBADMSG);
            }
        }
    }

    /* Axis C: Read `uma_hot.bin` */
    BinaryReader ur{uma_raw.data(), uma_raw.size(), 0U};
    uint32_t umagic = 0U, uver = 0U, ulayers = 0U, uexps = 0U;
    if (!ur.read_u32(&umagic) || !ur.read_u32(&uver) ||
        !ur.read_u32(&ulayers) || !ur.read_u32(&uexps) ||
        umagic != kUmaHotFileMagic || uver != kBundleFileVer ||
        ulayers != cfg.num_layers || uexps != cfg.num_experts) {
        return fail_cleanup(-EBADMSG);
    }

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        StrataLayerWeights& lw = bundle_out->layers[l];
        const size_t E = cfg.num_experts;

        lw.is_uma_hot       = static_cast<bool*>(std::calloc(E, sizeof(bool)));
        lw.tier_kind        = static_cast<uint16_t*>(std::calloc(E, sizeof(uint16_t)));
        lw.hot_or_base_idx  = static_cast<uint16_t*>(std::calloc(E, sizeof(uint16_t)));
        lw.cold_extent_idx  = static_cast<int32_t*>(std::calloc(E, sizeof(int32_t)));
        lw.default_slot_map = static_cast<StrataSlotEntry*>(std::calloc(E, sizeof(StrataSlotEntry)));
        if (!lw.is_uma_hot || !lw.tier_kind || !lw.hot_or_base_idx || !lw.cold_extent_idx || !lw.default_slot_map) {
            return fail_cleanup(-ENOMEM);
        }

        std::vector<uint8_t> hot_u8(E, 0U);
        if (!ur.align_to(16U) || !ur.read_bytes(hot_u8.data(), E)) return fail_cleanup(-EBADMSG);
        for (size_t e = 0; e < E; ++e) lw.is_uma_hot[e] = (hot_u8[e] != 0U);

        if (!ur.align_to(16U) ||
            !ur.read_bytes(lw.tier_kind, E * sizeof(uint16_t)) ||
            !ur.read_bytes(lw.hot_or_base_idx, E * sizeof(uint16_t)) ||
            !ur.read_bytes(lw.cold_extent_idx, E * sizeof(int32_t))) {
            return fail_cleanup(-EBADMSG);
        }

        for (size_t e = 0; e < E; ++e) {
            lw.default_slot_map[e].hot_or_base_idx = lw.hot_or_base_idx[e];
            lw.default_slot_map[e].ring_slot_idx   = STRATA_SLOT_INVALID;
            lw.default_slot_map[e].tier_kind       = lw.tier_kind[e];
            lw.default_slot_map[e].reserved        = 0U;
            lw.default_slot_map[e].expected_epoch  = 0U;
        }

        uint64_t hp_bytes = 0ULL;
        if (!ur.read_u32(&lw.hot_pool_slots) ||
            !ur.read_u64(&lw.hot_expert_bytes) ||
            !ur.read_u64(&hp_bytes) ||
            !ur.align_to(256U) ||
            lw.hot_pool_slots == 0U ||
            lw.hot_pool_slots > cfg.num_experts ||
            lw.hot_expert_bytes != exp_planar_bytes ||
            hp_bytes > ur.remaining() ||
            hp_bytes < static_cast<uint64_t>(lw.hot_pool_slots) * exp_planar_bytes) {
            return fail_cleanup(-EBADMSG);
        }
        lw.hot_pool_bytes = static_cast<size_t>(hp_bytes);
        lw.hot_pool_buf   = static_cast<uint8_t*>(std::calloc(1U, std::max<size_t>(lw.hot_pool_bytes, 256U)));
        if (!lw.hot_pool_buf || !ur.read_bytes(lw.hot_pool_buf, lw.hot_pool_bytes)) {
            return fail_cleanup(-EBADMSG);
        }

        uint64_t bp_bytes = 0ULL;
        const uint64_t min_bp_slots_stored = (cfg.regime == STRATA_REGIME_I_BITNEST)
                                                 ? static_cast<uint64_t>(cfg.num_experts)
                                                 : 1ULL;
        if (!ur.read_u32(&lw.base_pool_slots) ||
            !ur.read_u64(&lw.base_expert_bytes) ||
            !ur.read_u64(&bp_bytes) ||
            !ur.align_to(256U) ||
            lw.base_pool_slots == 0U ||
            lw.base_pool_slots > cfg.num_experts ||
            lw.base_expert_bytes != exp_planar_bytes ||
            bp_bytes > ur.remaining() ||
            bp_bytes < min_bp_slots_stored * exp_planar_bytes ||
            (cfg.regime == STRATA_REGIME_I_BITNEST &&
             (lw.base_pool_slots != cfg.num_experts ||
              bp_bytes < static_cast<uint64_t>(lw.base_pool_slots) * exp_planar_bytes))) {
            return fail_cleanup(-EBADMSG);
        }
        lw.base_pool_bytes = static_cast<size_t>(bp_bytes);
        lw.base_pool_buf   = static_cast<uint8_t*>(std::calloc(1U, std::max<size_t>(lw.base_pool_bytes, 256U)));
        if (!lw.base_pool_buf || !ur.read_bytes(lw.base_pool_buf, lw.base_pool_bytes)) {
            return fail_cleanup(-EBADMSG);
        }

        uint64_t u_w6_sz = 0ULL, v_f16_sz = 0ULL, vs_f16_sz = 0ULL;
        if (!ur.read_u32(&lw.svd_rank_aligned) ||
            !ur.read_u32(&lw.svd_u_slots) ||
            !ur.read_u64(&u_w6_sz) ||
            !ur.read_u64(&v_f16_sz) ||
            !ur.read_u64(&vs_f16_sz) ||
            !ur.align_to(64U) ||
            lw.svd_rank_aligned != exp_svd_alg ||
            lw.svd_u_slots > cfg.num_experts ||
            (exp_svd_alg > 0U && lw.svd_u_slots == 0U) ||
            u_w6_sz > ur.remaining()) {
            return fail_cleanup(-EBADMSG);
        }
        const uint64_t min_u_w6 =
            static_cast<uint64_t>(lw.svd_u_slots) * cfg.d_ff * (lw.svd_rank_aligned / 64U) * 50ULL;
        const uint64_t min_v_f16 =
            static_cast<uint64_t>(cfg.num_experts) * cfg.svd_rank * cfg.d_model * sizeof(uint16_t);
        const uint64_t min_vs_f16 =
            static_cast<uint64_t>(lw.svd_rank_aligned) * cfg.d_model * sizeof(uint16_t);
        if ((u_w6_sz > 0ULL && u_w6_sz < min_u_w6) ||
            (exp_svd_alg > 0U && u_w6_sz < min_u_w6) ||
            (v_f16_sz > 0ULL && v_f16_sz < min_v_f16) ||
            (vs_f16_sz > 0ULL && vs_f16_sz < min_vs_f16)) {
            return fail_cleanup(-EBADMSG);
        }
        for (size_t e = 0; e < E; ++e) {
            const uint16_t idx = lw.hot_or_base_idx[e];
            if (lw.tier_kind[e] == STRATA_TIER_HOT_FULL_4B) {
                if (idx >= lw.hot_pool_slots) return fail_cleanup(-EBADMSG);
            } else {
                if (idx >= lw.base_pool_slots) return fail_cleanup(-EBADMSG);
            }
            if (exp_svd_alg > 0U && lw.svd_u_slots > 0U && idx >= lw.svd_u_slots &&
                lw.tier_kind[e] != STRATA_TIER_HOT_FULL_4B) {
                return fail_cleanup(-EBADMSG);
            }
        }
        lw.svd_u_w6_bytes = static_cast<size_t>(u_w6_sz);
        if (u_w6_sz > 0ULL) {
            lw.svd_u_w6_buf = static_cast<uint8_t*>(std::calloc(1U, lw.svd_u_w6_bytes));
            if (!lw.svd_u_w6_buf || !ur.read_bytes(lw.svd_u_w6_buf, lw.svd_u_w6_bytes)) {
                return fail_cleanup(-EBADMSG);
            }
        }
        if (!ur.align_to(16U) || v_f16_sz > ur.remaining()) return fail_cleanup(-EBADMSG);
        if (v_f16_sz > 0ULL) {
            lw.svd_v_f16 = static_cast<uint16_t*>(std::calloc(1U, static_cast<size_t>(v_f16_sz)));
            if (!lw.svd_v_f16 || !ur.read_bytes(lw.svd_v_f16, static_cast<size_t>(v_f16_sz))) {
                return fail_cleanup(-EBADMSG);
            }
        }
        if (vs_f16_sz > ur.remaining()) return fail_cleanup(-EBADMSG);
        if (vs_f16_sz > 0ULL) {
            lw.svd_v_shared_f16 = static_cast<uint16_t*>(std::calloc(1U, static_cast<size_t>(vs_f16_sz)));
            if (!lw.svd_v_shared_f16 || !ur.read_bytes(lw.svd_v_shared_f16, static_cast<size_t>(vs_f16_sz))) {
                return fail_cleanup(-EBADMSG);
            }
        }
    }

    /* Axis A: Verify `engram.ngram` if enabled */
    if (cfg.has_ngram_table) {
        std::string ng_p = bdir + "/engram.ngram";
        struct stat nst;
        if (::stat(ng_p.c_str(), &nst) == 0 && nst.st_size >= static_cast<off_t>(kNgramPageBytes)) {
            std::snprintf(bundle_out->ngram_path, sizeof(bundle_out->ngram_path), "%s", ng_p.c_str());
            bundle_out->ngram_file_bytes = static_cast<uint64_t>(nst.st_size);
        } else {
            cfg.has_ngram_table = false;
        }
    }

    /* Axis D: Read & verify `cold_experts.strata` manifest */
    std::string cpath = bdir + "/cold_experts.strata";
    std::snprintf(bundle_out->cold_strata_path, sizeof(bundle_out->cold_strata_path), "%s", cpath.c_str());
    {
        int cfd = ::open(cpath.c_str(), O_RDONLY);
        if (cfd < 0) return fail_cleanup(-errno);
        struct stat cst;
        if (::fstat(cfd, &cst) != 0 ||
            !read_all_fd(cfd, &bundle_out->cold_header, sizeof(StrataContainerHeader), 0)) {
            ::close(cfd);
            return fail_cleanup(-EIO);
        }
        const StrataContainerHeader& ch = bundle_out->cold_header;
        if (ch.magic != STRATA_MAGIC_U64 ||
            ch.abi_version != STRATA_ABI_VERSION ||
            ch.num_extents > 1048576U ||
            (ch.data_start_offset % STRATA_LBA_ALIGN_BYTES) != 0ULL ||
            ch.total_file_bytes != static_cast<uint64_t>(cst.st_size) ||
            ch.data_start_offset > ch.total_file_bytes ||
            sizeof(StrataContainerHeader) + static_cast<uint64_t>(ch.num_extents) * sizeof(StrataExtentRecord) >
                ch.data_start_offset) {
            ::close(cfd);
            return fail_cleanup(-EBADMSG);
        }
        bundle_out->num_cold_extents = ch.num_extents;
        if (ch.num_extents > 0U) {
            bundle_out->cold_extents = static_cast<StrataExtentRecord*>(
                std::calloc(ch.num_extents, sizeof(StrataExtentRecord)));
            if (!bundle_out->cold_extents ||
                !read_all_fd(cfd, bundle_out->cold_extents,
                             static_cast<size_t>(ch.num_extents) * sizeof(StrataExtentRecord),
                             static_cast<off_t>(sizeof(StrataContainerHeader)))) {
                ::close(cfd);
                return fail_cleanup(-EIO);
            }
            for (uint32_t i = 0; i < ch.num_extents; ++i) {
                const StrataExtentRecord& r = bundle_out->cold_extents[i];
                if ((r.offset_bytes % STRATA_LBA_ALIGN_BYTES) != 0ULL ||
                    (r.aligned_bytes % STRATA_LBA_ALIGN_BYTES) != 0ULL ||
                    r.offset_bytes < ch.data_start_offset ||
                    r.offset_bytes > ch.total_file_bytes ||
                    r.aligned_bytes > ch.total_file_bytes - r.offset_bytes) {
                    ::close(cfd);
                    return fail_cleanup(-EBADMSG);
                }
            }
            cfg.cold_extent_bytes = bundle_out->cold_extents[0].aligned_bytes;
        } else {
            cfg.cold_extent_bytes = STRATA_LBA_ALIGN_BYTES;
        }
        ::close(cfd);
        for (uint32_t l = 0U; l < cfg.num_layers; ++l) {
            for (uint32_t e = 0U; e < cfg.num_experts; ++e) {
                const int32_t ce = bundle_out->layers[l].cold_extent_idx[e];
                if (ce < -1 || (ce >= 0 && static_cast<uint32_t>(ce) >= bundle_out->num_cold_extents)) {
                    return fail_cleanup(-EBADMSG);
                }
            }
        }
    }

    /* Tokenizer / Vocabulary */
    std::vector<std::string> vocab;
    load_vocab_from_dir(bdir, cfg.vocab_size, &vocab);
    bundle_out->vocab_tokens = static_cast<char**>(std::calloc(cfg.vocab_size, sizeof(char*)));
    bundle_out->vocab_scores = static_cast<float*>(std::calloc(cfg.vocab_size, sizeof(float)));
    if (!bundle_out->vocab_tokens || !bundle_out->vocab_scores) return fail_cleanup(-ENOMEM);

    for (uint32_t v = 0; v < cfg.vocab_size; ++v) {
        const std::string& s = (v < vocab.size()) ? vocab[v] : "";
        char* copy = static_cast<char*>(std::calloc(s.size() + 1U, 1U));
        if (!copy) return fail_cleanup(-ENOMEM);
        if (!s.empty()) std::memcpy(copy, s.data(), s.size());
        bundle_out->vocab_tokens[v] = copy;
        bundle_out->vocab_scores[v] = static_cast<float>(cfg.vocab_size - v);
    }
    bundle_out->unk_token_id = 0U;
    bundle_out->bos_token_id = (cfg.vocab_size > 1U) ? 1U : 0U;
    bundle_out->eos_token_id = (cfg.vocab_size > 2U) ? 2U : 0U;
    return 0;
}

void strata_bundle_free(StrataCheckpointBundle* bundle) {
    if (!bundle) return;

    std::free(bundle->tok_embed_f16);
    std::free(bundle->final_norm_f16);
    std::free(bundle->lm_head_rans_blob);
    std::free(bundle->lm_head_f16);
    std::free(bundle->mtp_norm_f16);
    std::free(bundle->mtp_proj_f16);
    std::free(bundle->mtp_head_f16);
    std::free(bundle->mtp_head_rans_blob);
    std::free(bundle->shortlist_vocab_ids);

    if (bundle->layers) {
        for (uint32_t l = 0; l < bundle->cfg.num_layers; ++l) {
            StrataLayerWeights& lw = bundle->layers[l];
            std::free(lw.attn_norm_f16);
            std::free(lw.ffn_norm_f16);
            std::free(lw.q_rans_blob);
            std::free(lw.k_rans_blob);
            std::free(lw.v_rans_blob);
            std::free(lw.o_rans_blob);
            std::free(lw.router_w_f16);
            std::free(lw.is_uma_hot);
            std::free(lw.tier_kind);
            std::free(lw.hot_or_base_idx);
            std::free(lw.cold_extent_idx);
            std::free(lw.default_slot_map);
            std::free(lw.hot_pool_buf);
            std::free(lw.base_pool_buf);
            std::free(lw.svd_u_w6_buf);
            std::free(lw.svd_v_f16);
            std::free(lw.svd_v_shared_f16);
        }
        std::free(bundle->layers);
    }

    std::free(bundle->cold_extents);

    if (bundle->vocab_tokens) {
        for (uint32_t v = 0; v < bundle->cfg.vocab_size; ++v) {
            std::free(bundle->vocab_tokens[v]);
        }
        std::free(bundle->vocab_tokens);
    }
    std::free(bundle->vocab_scores);

    std::memset(bundle, 0, sizeof(*bundle));
}

int strata_tokenize_prompt(
    const StrataCheckpointBundle* bundle,
    const char* prompt_utf8,
    bool add_bos,
    uint32_t* token_ids_out,
    uint32_t max_tokens,
    uint32_t* num_tokens_out
) {
    if (!bundle || !prompt_utf8 || !token_ids_out || !num_tokens_out || max_tokens == 0U) {
        return -EINVAL;
    }

    uint32_t count = 0U;
    if (add_bos) {
        token_ids_out[count++] = bundle->bos_token_id;
    }

    /* Prefix-prepend a leading space if non-empty and not starting with space, matching SentencePiece/GPT subwords */
    std::string text(prompt_utf8);
    if (!text.empty() && text[0] != ' ' && text[0] != '\n') {
        text = " " + text;
    }

    const uint32_t V = bundle->cfg.vocab_size;
    size_t pos = 0U;
    while (pos < text.size() && count < max_tokens) {
        size_t best_len = 0U;
        uint32_t best_id = bundle->unk_token_id;

        /* Skip special control tokens 0..2 ("<unk>", "<s>", "</s>") during plain text matching */
        for (uint32_t v = 3U; v < V; ++v) {
            const char* piece = bundle->vocab_tokens[v];
            if (!piece || piece[0] == '\0') continue;
            size_t plen = std::strlen(piece);
            if (plen > best_len && pos + plen <= text.size() &&
                std::memcmp(text.data() + pos, piece, plen) == 0) {
                best_len = plen;
                best_id  = v;
            }
        }

        if (best_len == 0U) {
            best_len = 1U;
            best_id  = bundle->unk_token_id;
        }
        token_ids_out[count++] = best_id;
        pos += best_len;
    }

    *num_tokens_out = count;
    return 0;
}

const char* strata_detokenize_token(
    const StrataCheckpointBundle* bundle,
    uint32_t token_id
) {
    if (!bundle || !bundle->vocab_tokens || token_id >= bundle->cfg.vocab_size) {
        return "";
    }
    if (token_id == bundle->bos_token_id || token_id == bundle->eos_token_id) {
        return "";
    }
    return bundle->vocab_tokens[token_id] ? bundle->vocab_tokens[token_id] : "";
}

}  // extern "C"

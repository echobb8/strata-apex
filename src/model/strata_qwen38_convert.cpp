/*
 * STRATA-M5 v2.0 — Qwen3.8-27B Multi-Shard Streaming Converter, Zero-Copy Loader & BPE Tokenizer
 * File: src/model/strata_qwen38_convert.cpp
 *
 * Implements:
 *   1. Multi-shard Hugging Face `model.safetensors.index.json` -> `weights.strata` streaming
 *      converter (`F_NOCACHE` 16 MiB chunked I/O, < 200 MiB Peak RSS, 16 KiB VM-page alignment,
 *      hardware NEON BF16 -> IEEE-FP16 scales/biases, lossless BF16 -> FP32 norms/conv1d/gates).
 *   2. Zero-copy read-only `mmap` `.strata` bundle loader (`strata_qwen38_load_bundle`).
 *   3. Exact Hugging Face `tokenizer.json` Qwen2/Qwen3.8 Byte-Level BPE Tokenizer (`248,044` base
 *      vocab + `247,587` merges + `33` added/special tokens), `O(N log N)` min-heap BPE merger,
 *      ChatML `<think>` / `--no-think` formatter, and stateful UTF-8 streaming decoder.
 */

#include "../../include/strata_qwen38.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

/* ============================================================================
 * 1. Alignment, Endian, BF16/FP16 Conversion & Filesystem Helpers
 * ============================================================================ */

static inline uint64_t align_up_u64(uint64_t v, uint64_t align) {
    return (v + (align - 1ULL)) & ~(align - 1ULL);
}

static inline float bf16_to_f32_scalar(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

static inline float fp16_to_f32_scalar(uint16_t h) {
#if defined(__aarch64__)
    float16x4_t v = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(v), 0);
#else
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
#endif
}

static inline uint16_t f32_to_fp16_scalar(float f) {
#if defined(__aarch64__)
    float16x4_t h = vcvt_f16_f32(vdupq_n_f32(f));
    return vget_lane_u16(vreinterpret_u16_f16(h), 0);
#else
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(uint32_t));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x00800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_bit = 1u << (shift - 1);
        uint32_t rem_mask = half_bit - 1u;
        uint32_t res = mant >> shift;
        if ((mant & half_bit) && ((mant & rem_mask) || (res & 1u))) ++res;
        return static_cast<uint16_t>(sign | res);
    } else if (exp == 0xFF - 127 + 15) {
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    } else if (exp > 30) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    uint32_t res = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    if ((mant & 0x1000u) && ((mant & 0x0FFFu) || (res & 1u))) ++res;
    return static_cast<uint16_t>(sign | res);
#endif
}

static void convert_bf16_to_fp16_vec(const uint16_t* src_bf16, uint16_t* dst_fp16, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bf = vld1q_u16(src_bf16 + i);
        uint32x4_t lo_u32 = vshll_n_u16(vget_low_u16(bf), 16);
        uint32x4_t hi_u32 = vshll_n_u16(vget_high_u16(bf), 16);
        float16x4_t lo_f16 = vcvt_f16_f32(vreinterpretq_f32_u32(lo_u32));
        float16x4_t hi_f16 = vcvt_f16_f32(vreinterpretq_f32_u32(hi_u32));
        vst1q_u16(dst_fp16 + i, vreinterpretq_u16_f16(vcombine_f16(lo_f16, hi_f16)));
    }
#endif
    for (; i < n; ++i) {
        dst_fp16[i] = f32_to_fp16_scalar(bf16_to_f32_scalar(src_bf16[i]));
    }
}

static double current_peak_rss_gib() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#if defined(__APPLE__)
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
#else
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
#endif
}

static bool pread_exact(int fd, void* buf, size_t count, uint64_t offset) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t r = pread(fd, p + done, count - done, static_cast<off_t>(offset + done));
        if (r <= 0) return false;
        done += static_cast<size_t>(r);
    }
    return true;
}

static bool pwrite_exact(int fd, const void* buf, size_t count, uint64_t offset) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t w = pwrite(fd, p + done, count - done, static_cast<off_t>(offset + done));
        if (w <= 0) return false;
        done += static_cast<size_t>(w);
    }
    return true;
}

static bool mkdirs(const std::string& dir) {
    if (dir.empty()) return false;
    std::string cur;
    if (dir[0] == '/') cur = "/";
    size_t i = (dir[0] == '/') ? 1 : 0;
    while (i <= dir.size()) {
        size_t slash = dir.find('/', i);
        std::string part = (slash == std::string::npos) ? dir.substr(i) : dir.substr(i, slash - i);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') cur.push_back('/');
            cur += part;
            struct stat st;
            if (stat(cur.c_str(), &st) != 0) {
                if (mkdir(cur.c_str(), 0755) != 0) return false;
            } else if (!S_ISDIR(st.st_mode)) {
                return false;
            }
        }
        if (slash == std::string::npos) break;
        i = slash + 1;
    }
    return true;
}

static bool read_whole_file(const std::string& path, std::string& out) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        return false;
    }
    out.resize(static_cast<size_t>(st.st_size));
    bool ok = pread_exact(fd, out.data(), out.size(), 0);
    close(fd);
    return ok;
}

static bool copy_file_if_exists(const std::string& src, const std::string& dst, bool required) {
    int in_fd = open(src.c_str(), O_RDONLY);
    if (in_fd < 0) return !required;
    struct stat st;
    if (fstat(in_fd, &st) != 0) {
        close(in_fd);
        return false;
    }
    int out_fd = open(dst.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (out_fd < 0) {
        close(in_fd);
        return false;
    }
    std::vector<uint8_t> buf(1024 * 1024);
    uint64_t off = 0;
    uint64_t total = static_cast<uint64_t>(st.st_size);
    bool ok = true;
    while (off < total) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(buf.size(), total - off));
        if (!pread_exact(in_fd, buf.data(), chunk, off) ||
            !pwrite_exact(out_fd, buf.data(), chunk, off)) {
            ok = false;
            break;
        }
        off += chunk;
    }
    close(out_fd);
    close(in_fd);
    return ok;
}

/* ============================================================================
 * 2. UTF-8 Encoding/Decoding & Lightweight RFC-8259 JSON Scanner + Small DOM
 * ============================================================================ */

static inline void append_utf8_codepoint(std::string& out, uint32_t cp) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

static inline uint32_t decode_utf8_one(const uint8_t* s, size_t len, size_t* adv) {
    if (len == 0) {
        *adv = 0;
        return 0;
    }
    uint8_t b0 = s[0];
    if (b0 < 0x80u) {
        *adv = 1;
        return b0;
    }
    if ((b0 & 0xE0u) == 0xC0u && len >= 2 && (s[1] & 0xC0u) == 0x80u) {
        uint32_t cp = ((static_cast<uint32_t>(b0 & 0x1Fu)) << 6) |
                      static_cast<uint32_t>(s[1] & 0x3Fu);
        if (cp >= 0x80u) {
            *adv = 2;
            return cp;
        }
    } else if ((b0 & 0xF0u) == 0xE0u && len >= 3 &&
               (s[1] & 0xC0u) == 0x80u && (s[2] & 0xC0u) == 0x80u) {
        uint32_t cp = ((static_cast<uint32_t>(b0 & 0x0Fu)) << 12) |
                      ((static_cast<uint32_t>(s[1] & 0x3Fu)) << 6) |
                      static_cast<uint32_t>(s[2] & 0x3Fu);
        if (cp >= 0x800u) {
            *adv = 3;
            return cp;
        }
    } else if ((b0 & 0xF8u) == 0xF0u && len >= 4 &&
               (s[1] & 0xC0u) == 0x80u && (s[2] & 0xC0u) == 0x80u && (s[3] & 0xC0u) == 0x80u) {
        uint32_t cp = ((static_cast<uint32_t>(b0 & 0x07u)) << 18) |
                      ((static_cast<uint32_t>(s[1] & 0x3Fu)) << 12) |
                      ((static_cast<uint32_t>(s[2] & 0x3Fu)) << 6) |
                      static_cast<uint32_t>(s[3] & 0x3Fu);
        if (cp >= 0x10000u && cp <= 0x10FFFFu) {
            *adv = 4;
            return cp;
        }
    }
    *adv = 1;
    return 0xFFFDu;
}

struct JsonCursor {
    const char* s;
    size_t      n;
    size_t      p;

    JsonCursor(const char* str, size_t len) : s(str), n(len), p(0) {}

    inline void skip_ws() {
        while (p < n) {
            unsigned char c = static_cast<unsigned char>(s[p]);
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++p;
            } else {
                break;
            }
        }
    }

    inline bool consume(char expected) {
        skip_ws();
        if (p < n && s[p] == expected) {
            ++p;
            return true;
        }
        return false;
    }

    inline char peek() {
        skip_ws();
        return (p < n) ? s[p] : '\0';
    }

    static inline int hex_val(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    }

    bool parse_hex4(uint32_t& out) {
        if (p + 4 > n) return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            int h = hex_val(s[p + i]);
            if (h < 0) return false;
            v = (v << 4) | static_cast<uint32_t>(h);
        }
        p += 4;
        out = v;
        return true;
    }

    bool parse_string(std::string& out) {
        skip_ws();
        if (p >= n || s[p] != '"') return false;
        ++p;
        out.clear();
        while (p < n) {
            const char* start = s + p;
            size_t run = 0;
            while (p + run < n) {
                char c = s[p + run];
                if (c == '"' || c == '\\') break;
                ++run;
            }
            if (run > 0) {
                out.append(start, run);
                p += run;
            }
            if (p >= n) return false;
            char c = s[p++];
            if (c == '"') return true;
            if (p >= n) return false;
            char esc = s[p++];
            switch (esc) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t u1 = 0;
                    if (!parse_hex4(u1)) return false;
                    if (u1 >= 0xD800u && u1 <= 0xDBFFu) {
                        if (p + 2 <= n && s[p] == '\\' && s[p + 1] == 'u') {
                            p += 2;
                            uint32_t u2 = 0;
                            if (!parse_hex4(u2)) return false;
                            if (u2 >= 0xDC00u && u2 <= 0xDFFFu) {
                                uint32_t cp = 0x10000u + (((u1 - 0xD800u) << 10) | (u2 - 0xDC00u));
                                append_utf8_codepoint(out, cp);
                                break;
                            }
                        }
                    }
                    append_utf8_codepoint(out, u1);
                    break;
                }
                default:
                    return false;
            }
        }
        return false;
    }

    bool parse_number(double& out_d, int64_t& out_i) {
        skip_ws();
        if (p >= n) return false;
        const char* start = s + p;
        char* end_d = nullptr;
        out_d = std::strtod(start, &end_d);
        if (end_d == start) return false;
        char* end_i = nullptr;
        out_i = std::strtoll(start, &end_i, 10);
        if (end_i != end_d) out_i = static_cast<int64_t>(out_d);
        p = static_cast<size_t>(end_d - s);
        return true;
    }

    bool parse_uint64(uint64_t& out_u) {
        skip_ws();
        if (p >= n) return false;
        const char* start = s + p;
        char* end_u = nullptr;
        unsigned long long v = std::strtoull(start, &end_u, 10);
        if (end_u == start) return false;
        out_u = static_cast<uint64_t>(v);
        p = static_cast<size_t>(end_u - s);
        return true;
    }

    bool parse_bool(bool& out_b) {
        skip_ws();
        if (p + 4 <= n && std::memcmp(s + p, "true", 4) == 0) {
            p += 4;
            out_b = true;
            return true;
        }
        if (p + 5 <= n && std::memcmp(s + p, "false", 5) == 0) {
            p += 5;
            out_b = false;
            return true;
        }
        return false;
    }

    bool skip_value(int depth = 0) {
        if (depth > 128) return false;
        skip_ws();
        if (p >= n) return false;
        char c = s[p];
        if (c == '"') {
            std::string tmp;
            return parse_string(tmp);
        }
        if (c == '{') {
            ++p;
            if (consume('}')) return true;
            std::string key;
            while (true) {
                if (!parse_string(key)) return false;
                if (!consume(':')) return false;
                if (!skip_value(depth + 1)) return false;
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == '[') {
            ++p;
            if (consume(']')) return true;
            while (true) {
                if (!skip_value(depth + 1)) return false;
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        if (c == 't' || c == 'f') {
            bool b = false;
            return parse_bool(b);
        }
        if (c == 'n') {
            if (p + 4 <= n && std::memcmp(s + p, "null", 4) == 0) {
                p += 4;
                return true;
            }
            return false;
        }
        double d = 0.0;
        int64_t iv = 0;
        return parse_number(d, iv);
    }
};

struct SmallJson {
    enum Type { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } type = J_NULL;
    bool b_val = false;
    double d_val = 0.0;
    int64_t i_val = 0;
    std::string s_val;
    std::vector<SmallJson> arr;
    std::map<std::string, SmallJson> obj;

    const SmallJson* get(const std::string& k) const {
        if (type != J_OBJ) return nullptr;
        auto it = obj.find(k);
        return (it == obj.end()) ? nullptr : &it->second;
    }
    int64_t get_i64(const std::string& k, int64_t def) const {
        const SmallJson* v = get(k);
        return (v && v->type == J_NUM) ? v->i_val : def;
    }
    double get_f64(const std::string& k, double def) const {
        const SmallJson* v = get(k);
        return (v && v->type == J_NUM) ? v->d_val : def;
    }
    bool get_bool(const std::string& k, bool def) const {
        const SmallJson* v = get(k);
        return (v && v->type == J_BOOL) ? v->b_val : def;
    }
    std::string get_str(const std::string& k, const std::string& def = "") const {
        const SmallJson* v = get(k);
        return (v && v->type == J_STR) ? v->s_val : def;
    }
};

static bool parse_small_json_rec(JsonCursor& cur, SmallJson& out, int depth = 0) {
    if (depth > 128) return false;
    char c = cur.peek();
    if (c == '\0') return false;
    if (c == '"') {
        out.type = SmallJson::J_STR;
        return cur.parse_string(out.s_val);
    }
    if (c == '{') {
        cur.consume('{');
        out.type = SmallJson::J_OBJ;
        if (cur.consume('}')) return true;
        while (true) {
            std::string k;
            if (!cur.parse_string(k)) return false;
            if (!cur.consume(':')) return false;
            SmallJson child;
            if (!parse_small_json_rec(cur, child, depth + 1)) return false;
            out.obj.emplace(std::move(k), std::move(child));
            if (cur.consume('}')) return true;
            if (!cur.consume(',')) return false;
        }
    }
    if (c == '[') {
        cur.consume('[');
        out.type = SmallJson::J_ARR;
        if (cur.consume(']')) return true;
        while (true) {
            SmallJson elem;
            if (!parse_small_json_rec(cur, elem, depth + 1)) return false;
            out.arr.push_back(std::move(elem));
            if (cur.consume(']')) return true;
            if (!cur.consume(',')) return false;
        }
    }
    if (c == 't' || c == 'f') {
        out.type = SmallJson::J_BOOL;
        return cur.parse_bool(out.b_val);
    }
    if (c == 'n') {
        if (cur.p + 4 <= cur.n && std::memcmp(cur.s + cur.p, "null", 4) == 0) {
            cur.p += 4;
            out.type = SmallJson::J_NULL;
            return true;
        }
        return false;
    }
    out.type = SmallJson::J_NUM;
    return cur.parse_number(out.d_val, out.i_val);
}

static bool parse_small_json(const std::string& text, SmallJson& out) {
    JsonCursor cur(text.data(), text.size());
    return parse_small_json_rec(cur, out);
}

/* ============================================================================
 * 3. Streaming Safetensors Header Directory & Multi-Shard Converter
 * ============================================================================ */

enum SafetensorDtype {
    ST_UNKNOWN = 0,
    ST_BF16    = 1,
    ST_F16     = 2,
    ST_F32     = 3,
    ST_U32     = 4
};

struct SafetensorEntry {
    uint32_t              shard_idx = 0;
    SafetensorDtype       dtype     = ST_UNKNOWN;
    std::vector<uint64_t> shape;
    uint64_t              abs_offset = 0;
    uint64_t              byte_len   = 0;
};

static SafetensorDtype parse_st_dtype(const std::string& s) {
    if (s == "BF16") return ST_BF16;
    if (s == "F16")  return ST_F16;
    if (s == "F32")  return ST_F32;
    if (s == "U32")  return ST_U32;
    return ST_UNKNOWN;
}

static bool parse_safetensors_header(
    int fd,
    uint32_t shard_idx,
    uint64_t shard_file_bytes,
    std::unordered_map<std::string, SafetensorEntry>& index_out,
    uint32_t& vision_skipped_inout
) {
    uint8_t len_buf[8];
    if (!pread_exact(fd, len_buf, 8, 0)) return false;
    uint64_t hdr_len = 0;
    for (int i = 0; i < 8; ++i) {
        hdr_len |= static_cast<uint64_t>(len_buf[i]) << (8 * i);
    }
    if (hdr_len == 0 || hdr_len > 128ULL * 1024ULL * 1024ULL || 8ULL + hdr_len > shard_file_bytes) {
        return false;
    }
    std::string hdr_json(static_cast<size_t>(hdr_len), '\0');
    if (!pread_exact(fd, hdr_json.data(), static_cast<size_t>(hdr_len), 8)) return false;

    uint64_t data_base = 8ULL + hdr_len;
    JsonCursor cur(hdr_json.data(), hdr_json.size());
    if (!cur.consume('{')) return false;
    if (cur.consume('}')) return true;

    std::string key, prop, dt_str;
    while (true) {
        if (!cur.parse_string(key)) return false;
        if (!cur.consume(':')) return false;
        if (key == "__metadata__") {
            if (!cur.skip_value()) return false;
        } else if (key.rfind("vision_tower.", 0) == 0 || key.rfind("visual.", 0) == 0) {
            if (!cur.skip_value()) return false;
            ++vision_skipped_inout;
        } else {
            if (!cur.consume('{')) return false;
            SafetensorEntry ent;
            ent.shard_idx = shard_idx;
            uint64_t off0 = 0, off1 = 0;
            if (!cur.consume('}')) {
                while (true) {
                    if (!cur.parse_string(prop)) return false;
                    if (!cur.consume(':')) return false;
                    if (prop == "dtype") {
                        if (!cur.parse_string(dt_str)) return false;
                        ent.dtype = parse_st_dtype(dt_str);
                    } else if (prop == "shape") {
                        if (!cur.consume('[')) return false;
                        if (!cur.consume(']')) {
                            while (true) {
                                uint64_t dim = 0;
                                if (!cur.parse_uint64(dim)) return false;
                                ent.shape.push_back(dim);
                                if (cur.consume(']')) break;
                                if (!cur.consume(',')) return false;
                            }
                        }
                    } else if (prop == "data_offsets") {
                        if (!cur.consume('[')) return false;
                        if (!cur.parse_uint64(off0)) return false;
                        if (!cur.consume(',')) return false;
                        if (!cur.parse_uint64(off1)) return false;
                        if (!cur.consume(']')) return false;
                    } else {
                        if (!cur.skip_value()) return false;
                    }
                    if (cur.consume('}')) break;
                    if (!cur.consume(',')) return false;
                }
            }
            if (off1 < off0 || data_base + off1 > shard_file_bytes) return false;
            ent.abs_offset = data_base + off0;
            ent.byte_len   = off1 - off0;

            /* Canonicalize `language_model.` prefix so both VLM & text-only checkpoints match */
            std::string canon = key;
            if (canon.rfind("language_model.", 0) == 0) {
                canon.erase(0, std::strlen("language_model."));
            }
            index_out.emplace(std::move(canon), std::move(ent));
        }
        if (cur.consume('}')) break;
        if (!cur.consume(',')) return false;
    }
    return true;
}

struct ConvertStreamCtx {
    std::vector<int> shard_fds;
    std::unordered_map<std::string, SafetensorEntry> tensors;
    int              out_fd = -1;
    uint64_t         write_off = 0;
    uint32_t         converted_tensors = 0;
    std::vector<uint8_t> io_in;
    std::vector<uint8_t> io_out;
    std::vector<uint8_t> zero_buf;

    ConvertStreamCtx() {
        io_in.resize(16 * 1024 * 1024);
        io_out.resize(16 * 1024 * 1024);
        zero_buf.assign(65536, 0);
    }

    bool pad_to(uint64_t align_bytes) {
        uint64_t next = align_up_u64(write_off, align_bytes);
        while (write_off < next) {
            size_t chunk = static_cast<size_t>(std::min<uint64_t>(zero_buf.size(), next - write_off));
            if (!pwrite_exact(out_fd, zero_buf.data(), chunk, write_off)) return false;
            write_off += chunk;
        }
        return true;
    }

    const SafetensorEntry* find_tensor(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            std::fprintf(stderr, "[qwen38-convert] ERROR: missing tensor '%s'\n", name.c_str());
            return nullptr;
        }
        return &it->second;
    }

    bool write_f32_slice(
        const std::string& name,
        uint32_t expected_logical_elems,
        uint32_t padded_elems,
        uint32_t shape0,
        uint32_t shape1,
        StrataQwen38TensorSlice& desc_out
    ) {
        const SafetensorEntry* ent = find_tensor(name);
        if (!ent) return false;
        uint64_t actual_elems = 1;
        for (uint64_t d : ent->shape) actual_elems *= d;
        if (actual_elems != expected_logical_elems || padded_elems < expected_logical_elems) {
            std::fprintf(stderr, "[qwen38-convert] ERROR: shape mismatch on '%s' (got %llu, expected %u)\n",
                         name.c_str(), static_cast<unsigned long long>(actual_elems), expected_logical_elems);
            return false;
        }
        if (!pad_to(STRATA_QWEN38_TENSOR_ALIGN)) return false;

        std::vector<uint8_t> raw(static_cast<size_t>(ent->byte_len));
        if (!pread_exact(shard_fds[ent->shard_idx], raw.data(), raw.size(), ent->abs_offset)) {
            return false;
        }
        std::vector<float> f32(padded_elems, 0.0f);
        if (ent->dtype == ST_BF16) {
            if (ent->byte_len != static_cast<uint64_t>(expected_logical_elems) * 2ULL) return false;
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(raw.data());
            for (uint32_t i = 0; i < expected_logical_elems; ++i) {
                f32[i] = bf16_to_f32_scalar(s16[i]);
            }
        } else if (ent->dtype == ST_F16) {
            if (ent->byte_len != static_cast<uint64_t>(expected_logical_elems) * 2ULL) return false;
            const uint16_t* s16 = reinterpret_cast<const uint16_t*>(raw.data());
            for (uint32_t i = 0; i < expected_logical_elems; ++i) {
                f32[i] = fp16_to_f32_scalar(s16[i]);
            }
        } else if (ent->dtype == ST_F32) {
            if (ent->byte_len != static_cast<uint64_t>(expected_logical_elems) * 4ULL) return false;
            std::memcpy(f32.data(), raw.data(), expected_logical_elems * sizeof(float));
        } else {
            std::fprintf(stderr, "[qwen38-convert] ERROR: unsupported dtype on '%s'\n", name.c_str());
            return false;
        }

        uint64_t bytes = static_cast<uint64_t>(padded_elems) * sizeof(float);
        if (!pwrite_exact(out_fd, f32.data(), static_cast<size_t>(bytes), write_off)) return false;

        desc_out.offset     = write_off;
        desc_out.bytes      = bytes;
        desc_out.elem_count = padded_elems;
        desc_out.dtype      = STRATA_QWEN38_DTYPE_F32;
        desc_out.shape0     = shape0;
        desc_out.shape1     = shape1;

        write_off += bytes;
        ++converted_tensors;
        return true;
    }

    bool stream_fp16_scales_or_biases(
        const SafetensorEntry& ent,
        uint64_t logical_elems,
        uint64_t padded_elems,
        uint64_t& out_off,
        uint64_t& out_bytes
    ) {
        if (!pad_to(STRATA_QWEN38_TENSOR_ALIGN)) return false;
        out_off   = write_off;
        out_bytes = padded_elems * sizeof(uint16_t);

        if (ent.dtype == ST_F16) {
            if (ent.byte_len != logical_elems * 2ULL) return false;
            uint64_t rem = ent.byte_len;
            uint64_t r_off = ent.abs_offset;
            while (rem > 0) {
                size_t chunk = static_cast<size_t>(std::min<uint64_t>(io_in.size(), rem));
                if (!pread_exact(shard_fds[ent.shard_idx], io_in.data(), chunk, r_off)) return false;
                if (!pwrite_exact(out_fd, io_in.data(), chunk, write_off)) return false;
                r_off     += chunk;
                write_off += chunk;
                rem       -= chunk;
            }
        } else if (ent.dtype == ST_BF16) {
            if (ent.byte_len != logical_elems * 2ULL) return false;
            uint64_t rem_elems = logical_elems;
            uint64_t r_off = ent.abs_offset;
            size_t max_elems = io_in.size() / sizeof(uint16_t);
            while (rem_elems > 0) {
                size_t n = static_cast<size_t>(std::min<uint64_t>(max_elems, rem_elems));
                size_t b = n * sizeof(uint16_t);
                if (!pread_exact(shard_fds[ent.shard_idx], io_in.data(), b, r_off)) return false;
                convert_bf16_to_fp16_vec(
                    reinterpret_cast<const uint16_t*>(io_in.data()),
                    reinterpret_cast<uint16_t*>(io_out.data()),
                    n
                );
                if (!pwrite_exact(out_fd, io_out.data(), b, write_off)) return false;
                r_off     += b;
                write_off += b;
                rem_elems -= n;
            }
        } else {
            return false;
        }

        uint64_t pad_bytes = (padded_elems - logical_elems) * sizeof(uint16_t);
        while (pad_bytes > 0) {
            size_t chunk = static_cast<size_t>(std::min<uint64_t>(zero_buf.size(), pad_bytes));
            if (!pwrite_exact(out_fd, zero_buf.data(), chunk, write_off)) return false;
            write_off += chunk;
            pad_bytes -= chunk;
        }
        return true;
    }

    bool write_quant_matrix(
        const std::string& prefix,
        uint32_t expected_rows,
        uint32_t expected_cols,
        StrataQwen38QuantMatrix& qm_out
    ) {
        const SafetensorEntry* w_ent = find_tensor(prefix + ".weight");
        const SafetensorEntry* s_ent = find_tensor(prefix + ".scales");
        const SafetensorEntry* b_ent = find_tensor(prefix + ".biases");
        if (!w_ent || !s_ent || !b_ent) return false;

        if (w_ent->dtype != ST_U32 || w_ent->shape.size() != 2 ||
            s_ent->shape.size() != 2 || b_ent->shape.size() != 2) {
            std::fprintf(stderr, "[qwen38-convert] ERROR: invalid quant tensor layout on '%s'\n", prefix.c_str());
            return false;
        }
        uint32_t u32_cols = expected_cols / 8u;
        uint32_t grp_cols = expected_cols / STRATA_QWEN38_QUANT_GROUP_SIZE;
        if (w_ent->shape[0] != expected_rows || w_ent->shape[1] != u32_cols ||
            s_ent->shape[0] != expected_rows || s_ent->shape[1] != grp_cols ||
            b_ent->shape[0] != expected_rows || b_ent->shape[1] != grp_cols) {
            std::fprintf(stderr, "[qwen38-convert] ERROR: quant dim mismatch on '%s' ([%llu,%llu] vs [%u,%u])\n",
                         prefix.c_str(),
                         static_cast<unsigned long long>(w_ent->shape[0]),
                         static_cast<unsigned long long>(w_ent->shape[1]),
                         expected_rows, u32_cols);
            return false;
        }

        uint32_t rows_padded = static_cast<uint32_t>(align_up_u64(expected_rows, 64ULL));

        /* 1. Packed U32 nibbles (`[rows_padded, cols / 8]`) */
        if (!pad_to(STRATA_QWEN38_TENSOR_ALIGN)) return false;
        uint64_t w_logical_bytes = static_cast<uint64_t>(expected_rows) * u32_cols * sizeof(uint32_t);
        uint64_t w_padded_bytes  = static_cast<uint64_t>(rows_padded) * u32_cols * sizeof(uint32_t);
        if (w_ent->byte_len != w_logical_bytes) return false;

        qm_out.weight_off   = write_off;
        qm_out.weight_bytes = w_padded_bytes;

        uint64_t rem = w_logical_bytes;
        uint64_t r_off = w_ent->abs_offset;
        while (rem > 0) {
            size_t chunk = static_cast<size_t>(std::min<uint64_t>(io_in.size(), rem));
            if (!pread_exact(shard_fds[w_ent->shard_idx], io_in.data(), chunk, r_off)) return false;
            if (!pwrite_exact(out_fd, io_in.data(), chunk, write_off)) return false;
            r_off     += chunk;
            write_off += chunk;
            rem       -= chunk;
        }
        uint64_t w_pad_rem = w_padded_bytes - w_logical_bytes;
        while (w_pad_rem > 0) {
            size_t chunk = static_cast<size_t>(std::min<uint64_t>(zero_buf.size(), w_pad_rem));
            if (!pwrite_exact(out_fd, zero_buf.data(), chunk, write_off)) return false;
            write_off += chunk;
            w_pad_rem -= chunk;
        }

        /* 2. IEEE-FP16 scales & biases (`[rows_padded, cols / 64]`) */
        uint64_t sb_logical_elems = static_cast<uint64_t>(expected_rows) * grp_cols;
        uint64_t sb_padded_elems  = static_cast<uint64_t>(rows_padded) * grp_cols;
        if (!stream_fp16_scales_or_biases(*s_ent, sb_logical_elems, sb_padded_elems,
                                          qm_out.scales_off, qm_out.scales_bytes)) {
            return false;
        }
        if (!stream_fp16_scales_or_biases(*b_ent, sb_logical_elems, sb_padded_elems,
                                          qm_out.biases_off, qm_out.biases_bytes)) {
            return false;
        }

        qm_out.rows_logical = expected_rows;
        qm_out.rows_padded  = rows_padded;
        qm_out.cols         = expected_cols;
        qm_out.group_size   = STRATA_QWEN38_QUANT_GROUP_SIZE;

        converted_tensors += 3;
        return true;
    }
};

static inline uint64_t qm_active_bytes(const StrataQwen38QuantMatrix& qm) {
    uint64_t row_bytes = static_cast<uint64_t>(qm.cols / 8u) * sizeof(uint32_t) +
                         2ULL * static_cast<uint64_t>(qm.cols / 64u) * sizeof(uint16_t);
    return static_cast<uint64_t>(qm.rows_logical) * row_bytes;
}

/* ============================================================================
 * 4. Qwen2 / Qwen3.8 Byte-Level BPE Tokenizer (`tokenizer.json`)
 * ============================================================================ */

/* Unicode General Category Helpers for exact Qwen2 Pre-Tokenizer Regex:
 * `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
 */
static inline bool is_unicode_whitespace(uint32_t cp) {
    if (cp <= 0x20u) {
        return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f';
    }
    return cp == 0x0085u || cp == 0x00A0u || cp == 0x1680u ||
           (cp >= 0x2000u && cp <= 0x200Au) ||
           cp == 0x2028u || cp == 0x2029u || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u;
}

static inline bool is_unicode_number(uint32_t cp) {
    if (cp < 0x80u) return cp >= '0' && cp <= '9';
    if (cp == 0x00B2u || cp == 0x00B3u || cp == 0x00B9u || (cp >= 0x00BCu && cp <= 0x00BEu)) return true;
    if ((cp >= 0x0660u && cp <= 0x0669u) || (cp >= 0x06F0u && cp <= 0x06F9u)) return true;
    if ((cp >= 0x07C0u && cp <= 0x07C9u) || (cp >= 0x0966u && cp <= 0x096Fu)) return true;
    if ((cp >= 0x09E6u && cp <= 0x09EFu) || (cp >= 0x0A66u && cp <= 0x0A6Fu)) return true;
    if ((cp >= 0x0AE6u && cp <= 0x0AEFu) || (cp >= 0x0B66u && cp <= 0x0B6Fu)) return true;
    if ((cp >= 0x0BE6u && cp <= 0x0BEFu) || (cp >= 0x0C66u && cp <= 0x0C6Fu)) return true;
    if ((cp >= 0x0CE6u && cp <= 0x0CEFu) || (cp >= 0x0D66u && cp <= 0x0D6Fu)) return true;
    if ((cp >= 0x0E50u && cp <= 0x0E59u) || (cp >= 0x0ED0u && cp <= 0x0ED9u)) return true;
    if ((cp >= 0x0F20u && cp <= 0x0F33u) || (cp >= 0x1040u && cp <= 0x1049u)) return true;
    if ((cp >= 0x17E0u && cp <= 0x17E9u) || (cp >= 0x1810u && cp <= 0x1819u)) return true;
    if (cp == 0x2070u || (cp >= 0x2074u && cp <= 0x2079u) || (cp >= 0x2080u && cp <= 0x2089u)) return true;
    if (cp >= 0x2150u && cp <= 0x218Bu) return true;
    if ((cp >= 0x2460u && cp <= 0x249Bu) || cp == 0x24EAu || (cp >= 0x24EB && cp <= 0x24FFu)) return true;
    if ((cp >= 0x2776u && cp <= 0x2793u) || cp == 0x3007u || (cp >= 0x3021u && cp <= 0x3029u)) return true;
    if ((cp >= 0x3038u && cp <= 0x303Au) || (cp >= 0x3220u && cp <= 0x3229u)) return true;
    if ((cp >= 0xFF10u && cp <= 0xFF19u) || (cp >= 0x1D7CEu && cp <= 0x1D7FFu)) return true;
    return false;
}

static inline bool is_unicode_letter(uint32_t cp) {
    if (cp < 0x80u) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    if (is_unicode_whitespace(cp) || is_unicode_number(cp)) return false;
    /* Latin-1 Supplement letters */
    if (cp == 0x00AAu || cp == 0x00B5u || cp == 0x00BAu) return true;
    if (cp >= 0x00C0u && cp <= 0x00D6u) return true;
    if (cp >= 0x00D8u && cp <= 0x00F6u) return true;
    if (cp >= 0x00F8u && cp <= 0x02C1u) return true;
    if ((cp >= 0x02C6u && cp <= 0x02D1u) || (cp >= 0x02E0u && cp <= 0x02E4u) ||
        cp == 0x02ECu || cp == 0x02EEu) return true;
    /* Greek, Cyrillic, Armenian, Hebrew, Arabic, Syriac, Indic, Thai, Lao, Tibetan, Georgian, Hangul Jamo */
    if (cp >= 0x0370u && cp <= 0x0374u) return true;
    if (cp >= 0x0376u && cp <= 0x0377u) return true;
    if (cp >= 0x037Au && cp <= 0x037Du) return true;
    if (cp == 0x037Fu || cp == 0x0386u || (cp >= 0x0388u && cp <= 0x038Au) ||
        cp == 0x038Cu || (cp >= 0x038Eu && cp <= 0x03A1u) || (cp >= 0x03A3u && cp <= 0x03F5u) ||
        (cp >= 0x03F7u && cp <= 0x0481u) || (cp >= 0x048Au && cp <= 0x052Fu)) return true;
    if ((cp >= 0x0531u && cp <= 0x0556u) || cp == 0x0559u || (cp >= 0x0560u && cp <= 0x0588u)) return true;
    if ((cp >= 0x05D0u && cp <= 0x05EAu) || (cp >= 0x05EFu && cp <= 0x05F2u)) return true;
    if ((cp >= 0x0620u && cp <= 0x064Au) || (cp >= 0x066Eu && cp <= 0x066Fu) ||
        (cp >= 0x0671u && cp <= 0x06D3u) || cp == 0x06D5u || (cp >= 0x06E5u && cp <= 0x06E6u) ||
        (cp >= 0x06EEu && cp <= 0x06EFu) || (cp >= 0x06FAu && cp <= 0x06FCu) || cp == 0x06FFu) return true;
    if (cp >= 0x0904u && cp <= 0x0939u) return true;
    if (cp >= 0x0E01u && cp <= 0x0E30u) return true;
    if (cp >= 0x10A0u && cp <= 0x1248u) return true;
    /* Phonetic, Latin Extended Additional, Greek Extended */
    if (cp >= 0x1D00u && cp <= 0x1DBF) return true;
    if (cp >= 0x1E00u && cp <= 0x1FBCu) return true;
    /* Letterlike symbols (e.g. ℕ, ℝ, ℤ, ℓ, etc.) */
    if (cp == 0x2071u || cp == 0x207Fu || (cp >= 0x2090u && cp <= 0x209Cu)) return true;
    if (cp == 0x2102u || cp == 0x2107u || (cp >= 0x210Au && cp <= 0x2113u) || cp == 0x2115u ||
        (cp >= 0x2119u && cp <= 0x211Du) || cp == 0x2124u || cp == 0x2126u || cp == 0x2128u ||
        (cp >= 0x212Au && cp <= 0x212Du) || (cp >= 0x212Fu && cp <= 0x2139u) ||
        (cp >= 0x213Cu && cp <= 0x213Fu) || (cp >= 0x2145u && cp <= 0x2149u)) return true;
    /* Hiragana, Katakana, Bopomofo, Hangul, CJK Unified Ideographs */
    if ((cp >= 0x3005u && cp <= 0x3006u) || (cp >= 0x3031u && cp <= 0x3035u) ||
        cp == 0x303Bu || cp == 0x303Cu) return true;
    if ((cp >= 0x3041u && cp <= 0x3096u) || (cp >= 0x309Du && cp <= 0x309Fu)) return true;
    if ((cp >= 0x30A1u && cp <= 0x30FAu) || (cp >= 0x30FCu && cp <= 0x30FFu)) return true;
    if (cp >= 0x3105u && cp <= 0x318Eu) return true;
    if (cp >= 0x3400u && cp <= 0x9FFFu) return true;
    if (cp >= 0xAC00u && cp <= 0xD7A3u) return true;
    if (cp >= 0xF900u && cp <= 0xFAFFu) return true;
    if ((cp >= 0xFF21u && cp <= 0xFF3Au) || (cp >= 0xFF41u && cp <= 0xFF5Au) ||
        (cp >= 0xFF66u && cp <= 0xFFBEu)) return true;
    if ((cp >= 0x1D400u && cp <= 0x1D7CB) || (cp >= 0x20000u && cp <= 0x2FA1Fu)) return true;
    return false;
}

static inline char ascii_lower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return static_cast<char>(cp + ('a' - 'A'));
    return (cp < 0x80u) ? static_cast<char>(cp) : '\0';
}

struct CodepointSpan {
    uint32_t cp;
    uint32_t byte_off;
    uint32_t byte_len;
};

static void qwen2_pretokenize_span(
    const char* text,
    size_t len,
    std::vector<std::pair<uint32_t, uint32_t>>& pieces_out
) {
    pieces_out.clear();
    if (len == 0) return;

    std::vector<CodepointSpan> cps;
    cps.reserve(len);
    size_t off = 0;
    const uint8_t* u = reinterpret_cast<const uint8_t*>(text);
    while (off < len) {
        size_t adv = 0;
        uint32_t cp = decode_utf8_one(u + off, len - off, &adv);
        cps.push_back({cp, static_cast<uint32_t>(off), static_cast<uint32_t>(adv)});
        off += adv;
    }

    const size_t N = cps.size();
    size_t i = 0;
    while (i < N) {
        const uint32_t c0 = cps[i].cp;

        /* 1. `(?i:'s|'t|'re|'ve|'m|'ll|'d)` */
        if (c0 == '\'' && i + 1 < N) {
            char c1 = ascii_lower(cps[i + 1].cp);
            if (i + 2 < N) {
                char c2 = ascii_lower(cps[i + 2].cp);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    uint32_t b0 = cps[i].byte_off;
                    uint32_t b1 = cps[i + 2].byte_off + cps[i + 2].byte_len;
                    pieces_out.emplace_back(b0, b1 - b0);
                    i += 3;
                    continue;
                }
            }
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                uint32_t b0 = cps[i].byte_off;
                uint32_t b1 = cps[i + 1].byte_off + cps[i + 1].byte_len;
                pieces_out.emplace_back(b0, b1 - b0);
                i += 2;
                continue;
            }
        }

        /* 2. `[^\r\n\p{L}\p{N}]?\p{L}+` */
        if (is_unicode_letter(c0)) {
            size_t j = i + 1;
            while (j < N && is_unicode_letter(cps[j].cp)) ++j;
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }
        if (c0 != '\r' && c0 != '\n' && !is_unicode_number(c0) &&
            i + 1 < N && is_unicode_letter(cps[i + 1].cp)) {
            size_t j = i + 2;
            while (j < N && is_unicode_letter(cps[j].cp)) ++j;
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }

        /* 3. `\p{N}` (single digit per token!) */
        if (is_unicode_number(c0)) {
            pieces_out.emplace_back(cps[i].byte_off, cps[i].byte_len);
            ++i;
            continue;
        }

        /* 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*` */
        auto is_punct_sym = [](uint32_t cp) {
            return !is_unicode_whitespace(cp) && !is_unicode_letter(cp) && !is_unicode_number(cp);
        };
        if (is_punct_sym(c0) || (c0 == ' ' && i + 1 < N && is_punct_sym(cps[i + 1].cp))) {
            size_t j = (c0 == ' ') ? (i + 2) : (i + 1);
            while (j < N && is_punct_sym(cps[j].cp)) ++j;
            while (j < N && (cps[j].cp == '\r' || cps[j].cp == '\n')) ++j;
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[j - 1].byte_off + cps[j - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = j;
            continue;
        }

        /* 5/6/7. `\s*[\r\n]+ | \s+(?!\S) | \s+` */
        if (is_unicode_whitespace(c0)) {
            size_t ws_end = i + 1;
            ssize_t last_crlf = (c0 == '\r' || c0 == '\n') ? static_cast<ssize_t>(i) : -1;
            while (ws_end < N && is_unicode_whitespace(cps[ws_end].cp)) {
                if (cps[ws_end].cp == '\r' || cps[ws_end].cp == '\n') {
                    last_crlf = static_cast<ssize_t>(ws_end);
                }
                ++ws_end;
            }
            size_t match_end = ws_end;
            if (last_crlf >= 0) {
                /* Branch 5: `\s*[\r\n]+` consumes through the last CR/LF in this whitespace run */
                match_end = static_cast<size_t>(last_crlf) + 1;
            } else if (ws_end < N && (ws_end - i) >= 2) {
                /* Branch 6: `\s+(?!\S)` leaves the final space before `\S` */
                match_end = ws_end - 1;
            } else {
                /* Branch 6 at EOS (`ws_end == N`) or Branch 7 (`ws_end - i == 1`) */
                match_end = ws_end;
            }
            uint32_t b0 = cps[i].byte_off;
            uint32_t b1 = cps[match_end - 1].byte_off + cps[match_end - 1].byte_len;
            pieces_out.emplace_back(b0, b1 - b0);
            i = match_end;
            continue;
        }

        /* Fallback (unreachable, guarantees forward progress) */
        pieces_out.emplace_back(cps[i].byte_off, cps[i].byte_len);
        ++i;
    }
}

struct MergeSlot {
    uint64_t pair_key;   /* (uint64_t(left_id) << 32) | uint64_t(right_id); UINT64_MAX = empty */
    uint32_t rank;
    uint32_t merged_id;
};

struct AddedTokenEntry {
    std::string content;
    uint32_t    id;
    bool        special;
};

} /* anonymous namespace */

struct StrataQwen38Tokenizer {
    uint32_t                     vocab_size = 0;
    uint32_t                     byte_token_ids[256] = {};
    std::vector<std::string>     id_to_raw_bytes;
    std::vector<uint8_t>         id_is_special;
    std::vector<MergeSlot>       merge_ht;       /* Power-of-two open-addressed table */
    uint64_t                     merge_mask = 0;
    std::vector<AddedTokenEntry> added_sorted;   /* Sorted by descending byte length */
    std::vector<uint32_t>        added_by_b0[256]; /* Quick index by first byte */

    inline bool lookup_merge(uint32_t left_id, uint32_t right_id, uint32_t* rank_out, uint32_t* merged_out) const {
        uint64_t key = (static_cast<uint64_t>(left_id) << 32) | static_cast<uint64_t>(right_id);
        uint64_t h = (key * 11400714819323198485ULL) >> 45;
        uint64_t idx = h & merge_mask;
        while (true) {
            const MergeSlot& s = merge_ht[idx];
            if (s.pair_key == key) {
                *rank_out   = s.rank;
                *merged_out = s.merged_id;
                return true;
            }
            if (s.pair_key == std::numeric_limits<uint64_t>::max()) {
                return false;
            }
            idx = (idx + 1ULL) & merge_mask;
        }
    }

    inline void insert_merge(uint32_t left_id, uint32_t right_id, uint32_t rank, uint32_t merged_id) {
        uint64_t key = (static_cast<uint64_t>(left_id) << 32) | static_cast<uint64_t>(right_id);
        uint64_t h = (key * 11400714819323198485ULL) >> 45;
        uint64_t idx = h & merge_mask;
        while (true) {
            MergeSlot& s = merge_ht[idx];
            if (s.pair_key == std::numeric_limits<uint64_t>::max()) {
                s.pair_key  = key;
                s.rank      = rank;
                s.merged_id = merged_id;
                return;
            }
            if (s.pair_key == key) {
                if (rank < s.rank) {
                    s.rank      = rank;
                    s.merged_id = merged_id;
                }
                return;
            }
            idx = (idx + 1ULL) & merge_mask;
        }
    }
};

namespace {

static void build_gpt2_byte_unicode_tables(
    std::string b2u_utf8[256],
    int16_t u2b_cp[512]
) {
    for (int i = 0; i < 512; ++i) u2b_cp[i] = -1;
    bool direct[256] = {};
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;

    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++);
        b2u_utf8[b].clear();
        append_utf8_codepoint(b2u_utf8[b], cp);
        u2b_cp[cp] = static_cast<int16_t>(b);
    }
}

static bool decode_bpe_vocab_key_to_bytes(
    const std::string& vocab_key,
    const int16_t u2b_cp[512],
    std::string& raw_out
) {
    raw_out.clear();
    raw_out.reserve(vocab_key.size());
    const uint8_t* u = reinterpret_cast<const uint8_t*>(vocab_key.data());
    size_t len = vocab_key.size();
    size_t off = 0;
    while (off < len) {
        size_t adv = 0;
        uint32_t cp = decode_utf8_one(u + off, len - off, &adv);
        if (cp >= 512u || u2b_cp[cp] < 0) return false;
        raw_out.push_back(static_cast<char>(static_cast<uint8_t>(u2b_cp[cp])));
        off += adv;
    }
    return true;
}

static void bpe_encode_piece(
    const StrataQwen38Tokenizer* tok,
    const uint8_t* bytes,
    uint32_t len,
    std::vector<uint32_t>& out_ids
) {
    if (len == 0) return;
    if (len == 1) {
        out_ids.push_back(tok->byte_token_ids[bytes[0]]);
        return;
    }

    /* Fast stack path for typical words (`len <= 64`) */
    if (len <= 64) {
        uint32_t ids[64];
        uint32_t n = len;
        for (uint32_t i = 0; i < n; ++i) ids[i] = tok->byte_token_ids[bytes[i]];

        while (n >= 2) {
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            uint32_t best_merged = 0;
            uint32_t best_pos = 0;
            for (uint32_t i = 0; i + 1 < n; ++i) {
                uint32_t r = 0, m = 0;
                if (tok->lookup_merge(ids[i], ids[i + 1], &r, &m) && r < best_rank) {
                    best_rank   = r;
                    best_merged = m;
                    best_pos    = i;
                }
            }
            if (best_rank == std::numeric_limits<uint32_t>::max()) break;

            ids[best_pos] = best_merged;
            for (uint32_t j = best_pos + 1; j + 1 < n; ++j) {
                ids[j] = ids[j + 1];
            }
            --n;
        }
        for (uint32_t i = 0; i < n; ++i) out_ids.push_back(ids[i]);
        return;
    }

    /* `O(N log N)` doubly-linked list + min-heap priority queue for long runs (`> 64` bytes) */
    struct Node {
        uint32_t id;
        int32_t  prev;
        int32_t  next;
        uint32_t ver;
        bool     alive;
    };
    struct HeapItem {
        uint32_t rank;
        int32_t  left;
        uint32_t left_ver;
        uint32_t right_ver;
        uint32_t merged_id;
        bool operator>(const HeapItem& o) const {
            return (rank != o.rank) ? (rank > o.rank) : (left > o.left);
        }
    };

    std::vector<Node> nodes(len);
    for (uint32_t i = 0; i < len; ++i) {
        nodes[i] = {
            tok->byte_token_ids[bytes[i]],
            static_cast<int32_t>(i) - 1,
            (i + 1 < len) ? static_cast<int32_t>(i + 1) : -1,
            0u,
            true
        };
    }

    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> pq;
    auto try_push = [&](int32_t l) {
        if (l < 0) return;
        int32_t r = nodes[l].next;
        if (r < 0) return;
        uint32_t rank = 0, merged = 0;
        if (tok->lookup_merge(nodes[l].id, nodes[r].id, &rank, &merged)) {
            pq.push({rank, l, nodes[l].ver, nodes[r].ver, merged});
        }
    };
    for (uint32_t i = 0; i + 1 < len; ++i) {
        try_push(static_cast<int32_t>(i));
    }

    while (!pq.empty()) {
        HeapItem top = pq.top();
        pq.pop();
        int32_t l = top.left;
        if (!nodes[l].alive || nodes[l].ver != top.left_ver) continue;
        int32_t r = nodes[l].next;
        if (r < 0 || !nodes[r].alive || nodes[r].ver != top.right_ver) continue;

        nodes[l].id = top.merged_id;
        ++nodes[l].ver;
        nodes[r].alive = false;
        int32_t rn = nodes[r].next;
        nodes[l].next = rn;
        if (rn >= 0) nodes[rn].prev = l;

        try_push(nodes[l].prev);
        try_push(l);
    }

    for (int32_t cur = 0; cur >= 0; cur = nodes[cur].next) {
        if (nodes[cur].alive) out_ids.push_back(nodes[cur].id);
    }
}

} /* anonymous namespace */

/* ============================================================================
 * 5. Public C ABI Implementations (`include/strata_qwen38.h`)
 * ============================================================================ */

extern "C" {

StrataQwen38Tokenizer* strata_qwen38_tokenizer_load(const char* tokenizer_json_path) {
    if (!tokenizer_json_path) return nullptr;
    std::string json_str;
    if (!read_whole_file(tokenizer_json_path, json_str)) {
        std::fprintf(stderr, "[qwen38-tok] ERROR: failed to read '%s'\n", tokenizer_json_path);
        return nullptr;
    }

    std::string b2u_utf8[256];
    int16_t u2b_cp[512];
    build_gpt2_byte_unicode_tables(b2u_utf8, u2b_cp);

    std::unordered_map<std::string, uint32_t> str_to_id;
    str_to_id.reserve(300000);

    struct RawMerge {
        std::string a;
        std::string b;
    };
    std::vector<RawMerge> raw_merges;
    raw_merges.reserve(260000);
    std::vector<AddedTokenEntry> added_list;

    JsonCursor cur(json_str.data(), json_str.size());
    if (!cur.consume('{')) return nullptr;

    std::string top_key, sub_key, k_str, v_str;
    uint32_t max_id_seen = 0;

    if (!cur.consume('}')) {
        while (true) {
            if (!cur.parse_string(top_key)) return nullptr;
            if (!cur.consume(':')) return nullptr;

            if (top_key == "added_tokens") {
                if (!cur.consume('[')) return nullptr;
                if (!cur.consume(']')) {
                    while (true) {
                        if (!cur.consume('{')) return nullptr;
                        AddedTokenEntry at{"", 0, false};
                        if (!cur.consume('}')) {
                            while (true) {
                                if (!cur.parse_string(sub_key)) return nullptr;
                                if (!cur.consume(':')) return nullptr;
                                if (sub_key == "id") {
                                    uint64_t id_u64 = 0;
                                    if (!cur.parse_uint64(id_u64) || id_u64 >= (1ULL << 22)) return nullptr;
                                    at.id = static_cast<uint32_t>(id_u64);
                                } else if (sub_key == "content") {
                                    if (!cur.parse_string(at.content)) return nullptr;
                                } else if (sub_key == "special") {
                                    if (!cur.parse_bool(at.special)) return nullptr;
                                } else {
                                    if (!cur.skip_value()) return nullptr;
                                }
                                if (cur.consume('}')) break;
                                if (!cur.consume(',')) return nullptr;
                            }
                        }
                        if (!at.content.empty()) {
                            max_id_seen = std::max(max_id_seen, at.id);
                            added_list.push_back(std::move(at));
                        }
                        if (cur.consume(']')) break;
                        if (!cur.consume(',')) return nullptr;
                    }
                }
            } else if (top_key == "model") {
                if (!cur.consume('{')) return nullptr;
                if (!cur.consume('}')) {
                    while (true) {
                        if (!cur.parse_string(sub_key)) return nullptr;
                        if (!cur.consume(':')) return nullptr;
                        if (sub_key == "vocab") {
                            if (!cur.consume('{')) return nullptr;
                            if (!cur.consume('}')) {
                                while (true) {
                                    if (!cur.parse_string(k_str)) return nullptr;
                                    if (!cur.consume(':')) return nullptr;
                                    uint64_t tid = 0;
                                    if (!cur.parse_uint64(tid) || tid >= (1ULL << 22)) return nullptr;
                                    uint32_t id32 = static_cast<uint32_t>(tid);
                                    max_id_seen = std::max(max_id_seen, id32);
                                    str_to_id.emplace(k_str, id32);
                                    if (cur.consume('}')) break;
                                    if (!cur.consume(',')) return nullptr;
                                }
                            }
                        } else if (sub_key == "merges") {
                            if (!cur.consume('[')) return nullptr;
                            if (!cur.consume(']')) {
                                while (true) {
                                    char pk = cur.peek();
                                    if (pk == '[') {
                                        cur.consume('[');
                                        RawMerge rm;
                                        if (!cur.parse_string(rm.a)) return nullptr;
                                        if (!cur.consume(',')) return nullptr;
                                        if (!cur.parse_string(rm.b)) return nullptr;
                                        if (!cur.consume(']')) return nullptr;
                                        raw_merges.push_back(std::move(rm));
                                    } else if (pk == '"') {
                                        if (!cur.parse_string(v_str)) return nullptr;
                                        size_t sp = v_str.find(' ');
                                        if (sp != std::string::npos) {
                                            raw_merges.push_back({v_str.substr(0, sp), v_str.substr(sp + 1)});
                                        }
                                    } else {
                                        return nullptr;
                                    }
                                    if (cur.consume(']')) break;
                                    if (!cur.consume(',')) return nullptr;
                                }
                            }
                        } else {
                            if (!cur.skip_value()) return nullptr;
                        }
                        if (cur.consume('}')) break;
                        if (!cur.consume(',')) return nullptr;
                    }
                }
            } else {
                if (!cur.skip_value()) return nullptr;
            }

            if (cur.consume('}')) break;
            if (!cur.consume(',')) return nullptr;
        }
    }

    if (str_to_id.empty() || max_id_seen >= (1U << 22)) {
        std::fprintf(stderr, "[qwen38-tok] ERROR: empty or out-of-range vocab in '%s'\n", tokenizer_json_path);
        return nullptr;
    }

    auto* tok = new StrataQwen38Tokenizer();
    tok->vocab_size = max_id_seen + 1u;
    tok->id_to_raw_bytes.assign(tok->vocab_size, std::string());
    tok->id_is_special.assign(tok->vocab_size, 0u);

    std::string raw_tmp;
    for (const auto& kv : str_to_id) {
        if (kv.second >= tok->vocab_size) {
            delete tok;
            return nullptr;
        }
        if (decode_bpe_vocab_key_to_bytes(kv.first, u2b_cp, raw_tmp)) {
            tok->id_to_raw_bytes[kv.second] = raw_tmp;
        } else {
            tok->id_to_raw_bytes[kv.second] = kv.first;
        }
    }

    for (int b = 0; b < 256; ++b) {
        auto it = str_to_id.find(b2u_utf8[b]);
        if (it == str_to_id.end()) {
            delete tok;
            return nullptr;
        }
        tok->byte_token_ids[b] = it->second;
    }

    /* Populate added/special tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<think>`, etc.) */
    for (const auto& at : added_list) {
        if (at.id < tok->vocab_size) {
            tok->id_to_raw_bytes[at.id] = at.content;
            tok->id_is_special[at.id]   = at.special ? 1u : 0u;
        }
    }
    tok->added_sorted = std::move(added_list);
    std::sort(tok->added_sorted.begin(), tok->added_sorted.end(),
              [](const AddedTokenEntry& x, const AddedTokenEntry& y) {
                  if (x.content.size() != y.content.size()) return x.content.size() > y.content.size();
                  return x.id < y.id;
              });
    for (uint32_t idx = 0; idx < tok->added_sorted.size(); ++idx) {
        uint8_t b0 = static_cast<uint8_t>(tok->added_sorted[idx].content[0]);
        tok->added_by_b0[b0].push_back(idx);
    }

    /* Build flat open-addressed merge hash table (`size = next_pow2(merges * 2)`) */
    uint64_t ht_cap = 1024;
    while (ht_cap < raw_merges.size() * 2ULL + 64ULL) ht_cap <<= 1;
    tok->merge_ht.assign(static_cast<size_t>(ht_cap),
                         MergeSlot{std::numeric_limits<uint64_t>::max(), 0u, 0u});
    tok->merge_mask = ht_cap - 1ULL;

    std::string combined;
    for (uint32_t rank = 0; rank < raw_merges.size(); ++rank) {
        const auto& rm = raw_merges[rank];
        auto it_a = str_to_id.find(rm.a);
        auto it_b = str_to_id.find(rm.b);
        if (it_a == str_to_id.end() || it_b == str_to_id.end()) continue;
        combined.assign(rm.a);
        combined.append(rm.b);
        auto it_m = str_to_id.find(combined);
        if (it_m == str_to_id.end()) continue;
        tok->insert_merge(it_a->second, it_b->second, rank, it_m->second);
    }

    return tok;
}

void strata_qwen38_tokenizer_free(StrataQwen38Tokenizer* tok) {
    delete tok;
}

uint32_t strata_qwen38_tokenizer_vocab_size(const StrataQwen38Tokenizer* tok) {
    return tok ? tok->vocab_size : 0u;
}

int strata_qwen38_encode(
    const StrataQwen38Tokenizer* tok,
    const char* text,
    size_t text_len,
    bool parse_special,
    uint32_t* out_ids,
    uint32_t max_ids,
    uint32_t* out_num_ids
) {
    if (!tok || !text || !out_num_ids) return -1;
    if (text_len == static_cast<size_t>(-1)) {
        text_len = std::strlen(text);
    }
    *out_num_ids = 0;
    if (text_len == 0) return 0;

    std::vector<uint32_t> produced;
    produced.reserve(text_len / 2 + 16);
    std::vector<std::pair<uint32_t, uint32_t>> pieces;

    auto encode_plain_span = [&](const char* span_ptr, size_t span_len) {
        if (span_len == 0) return;
        qwen2_pretokenize_span(span_ptr, span_len, pieces);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(span_ptr);
        for (const auto& pc : pieces) {
            bpe_encode_piece(tok, base + pc.first, pc.second, produced);
        }
    };

    if (!parse_special || tok->added_sorted.empty()) {
        encode_plain_span(text, text_len);
    } else {
        size_t span_start = 0;
        size_t pos = 0;
        while (pos < text_len) {
            uint8_t b0 = static_cast<uint8_t>(text[pos]);
            const auto& candidates = tok->added_by_b0[b0];
            int32_t matched_idx = -1;
            for (uint32_t c_idx : candidates) {
                const AddedTokenEntry& at = tok->added_sorted[c_idx];
                size_t mlen = at.content.size();
                if (pos + mlen <= text_len &&
                    std::memcmp(text + pos, at.content.data(), mlen) == 0) {
                    matched_idx = static_cast<int32_t>(c_idx);
                    break; /* `added_sorted` is sorted longest-first */
                }
            }
            if (matched_idx >= 0) {
                if (pos > span_start) {
                    encode_plain_span(text + span_start, pos - span_start);
                }
                const AddedTokenEntry& at = tok->added_sorted[static_cast<size_t>(matched_idx)];
                produced.push_back(at.id);
                pos += at.content.size();
                span_start = pos;
            } else {
                ++pos;
            }
        }
        if (span_start < text_len) {
            encode_plain_span(text + span_start, text_len - span_start);
        }
    }

    *out_num_ids = static_cast<uint32_t>(produced.size());
    if (!out_ids) return 0;
    if (produced.size() > max_ids) return -2;
    if (!produced.empty()) {
        std::memcpy(out_ids, produced.data(), produced.size() * sizeof(uint32_t));
    }
    return 0;
}

int strata_qwen38_format_chat_prompt(
    const char* user_prompt,
    const char* system_prompt,
    bool enable_thinking,
    char* out_buf,
    size_t out_buf_cap,
    size_t* out_len
) {
    if (!user_prompt) return -1;
    std::string res;
    res.reserve(std::strlen(user_prompt) + 256);

    if (system_prompt && system_prompt[0] != '\0') {
        res += "<|im_start|>system\n";
        res += system_prompt;
        res += "<|im_end|>\n";
    }
    res += "<|im_start|>user\n";
    res += user_prompt;
    res += "<|im_end|>\n<|im_start|>assistant\n";
    if (enable_thinking) {
        res += "<think>\n";
    } else {
        res += "<think>\n\n</think>\n\n";
    }

    if (out_len) *out_len = res.size();
    if (!out_buf) return 0;
    if (res.size() + 1 > out_buf_cap) return -2;
    std::memcpy(out_buf, res.data(), res.size() + 1);
    return 0;
}

int strata_qwen38_stream_decode_step(
    const StrataQwen38Tokenizer* tok,
    StrataQwen38StreamDecoder* dec,
    uint32_t token_id,
    bool skip_special,
    char* out_utf8,
    size_t out_cap,
    size_t* out_len
) {
    if (!tok || !dec || !out_utf8 || out_cap == 0) return -1;
    out_utf8[0] = '\0';
    if (out_len) *out_len = 0;

    if (token_id >= tok->vocab_size) return 0;
    if (skip_special && tok->id_is_special[token_id]) return 0;

    const std::string& raw = tok->id_to_raw_bytes[token_id];
    if (raw.empty() && dec->pending_len == 0) return 0;

    std::string combined;
    combined.reserve(dec->pending_len + raw.size());
    if (dec->pending_len > 0) {
        combined.append(reinterpret_cast<const char*>(dec->pending_bytes), dec->pending_len);
        dec->pending_len = 0;
    }
    combined.append(raw);

    /* Inspect trailing 1..3 bytes for an incomplete multi-byte UTF-8 sequence */
    const uint8_t* u = reinterpret_cast<const uint8_t*>(combined.data());
    size_t n = combined.size();
    size_t hold = 0;
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        uint8_t b = u[n - back];
        if ((b & 0xC0u) == 0x80u) {
            continue; /* UTF-8 continuation byte */
        }
        size_t need = 1;
        if ((b & 0xE0u) == 0xC0u)      need = 2;
        else if ((b & 0xF0u) == 0xE0u) need = 3;
        else if ((b & 0xF8u) == 0xF0u) need = 4;
        if (need > back) {
            hold = back;
        }
        break;
    }

    size_t emit_len = n - hold;
    if (hold > 0) {
        std::memcpy(dec->pending_bytes, u + emit_len, hold);
        dec->pending_len = static_cast<uint8_t>(hold);
    }
    if (emit_len + 1 > out_cap) return -2;
    if (emit_len > 0) {
        std::memcpy(out_utf8, combined.data(), emit_len);
    }
    out_utf8[emit_len] = '\0';
    if (out_len) *out_len = emit_len;
    return 0;
}

int strata_qwen38_decode(
    const StrataQwen38Tokenizer* tok,
    const uint32_t* ids,
    uint32_t num_ids,
    bool skip_special,
    char* out_utf8,
    size_t out_cap,
    size_t* out_len
) {
    if (!tok || (!ids && num_ids > 0)) return -1;
    std::string acc;
    acc.reserve(static_cast<size_t>(num_ids) * 4 + 16);
    for (uint32_t i = 0; i < num_ids; ++i) {
        uint32_t tid = ids[i];
        if (tid >= tok->vocab_size) continue;
        if (skip_special && tok->id_is_special[tid]) continue;
        acc.append(tok->id_to_raw_bytes[tid]);
    }
    if (out_len) *out_len = acc.size();
    if (!out_utf8) return 0;
    if (acc.size() + 1 > out_cap) return -2;
    std::memcpy(out_utf8, acc.data(), acc.size() + 1);
    return 0;
}

/* ============================================================================
 * 6. Multi-Shard HF Converter & Zero-Copy Bundle Loader
 * ============================================================================ */

int strata_qwen38_convert_hf_directory(
    const char* hf_dir,
    const char* out_bundle_dir,
    StrataQwen38ConvertStats* stats_out
) {
    if (!hf_dir || !out_bundle_dir) return -1;
    auto t0 = std::chrono::steady_clock::now();
    if (stats_out) std::memset(stats_out, 0, sizeof(*stats_out));

    std::string in_dir(hf_dir);
    std::string out_dir(out_bundle_dir);
    if (!mkdirs(out_dir)) {
        std::fprintf(stderr, "[qwen38-convert] ERROR: failed to create directory '%s'\n", out_dir.c_str());
        return -1;
    }

    /* 1. Parse `config.json` */
    std::string cfg_raw;
    if (!read_whole_file(in_dir + "/config.json", cfg_raw)) {
        std::fprintf(stderr, "[qwen38-convert] ERROR: missing '%s/config.json'\n", in_dir.c_str());
        return -1;
    }
    SmallJson root_cfg;
    if (!parse_small_json(cfg_raw, root_cfg) || root_cfg.type != SmallJson::J_OBJ) {
        std::fprintf(stderr, "[qwen38-convert] ERROR: invalid JSON in config.json\n");
        return -1;
    }
    const SmallJson* tc = root_cfg.get("text_config");
    if (!tc || tc->type != SmallJson::J_OBJ) tc = &root_cfg;

    StrataQwen38Config mcfg;
    std::memset(&mcfg, 0, sizeof(mcfg));
    mcfg.num_hidden_layers       = static_cast<uint32_t>(tc->get_i64("num_hidden_layers", 64));
    mcfg.hidden_size             = static_cast<uint32_t>(tc->get_i64("hidden_size", 5120));
    mcfg.intermediate_size       = static_cast<uint32_t>(tc->get_i64("intermediate_size", 17408));
    mcfg.vocab_size              = static_cast<uint32_t>(tc->get_i64("vocab_size", 248320));
    mcfg.full_attention_interval = static_cast<uint32_t>(tc->get_i64("full_attention_interval", 4));
    mcfg.num_attention_heads     = static_cast<uint32_t>(tc->get_i64("num_attention_heads", 24));
    mcfg.num_key_value_heads     = static_cast<uint32_t>(tc->get_i64("num_key_value_heads", 4));
    mcfg.head_dim                = static_cast<uint32_t>(tc->get_i64("head_dim", 256));
    double prf                   = tc->get_f64("partial_rotary_factor", 0.25);
    mcfg.rope_theta              = 10000000.0f;
    if (const SmallJson* rp = tc->get("rope_parameters")) {
        prf             = rp->get_f64("partial_rotary_factor", prf);
        mcfg.rope_theta = static_cast<float>(rp->get_f64("rope_theta", 10000000.0));
    } else {
        mcfg.rope_theta = static_cast<float>(tc->get_f64("rope_theta", 10000000.0));
    }
    mcfg.rotary_dim             = static_cast<uint32_t>(std::llround(mcfg.head_dim * prf));
    mcfg.attn_output_gate       = tc->get_bool("attn_output_gate", true) ? 1u : 0u;
    mcfg.linear_num_key_heads   = static_cast<uint32_t>(tc->get_i64("linear_num_key_heads", 16));
    mcfg.linear_key_head_dim    = static_cast<uint32_t>(tc->get_i64("linear_key_head_dim", 128));
    mcfg.linear_num_value_heads = static_cast<uint32_t>(tc->get_i64("linear_num_value_heads", 48));
    mcfg.linear_value_head_dim  = static_cast<uint32_t>(tc->get_i64("linear_value_head_dim", 128));
    mcfg.linear_conv_kernel_dim = static_cast<uint32_t>(tc->get_i64("linear_conv_kernel_dim", 4));
    mcfg.rms_norm_eps           = static_cast<float>(tc->get_f64("rms_norm_eps", 1e-6));
    mcfg.tie_word_embeddings    = tc->get_bool("tie_word_embeddings", false) ? 1u : 0u;
    mcfg.max_position_embeddings = static_cast<uint32_t>(tc->get_i64("max_position_embeddings", 262144));
    mcfg.bos_token_id           = static_cast<uint32_t>(tc->get_i64("bos_token_id", 248044));

    mcfg.quant_group_size = 64;
    mcfg.quant_bits       = 4;
    if (const SmallJson* qc = root_cfg.get("quantization")) {
        mcfg.quant_group_size = static_cast<uint32_t>(qc->get_i64("group_size", 64));
        mcfg.quant_bits       = static_cast<uint32_t>(qc->get_i64("bits", 4));
    }
    if (mcfg.num_hidden_layers == 0 || mcfg.num_hidden_layers > STRATA_QWEN38_MAX_LAYERS ||
        mcfg.quant_group_size != 64 || mcfg.quant_bits != 4) {
        std::fprintf(stderr, "[qwen38-convert] ERROR: unsupported config layers=%u gs=%u bits=%u\n",
                     mcfg.num_hidden_layers, mcfg.quant_group_size, mcfg.quant_bits);
        return -1;
    }

    /* Extract EOS token IDs from root `eos_token_id` or `text_config.eos_token_id` */
    const SmallJson* eos_node = root_cfg.get("eos_token_id");
    if (!eos_node) eos_node = tc->get("eos_token_id");
    if (eos_node && eos_node->type == SmallJson::J_ARR) {
        for (size_t i = 0; i < eos_node->arr.size() && mcfg.num_eos_token_ids < 4; ++i) {
            if (eos_node->arr[i].type == SmallJson::J_NUM) {
                mcfg.eos_token_ids[mcfg.num_eos_token_ids++] =
                    static_cast<uint32_t>(eos_node->arr[i].i_val);
            }
        }
    } else if (eos_node && eos_node->type == SmallJson::J_NUM) {
        mcfg.eos_token_ids[mcfg.num_eos_token_ids++] = static_cast<uint32_t>(eos_node->i_val);
    }
    if (mcfg.num_eos_token_ids == 0) {
        mcfg.eos_token_ids[0]  = STRATA_QWEN38_TOKEN_IM_END;
        mcfg.eos_token_ids[1]  = STRATA_QWEN38_TOKEN_ENDOFTEXT;
        mcfg.num_eos_token_ids = 2;
    }

    std::vector<uint32_t> layer_types(mcfg.num_hidden_layers, STRATA_QWEN38_LAYER_LINEAR_ATTN);
    const SmallJson* lt_arr = tc->get("layer_types");
    for (uint32_t l = 0; l < mcfg.num_hidden_layers; ++l) {
        bool is_full = ((l + 1u) % mcfg.full_attention_interval) == 0u;
        if (lt_arr && lt_arr->type == SmallJson::J_ARR && l < lt_arr->arr.size()) {
            is_full = (lt_arr->arr[l].s_val == "full_attention");
        }
        layer_types[l] = is_full ? STRATA_QWEN38_LAYER_FULL_ATTN : STRATA_QWEN38_LAYER_LINEAR_ATTN;
        if (is_full) ++mcfg.num_full_attn_layers;
        else         ++mcfg.num_linear_layers;
    }

    /* 2. Discover `.safetensors` shard list from `model.safetensors.index.json` or single file */
    std::vector<std::string> shard_names;
    std::string idx_raw;
    if (read_whole_file(in_dir + "/model.safetensors.index.json", idx_raw)) {
        SmallJson idx_json;
        if (!parse_small_json(idx_raw, idx_json)) return -1;
        const SmallJson* wm = idx_json.get("weight_map");
        if (!wm || wm->type != SmallJson::J_OBJ) return -1;
        auto is_safe_shard_basename = [](const std::string& name) -> bool {
            if (name.empty() || name.size() > 255U || name == "." || name == ".." || name[0] == '-') return false;
            for (unsigned char c : name) {
                if (c == '/' || c == '\\' || c == '[' || c == ']' || c == '{' || c == '}' ||
                    c < 0x20U || c == 0x7FU) return false;
            }
            return true;
        };
        std::map<std::string, bool> uniq;
        for (const auto& kv : wm->obj) {
            if (kv.second.type == SmallJson::J_STR) {
                if (!is_safe_shard_basename(kv.second.s_val)) {
                    std::fprintf(stderr, "[qwen38-convert] ERROR: unsafe shard filename '%s' in index\n",
                                 kv.second.s_val.c_str());
                    return -1;
                }
                uniq[kv.second.s_val] = true;
            }
        }
        for (const auto& kv : uniq) shard_names.push_back(kv.first);
    } else {
        shard_names.push_back("model.safetensors");
    }

    ConvertStreamCtx ctx;
    uint64_t total_hf_bytes = 0;
    uint32_t vision_skipped = 0;

    for (uint32_t s = 0; s < shard_names.size(); ++s) {
        std::string spath = in_dir + "/" + shard_names[s];
        int fd = open(spath.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "[qwen38-convert] ERROR: cannot open shard '%s'\n", spath.c_str());
            for (int ofd : ctx.shard_fds) close(ofd);
            return -1;
        }
#if defined(F_NOCACHE)
        (void)fcntl(fd, F_NOCACHE, 1);
#endif
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 8) {
            close(fd);
            for (int ofd : ctx.shard_fds) close(ofd);
            return -1;
        }
        uint64_t sbytes = static_cast<uint64_t>(st.st_size);
        total_hf_bytes += sbytes;
        ctx.shard_fds.push_back(fd);

        if (!parse_safetensors_header(fd, s, sbytes, ctx.tensors, vision_skipped)) {
            std::fprintf(stderr, "[qwen38-convert] ERROR: corrupt safetensors header in '%s'\n", spath.c_str());
            for (int ofd : ctx.shard_fds) close(ofd);
            return -1;
        }
        std::fprintf(stdout, "[qwen38-convert] Indexed shard %u/%zu (%s, %.2f GiB)\n",
                     s + 1, shard_names.size(), shard_names[s].c_str(),
                     static_cast<double>(sbytes) / (1024.0 * 1024.0 * 1024.0));
        std::fflush(stdout);
    }

    /* 3. Open `<out_bundle_dir>/weights.strata.tmp` with `F_NOCACHE` for low-RSS streaming write */
    std::string final_weights_path = out_dir + "/weights.strata";
    std::string tmp_weights_path   = final_weights_path + ".tmp";
    ctx.out_fd = open(tmp_weights_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (ctx.out_fd < 0) {
        for (int ofd : ctx.shard_fds) close(ofd);
        return -1;
    }
#if defined(F_NOCACHE)
    (void)fcntl(ctx.out_fd, F_NOCACHE, 1);
#endif

    auto cleanup_fail = [&]() -> int {
        if (ctx.out_fd >= 0) close(ctx.out_fd);
        unlink(tmp_weights_path.c_str());
        for (int ofd : ctx.shard_fds) close(ofd);
        return -1;
    };

    StrataQwen38WeightsHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    hdr.magic               = STRATA_QWEN38_MAGIC_U64;
    hdr.version             = STRATA_QWEN38_FORMAT_VERSION;
    hdr.header_padded_bytes = STRATA_QWEN38_HEADER_PADDED_BYTES;
    hdr.cfg                 = mcfg;

    /* Reserve 80 KiB (`5 * 16 KiB`) at start of file for `StrataQwen38WeightsHeader` */
    ctx.write_off = 0;
    while (ctx.write_off < STRATA_QWEN38_HEADER_PADDED_BYTES) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(
            ctx.zero_buf.size(),
            static_cast<uint64_t>(STRATA_QWEN38_HEADER_PADDED_BYTES) - ctx.write_off));
        if (!pwrite_exact(ctx.out_fd, ctx.zero_buf.data(), chunk, ctx.write_off)) {
            return cleanup_fail();
        }
        ctx.write_off += chunk;
    }

    uint64_t active_bytes = 0;

    /* 4. Stream `embed_tokens` (`[248320, 5120]` 4-bit affine) */
    hdr.embed_page_off = ctx.write_off;
    if (!ctx.write_quant_matrix("model.embed_tokens", mcfg.vocab_size, mcfg.hidden_size, hdr.embed_tokens)) {
        return cleanup_fail();
    }
    if (!ctx.pad_to(STRATA_QWEN38_PAGE_BYTES)) return cleanup_fail();
    hdr.embed_page_bytes = ctx.write_off - hdr.embed_page_off;
    /* Single-token forward reads 1 embedding row */
    active_bytes += static_cast<uint64_t>(mcfg.hidden_size / 8u) * sizeof(uint32_t) +
                    2ULL * static_cast<uint64_t>(mcfg.hidden_size / 64u) * sizeof(uint16_t);

    /* 5. Stream all 64 hybrid decoder layers (`linear_attention` + `full_attention`) */
    const uint32_t D    = mcfg.hidden_size;
    const uint32_t D_FF = mcfg.intermediate_size;
    const uint32_t K_DIM = mcfg.linear_num_key_heads * mcfg.linear_key_head_dim;     /* 2048 */
    const uint32_t V_DIM = mcfg.linear_num_value_heads * mcfg.linear_value_head_dim; /* 6144 */
    const uint32_t QKV_DIM = 2u * K_DIM + V_DIM;                                     /* 10240 */
    const uint32_t NV_HEADS = mcfg.linear_num_value_heads;                           /* 48 */
    const uint32_t NV_PAD   = static_cast<uint32_t>(align_up_u64(NV_HEADS, 64ULL));  /* 64 */
    const uint32_t Q_DIM    = mcfg.num_attention_heads * mcfg.head_dim;              /* 6144 */
    const uint32_t QG_DIM   = mcfg.attn_output_gate ? (2u * Q_DIM) : Q_DIM;          /* 12288 */
    const uint32_t KV_DIM   = mcfg.num_key_value_heads * mcfg.head_dim;              /* 1024 */

    for (uint32_t l = 0; l < mcfg.num_hidden_layers; ++l) {
        StrataQwen38LayerDesc& ld = hdr.layers[l];
        ld.layer_idx      = l;
        ld.layer_type     = layer_types[l];
        ld.layer_page_off = ctx.write_off;

        std::string lp = "model.layers." + std::to_string(l) + ".";
        if (!ctx.write_f32_slice(lp + "input_layernorm.weight", D, D, D, 1, ld.input_layernorm) ||
            !ctx.write_f32_slice(lp + "post_attention_layernorm.weight", D, D, D, 1, ld.post_attention_layernorm)) {
            return cleanup_fail();
        }
        active_bytes += ld.input_layernorm.bytes + ld.post_attention_layernorm.bytes;

        if (ld.layer_type == STRATA_QWEN38_LAYER_LINEAR_ATTN) {
            std::string ap = lp + "linear_attn.";
            if (!ctx.write_quant_matrix(ap + "in_proj_qkv", QKV_DIM, D, ld.lin_in_proj_qkv) ||
                !ctx.write_quant_matrix(ap + "in_proj_z",   V_DIM,   D, ld.lin_in_proj_z) ||
                !ctx.write_quant_matrix(ap + "in_proj_a",   NV_HEADS, D, ld.lin_in_proj_a) ||
                !ctx.write_quant_matrix(ap + "in_proj_b",   NV_HEADS, D, ld.lin_in_proj_b) ||
                !ctx.write_quant_matrix(ap + "out_proj",    D,   V_DIM, ld.lin_out_proj) ||
                !ctx.write_f32_slice(ap + "conv1d.weight", QKV_DIM * mcfg.linear_conv_kernel_dim,
                                     QKV_DIM * mcfg.linear_conv_kernel_dim,
                                     QKV_DIM, mcfg.linear_conv_kernel_dim, ld.lin_conv1d_weight) ||
                !ctx.write_f32_slice(ap + "A_log",   NV_HEADS, NV_PAD, NV_PAD, 1, ld.lin_A_log) ||
                !ctx.write_f32_slice(ap + "dt_bias", NV_HEADS, NV_PAD, NV_PAD, 1, ld.lin_dt_bias) ||
                !ctx.write_f32_slice(ap + "norm.weight", mcfg.linear_value_head_dim,
                                     mcfg.linear_value_head_dim, mcfg.linear_value_head_dim, 1, ld.lin_norm)) {
                return cleanup_fail();
            }
            active_bytes += qm_active_bytes(ld.lin_in_proj_qkv) +
                            qm_active_bytes(ld.lin_in_proj_z) +
                            qm_active_bytes(ld.lin_in_proj_a) +
                            qm_active_bytes(ld.lin_in_proj_b) +
                            qm_active_bytes(ld.lin_out_proj) +
                            ld.lin_conv1d_weight.bytes +
                            ld.lin_A_log.bytes +
                            ld.lin_dt_bias.bytes +
                            ld.lin_norm.bytes;
        } else {
            std::string ap = lp + "self_attn.";
            if (!ctx.write_quant_matrix(ap + "q_proj", QG_DIM, D, ld.attn_q_proj) ||
                !ctx.write_quant_matrix(ap + "k_proj", KV_DIM, D, ld.attn_k_proj) ||
                !ctx.write_quant_matrix(ap + "v_proj", KV_DIM, D, ld.attn_v_proj) ||
                !ctx.write_quant_matrix(ap + "o_proj", D,  Q_DIM, ld.attn_o_proj) ||
                !ctx.write_f32_slice(ap + "q_norm.weight", mcfg.head_dim, mcfg.head_dim, mcfg.head_dim, 1, ld.attn_q_norm) ||
                !ctx.write_f32_slice(ap + "k_norm.weight", mcfg.head_dim, mcfg.head_dim, mcfg.head_dim, 1, ld.attn_k_norm)) {
                return cleanup_fail();
            }
            active_bytes += qm_active_bytes(ld.attn_q_proj) +
                            qm_active_bytes(ld.attn_k_proj) +
                            qm_active_bytes(ld.attn_v_proj) +
                            qm_active_bytes(ld.attn_o_proj) +
                            ld.attn_q_norm.bytes +
                            ld.attn_k_norm.bytes;
        }

        std::string mp = lp + "mlp.";
        if (!ctx.write_quant_matrix(mp + "gate_proj", D_FF, D, ld.mlp_gate_proj) ||
            !ctx.write_quant_matrix(mp + "up_proj",   D_FF, D, ld.mlp_up_proj) ||
            !ctx.write_quant_matrix(mp + "down_proj", D, D_FF, ld.mlp_down_proj)) {
            return cleanup_fail();
        }
        active_bytes += qm_active_bytes(ld.mlp_gate_proj) +
                        qm_active_bytes(ld.mlp_up_proj) +
                        qm_active_bytes(ld.mlp_down_proj);

        if (!ctx.pad_to(STRATA_QWEN38_PAGE_BYTES)) return cleanup_fail();
        ld.layer_page_bytes = ctx.write_off - ld.layer_page_off;

        if (((l + 1u) % 16u) == 0u || l + 1u == mcfg.num_hidden_layers) {
            std::fprintf(stdout, "[qwen38-convert] Streamed layers 0..%u / %u (written %.2f GiB, RSS %.3f GiB)\n",
                         l, mcfg.num_hidden_layers,
                         static_cast<double>(ctx.write_off) / (1024.0 * 1024.0 * 1024.0),
                         current_peak_rss_gib());
            std::fflush(stdout);
        }
    }

    /* 6. Stream `model.norm` + `lm_head` (`[248320, 5120]` 4-bit affine) */
    hdr.head_page_off = ctx.write_off;
    if (!ctx.write_f32_slice("model.norm.weight", D, D, D, 1, hdr.final_norm)) {
        return cleanup_fail();
    }
    active_bytes += hdr.final_norm.bytes;

    if (mcfg.tie_word_embeddings) {
        hdr.lm_head = hdr.embed_tokens;
    } else {
        if (!ctx.write_quant_matrix("lm_head", mcfg.vocab_size, D, hdr.lm_head)) {
            return cleanup_fail();
        }
    }
    active_bytes += qm_active_bytes(hdr.lm_head);

    if (!ctx.pad_to(STRATA_QWEN38_PAGE_BYTES)) return cleanup_fail();
    hdr.head_page_bytes      = ctx.write_off - hdr.head_page_off;
    hdr.total_file_bytes     = ctx.write_off;
    hdr.active_bytes_per_tok = active_bytes;

    /* 7. Stamp `StrataQwen38WeightsHeader` at offset 0 and atomically rename */
    if (!pwrite_exact(ctx.out_fd, &hdr, sizeof(hdr), 0)) return cleanup_fail();
    close(ctx.out_fd);
    ctx.out_fd = -1;
    for (int ofd : ctx.shard_fds) close(ofd);
    ctx.shard_fds.clear();

    if (rename(tmp_weights_path.c_str(), final_weights_path.c_str()) != 0) {
        unlink(tmp_weights_path.c_str());
        return -1;
    }

    /* 8. Copy `tokenizer.json`, `tokenizer_config.json`, `config.json`, `chat_template.jinja` */
    if (!copy_file_if_exists(in_dir + "/tokenizer.json", out_dir + "/tokenizer.json", true) ||
        !copy_file_if_exists(in_dir + "/tokenizer_config.json", out_dir + "/tokenizer_config.json", false) ||
        !copy_file_if_exists(in_dir + "/config.json", out_dir + "/config.json", false) ||
        !copy_file_if_exists(in_dir + "/chat_template.jinja", out_dir + "/chat_template.jinja", false)) {
        std::fprintf(stderr, "[qwen38-convert] ERROR: failed to copy tokenizer/config files\n");
        return -1;
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double rss_gib = current_peak_rss_gib();

    /* 9. Write human-readable `manifest.json` */
    {
        std::string mf_path = out_dir + "/manifest.json";
        FILE* mf = std::fopen(mf_path.c_str(), "wb");
        if (mf) {
            std::fprintf(mf,
                "{\n"
                "  \"format\": \"STRATA_QWEN38_V1\",\n"
                "  \"architecture\": \"Qwen3_5ForConditionalGeneration\",\n"
                "  \"num_hidden_layers\": %u,\n"
                "  \"num_linear_layers\": %u,\n"
                "  \"num_full_attn_layers\": %u,\n"
                "  \"hidden_size\": %u,\n"
                "  \"intermediate_size\": %u,\n"
                "  \"vocab_size\": %u,\n"
                "  \"quant_bits\": %u,\n"
                "  \"quant_group_size\": %u,\n"
                "  \"page_align_bytes\": %llu,\n"
                "  \"weights_file_bytes\": %llu,\n"
                "  \"active_bytes_per_tok\": %llu,\n"
                "  \"num_shards_read\": %zu,\n"
                "  \"text_tensors_converted\": %u,\n"
                "  \"vision_tensors_skipped\": %u\n"
                "}\n",
                mcfg.num_hidden_layers, mcfg.num_linear_layers, mcfg.num_full_attn_layers,
                mcfg.hidden_size, mcfg.intermediate_size, mcfg.vocab_size,
                mcfg.quant_bits, mcfg.quant_group_size,
                static_cast<unsigned long long>(STRATA_QWEN38_PAGE_BYTES),
                static_cast<unsigned long long>(hdr.total_file_bytes),
                static_cast<unsigned long long>(hdr.active_bytes_per_tok),
                shard_names.size(), ctx.converted_tensors, vision_skipped);
            std::fclose(mf);
        }
    }

    if (stats_out) {
        stats_out->num_shards                 = static_cast<uint32_t>(shard_names.size());
        stats_out->num_text_tensors_converted = ctx.converted_tensors;
        stats_out->num_vision_tensors_skipped = vision_skipped;
        stats_out->hf_safetensors_bytes       = total_hf_bytes;
        stats_out->out_weights_bytes          = hdr.total_file_bytes;
        stats_out->active_bytes_per_tok       = hdr.active_bytes_per_tok;
        stats_out->elapsed_sec                = elapsed;
        stats_out->peak_rss_gib               = rss_gib;
        stats_out->page_aligned_verified      =
            (hdr.embed_page_off == STRATA_QWEN38_HEADER_PADDED_BYTES) &&
            ((hdr.total_file_bytes % STRATA_QWEN38_PAGE_BYTES) == 0ULL);
    }

    std::fprintf(stdout,
                 "[qwen38-convert] DONE in %.2f s: %u text tensors (%u vision skipped) -> %.2f GiB (Peak RSS %.3f GiB)\n",
                 elapsed, ctx.converted_tensors, vision_skipped,
                 static_cast<double>(hdr.total_file_bytes) / (1024.0 * 1024.0 * 1024.0),
                 rss_gib);
    std::fflush(stdout);
    return 0;
}

int strata_qwen38_load_bundle(
    const char* bundle_dir,
    StrataQwen38Bundle* bundle_out
) {
    if (!bundle_dir || !bundle_out) return -1;
    std::memset(bundle_out, 0, sizeof(*bundle_out));
    bundle_out->mmap_fd = -1;

    std::snprintf(bundle_out->bundle_dir, sizeof(bundle_out->bundle_dir), "%s", bundle_dir);
    std::snprintf(bundle_out->weights_path, sizeof(bundle_out->weights_path), "%s/weights.strata", bundle_dir);

    int fd = open(bundle_out->weights_path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[qwen38-load] ERROR: cannot open '%s'\n", bundle_out->weights_path);
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 ||
        static_cast<uint64_t>(st.st_size) < STRATA_QWEN38_HEADER_PADDED_BYTES ||
        (static_cast<uint64_t>(st.st_size) % STRATA_QWEN38_PAGE_BYTES) != 0ULL) {
        std::fprintf(stderr, "[qwen38-load] ERROR: invalid or unaligned weights.strata size\n");
        close(fd);
        return -1;
    }

    size_t map_len = static_cast<size_t>(st.st_size);
    void* mapped = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED ||
        (reinterpret_cast<uintptr_t>(mapped) % STRATA_QWEN38_PAGE_BYTES) != 0u) {
        if (mapped != MAP_FAILED) munmap(mapped, map_len);
        close(fd);
        return -1;
    }

    const auto* hdr = reinterpret_cast<const StrataQwen38WeightsHeader*>(mapped);
    auto validate_qwen38_hdr = [&]() -> bool {
        if (hdr->magic != STRATA_QWEN38_MAGIC_U64 ||
            hdr->version != STRATA_QWEN38_FORMAT_VERSION ||
            hdr->header_padded_bytes != STRATA_QWEN38_HEADER_PADDED_BYTES ||
            hdr->embed_page_off < STRATA_QWEN38_HEADER_PADDED_BYTES ||
            hdr->total_file_bytes != map_len) {
            return false;
        }
        const auto& c = hdr->cfg;
        if (c.num_hidden_layers == 0U || c.num_hidden_layers > STRATA_QWEN38_MAX_LAYERS ||
            c.hidden_size == 0U || c.hidden_size > 32768U ||
            c.intermediate_size == 0U || c.intermediate_size > 131072U ||
            c.vocab_size == 0U || c.vocab_size > (1U << 22) ||
            c.num_attention_heads == 0U || c.num_attention_heads > 256U ||
            c.num_key_value_heads == 0U || c.num_key_value_heads > c.num_attention_heads ||
            c.head_dim == 0U || c.head_dim > 1024U ||
            c.quant_group_size != STRATA_QWEN38_QUANT_GROUP_SIZE ||
            c.num_eos_token_ids > 4U) {
            return false;
        }
        auto in_rng = [&](uint64_t off, uint64_t len) -> bool {
            return off >= STRATA_QWEN38_HEADER_PADDED_BYTES && off <= map_len && len <= map_len - off;
        };
        auto chk_sl = [&](const StrataQwen38TensorSlice& s, uint32_t exp_dtype, uint64_t exp_rows, uint64_t exp_cols) -> bool {
            if (s.dtype != exp_dtype) return false;
            if (s.shape0 == 0U || static_cast<uint64_t>(s.shape0) != exp_rows) return false;
            const uint64_t s1 = (s.shape1 > 0U) ? static_cast<uint64_t>(s.shape1) : 1ULL;
            if (s1 != exp_cols) return false;
            const uint64_t bpe = (s.dtype == STRATA_QWEN38_DTYPE_F32 || s.dtype == STRATA_QWEN38_DTYPE_U32) ? 4ULL : 2ULL;
            if (static_cast<uint64_t>(s.shape0) > (map_len / bpe) / s1) return false;
            const uint64_t need = static_cast<uint64_t>(s.shape0) * s1 * bpe;
            return need > 0ULL && s.bytes >= need && in_rng(s.offset, s.bytes);
        };
        auto chk_q4 = [&](const StrataQwen38QuantMatrix& m, uint64_t exp_rows, uint64_t exp_cols) -> bool {
            if (exp_rows == 0ULL || exp_cols == 0ULL || exp_rows > 0xFFFFFFFFULL || exp_cols > 0xFFFFFFFFULL) return false;
            if (m.group_size != STRATA_QWEN38_QUANT_GROUP_SIZE) return false;
            if (m.cols == 0U || (m.cols % m.group_size) != 0U || (m.cols & 7U) != 0U) return false;
            if (static_cast<uint64_t>(m.rows_logical) != exp_rows || static_cast<uint64_t>(m.cols) != exp_cols) return false;
            const uint64_t exp_rp = (exp_rows + 63ULL) & ~63ULL;
            if (static_cast<uint64_t>(m.rows_padded) != exp_rp) return false;
            const uint64_t min_w = (static_cast<uint64_t>(m.rows_padded) * m.cols) / 2ULL;
            const uint64_t min_s = static_cast<uint64_t>(m.rows_padded) * (m.cols / m.group_size) * 2ULL;
            if (m.weight_bytes < min_w || m.scales_bytes < min_s || m.biases_bytes < min_s) return false;
            return in_rng(m.weight_off, m.weight_bytes) &&
                   in_rng(m.scales_off, m.scales_bytes) &&
                   in_rng(m.biases_off, m.biases_bytes);
        };
        const uint64_t d = c.hidden_size;
        if (!in_rng(hdr->embed_page_off, hdr->embed_page_bytes) || !chk_q4(hdr->embed_tokens, c.vocab_size, d) ||
            !in_rng(hdr->head_page_off, hdr->head_page_bytes) ||
            !chk_sl(hdr->final_norm, STRATA_QWEN38_DTYPE_F32, d, 1ULL) ||
            !chk_q4(hdr->lm_head, c.vocab_size, d)) {
            return false;
        }
        uint32_t n_lin = 0U, n_full = 0U;
        for (uint32_t l = 0U; l < c.num_hidden_layers; ++l) {
            const auto& ld = hdr->layers[l];
            if (!in_rng(ld.layer_page_off, ld.layer_page_bytes) ||
                !chk_sl(ld.input_layernorm, STRATA_QWEN38_DTYPE_F32, d, 1ULL) ||
                !chk_sl(ld.post_attention_layernorm, STRATA_QWEN38_DTYPE_F32, d, 1ULL) ||
                !chk_q4(ld.mlp_gate_proj, c.intermediate_size, d) ||
                !chk_q4(ld.mlp_up_proj, c.intermediate_size, d) ||
                !chk_q4(ld.mlp_down_proj, d, c.intermediate_size)) {
                return false;
            }
            if (ld.layer_type == STRATA_QWEN38_LAYER_LINEAR_ATTN) {
                ++n_lin;
                if (c.linear_num_key_heads == 0U || c.linear_num_key_heads > 256U ||
                    c.linear_key_head_dim == 0U || c.linear_key_head_dim > 1024U ||
                    c.linear_num_value_heads == 0U || c.linear_num_value_heads > 256U ||
                    c.linear_value_head_dim == 0U || c.linear_value_head_dim > 1024U ||
                    c.linear_conv_kernel_dim == 0U || c.linear_conv_kernel_dim > 64U) return false;
                const uint64_t k_dim    = static_cast<uint64_t>(c.linear_num_key_heads) * c.linear_key_head_dim;
                const uint64_t v_dim    = static_cast<uint64_t>(c.linear_num_value_heads) * c.linear_value_head_dim;
                const uint64_t conv_dim = 2ULL * k_dim + v_dim;
                const uint64_t nv       = c.linear_num_value_heads;
                const uint64_t nv_pad   = (nv + 63ULL) & ~63ULL;
                if (!chk_q4(ld.lin_in_proj_qkv, conv_dim, d) || !chk_q4(ld.lin_in_proj_z, v_dim, d) ||
                    !chk_q4(ld.lin_in_proj_a, nv, d) || !chk_q4(ld.lin_in_proj_b, nv, d) ||
                    !chk_q4(ld.lin_out_proj, d, v_dim) ||
                    !chk_sl(ld.lin_conv1d_weight, STRATA_QWEN38_DTYPE_F32, conv_dim, c.linear_conv_kernel_dim) ||
                    !chk_sl(ld.lin_A_log, STRATA_QWEN38_DTYPE_F32, nv_pad, 1ULL) ||
                    !chk_sl(ld.lin_dt_bias, STRATA_QWEN38_DTYPE_F32, nv_pad, 1ULL) ||
                    !chk_sl(ld.lin_norm, STRATA_QWEN38_DTYPE_F32, c.linear_value_head_dim, 1ULL)) {
                    return false;
                }
            } else if (ld.layer_type == STRATA_QWEN38_LAYER_FULL_ATTN) {
                ++n_full;
                const uint64_t q_out  = static_cast<uint64_t>(c.num_attention_heads) * c.head_dim * (c.attn_output_gate ? 2ULL : 1ULL);
                const uint64_t kv_out = static_cast<uint64_t>(c.num_key_value_heads) * c.head_dim;
                const uint64_t o_in   = static_cast<uint64_t>(c.num_attention_heads) * c.head_dim;
                if (!chk_q4(ld.attn_q_proj, q_out, d) || !chk_q4(ld.attn_k_proj, kv_out, d) ||
                    !chk_q4(ld.attn_v_proj, kv_out, d) || !chk_q4(ld.attn_o_proj, d, o_in) ||
                    !chk_sl(ld.attn_q_norm, STRATA_QWEN38_DTYPE_F32, c.head_dim, 1ULL) ||
                    !chk_sl(ld.attn_k_norm, STRATA_QWEN38_DTYPE_F32, c.head_dim, 1ULL)) {
                    return false;
                }
            } else {
                return false;
            }
        }
        if (c.num_linear_layers != n_lin || c.num_full_attn_layers != n_full) return false;
        return true;
    };
    if (!validate_qwen38_hdr()) {
        std::fprintf(stderr, "[qwen38-load] ERROR: header magic/version/offset/size mismatch\n");
        munmap(mapped, map_len);
        close(fd);
        return -1;
    }

    std::string tok_path = std::string(bundle_dir) + "/tokenizer.json";
    StrataQwen38Tokenizer* tok = strata_qwen38_tokenizer_load(tok_path.c_str());
    if (!tok) {
        munmap(mapped, map_len);
        close(fd);
        return -1;
    }

    bundle_out->mmap_fd    = fd;
    bundle_out->mmap_base  = static_cast<const uint8_t*>(mapped);
    bundle_out->mmap_bytes = map_len;
    bundle_out->header     = hdr;
    bundle_out->cfg        = hdr->cfg;
    bundle_out->tokenizer  = tok;
    return 0;
}

void strata_qwen38_free_bundle(StrataQwen38Bundle* bundle) {
    if (!bundle) return;
    if (bundle->tokenizer) {
        strata_qwen38_tokenizer_free(bundle->tokenizer);
        bundle->tokenizer = nullptr;
    }
    if (bundle->mmap_base && bundle->mmap_bytes > 0) {
        munmap(const_cast<uint8_t*>(bundle->mmap_base), bundle->mmap_bytes);
    }
    if (bundle->mmap_fd >= 0) {
        close(bundle->mmap_fd);
    }
    std::memset(bundle, 0, sizeof(*bundle));
    bundle->mmap_fd = -1;
}

} /* extern "C" */
